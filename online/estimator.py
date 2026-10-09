from datetime import datetime
import json
from pathlib import Path
import numpy as np
from project_paths import OUTPUTS
from .session import ControlSession
from .simulation import Simulation
from .onnx import OnlineOnnxModel, OnnxReadout
from .worker import OnlineWindow, OnlineWorker

class OnlineSession(ControlSession):
    def __init__(self, args):
        self.args = args
        self.sim = Simulation(
            getattr(args, "scenario", "stand"),
            getattr(args, "walk_speed", None),
            lie_seed=getattr(args, "lie_seed", 182),
            lie_randomize=not getattr(args, "lie_nominal", False),
            lie_elbow_angle=getattr(args, "lie_elbow_angle", 1.2),
            motion=getattr(args, "motion", 1),
            dance=getattr(args, "dance", None),
        )
        self.predictor = OnlineOnnxModel(threads=getattr(args, "onnx_threads", 4),
            strategy=getattr(args, "sampling_strategy", "adaptive"),
            candidates=getattr(args, "candidate_count", 4),
            seed=getattr(args, "sampling_seed", None))
        self.stride, self.lookahead = self.predictor.stride, self.predictor.lag
        self.links = self.predictor.links
        self.bodies = [self.sim.model.body(name).id for name in self.links]
        self.body_to_link = {body: i for i, body in enumerate(self.bodies)}
        self.worker = None if args.headless else OnlineWorker(self.predictor)
        self.generation = -1
        self.samples, self.predictions, self.window_records = [], [], []
        self.ignored_old_results = 0
        self.stop_events = []
        self.walk_stop_delay = float(getattr(args, "walk_stop_delay", 0.0))
        self.reset()

    def reset(self):
        ControlSession.reset(self)
        if self.worker is not None:
            self.worker.reset()
        self.input_samples = []
        self.fusion = OnnxReadout(self.predictor)
        self.last_prediction = self.last_observation_time = None
        self.window_number = 0
        self.pending_walk_stop = None

    def request_walk_stop(self, event):
        if (
            self.sim.scenario != "walk"
            or self.sim.walk_stopped
            or getattr(self, "pending_walk_stop", None) is not None
        ):
            return
        delay = getattr(self, "walk_stop_delay", 0.0)
        self.pending_walk_stop = dict(
            event, scheduled_stop_time=float(self.sim.data.time) + delay, requested_delay_s=delay
        )
        self.apply_pending_walk_stop()

    def apply_pending_walk_stop(self):
        event = getattr(self, "pending_walk_stop", None)
        if event is None:
            return
        if event["generation"] != self.generation:
            self.pending_walk_stop = None
            return
        now = float(self.sim.data.time)
        if now + 1e-8 < event["scheduled_stop_time"]:
            return
        self.pending_walk_stop = None
        if self.sim.stop_walk():
            event.update(command_zero_time=now, command_after=self.sim.command.tolist())
            self.stop_events.append(event)
            if self.args.headless:
                print(
                    f"WALK STOP: estimated contact at t={event['observation_time']:.2f}s; cmd zero at t={now:.2f}s. R resets.",
                    flush=True,
                )

    @property
    def ready(self):
        warmed = self.sim.data.time >= 3.0 - 1e-8 and len(self.input_samples) >= 50
        return warmed and (self.sim.scenario != "lie" or self.sim.lie_readiness.ready)

    def observe_sample(self, sample):
        if sample.time < 2.0 - 1e-8:
            return
        if sample.sensors.shape != (96,) or not np.isfinite(sample.sensors).all():
            raise ValueError("Online input must contain 96 finite robot signals")
        if (
            self.last_observation_time is not None
            and abs(sample.time - self.last_observation_time - 0.02) > 1e-7
        ):
            raise ValueError("Observation gap; reset before continuing")
        self.last_observation_time = sample.time
        self.input_samples.append(sample)
        count = len(self.input_samples)
        if count < 50 or (count - 50) % self.stride:
            return
        window = OnlineWindow(
            self.generation,
            self.window_number,
            np.arange(count - 50, count),
            np.stack([s.sensors for s in self.input_samples[-50:]]),
        )
        self.window_number += 1
        if self.worker is None:
            self.accept_online((window, self.predictor.predict_online(window)))
        else:
            self.worker.submit(window)

    def accept_online(self, packet):
        window, result = packet
        if window.generation != self.generation:
            self.ignored_old_results += 1
            return
        rows = self.fusion.accept(window, result)
        received = float(self.sim.data.time)
        for row in rows:
            sample = self.input_samples[row.pop("index")]
            event_sample = self.input_samples[row["event_source_window_end"]]
            row.update(
                event_target_time=event_sample.time,
                event_link=int(result["sequence"]["link"][-1]),
                event_force=np.asarray(result["sequence"]["force"][-1]).tolist(),
            )

            if self.ready and row["event_active"]:
                event = dict(
                    generation=self.generation,
                    received_sim_time=received,
                    observation_time=event_sample.time,
                    window_number=window.number,
                    estimated_link=row["event_link"],
                    estimated_force_N=row["event_force"],
                    reason="estimated_latest_frame_contact",
                )
                self.request_walk_stop(event)

            row.update(
                generation=self.generation,
                window_number=window.number,
                target_time=sample.time,
                received_sim_time=received,
                compute_ms=result["compute_ms"],
                sampling_strategy=result.get("sampling_strategy", "baseline"),
                candidate_count=result.get("candidate_count", 1),
                selected_candidate=result.get("selected_candidate", 0),
                extra_sampling_reason=result.get("extra_sampling_reason", "disabled"),
                true_force=sample.force.tolist(),
                true_torque=sample.torque.tolist(),
                true_wrench=np.concatenate([sample.force, sample.torque]).tolist(),
                true_link=sample.link,
            )
            self.predictions.append(row)
            self.last_prediction = row
        self.window_records.append(
            dict(
                generation=window.generation,
                number=window.number,
                start_time=self.input_samples[window.indices[0]].time,
                end_time=self.input_samples[window.indices[-1]].time,
                received_sim_time=received,
                compute_ms=result["compute_ms"],
                sampling_strategy=result.get("sampling_strategy", "baseline"),
                candidate_count=result.get("candidate_count", 1),
                selected_candidate=result.get("selected_candidate", 0),
                extra_sampling_reason=result.get("extra_sampling_reason", "disabled"),
                input_sha256=result["input_sha256"],
                emitted_frames=len(rows),
            )
        )

    def poll_inference(self):
        if self.worker is not None:
            for packet in self.worker.drain():
                self.accept_online(packet)
        self.apply_pending_walk_stop()

    def display_prediction(self, now):
        row = self.last_prediction
        if row is not None and now - row["received_sim_time"] > 1.0:
            return None
        return row

    def save(self):
        output = Path(
            self.args.output
            or OUTPUTS
            / "online"
            / (self.predictor.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".npz")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        samples = [sample for _, sample in self.samples]
        metadata = dict(
            mode=self.predictor.mode,
            scenario=self.sim.scenario,
            tracking_motion=(self.sim.motion_manifest[self.sim.motion_index] if self.sim.scenario == "tracking" else None),
            model_sha256=self.predictor.model_sha256,
            sampling_seed=self.predictor.sampling_seed,
            sampling_policy=("one fresh Gaussian source per window; no smoothing" if self.predictor.strategy == "baseline" else "complete candidate selection by consensus and causal timestamp-aligned history"),
            sampling_strategy=self.predictor.strategy,
            candidate_count=self.predictor.candidate_count,
            device=self.predictor.device,
            onnx_threads=self.predictor.session.get_session_options().intra_op_num_threads,
            target_links=self.links,
            force_frame="pelvis",
            torque_frame="pelvis",
            moment_origin="link body origin",
            wrench_components=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            input_signals=[
                "IMU",
                "joint_position",
                "joint_velocity",
                "software_motor_output_torque",
            ],
            observation_rate_hz=50,
            window_frames=50,
            stride_frames=self.stride,
            lookahead_frames=self.lookahead,
            predictions=self.predictions,
            windows=self.window_records,
            ignored_old_results=self.ignored_old_results,
            dropped_windows=self.worker.dropped_windows if self.worker else 0,
        )
        metadata.update(
            scenario=self.sim.scenario,
            controller="joint damping tau=-8*qdot"
            if self.sim.scenario == "lie"
            else ("Tracking ONNX + PD; 50 Hz; reference first-frame initialization" if self.sim.scenario == "tracking" else "shared locomotion ONNX"),
            initial_walk_speed_m_s=self.sim.walk_speed if self.sim.scenario == "walk" else 0.0,
            walk_stop_source="latest estimated contact; no reference force/link input",
            walk_stop_delay_s=getattr(self, "walk_stop_delay", 0.0),
            walk_stop_delay_reference="estimated-event receipt in simulation time; physical contact-to-stop also includes detector latency",
            walk_stop_latched_until_reset=True,
            walk_stop_events=self.stop_events,
            stop_event_readout="latest frame; separate from 0.5 s delayed plotted force",
            pose_limit_reached=bool(self.fallen),
        )
        metadata.update(self.log_metadata())
        if self.sim.scenario == "lie":
            from .lie_settling import PROTOCOL

            metadata.update(
                lie_protocol=dict(
                    PROTOCOL,
                    name="lie_deploy",
                    joint_reference="lie_nominal_qpos from deployment initializer",
                    original_settling_protocol=PROTOCOL["name"],
                ),
                lie_seed=self.sim.lie_seed,
                lie_randomized=self.sim.lie_randomize,
                lie_elbow_angle_rad=self.sim.lie_elbow_angle,
                lie_initial_qpos=self.sim.lie_initial_qpos.tolist(),
                lie_nominal_qpos=self.sim.lie_nominal_qpos.tolist(),
                lie_readiness=self.sim.lie_readiness.metadata(),
                lie_initialization_note="Bent elbows with bounded pose jitter.",
                lie_readiness_scope="Current reset generation; only controls pulling, never gates predictions",
            )
        metadata.update(
            direct_output=True,
            confidence_output=False,
            output_semantics="contact_active/link/force/torque/wrench share index; event_active uses event_source_window_end",
        )
        np.savez_compressed(
            output,
            metadata=json.dumps(metadata),
            generation=np.array([generation for generation, _ in self.samples]),
            time_s=np.array([s.time for s in samples]),
            qpos=np.array([s.qpos for s in samples]),
            command=np.array([s.command for s in samples]),
            base_velocity_pelvis_m_s=np.array([s.base_velocity for s in samples]),
            robot_observations=np.array([s.sensors for s in samples]),
            true_link=np.array([s.link for s in samples]),
            true_force_pelvis_N=np.array([s.force for s in samples]),
            true_torque_pelvis_Nm=np.array([s.torque for s in samples]),
            true_wrench_pelvis_N_Nm=np.array([np.concatenate([s.force, s.torque]) for s in samples]),
            point_link_m=np.array([s.point for s in samples]),
        )
        print(
            f"Saved {output} ({len(samples)} observations, {len(self.predictions)} online estimates)",
            flush=True,
        )
        return output
