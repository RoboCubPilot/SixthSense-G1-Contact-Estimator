from queue import SimpleQueue
from pathlib import Path
import argparse
import time
import glfw
import mujoco
import mujoco.viewer
import numpy as np
from online.display import prediction_arrows, applied_pull_handle, base_speed_label, torque_arrow, RED
from online.estimator import OnlineSession
from project_paths import SIM_SCENARIO_NAMES, normalize_sim_scenario
from online.plot import OnlinePlot

class NativeSession(OnlineSession):
    def __init__(self, args):
        self.handle = None
        self.played_motions = []
        self.elapsed_simulation = 0.0
        self.events = SimpleQueue()
        self.native_wrench = False
        self.wrench_log = []
        self.random_contact_log = []
        self.random_contact_rng = np.random.default_rng(getattr(args, "contact_seed", 42))
        self.plot = OnlinePlot()
        self.torque_plot = OnlinePlot(component="torque")
        self.show_plot = not getattr(args, "hide_plot", False)
        super().__init__(args)
        label = self.predictor.config.get("display_name", self.predictor.name)
        self.plot.title = f"{label} | {self.sim.scenario.upper()} | force estimate"
        self.torque_plot.title = f"{label} | {self.sim.scenario.upper()} | torque estimate"

    def camera_height(self):
        if self.sim.scenario == "lie":
            return -0.18 if self.show_plot else 0.2
        return 0.45 if self.show_plot else 0.8

    def interaction_help(self):
        return (
            "Double-click a body, then Ctrl + RIGHT drag to apply force. F: test push, R: reset, S: save snapshot, Space: pause, Q: quit.\n" +
            ("Tracking: 1-6 choose motion; D: next dance.\n" if self.sim.scenario == "tracking" else "") + f"50 Hz IMU/q/dq/motor effort -> {self.predictor.config.get('display_name', self.predictor.name)} ONNX -> event + fixed-lag link/wrench."
        )

    def reset(self):
        user_paused = bool(getattr(self, "paused", False)) and not getattr(self, "fallen", False)
        super().reset()
        if self.handle is not None:
            self.paused = user_paused
        self.next_random_contact = 4.0
        if getattr(self.args, "random_contacts", False):
            self.push_rng = self.random_contact_rng
        self.native_wrench = False
        self.applied_force_display = None
        self.camera_base_xy = self.sim.data.qpos[:2].copy()
        if self.sim.scenario == "tracking":
            self.played_motions.append(dict(generation=self.generation, **self.sim.motion_manifest[self.sim.motion_index]))
        if self.handle is not None:
            self.handle.perturb.active = self.handle.perturb.active2 = 0
            self.handle.perturb.select = 0
            self.handle.cam.lookat[:] = [*self.sim.data.qpos[:2], self.camera_height()]

    def key(self, key):
        if self.sim.scenario == "tracking" and glfw.KEY_1 <= key < glfw.KEY_1 + len(self.sim.motion_manifest):
            self.select_motion(key - glfw.KEY_1)
        elif self.sim.scenario == "tracking" and key == glfw.KEY_D and self.sim.dances:
            self.select_dance((self.sim.dance_index + 1) % len(self.sim.dances))
        elif key == glfw.KEY_P:
            self.show_plot = not self.show_plot
        else:
            super().key(key)

    def step(self):
        if (getattr(self.args, "random_contacts", False) and self.ready
                and not self.paused and not self.fallen
                and self.sim.data.time >= self.next_random_contact):
            self.push()
            rng = self.random_contact_rng
            angle = rng.uniform(-np.pi, np.pi)
            self.test_push_direction = [float(np.cos(angle)), float(np.sin(angle)), 0.0]
            self.test_push_force = float(rng.uniform(20.0, 45.0))
            self.test_push_duration = float(rng.uniform(0.25, 0.55))
            self.next_random_contact = self.sim.data.time + self.test_push_duration + rng.uniform(1.5, 2.5)
            event = dict(generation=self.generation, time_s=float(self.sim.data.time),
                         motion_id=self.sim.motion_index + 1 if self.sim.scenario == "tracking" else None,
                         link=self.sim.model.body(self.push_body).name,
                         direction=self.test_push_direction, frame=self.args.push_frame,
                         peak_force_N=self.test_push_force, duration_s=self.test_push_duration)
            self.random_contact_log.append(event)
            print(f"RANDOM CONTACT: {event}", flush=True)
        super().step()
        self.elapsed_simulation += self.sim.model.opt.timestep
        if self.sim.scenario == "tracking" and not self.fallen and self.sim.tick >= int(self.sim.twist2.motion.duration / self.sim.model.opt.timestep):
            if self.args.headless and self.elapsed_simulation >= self.args.seconds - 1e-7:
                return
            if getattr(self.args, "dance_playlist", False):
                self.select_dance((self.sim.dance_index + 1) % len(self.sim.dances))
            elif not self.args.single_motion:
                self.select_motion((self.sim.motion_index + 1) % len(self.sim.motion_manifest))
            else:
                self.reset()

    def run_headless(self):
        if self.sim.scenario != "tracking":
            return super().run_headless()
        try:
            while self.elapsed_simulation < self.args.seconds and not self.fallen:
                self.step()
        finally:
            self.finish_inference()
            self.save()
        if self.fallen:
            raise RuntimeError("Tracking rollout fell before completion")

    def select_motion(self, index):
        from .twist2 import Motion, ROOT
        self.sim.motion_index = index
        item = self.sim.motion_manifest[index]
        if "dance_id" in item:
            self.sim.dance_index = item["dance_id"] - 1
        self.sim.twist2.motion = Motion(ROOT / "motions" / item["file"])
        self.sim.twist2.reference_velocity = item.get("reference_velocity", False)
        print(f"Tracking {index+1}/{len(self.sim.motion_manifest)}: {item["label"]}", flush=True)
        self.reset()

    def select_dance(self, index):
        self.sim.dance_index = index
        self.sim.motion_manifest[2] = dict(self.sim.dances[index], id=3)
        self.select_motion(2)

    def push_readout(self):
        if self.push_start is None:
            if self.sim.scenario == "tracking":
                mode = "dance playlist" if getattr(self.args, "dance_playlist", False) else ("repeat selected" if self.args.single_motion else "motion playlist")
                return f"Tracking {self.sim.motion_index+1}/{len(self.sim.motion_manifest)}: {self.sim.motion_manifest[self.sim.motion_index]["label"]} ({self.sim.twist2.motion.duration:.1f}s)\n1-6: choose motion | D: next dance | {mode}"
            return ""
        link, force, _ = self.current
        selected = self.body_to_link[self.push_body]
        magnitude = np.linalg.norm(force) if link == selected else 0.0
        duration = self.test_push_duration
        active = 0 <= self.sim.data.time - self.push_start < duration
        state = ("PUSH ACTIVE" if active else "RECOVERING") if getattr(self.args, "random_contacts", False) else ("F PUSH" if active else "LAST F PUSH")
        return (
            f"{state}\n"
            f"LINK: {self.sim.model.body(self.push_body).name}\n"
            f"FORCE NOW: {magnitude:.1f} N\n"
            f"PULSE: {duration:.2f} s"
        )

    def external(self):
        self.native_wrench = False
        self.applied_force_display = None
        if self.handle is not None and self.ready:
            pert, d = self.handle.perturb, self.sim.data
            selected = int(pert.select)
            mapped = self.select(selected)
            if mapped > 0 and (pert.active or pert.active2):
                mujoco.mjv_applyPerturbForce(self.sim.model, d, pert)
                self.native_wrench = True
                world = d.xpos[selected] + d.xmat[selected].reshape(3, 3) @ pert.localpos
                point = d.xmat[mapped].reshape(3, 3).T @ (world - d.xpos[mapped])
                if self.sim.tick % self.sim.config["control_decimation"] == 0:
                    self.wrench_log.append(
                        dict(
                            generation=self.generation,
                            time=float(d.time),
                            body_id=selected,
                            world_force_and_com_torque=d.xfrc_applied[selected].tolist(),
                            rotational=bool(
                                (pert.active | pert.active2) & int(mujoco.mjtPertBit.mjPERT_ROTATE)
                            ),
                        )
                    )
                return mapped, point, d.xfrc_applied[selected, :3].copy()
        return super().external()

    def apply_external(self, body, point, world_force):
        self.applied_force_display = (body, point.copy(), world_force.copy())
        if self.native_wrench:
            if self.sim.scenario == "lie" and np.linalg.norm(self.sim.data.xfrc_applied) > 1e-9:
                self.sim.lie_readiness.start_force(float(self.sim.data.time))

            return self.sim.data.xmat[self.sim.base].reshape(3, 3).T @ world_force
        return super().apply_external(body, point, world_force)

    def log_metadata(self):
        return dict(
            tracking_played_motions=self.played_motions,
            random_contacts=self.random_contact_log,
            random_contact_seed=getattr(self.args, "contact_seed", 42),
            viewer_type="mujoco_native_passive",
            native_wrench_samples=self.wrench_log,
            event_gate=dict(enabled=False, reason="direct contact class; no confidence gating"),
            estimate_plot="continuous true/estimated magnitudes at original target timestamps; no replay queue",
            arrows="latest finalized online wrench at each current predicted-link body origin; solid force and dashed curved torque",
            true_force_display="red MuJoCo-style capsule and endpoint sphere at actual force application point; current force, 0.015 m/N capped at 0.8 m; native mouse drags retain their native handle",
            true_torque_display="red right-hand curved arrow at the true link body origin; current moment in pelvis axes",
            native_torque_note="COM perturbation moment is shifted to the mapped link body origin; estimator force and torque use pelvis axes",
        )

    def external_torque(self, body, point, world_force):
        if not self.native_wrench:
            return super().external_torque(body, point, world_force)
        d = self.sim.data
        forces = d.xfrc_applied[:, :3]
        moments = d.xfrc_applied[:, 3:] + np.cross(d.xipos - d.xpos[body], forces)
        return d.xmat[self.sim.base].reshape(3, 3).T @ moments.sum(0)

    def wrench_readout(self):
        vector = lambda v: '[' + ', '.join(f'{x:+.2f}' for x in v) + ']'
        link, force, _ = self.current
        real_link = self.links[link] if link >= 0 else 'none'
        text = f"TRUE NOW: {real_link}\nF {vector(force)} N\nT {vector(self.current_torque)} N.m"
        row = self.display_prediction(float(self.sim.data.time))
        if row is not None:
            predicted_link = self.links[row['link']] if row['link'] >= 0 else 'none'
            text += f"\nEST t={row['target_time']:.2f}s: {predicted_link}\nF {vector(row['force'])} N\nT {vector(row['torque'])} N.m"
        return text

    def update_scene(self):
        h, d = self.handle, self.sim.data
        h.cam.lookat[:2] += d.qpos[:2] - self.camera_base_xy
        self.camera_base_xy = d.qpos[:2].copy()
        h.user_scn.ngeom = 0
        if self.sim.scenario == "walk":
            base_speed_label(h.user_scn, d, self.sim.base, float(self.sim.command[0]))

        if self.applied_force_display is not None and not self.native_wrench:
            applied_pull_handle(h.user_scn, self.sim.model, d, *self.applied_force_display)
        link = self.current[0]
        if link >= 0:
            torque_arrow(h.user_scn, d, self.sim.base, self.bodies[link], self.current_torque, RED)
        e = self.display_prediction(float(d.time))
        prediction_arrows(h.user_scn, d, self.sim.base, self.bodies, e)

    def run(self):
        if self.args.headless:
            return self.run_headless()
        print(self.interaction_help(), flush=True)
        h = mujoco.viewer.launch_passive(
            self.sim.model,
            self.sim.data,
            key_callback=self.events.put,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.handle = h
        if not hasattr(h, "set_figures"):
            self.show_plot = False
            print("Bottom plot requires MuJoCo >= 3.14; install requirements.txt.", flush=True)
        with h.lock():
            h.cam.lookat[:] = [*self.sim.data.qpos[:2], self.camera_height()]
            h.cam.distance, h.cam.azimuth, h.cam.elevation = (
                3.0 if self.show_plot else 2.7,
                135,
                -15,
            )
            if self.sim.scenario == "lie":
                h.cam.distance, h.cam.elevation = 2.9, -50
            h.opt.geomgroup[0] = 0
            h.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1
            h.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            self.sim.model.geom_group[self.sim.model.geom("floor").id] = 2
        start = previous = last_render = time.perf_counter()
        last_plot = -float("inf")
        budget, last_sim_time = 0.0, 0.0
        should_quit = False
        try:
            while h.is_running() and not should_quit:
                now = time.perf_counter()
                if self.args.seconds and now - start >= self.args.seconds:
                    break
                elapsed, previous = now - previous, now
                with h.lock():
                    while not self.events.empty():
                        key = self.events.get()
                        if key == glfw.KEY_Q:
                            should_quit = True
                        elif key == glfw.KEY_BACKSPACE:
                            self.reset()
                        else:
                            self.key(key)
                    if self.sim.data.time < last_sim_time - 1e-6 and self.sim.tick != 0:
                        self.reset()
                    budget = min(0.1, budget + elapsed) if not self.paused else 0.0
                    while budget >= self.sim.model.opt.timestep and not self.paused:
                        self.step()
                        budget -= self.sim.model.opt.timestep
                    if self.fallen:
                        h.perturb.active = h.perturb.active2 = 0
                    last_sim_time = float(self.sim.data.time)
                    self.poll_inference()
                    self.update_scene()
                if now - last_render >= 1 / 60:
                    if hasattr(h, "set_figures"):
                        if self.show_plot and h.viewport is not None and now - last_plot >= 0.1:
                            figure = self.plot.update_live(
                                float(self.sim.data.time),
                                self.generation,
                                self.predictions,
                                self.samples,
                            )
                            torque_figure = self.torque_plot.update_live(
                                float(self.sim.data.time), self.generation, self.predictions, self.samples,
                            )
                            h.set_figures([(self.plot.viewport(h.viewport, 0), figure),
                                           (self.plot.viewport(h.viewport, 1), torque_figure)])
                            last_plot = now
                        elif not self.show_plot:
                            h.clear_figures()
                    if hasattr(h, "set_texts"):
                        h.set_texts(
                            [
                                (
                                    mujoco.mjtFontScale.mjFONTSCALE_100,
                                    mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                    "Double-click body, then Ctrl + right drag\nF: random push | R: reset | S: save | Space: pause | Q: quit\nP: force / torque plots | Tab: native panels\nPlots: red True, green Estimated\nWrench: pelvis XYZ, each link body origin",
                                    "",
                                ),
                                (
                                    mujoco.mjtFontScale.mjFONTSCALE_100,
                                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                    self.push_readout() + '\n' + self.wrench_readout(),
                                    "",
                                ),
                            ]
                        )
                    h.sync()
                    last_render = now
                time.sleep(0.001)
        finally:
            try:
                self.finish_inference()
            finally:
                h.close()
                self.handle = None
                self.save()

def native_arguments():
    parser = argparse.ArgumentParser(description='Interactive MuJoCo viewer; launch with python sixthsense.py sim stand/walk/lie/tracking.')
    parser.add_argument("--scenario", type=normalize_sim_scenario, choices=SIM_SCENARIO_NAMES, default="stand")
    parser.add_argument(
        "--walk-speed",
        type=float,
        help="Fixed forward cmd in m/s; default random 0.4–0.9 at startup and reset",
    )
    parser.add_argument(
        "--walk-stop-delay",
        type=float,
        default=0.0,
        help="Seconds after estimated contact receipt before cmd zero (default 0)",
    )
    parser.add_argument(
        "--lie-seed",
        type=int,
        default=182,
        help="Reproducible lie initialization sequence; R samples the next pose",
    )
    parser.add_argument(
        "--lie-nominal",
        action="store_true",
        help="Disable lie pose randomization; settling checks still apply",
    )
    parser.add_argument(
        "--lie-elbow-angle", type=float, default=1.2,
        help="Initial lie elbow bend in radians (0 to 2; default 1.2)",
    )
    parser.add_argument("--motion", type=int, choices=range(1, 7), default=1, help="Tracking motion number (1–6)")
    parser.add_argument("--single-motion", action="store_true", help="Repeat selected Tracking motion instead of the full playlist")
    parser.add_argument("--dance", type=int, help="Select a verified long dance and repeat it")
    parser.add_argument("--dance-playlist", action="store_true", help="Cycle through the verified long dances")
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--onnx-threads", type=int, choices=range(1, 17), default=4,
        help="CPU inference threads (default 4); does not change the model or solver",
    )
    parser.add_argument("--sampling-strategy", choices=["baseline", "history", "adaptive"], default="adaptive", help="Stochastic candidate selection: baseline / scheme 3 / scheme 5 (default)")
    parser.add_argument("--candidate-count", type=int, choices=range(2,9), default=4)
    parser.add_argument("--sampling-seed", type=int, help="Reproducible estimator sampling seed")
    parser.add_argument("--demo-push", action="store_true")
    parser.add_argument("--random-contacts", action="store_true", help="Repeated random-link horizontal force pulses: 20–45 N, 0.25–0.55 s")
    parser.add_argument("--contact-seed", type=int, default=42, help="Random contact sequence seed")
    parser.add_argument(
        "--push-link", help="Fix the F target; default random (forearms, wrists, shins and ankles for lie)"
    )
    parser.add_argument(
        "--push-force", type=float, help="Peak test force N; defaults 35 N for lie, 60 N for stand/walk"
    )
    parser.add_argument(
        "--push-duration", type=float, help="Test pulse seconds; default 0.7, or 1.2 for lie"
    )
    parser.add_argument(
        "--push-direction",
        type=float,
        nargs=3,
        help="Default random forward/backward for stand/walk; outward/upward for lie",
    )
    parser.add_argument(
        "--push-frame",
        choices=["pelvis", "world"],
        help="Custom direction frame; default pelvis, or world for lie",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hide-plot", action="store_true")
    args = parser.parse_args()
    if args.dance is not None or args.dance_playlist:
        if args.scenario != "tracking":
            parser.error("Dance selection requires the tracking scenario")
        from .twist2 import ROOT
        import json
        dances = json.loads((ROOT / "dances.json").read_text())["dances"]
        if args.dance is None:
            args.dance = 1
        if not 1 <= args.dance <= len(dances):
            parser.error(f"Dance number must be between 1 and {len(dances)}")
        args.motion = 3
        args.single_motion = not args.dance_playlist
    if args.sampling_seed is not None and args.sampling_seed < 0:
        parser.error("Sampling seed must be nonnegative")
    if args.push_force is None:
        args.push_force = 35.0 if args.scenario == "lie" else 60.0
    if args.push_duration is None:
        args.push_duration = 1.2 if args.scenario == "lie" else 0.7
    args.push_frame = args.push_frame or ("world" if args.scenario == "lie" else "pelvis")
    direction_valid = args.push_direction is None or (
        np.isfinite(args.push_direction).all() and np.linalg.norm(args.push_direction) >= 1e-6
    )
    if (
        not np.isfinite(
            [
                args.seconds,
                args.push_force,
                args.push_duration,
                args.walk_stop_delay,
                args.lie_elbow_angle,
            ]
        ).all()
        or args.seconds < 0
        or not 0 < args.push_force <= (300 if args.scenario == "lie" else 110)
        or not 0.1 <= args.push_duration <= 5.0
        or (args.walk_speed is not None and
            (not np.isfinite(args.walk_speed) or not 0.3 <= args.walk_speed <= 1.0))
        or not 0 <= args.walk_stop_delay <= 2.0
        or not 0 <= args.lie_elbow_angle <= 2.0
        or not direction_valid
    ):
        parser.error("Invalid duration or push force/direction")
    from contact_model.metadata import load_metadata
    if args.push_link is not None and args.push_link not in load_metadata()["target_links"]:
        parser.error("Test force must select one of the model's 30 target links")
    if args.headless and args.seconds == 0:
        args.seconds = 10.0
        if args.scenario == "tracking":
            from .twist2 import ROOT, Motion
            import json
            motions = json.loads((ROOT / "manifest.json").read_text())["motions"]
            if args.dance is not None:
                dances = json.loads((ROOT / "dances.json").read_text())["dances"]
                motions[2] = dances[args.dance - 1]
            chosen = [motions[args.motion-1]] if args.single_motion else motions
            if args.dance_playlist:
                chosen = dances
            args.seconds = sum(int(Motion(ROOT / "motions" / item["file"]).duration / 0.001) * 0.001 for item in chosen)
    return args

if __name__ == "__main__":
    NativeSession(native_arguments()).run()
