import hashlib
import time
import numpy as np

class SequenceEstimator:
    def predict_sequence_windows(self, raw, ends):
        outputs = []
        for first in range(0, len(ends), 64):
            windows = np.stack([raw[end - 49 : end + 1] for end in ends[first : first + 64]])
            outputs.append(self.predict_windows(windows))
        return {k: np.concatenate([row[k] for row in outputs]) for k in outputs[0]}, {}

    def predict(self, sensors):
        started = time.perf_counter()
        first_draw = getattr(self, "draw_count", 0)
        raw = np.asarray(sensors, np.float32)
        if raw.ndim != 2 or raw.shape[1] != 96 or len(raw) < 50 or not np.isfinite(raw).all():
            raise ValueError("Expected finite [T>=50, 96]")
        stride = getattr(self, "stride", 5)
        ends = np.arange(49, len(raw), stride)
        sequence, sampling_details = self.predict_sequence_windows(raw, ends)
        n = len(raw)
        field = np.zeros((n, 30, 3), np.float32)
        torque_field = np.zeros_like(field)
        valid = np.zeros(n, bool)
        force_valid = np.zeros(n, bool)
        localization = np.full(n, -1, np.int64)
        detected = np.zeros(n, bool)
        event_run = 0
        for i, end in enumerate(ends):
            stop = min(n, end + stride)
            valid[end:stop] = True
            event_run = event_run + 1 if sequence["contact_active"][i, -1] else 0
            detected[end:stop] = event_run >= self.debounce_outputs
            target, offset = end - self.lag, 49 - self.lag
            force_stop = min(n - self.lag, target + stride)
            force_valid[target:force_stop] = True
            length = force_stop - target
            localization[target:force_stop] = sequence["link"][i, offset:offset + length]
            field[target:force_stop] = sequence["force_field"][i, offset:offset + length]
            torque_field[target:force_stop] = sequence["torque_field"][i, offset:offset + length]
        result = dict(
            field=field,
            torque_field=torque_field,
            wrench_field=np.concatenate([field, torque_field], -1),
            detection_active=detected,
            observation_valid=valid,
            force_observation_valid=force_valid,
            localization_link_idx=localization,
            debounce_outputs=self.debounce_outputs,
            event_confirmation_s=(self.debounce_outputs - 1) * stride * 0.02,
            inference_stride_frames=stride,
            sequence_readout="dense_center_block",
            inference_rate_hz=50 / stride,
            force_latency_s=self.lag * 0.02,
            target_links=self.links,
            device=self.device,
            model_sha256=self.model_sha256,
            compute_ms=(time.perf_counter() - started) * 1000,
            input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            mode=self.mode,
            sampling_seed=getattr(self, "sampling_seed", None),
            sampling_draw_start=first_draw,
            sampling_policy="one Gaussian source per window; no smoothing",
            window_end_frame=ends,
            window_output_valid=sequence.get("sampled_output_valid", np.ones((len(ends),50),bool)),
            window_link_idx=sequence["link"],
            window_force_pelvis_N=sequence["force"],
            window_torque_pelvis_Nm=sequence["torque"],
            window_wrench_pelvis_N_Nm=sequence["wrench"],
            direct_output=True,
            confidence_output=False,
            contact_active=force_valid & (localization >= 0),
            window_contact_active=sequence["contact_active"],
            window_force_field_pelvis_N=sequence["force_field"],
            window_torque_field_pelvis_Nm=sequence["torque_field"],
            window_wrench_field_pelvis_N_Nm=sequence["wrench_field"],
            dense_output=True,
            link_contact_active=np.linalg.norm(field, axis=-1) > 0,
            window_link_contact_active=sequence["link_contact_active"],
            window_link_force_pelvis_N=sequence["link_force"],
            window_link_torque_pelvis_Nm=sequence["link_torque"],
            moment_origin="link body origin",
            force_frame="pelvis",
            torque_frame="pelvis",
            output_semantics="30 contact decisions and 30 pelvis-frame wrenches about each link body origin; force and torque share the same mask and target timestamps",
        )
        result.update(sampling_details)
        return result
