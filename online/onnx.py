import hashlib
import time
import numpy as np
from contact_model.onnx import OnnxEstimator

class OnlineOnnxModel(OnnxEstimator):
    def __init__(self, *, strategy="adaptive", candidates=4, **kwargs):
        super().__init__(**kwargs)
        if strategy not in ("baseline", "history", "adaptive") or not 2 <= candidates <= 8:
            raise ValueError("Expected baseline/history/adaptive strategy and 2–8 candidates")
        from .sampling import CandidateSelector
        self.strategy, self.candidate_count = strategy, candidates
        self.selector = CandidateSelector()

    def predict_online(self, window):
        from .sampling import window_noise
        start = time.perf_counter()
        draw_id = self.draw_count
        end, generation = int(window.indices[-1]), window.generation
        self.selector.prepare(generation, end)

        def generate(indices):
            noise = np.stack([window_noise(self.sampling_seed,generation,end,k,self.noise_shape) for k in indices])
            output = self.predict_windows(np.repeat(window.observations[None],len(indices),axis=0),noise=noise)
            return [{k:v[i] for k,v in output.items()} for i in range(len(indices))]

        reason = "disabled"
        if self.strategy == "baseline":

            output = self.predict_windows(window.observations[None])
            sequence = {k:v[0] for k,v in output.items()}
            count, selection = 1, dict(selected_candidate=0,candidate_scores=[])
        else:
            if self.strategy == "history":
                candidates = generate(range(self.candidate_count))
                reason = "always"
            else:
                candidates = generate([0])
                trigger, reason = self.selector.trigger(candidates[0],end)
                if trigger:
                    candidates.extend(generate(range(1,self.candidate_count)))
            count = len(candidates)
            sequence,selection = self.selector.choose(candidates,generation,end)
        return dict(sequence=sequence, compute_ms=(time.perf_counter()-start)*1000,
                    input_sha256=hashlib.sha256(window.observations.tobytes()).hexdigest(),
                    sampling_draw_id=draw_id+selection["selected_candidate"],sampling_strategy=self.strategy,
                    candidate_count=count,extra_sampling_reason=reason,**selection)

class OnnxReadout:

    def __init__(self, model):
        self.model = model
        self.last_window_end = -1
        self.event_run = 0

    def accept(self, window, result):
        end = int(window.indices[-1])
        if end <= self.last_window_end:
            return []
        if self.last_window_end >= 0 and end - self.last_window_end != getattr(
            self.model, "stride", 5
        ):
            self.event_run = 0
        self.last_window_end = end
        sequence = result["sequence"]
        offset = 49 - self.model.lag
        latest_active = bool(sequence["contact_active"][-1])
        self.event_run = self.event_run + 1 if latest_active else 0
        raw = sequence["force"][offset]
        active = bool(sequence["contact_active"][offset])
        index = end - self.model.lag
        row = dict(
            index=index,
            link=int(sequence["link"][offset]) if active else -1,
            force=(raw if active else np.zeros(3, np.float32)).tolist(),
            raw_force=raw.tolist(),
            torque=sequence["torque"][offset].tolist(),
            wrench=sequence["wrench"][offset].tolist(),
            event_active=self.event_run >= self.model.debounce_outputs,
            event_source_window_end=end,
            overlap_count=1,
            link_history_frames=0,
            source_window_end=end,
            direct_output=True,
            confidence_output=False,
            contact_active=active,
            dense_output=True,
            link_contact_active=sequence["link_contact_active"][offset].tolist(),
            force_field=sequence["force_field"][offset].tolist(),
            torque_field=sequence["torque_field"][offset].tolist(),
            wrench_field=sequence["wrench_field"][offset].tolist(),
            link_force=sequence["link_force"][offset].tolist(),
            link_torque=sequence["link_torque"][offset].tolist(),
            moment_origin="link body origin",
            wrench_frame="pelvis",
        )
        if "sampling_draw_id" in result:
            row["sampling_draw_id"] = result["sampling_draw_id"]
        return [row]
