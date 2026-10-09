import numpy as np

from online.onnx import OnlineOnnxModel
from online.worker import OnlineWindow

class ReplayOnnxEstimator(OnlineOnnxModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.replay_generation = 0

    def predict_sequence_windows(self, raw, ends):

        generation = self.replay_generation
        self.replay_generation += 1
        self.selector.previous = None
        if self.strategy == "baseline":
            sequence, details = super().predict_sequence_windows(raw, ends)
            details.update(sampling_strategy="baseline", sampling_generation=generation)
            return sequence, details
        results = [self.predict_online(OnlineWindow(
            generation, number, np.arange(end - 49, end + 1), raw[end - 49:end + 1]
        )) for number, end in enumerate(ends)]
        sequence = {k: np.stack([r["sequence"][k] for r in results])
                    for k in results[0]["sequence"]}
        counts = np.array([r["candidate_count"] for r in results], np.int64)
        return sequence, dict(
            sampling_strategy=self.strategy,
            sampling_generation=generation,
            candidate_count=self.candidate_count,
            extra_sampling_fraction=float(np.mean(counts > 1)),
            sampling_policy="complete candidate selection by consensus and causal timestamp-aligned history; no cross-link averaging",
            window_candidate_count=counts,
            window_selected_candidate=np.array([r["selected_candidate"] for r in results], np.int64),
            window_extra_sampling_reason=np.array([r["extra_sampling_reason"] for r in results]),
            window_compute_ms=np.array([r["compute_ms"] for r in results]),
        )
