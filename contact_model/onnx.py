import secrets
import numpy as np
import onnxruntime as ort

from project_paths import MODEL
from .inference import SequenceEstimator
from .metadata import load_metadata

class OnnxEstimator(SequenceEstimator):
    def __init__(self, *, device="cpu", threads=2, seed=None):
        config = load_metadata()
        self.config = config
        self.model_sha256 = config["model_sha256"]
        self.noise_shape = tuple(config["noise_shape"])
        self.sampling_seed = secrets.randbits(63) if seed is None else int(seed)
        self.draw_count = 0
        self.links = config["target_links"]
        self.name = config["name"]
        self.debounce_outputs = config["debounce_outputs"]
        self.lag = config["report_lag_frames"]
        self.stride = config.get("inference_stride_frames", 5)
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("Use cpu, cuda, or auto")
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        if provider not in ort.get_available_providers():
            raise ValueError(f"{provider} is not installed")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        if provider == "CPUExecutionProvider":
            options.enable_mem_reuse = False
        ort.disable_telemetry_events()
        self.session = ort.InferenceSession(str(MODEL), options, providers=[(provider, {"use_tf32": 0})] if device == "cuda" else [provider])
        inputs = ["sensors", "noise"]
        if [value.name for value in self.session.get_inputs()] != inputs:
            raise ValueError(f"Model expects inputs {inputs}")
        if [value.shape[1:] for value in self.session.get_inputs()] != [[50, 96], list(self.noise_shape)]:
            raise ValueError("ONNX input shapes differ from sensor/noise metadata")
        if [value.name for value in self.session.get_outputs()] != [
            "wrench", "raw_wrench", "force", "torque", "link_contact_active",
            "contact_active", "link_index", "gate_score",
        ]:
            raise ValueError("ONNX outputs do not match the six-dimensional wrench contract")
        self.device = "cuda" if device == "cuda" else "cpu"
        self.mode = config["name"]

    def predict_windows(self, sensors, *, noise=None):
        x = np.ascontiguousarray(sensors, np.float32)
        if x.ndim != 3 or x.shape[1:] != (50, 96) or len(x) < 1 or not np.isfinite(x).all():
            raise ValueError("Expected finite [batch>=1,50,96] robot signals")
        if noise is None:
            noise = np.stack([
                np.random.default_rng(np.random.SeedSequence(
                    [self.sampling_seed, self.draw_count+i]
                )).standard_normal(self.noise_shape, dtype=np.float32)
                for i in range(len(x))
            ])
        noise = np.ascontiguousarray(noise, dtype=np.float32)
        if noise.shape != (len(x), *self.noise_shape) or not np.isfinite(noise).all():
            raise ValueError(f"Expected finite [batch,{self.noise_shape}] standard-normal source")
        inputs = {"sensors": x, "noise": noise}
        values = self.session.run(None, inputs)
        self.draw_count += len(x)
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError("Non-finite ONNX prediction")
        output = dict(zip((value.name for value in self.session.get_outputs()), values))
        if output["wrench"].shape != (len(x), 50, 6, 30) or output["raw_wrench"].shape != (len(x), 50, 6, 30):
            raise ValueError("Invalid per-link wrench layout")
        output["wrench_field"] = output.pop("wrench").transpose(0, 1, 3, 2)
        output["force_field"] = output.pop("force").reshape(-1, 50, 30, 3)
        output["torque_field"] = output.pop("torque").transpose(0, 1, 3, 2)
        output["link_force"] = output["raw_wrench"][..., :3, :]
        output["link_torque"] = output["raw_wrench"][..., 3:, :]
        output["link"] = output.pop("link_index")
        active = output["contact_active"].astype(bool)
        link_active = output["link_contact_active"].astype(bool)
        if output["link"].shape != (len(x), 50) or np.any(output["link"] < -1) or np.any(output["link"] >= 30):
            raise ValueError("Invalid strongest-link index")
        if not np.array_equal(output["link"] >= 0, active):
            raise ValueError("Contact/link results disagree")
        if link_active.shape != (len(x), 50, 30) or output["link_force"].shape != (
            len(x),
            50,
            3,
            30,
        ):
            raise ValueError("Invalid per-link contact or force layout")
        if not np.array_equal(link_active.any(-1), active):
            raise ValueError("Per-link and any-contact decisions disagree")
        magnitudes = np.linalg.norm(output["link_force"], axis=2)
        if np.any(magnitudes[link_active] < self.config["minimum_force_N"] - 1e-4):
            raise ValueError("Sampled contact force violates the model's positive-contact threshold")
        composed = output["link_force"].transpose(0, 1, 3, 2) * link_active[..., None]
        if not np.allclose(composed, output["force_field"], rtol=1e-6, atol=1e-6):
            raise ValueError("Dense output must be the product of per-link decisions and forces")
        composed_torque = output["link_torque"].transpose(0, 1, 3, 2) * link_active[..., None]
        if not np.allclose(composed_torque, output["torque_field"], rtol=1e-6, atol=1e-6):
            raise ValueError("Force and torque must use the same per-link contact decisions")
        if not np.array_equal(output["wrench_field"], np.concatenate([output["force_field"], output["torque_field"]], -1)):
            raise ValueError("Wrench components disagree with the force/torque fields")
        if np.any(output["wrench_field"][~link_active] != 0):
            raise ValueError("No-contact wrench must be zero")
        output["flow_gate"] = (output["gate_score"] >= self.config["gate_threshold"]) & (magnitudes >= self.config["minimum_force_N"])
        if np.any(link_active & ~output["flow_gate"]) or not np.array_equal(link_active[active], output["flow_gate"][active]):
            raise ValueError("Contact mask must preserve gate/force thresholds and frame-level presence policy")
        strongest = np.linalg.norm(output["force_field"], axis=-1).argmax(-1)
        if not np.array_equal(strongest[active], output["link"][active]):
            raise ValueError("Strongest-link index disagrees with the force field")
        output["wrench"] = output["wrench_field"][np.arange(len(x))[:, None], np.arange(50)[None], output["link"].clip(0)].copy()
        output["wrench"][~active] = 0
        output["force"] = output["wrench"][..., :3]
        output["torque"] = output["wrench"][..., 3:]
        return output
