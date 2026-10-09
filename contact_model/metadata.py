import hashlib
import json
from pathlib import Path

from project_paths import MODEL, MODEL_NAME

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def load_metadata(*, verify_checksum=True):
    path = MODEL
    config = json.loads(path.with_suffix(".json").read_text())
    expected = dict(
        name=MODEL_NAME, schema="sixthsense_stochastic_wrench_onnx/1",
        preprocessing="g1_sensors_96", deterministic=False, dense_output=True,
        source_distribution="standard_normal", observation_shape=[50, 96],
        presence_mode="scalar1", noise_shape=[50, 211], joint_noise_shape=[50, 210], presence_noise_shape=[50, 1],
        temporal_distribution="joint_50_frame_wrench_gate_flow",
        sampled_output_frames=list(range(50)), sequence_readout="dense_center_block",
        ode_solver="midpoint_rk2", presence_guard=True,
        minimum_force_N=5.0, gate_threshold=1.0, relative_force_threshold=0.0,
        contact_scope="multiple_external_links",
        wrench_output_shape=[50, 6, 30], torque_output_shape=[50, 3, 30],
        wrench_components=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        wrench_units=["N", "N", "N", "Nm", "Nm", "Nm"],
        force_frame="pelvis", torque_frame="pelvis", moment_origin="link body origin",
        force_generator_frozen=False,
        joint_force_torque_generator=True,
        generator_kind="joint_link_origin_wrench_cfm_v1",
        force_torque_coupling="bidirectional",
    )
    invalid = [key for key, value in expected.items() if config.get(key) != value]
    provenance = config.get("reference_provenance", {})
    if provenance.get("force_generator_frozen") != config.get("force_generator_frozen"):
        invalid.append("reference_provenance.force_generator_frozen")
    if invalid:
        raise ValueError(f"Invalid sixthsense model contract ({', '.join(invalid)}): {path}")
    links = config.get("target_links", [])
    lag = config.get("report_lag_frames")
    stride = config.get("inference_stride_frames", 5)
    debounce = config.get("debounce_outputs")
    steps = config.get("ode_steps")
    if (
        len(links) != 30
        or len(set(links)) != 30
        or type(lag) is not int
        or not 0 <= lag < 50
        or type(stride) is not int
        or not 1 <= stride <= 50
        or type(debounce) is not int
        or debounce < 1
        or 49 - lag + stride > 50
        or type(steps) is not int
        or steps < 1
        or type(config.get("presence_steps")) is not int
        or config["presence_steps"] < 1
    ):
        raise ValueError("Invalid sixthsense links, readout timing, or RK2 step count")
    if verify_checksum and sha256(path) != config.get("model_sha256"):
        raise ValueError(f"ONNX checksum mismatch: {path}")
    return config
