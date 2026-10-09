import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from project_paths import dataset_root, SCENARIO_NAMES
from offline.onnx import ReplayOnnxEstimator
from contact_model.metadata import sha256 as sha

def robot_input(path):
    from contact_model.signals import sensor_features

    with h5py.File(path) as f:
        obs = sensor_features(*[f["robot/" + k][:] for k in ("imu", "q", "qdot", "tau")])
        time_field = "time_from_event_s" if "time_from_event_s" in f else "time_s"
        times, pair_id = f[time_field][:], str(f.attrs["pair_id"])
    if (
        len(times) != len(obs)
        or not np.isfinite(times).all()
        or not np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-8)
    ):
        raise ValueError("Input timestamps must form a continuous 50 Hz sequence.")
    return obs, times, pair_id

def save_result(path, result, times, pair_id, data_sha256):
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    info = {k: v for k, v in result.items() if k not in arrays}
    arrays["force_field_pelvis_N"] = arrays.pop("field")
    arrays["torque_field_pelvis_Nm"] = arrays.pop("torque_field")
    arrays["wrench_field_pelvis_N_Nm"] = arrays.pop("wrench_field")
    arrays["time_from_event_s"] = times
    info.update(
        pair_id=pair_id,
        data_sha256=data_sha256,
        schema="g1_wrench_model_result/3.0",
        force_frame="pelvis",
        torque_frame="pelvis",
        moment_origin="link body origin",
        contact_active_readout="same target timestamps as force_field_pelvis_N and localization_link_idx",
        detection_active_readout="latest available window frame; independent timestamp from delayed force readout",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, metadata_json=json.dumps(info), **arrays)

def main():
    parser = argparse.ArgumentParser(description='Estimate one recorded pair with the sixthsense ONNX model.')
    parser.add_argument("pair", type=int)
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="stand")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file already exists; choose a new path.")
    try:
        root = dataset_root(args.scenario, args.root)
    except ValueError as exc:
        parser.error(str(exc))
    path = root / str(args.pair) / "data.h5"
    observations, times, pair_id = robot_input(path)
    estimator = ReplayOnnxEstimator(strategy="adaptive", candidates=4)
    result = estimator.predict(observations)
    save_result(args.output, result, times, pair_id, sha(path))
    print(
        json.dumps(
            dict(
                output=str(args.output),
                pair_id=pair_id,
                compute_ms=result["compute_ms"],
                model=estimator.mode,
            )
        )
    )

if __name__ == "__main__":
    main()
