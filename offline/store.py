from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import h5py
import mujoco
import numpy as np

from contact_model.metadata import load_metadata, sha256 as sha
from project_paths import MODEL, REPLAY_MODEL, PROJECT, OUTPUTS, DATASETS, SCENARIO_NAMES, dataset_root
from .contact_prediction import ContactPrediction
from .replay import Clip
from .web_scene import arrow_context

HERE = Path(__file__).resolve().parent

class StoreRegistry:

    def __init__(self, scenario, root):
        self.default = scenario
        self.roots = {name: DATASETS / name for name in SCENARIO_NAMES}
        self.roots[scenario] = root
        self.stores = {scenario: Store(root)}
        self.lock = threading.Lock()

    def get(self, scenario=None):
        scenario = self.default if scenario is None else scenario
        if scenario not in self.roots:
            raise ValueError("Unknown scenario.")
        with self.lock:
            if scenario not in self.stores:
                self.stores[scenario] = Store(dataset_root(scenario, self.roots[scenario]))
            return self.stores[scenario]

    def available(self):
        return [name for name, root in self.roots.items() if (root / "manifest.json").is_file()]

    def close(self, wait=True):
        for store in self.stores.values():
            store.close(wait=wait)

def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value

class Store:
    def __init__(self, root):
        self.root = root.resolve()
        self.model_path = MODEL
        self.model_metadata = load_metadata()
        self.model_label = self.model_metadata["name"]
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.rows = {row["number"]: row for row in self.manifest["pairs"]}
        self.scenario = self.manifest.get("scenario", "stand")
        self.cache_dir = OUTPUTS / "web_cache" / self.scenario
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.data_cache = OrderedDict()
        self.jobs = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
        self.inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.inference_jobs = OrderedDict()
        self.estimator = None
        self.model = mujoco.MjModel.from_xml_path(str(REPLAY_MODEL))
        self.manifest_stamp = (root / "manifest.json").stat().st_mtime_ns

        dependencies = [
            HERE / "replay.py",
            HERE / "web_render.py",
            HERE / "web_scene.py",
            REPLAY_MODEL,
            PROJECT / "assets/robot/g1_29dof.xml",
        ]
        dependencies += sorted((PROJECT / "assets/robot/meshes").glob("*.STL"))
        self.render_version = hashlib.sha256(
            "".join(sha(p) for p in dependencies).encode()
        ).hexdigest()
        inference_files = [
            self.model_path,
            self.model_path.with_suffix(".json"),
            HERE / "store.py",
            HERE / "estimate_contact.py",
            HERE / "contact_prediction.py",
            HERE / "onnx.py",
            PROJECT / "online/onnx.py",
            PROJECT / "online/sampling.py",
            PROJECT / "online/worker.py",
            *[
                PROJECT / "contact_model" / name
                for name in ("onnx.py", "inference.py", "signals.py", "metadata.py")
            ],
            PROJECT / "assets/robot/g1_29dof.xml",
        ]
        import onnxruntime

        versions = f"{mujoco.__version__}:{np.__version__}:{onnxruntime.__version__}"
        self.inference_version = hashlib.sha256(
            ("".join(sha(p) for p in inference_files) + versions).encode()
        ).hexdigest()

    def close(self, wait=True):

        with self.lock:
            for job in self.inference_jobs.values():
                job["cancelled"] = True
        self.pool.shutdown(wait=wait, cancel_futures=True)
        self.inference_pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def refresh_sources(self):
        stamp = (self.root / "manifest.json").stat().st_mtime_ns
        if stamp != self.manifest_stamp:
            with self.lock:
                manifest = json.loads((self.root / "manifest.json").read_text())
                self.manifest = manifest
                self.rows = {row["number"]: row for row in manifest["pairs"]}
                self.manifest_stamp = stamp
                self.data_cache.clear()

    def catalog(self):
        self.refresh_sources()
        return {
            "scenario": self.scenario,
            "pairs": [
                {
                    k: row[k]
                    for k in (
                        "number",
                        "pair_id",
                        "source_group",
                        "contact_link",
                        "valid_force_frames",
                    )
                }
                for row in self.rows.values()
            ],
            "base_modes": ["fixed"],
            "prediction_modes": ["model"],
            "model_label": self.model_label,
            "model_display_name": self.model_metadata.get("display_name", self.model_label),
            "checkpoint_step": self.model_metadata.get("checkpoint_internal_step"),
            "checkpoint_label": self.model_metadata.get("checkpoint_label"),
            "model_buffer_s": 1,
            "sampling_strategy": "adaptive",
        }

    def pair(self, number, mode):
        self.refresh_sources()
        if number not in self.rows or mode != "fixed":
            raise ValueError("Unknown sample or base mode.")
        path = self.root / self.rows[number]["data_file"]
        signature = (number, mode, path.stat().st_mtime_ns)
        with self.lock:
            if signature in self.data_cache:
                self.data_cache.move_to_end(signature)
                return self.data_cache[signature]
            source_sha = sha(path)
            if source_sha != self.rows[number]["data_sha256"]:
                raise ValueError(
                    "Sample data.h5 does not match the manifest. Check the data source."
                )
            clip = Clip(path)
            prediction = ContactPrediction.from_result(
                dict(
                    target_links=self.model_metadata["target_links"],
                    mode=self.model_label + "; waiting for inference",
                    model_sha256=self.model_metadata["model_sha256"],
                    field=np.zeros((clip.n, 30, 3), np.float32),
                    torque_field=np.zeros((clip.n, 30, 3), np.float32),
                    observation_valid=np.zeros(clip.n, bool),
                    force_observation_valid=np.zeros(clip.n, bool),
                    contact_active=np.zeros(clip.n, bool),
                    localization_link_idx=np.full(clip.n, -1, np.int64),
                    force_latency_s=self.model_metadata["report_lag_frames"] / 50,
                )
            )
            geometry = arrow_context(self.model, clip, prediction.links)
            payload = dict(
                number=number,
                pair_id=clip.pair_id,
                base_mode=mode,
                unified_model=True,
                scenario=self.scenario,
                force_labels_available=self.scenario == "stand",
                link=clip.link,
                point_link_m=clip.point_link,
                time_s=clip.time_s,
                event_time_s=clip.event_time_s,
                duration_s=clip.duration,
                frames=clip.n,
                joint_names=clip.joint_names,
                force_pelvis_N=clip.force_pelvis,
                force_valid=clip.force_valid,
                prediction=prediction.payload(),
                arrow_context=geometry,
            )
            with h5py.File(path, "r") as f:
                payload.update(
                    q=f["robot/q"][:],
                    qdot=f["robot/qdot"][:],
                    tau=f["robot/tau"][:],
                    imu=f["robot/imu"][:],
                )
                payload["force_magnitude_N"] = (
                    f["contact/force_magnitude_N"][:]
                    if "contact/force_magnitude_N" in f
                    else np.full(clip.n, np.nan)
                )
                payload["videos"] = []
                for camera in (1, 2):
                    info = json.loads(f[f"video/camera_{camera}_timing_json"].asstr()[()])

                    rows = list(
                        csv.DictReader(
                            io.StringIO(f[f"video/camera_{camera}_frames_csv"].asstr()[()])
                        )
                    )
                    time_field = "time_from_event_s" if "time_from_event_s" in rows[0] else "time_s"
                    targets = [float(row[time_field]) for row in rows]
                    output_fps = float(info["output_fps"])
                    media_times = [int(row["output_frame"]) / output_fps for row in rows]
                    available = info.get("available", True)
                    video_revision = (
                        self.rows[number]["videos_sha256"][f"camera_{camera}.mp4"][:16]
                        if available
                        else None
                    )
                    payload["videos"].append(
                        dict(
                            camera=camera,
                            available=available,
                            url=f"/video/{number}/{camera}.mp4?scenario={self.scenario}&v={video_revision}"
                            if available
                            else None,
                            missing_reason=info.get("missing_reason"),
                            status=info["timing_status"],
                            max_frame_error_ms=info.get("max_frame_error_ms"),
                            fps=output_fps,
                            event_times=targets,
                            media_times=media_times,
                        )
                    )
            valid_values = np.where(clip.force_valid, payload["force_magnitude_N"], -np.inf)
            payload["peak_frame"] = (
                int(np.argmax(valid_values))
                if clip.force_valid.any()
                else int(np.argmin(np.abs(clip.event_time_s)))
            )
            state_hash = hashlib.sha256(clip.qpos.tobytes() + clip.qvel.tobytes()).hexdigest()
            key = hashlib.sha256(
                (source_sha + self.render_version + state_hash + mode).encode()
            ).hexdigest()[:24]
            payload["render_key"] = f"{number}-{mode}-{key}"
            payload["prediction"]["origin"] = "pending"
            payload = clean(payload)
            self.data_cache[signature] = payload
            while len(self.data_cache) > 8:
                self.data_cache.popitem(last=False)
            return payload

    def start_inference(self, number):
        self.refresh_sources()
        if number not in self.rows:
            raise ValueError("Unknown sample.")
        token = uuid.uuid4().hex
        with self.lock:
            for job in self.inference_jobs.values():
                if job["state"] in ("queued", "buffering", "loading_model"):
                    job["cancelled"] = True
                    if job["future"].cancel():
                        job["state"] = "cancelled"
            job = dict(
                id=token,
                number=number,
                state="queued",
                received=0,
                total=self.rows[number]["frames"],
                cancelled=False,
                started=time.perf_counter(),
            )
            self.inference_jobs[token] = job
            job["future"] = self.inference_pool.submit(self.run_inference, job)
            while len(self.inference_jobs) > 16:
                oldest = next(iter(self.inference_jobs))
                if self.inference_jobs[oldest]["state"] not in ("ready", "error", "cancelled"):
                    break
                self.inference_jobs.pop(oldest)
        return self.inference_status(token)

    def inference_status(self, token):
        with self.lock:
            if token not in self.inference_jobs:
                raise ValueError("Inference job is missing or expired. Run inference again.")
            return clean(
                {
                    k: v
                    for k, v in self.inference_jobs[token].items()
                    if k not in ("future", "cancelled", "started")
                }
            )

    def run_inference(self, job):
        try:
            from offline.estimate_contact import robot_input, save_result
            from offline.onnx import ReplayOnnxEstimator

            path = self.root / self.rows[job["number"]]["data_file"]
            source_sha = sha(path)
            if source_sha != self.rows[job["number"]]["data_sha256"]:
                raise ValueError("Source data.h5 checksum verification failed.")

            key = hashlib.sha256(
                f"{source_sha}:{self.inference_version}:{job['id']}".encode()
            ).hexdigest()[:24]
            output = self.cache_dir / f"estimate_{job['number']}_{key}.npz"
            prediction_file = output.with_suffix(".json")
            obs, times, pair_id = robot_input(path)
            with self.lock:
                job["state"] = "loading_model" if self.estimator is None else "inference"
            if self.estimator is None:
                self.estimator = ReplayOnnxEstimator(strategy="adaptive", candidates=4)
            if job["cancelled"]:
                with self.lock:
                    job["state"] = "cancelled"
                return
            with self.lock:
                job.update(state="inference", received=len(obs))
            result = self.estimator.predict(obs)
            temporary = output.with_name(job["id"] + ".partial.npz")
            save_result(temporary, result, times, pair_id, source_sha)
            temporary.replace(output)
            prediction = clean(ContactPrediction.from_result(result).payload())
            prediction.update(
                origin="model",
                compute_ms=result["compute_ms"],
                buffer_s=1,
                device=result["device"],
                sampling_strategy=result["sampling_strategy"],
                candidate_count=result["candidate_count"],
                extra_sampling_fraction=result["extra_sampling_fraction"],
            )
            temporary = prediction_file.with_name(job["id"] + ".partial.json")
            temporary.write_text(json.dumps(prediction, allow_nan=False))
            temporary.replace(prediction_file)
            with self.lock:
                job.update(
                    state="ready",
                    prediction=dict(prediction, inference_id=job["id"]),
                    compute_ms=prediction["compute_ms"],
                    elapsed_ms=(time.perf_counter() - job["started"]) * 1000,
                    output=str(output.relative_to(PROJECT)),
                )
        except Exception as exc:
            with self.lock:
                job.update(state="error", message=str(exc))

    def render_status(self, number, mode, retry=False):
        payload = self.pair(number, mode)
        key = payload["render_key"]
        output = self.cache_dir / (key + ".mp4")
        if output.exists():
            return dict(state="ready", progress=1, url=f"/rendered/{key}.mp4?scenario={self.scenario}", key=key)
        with self.lock:
            job = self.jobs.get(key)
            if retry and job is not None and job["state"] == "error":
                self.jobs.pop(key)
                job = None
            if job is None or job["state"] == "cancelled":
                for other in self.jobs.values():
                    if other["state"] == "queued" and other["future"].cancel():
                        other["state"] = "cancelled"
                job = dict(state="queued", progress=0, key=key)
                self.jobs[key] = job
                job["future"] = self.pool.submit(self.render, number, key)
            status = {k: v for k, v in job.items() if k != "future"}
            progress = self.cache_dir / (key + ".progress.json")
            if status["state"] == "rendering" and progress.exists():
                try:
                    p = json.loads(progress.read_text())
                    status["progress"] = p["frames"] / p["total"]
                except (ValueError, OSError):
                    pass
            return status

    def render(self, number, key):
        output = self.cache_dir / (key + ".mp4")
        temporary = self.cache_dir / (key + ".partial.mp4")
        progress = self.cache_dir / (key + ".progress.json")
        with self.lock:
            self.jobs[key]["state"] = "rendering"
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "offline.web_render",
                    "--root",
                    str(self.root),
                    "--pair",
                    str(number),
                    "--output",
                    str(temporary),
                    "--progress",
                    str(progress),
                ],
                cwd=PROJECT,
                capture_output=True,
                text=True,
                timeout=240,
            )
            if process.returncode:
                raise RuntimeError(process.stderr[-2000:] or "MuJoCo rendering failed.")
            if not temporary.exists() or temporary.stat().st_size < 1000:
                raise RuntimeError("Replay video was not generated successfully.")
            temporary.replace(output)
            with self.lock:
                self.jobs[key].update(state="ready", progress=1, url=f"/rendered/{key}.mp4?scenario={self.scenario}")
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            with self.lock:
                self.jobs[key].update(state="error", message=str(exc))
