from pathlib import Path
import h5py
import mujoco
import numpy as np
from project_paths import REPLAY_MODEL

class Clip:
    def __init__(self, path):
        self.path = Path(path).resolve()
        with h5py.File(self.path, "r") as f:
            self.number = int(f.attrs["number"])
            self.pair_id = f.attrs["pair_id"]
            self.scenario = f.attrs.get("scenario", "stand")
            self.link = f["contact/contact_link_name"].asstr()[()]
            self.point_link = (
                f["contact/contact_point_link_m"][:]
                if "contact/contact_point_link_m" in f
                else np.full(3, np.nan)
            )
            self.force_pelvis = (
                f["contact/force_pelvis_N"][:].astype(float)
                if "contact/force_pelvis_N" in f
                else np.full((len(f["time_s"]), 3), np.nan)
            )
            self.force_valid = f["contact/force_valid"][:]
            self.torque_pelvis = (
                f["contact/moment_about_link_origin_pelvis_Nm"][:].astype(float)
                if "contact/moment_about_link_origin_pelvis_Nm" in f else None
            )
            self.qpos = f["replay/qpos"][:]
            self.qvel = f["replay/qvel"][:]
            self.valid = f["replay/valid"][:]
            self.time_s = f["time_s"][:]
            self.event_time_s = (
                f["time_from_event_s"][:]
                if "time_from_event_s" in f else self.time_s.copy()
            )
            self.joint_names = f["robot/joint_names"].asstr()[:].tolist()
            self.model_path = REPLAY_MODEL
            self.duration = float(f.attrs["duration_s"])
        if not self.valid.all():
            raise ValueError("This clip contains invalid robot poses; replay requires valid poses.")
        self.n = len(self.qpos)

    def validate_model(self, model):
        if (model.nq, model.nv) != (self.qpos.shape[1], self.qvel.shape[1]):
            raise ValueError("MuJoCo model dimensions do not match the recording.")
        if [model.joint(i).name for i in range(1, model.njnt)] != self.joint_names:
            raise ValueError("MuJoCo joint order does not match joint_names.")
        return (model.body(self.link).id if self.link != "unknown" else -1), model.body("pelvis").id

def apply_frame(model, data, clip, frame):

    body, pelvis = (
        (model.body(clip.link).id if clip.link != "unknown" else model.body("pelvis").id),
        model.body("pelvis").id,
    )
    data.qpos[:] = clip.qpos[frame]
    data.qvel[:] = clip.qvel[frame]
    data.time = float(clip.time_s[frame])
    data.ctrl[:] = 0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ clip.point_link
    valid = bool(clip.force_valid[frame])
    force = data.xmat[pelvis].reshape(3, 3) @ clip.force_pelvis[frame]
    if valid:
        if not np.isfinite(force).all():
            raise ValueError("Valid force contains non-finite values.")
        mujoco.mj_applyFT(model, data, force, np.zeros(3), point, body, data.qfrc_applied)
    mujoco.mj_forward(model, data)
    return point, force, valid

def add_motion_context(scene, clip, frame):

    def line(start, end, rgba, width=1):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            np.asarray(start, dtype=float),
            np.asarray(end, dtype=float),
        )
        scene.ngeom += 1

    for coordinate in np.arange(-1.5, 1.51, 0.25):
        line([coordinate, -1.5, 0.002], [coordinate, 1.5, 0.002], [0.42, 0.45, 0.48, 1])
        line([-1.5, coordinate, 0.002], [1.5, coordinate, 0.002], [0.42, 0.45, 0.48, 1])
    for k in range(1, frame + 1):
        start, end = clip.qpos[k - 1, :3].copy(), clip.qpos[k, :3].copy()
        start[2] = end[2] = 0.005
        if np.linalg.norm(end - start) > 1e-9:
            line(start, end, [0.2, 1.0, 0.6, 1], 3)

def configure_option(option):
    option.geomgroup[:] = 0
    option.geomgroup[0:2] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
