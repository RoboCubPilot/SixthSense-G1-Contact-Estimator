import json
import pickle
from collections import deque
from pathlib import Path
import numpy as np
import mujoco
import onnxruntime as ort
from scipy.spatial.transform import Rotation, Slerp
from project_paths import SCENARIOS

ROOT = SCENARIOS / 'twist2'

class Motion:
    def __init__(self, path):
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        self.pos = np.asarray(raw['root_pos'], dtype=float)
        self.quat = np.asarray(raw['root_rot'], dtype=float)
        self.joints = np.asarray(raw['dof_pos'], dtype=float)
        count = len(self.pos)
        if count < 3 or self.pos.shape != (count, 3) or self.quat.shape != (count, 4) or self.joints.shape != (count, 29):
            raise ValueError(f'Motion must contain root position, xyzw rotation and 29 joints: {path}')
        if not all(np.isfinite(v).all() for v in (self.pos, self.quat, self.joints)):
            raise ValueError(f'Motion contains nonfinite values: {path}')
        if not np.isfinite(float(raw['fps'])) or float(raw['fps']) <= 0:
            raise ValueError(f'Invalid motion frame rate: {path}')
        self.dt = 1.0 / float(raw['fps'])
        self.times = np.arange(len(self.pos)) * self.dt
        self.duration = float(self.times[-1])
        self.rot = Rotation.from_quat(self.quat)
        self.slerp = Slerp(self.times, self.rot)
        self.vel = np.gradient(self.pos, self.dt, axis=0)
        r = self.rot
        self.omega = np.empty_like(self.pos)
        self.omega[1:-1] = (r[2:] * r[:-2].inv()).as_rotvec() / (2 * self.dt)
        self.omega[0] = (r[1] * r[0].inv()).as_rotvec() / self.dt
        self.omega[-1] = (r[-1] * r[-2].inv()).as_rotvec() / self.dt

    def frame(self, t):
        t = float(np.clip(t, 0, self.duration))
        index = min(int(t / self.dt), len(self.pos)-2)
        blend = t / self.dt - index
        pos = (1-blend)*self.pos[index] + blend*self.pos[index+1]
        joints = (1-blend)*self.joints[index] + blend*self.joints[index+1]
        rot = self.slerp(t)
        rpy = rot.as_euler('xyz')
        obs = np.r_[rot.inv().apply(self.vel[index])[:2], pos[2], rpy[:2],
                    rot.inv().apply(self.omega[index])[2], joints]
        return pos, rot, joints, obs

class Twist2Controller:
    def __init__(self, model, motion):
        self.motion = Motion(motion)
        self.model = model
        manifest = json.loads((ROOT / "manifest.json").read_text())
        entries = list(manifest["motions"])
        catalog = ROOT / "dances.json"
        if catalog.exists():
            entries.extend(json.loads(catalog.read_text())["dances"])
        self.reference_velocity = next((item.get("reference_velocity", False) for item in entries if item["file"] == Path(motion).name), False)
        self.default = np.array([-.2,0,0,.4,-.2,0]*2 + [0,0,0] +
                                [0,.4,0,1.2,0,0,0] + [0,-.4,0,1.2,0,0,0])
        self.kp = np.array([100,100,100,150,40,40]*2+[150]*3+[40]*4+[4]*3+[40]*4+[4]*3)
        self.kd = np.array([2,2,2,4,2,2]*2+[4]*3+[5]*4+[.2]*3+[5]*4+[.2]*3)
        self.limits = self.kp.copy()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(str(ROOT/'policy.onnx'), opts, providers=['CPUExecutionProvider'])
        self.input = self.policy.get_inputs()[0].name
        if self.policy.get_inputs()[0].shape[-1] != 1432 or self.policy.get_outputs()[0].shape[-1] != 29:
            raise ValueError('TWIST2 policy must accept 1432 observations and return 29 actions')
        self.reset()

    def reset(self):
        self.action = np.zeros(29)
        self.history = deque([np.zeros(127) for _ in range(10)], maxlen=10)
        self.target = self.default.copy()

    def control(self, data):
        rpy = Rotation.from_quat(data.qpos[[4,5,6,3]]).as_euler('xyz')
        vel = data.qvel[6:].copy()
        vel[[4,5,10,11]] = 0
        mimic = self.motion.frame(data.time + .02)[3]
        proprio = np.r_[data.qvel[3:6]*.25, rpy[:2], data.qpos[7:]-self.default, vel*.05, self.action]
        full = np.r_[mimic, proprio]
        obs = np.r_[full, np.asarray(self.history).ravel(), mimic].astype(np.float32)
        self.history.append(full)
        self.action = self.policy.run(None, {self.input: obs[None]})[0][0]
        self.target = self.default + .5*np.clip(self.action, -10, 10)

    def torque(self, data):
        return np.clip((self.target-data.qpos[7:])*self.kp-data.qvel[6:]*self.kd, -self.limits, self.limits)

    def initialize(self, data, reference=False):
        self.reset()
        mujoco.mj_resetData(self.model, data)
        data.qpos[:3] = [0,0,.793]
        data.qpos[3:7] = [1,0,0,0]
        data.qpos[7:] = self.default
        data.qpos[7+16], data.qpos[7+23] = .2, -.2
        if reference:
            pos, rot, joints, _ = self.motion.frame(0)
            data.qpos[:3] = pos
            data.qpos[3:7] = rot.as_quat()[[3,0,1,2]]
            data.qpos[7:] = joints
            if self.reference_velocity:
                data.qvel[:3] = self.motion.vel[0]
                data.qvel[3:6] = rot.inv().apply(self.motion.omega[0])
                data.qvel[6:] = np.gradient(self.motion.joints, self.motion.dt, axis=0)[0]
        mujoco.mj_forward(self.model, data)
