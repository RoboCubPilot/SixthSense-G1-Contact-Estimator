import json
import mujoco
import numpy as np
import onnxruntime as ort

ort.disable_telemetry_events()

from project_paths import SCENARIOS, SIM_SCENARIO_NAMES, normalize_sim_scenario
from contact_model.metadata import load_metadata
from .lie_settling import PROTOCOL, randomize_pose, measure, gravity
from .lie_readiness import LieReadiness

ROOT = SCENARIOS / "stand"

class Simulation:
    def __init__(self, scenario="stand", walk_speed=None, lie_seed=182, lie_randomize=True,
                 lie_elbow_angle=1.2, motion=1, dance=None):
        scenario = normalize_sim_scenario(scenario)
        if scenario not in SIM_SCENARIO_NAMES:
            raise ValueError("Expected stand, walk, lie or tracking")
        if walk_speed is not None and (not np.isfinite(walk_speed) or not 0.3 <= walk_speed <= 1.0):
            raise ValueError("Walk speed must be between 0.3 and 1.0 m/s")
        self.scenario = scenario
        self.walk_speed_override = walk_speed
        self.walk_speed = 0.0
        self.walk_rng = np.random.default_rng()
        self.lie_seed, self.lie_randomize = lie_seed, lie_randomize
        if not np.isfinite(lie_elbow_angle) or not 0.0 <= lie_elbow_angle <= 2.0:
            raise ValueError("Lie elbow angle must be between 0 and 2 radians")
        self.lie_elbow_angle = float(lie_elbow_angle)
        self.lie_rng = np.random.default_rng(lie_seed)
        self.config = json.loads((ROOT / "online.json").read_text())
        self.model = mujoco.MjModel.from_xml_path(str((SCENARIOS / "twist2" if scenario == "tracking" else ROOT) / "scene.xml"))
        metadata = load_metadata(verify_checksum=False)
        imu_position = metadata.get("simulation_imu_position_m")
        if imu_position is not None:
            self.model.site_pos[self.model.site("imu").id] = np.asarray(imu_position, dtype=float)
        if scenario == "tracking":
            from .twist2 import Twist2Controller
            self.motion_manifest = json.loads((SCENARIOS / "twist2/manifest.json").read_text())["motions"]
            catalog = SCENARIOS / "twist2/dances.json"
            self.dances = json.loads(catalog.read_text())["dances"] if catalog.exists() else []
            self.dance_index = (dance or 1) - 1
            if dance is not None:
                self.motion_manifest[2] = dict(self.dances[self.dance_index], id=3)
            self.motion_index = motion - 1
            if "dance_id" in self.motion_manifest[self.motion_index]:
                self.dance_index = self.motion_manifest[self.motion_index]["dance_id"] - 1
            self.config.update(timestep=0.001, control_decimation=20)
            self.twist2 = Twist2Controller(self.model, SCENARIOS / "twist2/motions" / self.motion_manifest[self.motion_index]["file"])
            self.twist2.reference_velocity = self.motion_manifest[self.motion_index].get("reference_velocity", False)
        self.model.opt.timestep = self.config["timestep"]
        self.data = mujoco.MjData(self.model)
        self.p2x = np.array(self.config["policy_to_xml"])
        self.x2p = np.array(self.config["xml_to_policy"])
        self.default = np.array(self.config["default_angles"])[self.p2x]
        self.kp, self.kd = np.array(self.config["kps"]), np.array(self.config["kds"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.policy = (
            ort.InferenceSession(
                str(ROOT / "locomotion.onnx"), opts, providers=["CPUExecutionProvider"]
            )
            if scenario not in ("lie", "tracking")
            else None
        )
        self.base = self.model.body("pelvis").id
        self.acc_address = self.model.sensor_adr[self.model.sensor("imu_acc").id]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self.default
        if self.scenario == "tracking":
            self.twist2.initialize(self.data, reference=True)
        if self.scenario == "lie":
            self.data.qpos[2] = 0.35
            self.data.qpos[3:7] = [2**-0.5, 0.0, -(2**-0.5), 0.0]
            self.data.qpos[7:] = 0.0
            self.data.qpos[7 + 16], self.data.qpos[7 + 23] = 0.35, -0.35
            for side in ("left", "right"):
                address = int(self.model.joint(f"{side}_elbow_joint").qposadr[0])
                self.data.qpos[address] = self.lie_elbow_angle
            self.lie_nominal_qpos = self.data.qpos.copy()
            if self.lie_randomize:
                randomize_pose(self.model, self.data, self.lie_rng, self.lie_nominal_qpos)
            self.lie_initial_qpos = self.data.qpos.copy()
            self.lie_readiness = LieReadiness(
                self.model.opt.timestep, self.lie_rng.uniform(*PROTOCOL["baseline_s"])
            )
        if self.scenario == "walk":
            if self.walk_speed_override is None:
                speeds = [value / 100 for value in range(40, 91)
                          if value / 100 != self.walk_speed]
                self.walk_speed = float(self.walk_rng.choice(speeds))
            else:
                self.walk_speed = float(self.walk_speed_override)
        self.command = np.array(
            [self.walk_speed if self.scenario == "walk" else 0.0, 0.0, 0.0], np.float32
        )
        self.walk_stopped = False
        self.action = np.zeros(29, np.float32)
        self.target = self.default.copy()
        self.stack = np.zeros((5, 96), np.float32)
        self.tick = 0
        mujoco.mj_forward(self.model, self.data)

    def stop_walk(self):

        if self.scenario != "walk" or self.walk_stopped:
            return False
        self.command[:] = 0.0
        self.walk_stopped = True
        return True

    def invalid_pose(self):
        d = self.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            return True
        g = gravity(d.qpos[3:7])
        if self.scenario == "lie":
            return bool(d.time >= 3.0 and (abs(g[2]) > 0.5 or g[0] > -0.5))
        return bool(d.qpos[2] < 0.35 or g[2] > -0.35)

    def observe_and_control(self):

        d = self.data
        mujoco.mj_forward(self.model, d)
        obs = np.r_[
            d.qvel[3:6] * 0.2,
            gravity(d.qpos[3:7]),
            self.command,
            d.qpos[7:] - self.default,
            d.qvel[6:] * 0.05,
            self.action,
        ].astype(np.float32)
        if self.scenario == "tracking":
            self.twist2.control(d)
            self.target = self.twist2.target.copy()
            self.action = self.twist2.action.copy()
        if self.policy is not None:
            policy_obs = obs.copy()
            for start in (9, 38, 67):
                policy_obs[start : start + 29] = obs[start : start + 29][self.x2p]
            self.stack[:-1] = self.stack[1:]
            self.stack[-1] = policy_obs
            grouped = np.concatenate(
                [
                    self.stack[:, start:end].reshape(-1)
                    for start, end in [(0, 3), (3, 6), (6, 9), (9, 38), (38, 67), (67, 96)]
                ]
            )
            self.action = self.policy.run(None, {self.policy.get_inputs()[0].name: grouped[None]})[
                0
            ][0][self.p2x]
            self.target = self.default + self.config["action_scale"] * self.action
        acceleration = (
            d.sensordata[self.acc_address : self.acc_address + 3] + 9.81 * gravity(d.qpos[3:7])
        ) * 0.1
        return np.r_[obs, acceleration].astype(np.float32)

    def step(self):
        self.data.ctrl[:] = (
            self.twist2.torque(self.data) if self.scenario == "tracking" else
            -8.0 * self.data.qvel[6:]
            if self.scenario == "lie"
            else (self.target - self.data.qpos[7:]) * self.kp - self.data.qvel[6:] * self.kd
        )
        mujoco.mj_step(self.model, self.data)
        if self.scenario == "lie":
            mujoco.mj_forward(self.model, self.data)
            speeds, quiet = measure(self.model, self.data)
            self.lie_readiness.update(float(self.data.time), speeds, quiet)
        self.tick += 1

    def robot_sensors(self, observation):

        from contact_model.signals import simulation_features

        effort = np.clip(
            self.data.ctrl, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )
        return simulation_features(observation, self.data.qpos[7:], effort)

    def apply_point_force(self, body, local_point, world_force):

        d = self.data
        point = d.xpos[body] + d.xmat[body].reshape(3, 3) @ local_point
        mujoco.mj_applyFT(self.model, d, world_force, np.zeros(3), point, body, d.qfrc_applied)
        return d.xmat[self.base].reshape(3, 3).T @ world_force
