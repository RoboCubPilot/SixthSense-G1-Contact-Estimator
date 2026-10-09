from dataclasses import dataclass
import glfw
import numpy as np

LIE_PUSH_LINKS = (
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
)

@dataclass
class Sample:
    time: float
    qpos: np.ndarray
    link: int
    force: np.ndarray
    torque: np.ndarray
    point: np.ndarray
    sensors: np.ndarray
    command: np.ndarray
    base_velocity: np.ndarray

class ControlSession:
    def reset(self):
        self.sim.reset()
        self.generation += 1
        self.paused = self.fallen = False
        self.push_start = None
        self.demo_used = False
        self.current = (-1, np.zeros(3), np.zeros(3))
        self.current_torque = np.zeros(3)

    def select(self, body):
        while body > 0 and body not in self.body_to_link:
            body = self.sim.model.body_parentid[body]
        return int(body) if body in self.body_to_link else -1

    def push(self):
        if not self.ready or self.paused:
            return
        link = self.args.push_link
        if link is None:
            candidates = list(LIE_PUSH_LINKS) if self.sim.scenario == "lie" else self.links
            previous = getattr(self, "last_push_link", None)
            candidates = [name for name in candidates if name != previous]
            if not hasattr(self, "push_rng"):
                self.push_rng = np.random.default_rng()
            link = str(self.push_rng.choice(candidates))
        self.last_push_link = link
        self.push_start = self.sim.data.time
        self.test_push_force = self.args.push_force
        self.test_push_duration = self.args.push_duration
        self.push_body = self.sim.model.body(link).id
        self.push_point = np.array([0.0, 0.0, -0.05])
        self.test_push_direction = self.args.push_direction
        if self.sim.scenario != "lie" and self.test_push_direction is None:
            if not hasattr(self, "push_rng"):
                self.push_rng = np.random.default_rng()
            self.test_push_direction = [float(self.push_rng.choice([-1, 1])), 0.0, 0.0]

    def external(self):
        if not self.ready:
            return None
        if self.push_start is not None:
            elapsed = self.sim.data.time - self.push_start
            duration = self.test_push_duration
            if 0 <= elapsed < duration:
                rotation = self.sim.data.xmat[self.sim.base].reshape(3, 3)
                if self.sim.scenario == "lie" and self.args.push_direction is None:
                    direction = np.array([0.0, 0.0, 1.0])
                    if self.sim.model.body(self.push_body).name.startswith(("left_", "right_")):
                        d = self.sim.data
                        radial = d.xipos[self.push_body] - d.xipos[self.sim.model.body("torso_link").id]
                        radial /= max(np.linalg.norm(radial), 1e-12)
                        tangent = np.array([0.0, 0.0, 1.0]) - radial * radial[2]
                        tangent /= max(np.linalg.norm(tangent), 1e-12)
                        angle = min(np.pi / 3, max(np.pi / 4, np.arcsin(max(0.0, -radial[2])) + 0.01))
                        direction = radial * np.cos(angle) + tangent * np.sin(angle)
                        if direction[2] <= 0:
                            return None
                else:
                    direction = np.asarray(self.test_push_direction, float).copy()
                    direction /= np.linalg.norm(direction)
                    if self.args.push_frame == "pelvis":
                        direction = rotation @ direction
                force = direction * self.test_push_force * np.sin(np.pi * elapsed / duration)
                return self.push_body, self.push_point, force
        return None

    def apply_external(self, body, point, world_force):
        if self.sim.scenario == "lie" and np.linalg.norm(world_force) > 1e-9:
            self.sim.lie_readiness.start_force(float(self.sim.data.time))
        return self.sim.apply_point_force(body, point, world_force)

    def external_torque(self, body, point, world_force):

        d = self.sim.data
        arm_world = d.xmat[body].reshape(3, 3) @ point
        return d.xmat[self.sim.base].reshape(3, 3).T @ np.cross(arm_world, world_force)

    def key(self, key):
        if key == glfw.KEY_SPACE and not self.fallen:
            self.paused = not self.paused
        elif key == glfw.KEY_R:
            self.reset()
        elif key == glfw.KEY_F:
            self.push()
        elif key == glfw.KEY_S:
            self.save()

    def step(self):
        s, d = self.sim, self.sim.data
        if s.tick % s.config["control_decimation"] == 0:
            self.poll_inference()
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        if self.args.demo_push and not self.demo_used and d.time >= 4 and self.ready:
            self.push()
            self.demo_used = True
        ext = self.external()
        self.current = (-1, np.zeros(3), np.zeros(3))
        self.current_torque = np.zeros(3)
        if ext is not None:
            body, point, world_force = ext
            force = self.apply_external(body, point, world_force)
            self.current = (self.body_to_link[body], force, point.copy())
            self.current_torque = self.external_torque(body, point, world_force)
        if s.tick % s.config["control_decimation"] == 0:
            observation = s.observe_and_control()
            link, force, point = self.current
            sample = Sample(
                float(d.time),
                d.qpos.copy(),
                link,
                force.copy(),
                self.current_torque.copy(),
                point.copy(),
                s.robot_sensors(observation),
                s.command.copy(),
                d.xmat[s.base].reshape(3, 3).T @ d.qvel[:3],
            )
            self.samples.append((self.generation, sample))
            self.observe_sample(sample)
        s.step()
        if s.invalid_pose():
            self.fallen = self.paused = True

    def finish_inference(self):
        if self.worker is not None:
            self.worker.close()
            self.poll_inference()

    def run_headless(self):
        try:
            while self.sim.data.time < self.args.seconds and not self.fallen:
                self.step()
        finally:
            self.finish_inference()
            self.save()
