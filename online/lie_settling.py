import numpy as np
import mujoco

PROTOCOL = dict(
    name="lie_stationary",
    check_rate_hz=500,
    minimum_settle_s=3.0,
    continuous_quiet_s=1.0,
    timeout_s=20.0,
    max_base_linear_m_s=0.01,
    max_base_angular_rad_s=0.05,
    max_joint_rad_s=0.05,
    require_floor_contact=True,
    baseline_s=[1.5, 2.5],
    require_quiet_until_first_force=True,
    initial_xy_m=[-0.02, 0.02],
    initial_height_m=[0.34, 0.36],
    initial_yaw_rad=[-float(np.deg2rad(5)), float(np.deg2rad(5))],
    initial_tilt_rad=[-float(np.deg2rad(1)), float(np.deg2rad(1))],
    joint_jitter_std_rad=float(np.deg2rad(5)),
    max_joint_deviation_rad=float(np.deg2rad(30)),
    joint_reference="original nominal supine pose before R2S mixing",
    max_attempts=32,
    timeout_action="continue checking; user may reset",
    real_data_added=False,
)
LIMITS = np.array(
    [
        PROTOCOL["max_base_linear_m_s"],
        PROTOCOL["max_base_angular_rad_s"],
        PROTOCOL["max_joint_rad_s"],
    ]
)

def gravity(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])

def randomize_pose(model, data, rng, nominal_qpos):

    data.qpos[:2] = rng.uniform(*PROTOCOL["initial_xy_m"], 2)
    data.qpos[2] = rng.uniform(*PROTOCOL["initial_height_m"])
    yaw = rng.uniform(*PROTOCOL["initial_yaw_rad"])
    qyaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qyaw, data.qpos[3:7])
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        tilt = np.zeros(4)
        mujoco.mju_axisAngle2Quat(tilt, axis, rng.uniform(*PROTOCOL["initial_tilt_rad"]))
        result = np.zeros(4)
        mujoco.mju_mulQuat(result, q, tilt)
        q = result
    data.qpos[3:7] = q
    data.qpos[7:] += rng.normal(0, PROTOCOL["joint_jitter_std_rad"], model.nv - 6)
    ranges = model.jnt_range[1:]
    lower = np.maximum(ranges[:, 0] + 0.01, nominal_qpos[7:] - PROTOCOL["max_joint_deviation_rad"])
    upper = np.minimum(ranges[:, 1] - 0.01, nominal_qpos[7:] + PROTOCOL["max_joint_deviation_rad"])
    if np.any(lower > upper):
        raise ValueError("Nominal pose and joint limits have no valid intersection")

    data.qpos[7:] = np.clip(data.qpos[7:], lower, upper)
    mujoco.mj_forward(model, data)
    return data.qpos.copy()

def measure(model, data):

    speeds = np.array(
        [
            np.linalg.norm(data.qvel[:3]),
            np.linalg.norm(data.qvel[3:6]),
            np.max(np.abs(data.qvel[6:])),
        ]
    )
    g = gravity(data.qpos[3:7])
    floor = model.geom("floor").id
    on_floor = False
    for i in range(data.ncon):
        contact = data.contact[i]
        if floor in (contact.geom1, contact.geom2) and contact.efc_address >= 0:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 1.0:
                on_floor = True
                break
    supine = abs(g[2]) <= 0.5 and g[0] <= -0.5
    finite = np.isfinite(data.qpos).all() and np.isfinite(speeds).all()
    quiet = finite and supine and on_floor and bool(np.all(speeds <= LIMITS))
    return speeds, bool(quiet)

class QuietGate:
    def __init__(self, dt):
        self.dt = dt
        self.quiet_s = 0.0

    def update(self, elapsed, speeds, quiet):
        if elapsed < PROTOCOL["minimum_settle_s"] or not quiet:
            self.quiet_s = 0.0
            return False
        self.quiet_s += self.dt
        return self.quiet_s + 1e-9 >= PROTOCOL["continuous_quiet_s"]
