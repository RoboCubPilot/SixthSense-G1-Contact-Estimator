import numpy as np

def sensor_features(imu, q, qdot, tau):

    imu, q, qdot, tau = [np.asarray(v, np.float32) for v in (imu, q, qdot, tau)]
    if imu.shape[-1] != 10 or any(v.shape != (*imu.shape[:-1], 29) for v in (q, qdot, tau)):
        raise ValueError("Expected IMU [...,10] and joints [...,29] in XML order")
    norm = np.linalg.norm(imu[..., :4], axis=-1, keepdims=True)
    if np.any(norm < 0.5):
        raise ValueError("Invalid IMU quaternion")
    x, y, z, w = np.moveaxis(imu[..., :4] / norm, -1, 0)
    gravity = np.stack(
        [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)], axis=-1
    )
    features = np.concatenate(
        [
            imu[..., 4:7] * 0.2,
            gravity,
            (imu[..., 7:10] + 9.81 * gravity) * 0.1,
            q,
            qdot * 0.05,
            tau * 0.025,
        ],
        axis=-1,
    )
    if not np.isfinite(features).all():
        raise ValueError("Non-finite robot sensor input")
    return features.astype(np.float32)

def simulation_features(policy_obs, q, tau):

    return np.concatenate(
        [policy_obs[..., :6], policy_obs[..., 96:99], q, policy_obs[..., 38:67], tau * 0.025],
        axis=-1,
    ).astype(np.float32)
