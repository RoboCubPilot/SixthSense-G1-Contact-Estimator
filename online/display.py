import mujoco
import numpy as np

GREEN = np.array([0.12, 0.95, 0.42, 1.0], np.float32)
RED = np.array([1.0, 0.32, 0.29, 1.0], np.float32)

def base_speed_label(scene, data, base_body, speed):

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    point = data.xpos[base_body] + np.array([0.0, 0.0, .25])
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), point,
        np.eye(3).ravel(), np.ones(4, np.float32),
    )
    geom.label = f"CMD: {speed:.2f} m/s"
    scene.ngeom += 1

def applied_pull_handle(scene, model, data, body, local_point, world_force):

    if np.linalg.norm(world_force) < 0.5 or scene.ngeom + 2 > scene.maxgeom:
        return
    point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local_point
    delta = np.asarray(world_force) * 0.015
    delta *= min(1.0, 0.8 / max(np.linalg.norm(delta), 1e-9))
    tip = point + delta
    radius = model.stat.meansize * model.vis.scale.constraint
    red = model.vis.rgba.constraint
    rod = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        rod, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), red
    )
    mujoco.mjv_connector(rod, mujoco.mjtGeom.mjGEOM_ARROW, radius, point, tip)
    rod.label = ""
    ball = scene.geoms[scene.ngeom + 1]
    mujoco.mjv_initGeom(
        ball, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 2 * radius), tip, np.eye(3).ravel(), red
    )
    ball.label = ""
    scene.ngeom += 2

def predicted_forces(prediction):

    if prediction is None:
        return []
    field = np.asarray(prediction["force_field"], dtype=float)
    active = np.asarray(prediction["link_contact_active"], dtype=bool)
    if field.ndim != 2 or field.shape[1] != 3 or active.shape != (len(field),):
        raise ValueError("Invalid dense display field")
    return [(int(link), field[link]) for link in np.flatnonzero(active)]

def prediction_arrows(scene, data, base_body, bodies, prediction):
    for link, force in predicted_forces(prediction):
        prediction_arrow(scene, data, base_body, bodies[link], force)
        torque_arrow(scene, data, base_body, bodies[link], np.asarray(prediction["torque_field"])[link])

def prediction_arrow(scene, data, base_body, body, force):
    if np.linalg.norm(force) < 0.5 or scene.ngeom + 1 >= scene.maxgeom:
        return
    point = data.xpos[body].copy()
    world_force = data.xmat[base_body].reshape(3, 3) @ force
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), GREEN
    )
    delta = world_force * 0.015
    delta *= min(1.0, 0.8 / max(np.linalg.norm(delta), 1e-9))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.016, point, point + delta)

    geom.size[2] *= 2
    geom.label = ""
    scene.ngeom += 1

def torque_arrow(scene, data, base_body, body, torque, color=GREEN):

    magnitude = np.linalg.norm(torque)
    if not np.isfinite(magnitude) or magnitude < 1e-6:
        return
    axis = data.xmat[base_body].reshape(3, 3) @ (torque / magnitude)
    basis = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([0., 1., 0.])
    u = np.cross(basis, axis); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.linspace(0., 1.6 * np.pi, 33)
    radius = min(.22, .055 + .045 * np.sqrt(magnitude))
    points = data.xpos[body] + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    for i in range(0, len(points) - 1, 2):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .004, points[i], points[i + 1])
        geom.label = ""
        scene.ngeom += 1
    if scene.ngeom < scene.maxgeom:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .01, points[-3], points[-1])
        geom.label = ""
        scene.ngeom += 1
