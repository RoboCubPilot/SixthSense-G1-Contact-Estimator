import mujoco
import numpy as np

from offline.replay import apply_frame, configure_option

WIDTH, HEIGHT = 560, 500

def point_wrench(force, point, origin, rotation):

    arm = rotation.T @ (point - origin)
    return np.concatenate([force, np.cross(arm, force)])

def view_cameras(model, data, clip):
    apply_frame(model, data, clip, 0)
    rotation = data.xmat[model.body("pelvis").id].reshape(3, 3)
    heading = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    center = 0.5 * (clip.qpos[:, :3].min(0) + clip.qpos[:, :3].max(0))
    center[2] += 0.08
    distance = 2.7 + 1.1 * np.linalg.norm(np.ptp(clip.qpos[:, :2], axis=0))
    cameras = []

    for angle in (heading + 225, heading + 105):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = center
        camera.distance, camera.azimuth, camera.elevation = (
            distance,
            angle,
            -40 if clip.scenario == "lie" else -12,
        )
        cameras.append(camera)
    return cameras

def arrow_context(model, clip, links):
    data = mujoco.MjData(model)
    cameras = view_cameras(model, data, clip)
    option = mujoco.MjvOption()
    configure_option(option)
    scene = mujoco.MjvScene(model, maxgeom=2000)
    projections = []
    for camera in cameras:
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        left, right = scene.camera

        pos = 0.5 * (left.pos + right.pos)
        forward = np.asarray(left.forward, float)
        up = np.asarray(left.up, float)
        projections.append(
            dict(
                pos=pos,
                forward=forward,
                up=up,
                right=np.cross(forward, up),
                focal_px=HEIGHT * left.frustum_near / (2 * left.frustum_top),
            )
        )
    reference, origins, rotations, reference_origins, wrenches = [], [], [], [], []
    body_ids = [model.body(link).id for link in links]
    pelvis = model.body("pelvis").id
    reference_body = model.body(clip.link).id if clip.link in links else None
    for frame in range(clip.n):
        point, _, _ = apply_frame(model, data, clip, frame)
        rotation = data.xmat[pelvis].reshape(3, 3).copy()
        origin = data.xpos[reference_body].copy() if reference_body is not None else np.full(3, np.nan)
        force = clip.force_pelvis[frame]
        wrench = (np.concatenate([force, clip.torque_pelvis[frame]]) if clip.torque_pelvis is not None
                  else point_wrench(force, point, origin, rotation))
        if not clip.force_valid[frame]:
            wrench[:] = np.nan
        reference.append(point)
        origins.append(data.xpos[body_ids].copy())
        rotations.append(rotation)
        reference_origins.append(origin)
        wrenches.append(wrench)
    return dict(
        width=2 * WIDTH,
        height=HEIGHT,
        cameras=projections,
        reference_points_world_m=np.asarray(reference),
        reference_link_origin_world_m=np.asarray(reference_origins),
        reference_wrench_pelvis_N_Nm=np.asarray(wrenches),
        wrench_components=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        link_origin_world_m=np.asarray(origins),
        moment_origin="link body origin",
        reference_torque_source="recorded link-origin moment" if clip.torque_pelvis is not None else "contact point cross force",
        rotation_world_pelvis=np.asarray(rotations),
    )
