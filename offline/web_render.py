#!/usr/bin/env python3

from project_paths import DATASET, ffmpeg_executable
import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from offline.replay import Clip, apply_frame, add_motion_context, configure_option
from offline.web_scene import view_cameras, WIDTH, HEIGHT

FPS = 50

def render(root, number, output, progress):
    clip = Clip(root / str(number) / "data.h5")
    model = mujoco.MjModel.from_xml_path(str(clip.model_path))
    clip.validate_model(model)
    data = mujoco.MjData(model)
    cameras = view_cameras(model, data, clip)
    option = mujoco.MjvOption()
    configure_option(option)
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{2 * WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in range(clip.n):
                apply_frame(model, data, clip, frame)
                views = []
                for camera in cameras:
                    renderer.update_scene(data, camera=camera, scene_option=option)

                    add_motion_context(renderer.scene, clip, frame)
                    views.append(renderer.render().copy())
                process.stdin.write(np.concatenate(views, axis=1).tobytes())
                if frame % 5 == 0 or frame == clip.n - 1:
                    temporary = progress.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"frames": frame + 1, "total": clip.n}))
                    temporary.replace(progress)
        finally:
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"FFmpeg failed: {error[-2000:]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render one recorded pair for the browser, using the existing replay logic.')
    parser.add_argument("--root", type=Path, default=DATASET)
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.pair, args.output, args.progress)
