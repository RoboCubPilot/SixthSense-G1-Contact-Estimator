#!/usr/bin/env python3

import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

from contact_model.metadata import load_metadata
from project_paths import PROJECT, REPLAY_PORTS, SIM_SCENARIO_NAMES, normalize_sim_scenario

def main(argv=None):
    parser = argparse.ArgumentParser(description='sixthsense.py: real/sim/estimate stand/walk/lie/tracking [options].',
        epilog="Recorded replay and simulation use the sixthsense model with adaptive sampling.")
    parser.add_argument("mode", choices=["real", "sim", "estimate"])
    parser.add_argument("scenario", type=normalize_sim_scenario, choices=SIM_SCENARIO_NAMES)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if any(arg == "--scenario" or arg.startswith("--scenario=") for arg in args.args):
        parser.error("Specify the scenario once, after real/sim/estimate.")
    if args.scenario == "tracking" and args.mode != "sim":
        parser.error("Tracking provides policy-driven simulation: use sim tracking.")
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--port", type=int)
    options.add_argument("--headless", action="store_true")
    settings, _ = options.parse_known_args(args.args)
    help_only = any(arg in ("-h", "--help") for arg in args.args)
    if not help_only:
        missing = [
            name
            for name in ("numpy", "onnxruntime", "mujoco", "h5py", "glfw", "imageio_ffmpeg")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            parser.error(
                "Missing dependencies: " + ", ".join(missing)
                + ". Activate the project environment and install requirements.txt."
            )
        try:
            config = load_metadata()
        except (OSError, ValueError) as error:
            parser.error(f"Unable to load the sixthsense model: {error}")
        if args.mode == "sim" and not settings.headless:
            if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
                parser.error(
                    "Interactive simulation requires a graphical desktop. "
                    "For terminal testing, add --headless --seconds 10."
                )
            if sys.platform == "darwin":
                import mujoco.viewer

                if getattr(mujoco.viewer, "_MJPYTHON", None) is None:
                    launcher = Path(sys.executable).parent / "mjpython"
                    if not launcher.is_file() or os.environ.get("G1_NATIVE_MJPYTHON") == "1":
                        parser.error("Native MuJoCo requires mjpython from the same environment.")
                    os.execve(
                        str(launcher),
                        [str(launcher), str(PROJECT / "sixthsense.py"), "sim", args.scenario, *args.args],
                        dict(os.environ, G1_NATIVE_MJPYTHON="1"),
                    )
        label = {
            "real": "Recorded robot replay",
            "sim": "MuJoCo simulation",
            "estimate": "Recording inference",
        }[args.mode]
        if not (args.mode == "sim" and not settings.headless):
            print(f"{config.get('display_name', config['name'])} | {label} | {args.scenario}", flush=True)
    if args.mode == "real" and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    forwarded = ["--scenario", args.scenario, *args.args]
    if args.mode == "real" and settings.port is None:
        forwarded += ["--port", str(REPLAY_PORTS[args.scenario]), "--auto-port"]
    module = {
        "real": "offline.web_viewer",
        "sim": "online.deploy",
        "estimate": "offline.estimate_contact",
    }[args.mode]
    sys.argv = [f"sixthsense.py {args.mode}", *forwarded]
    try:
        runpy.run_module(module, run_name="__main__")
        return 0
    except KeyboardInterrupt:
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
