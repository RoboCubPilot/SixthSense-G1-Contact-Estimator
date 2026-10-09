from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parent
DATASETS = PROJECT / "dataset"
DATASET = DATASETS / "stand"
SCENARIOS = PROJECT / "assets"
SCENARIO_NAMES = ("stand", "walk", "lie")
SIM_SCENARIO_NAMES = (*SCENARIO_NAMES, "tracking")

MODEL = PROJECT / "models/sixthsense/sixthsense.onnx"
REPLAY_MODEL = PROJECT / "assets/robot/replay.xml"
MODEL_NAME = "sixthsense"
REPLAY_PORTS = {"stand": 8784, "walk": 8785, "lie": 8786}
OUTPUTS = PROJECT / "outputs"

def normalize_sim_scenario(scenario):

    return "tracking" if scenario == "twist2" else scenario

def dataset_root(scenario="stand", root=None):
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario: {scenario}")
    path = Path(root).resolve() if root is not None else DATASETS / scenario
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise ValueError(
            f"Missing {scenario} data: {manifest}. Extract the complete project archive."
        )
    if json.loads(manifest.read_text()).get("scenario", "stand") != scenario:
        raise ValueError(f"The data directory does not match scenario {scenario}.")
    return path

def ffmpeg_executable():
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()
