#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import threading
import webbrowser

from project_paths import dataset_root, SCENARIO_NAMES, ffmpeg_executable
from .http import open_server
from .store import StoreRegistry

def main():
    parser = argparse.ArgumentParser(description='Start the local recording viewer and manage its lifetime.')
    parser.add_argument("--root", type=Path)
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="stand")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Choose a free port if the requested port is already in use",
    )
    parser.add_argument("--pair", type=int, help="Sample number to open initially")
    parser.add_argument(
        "--no-browser", action="store_true", help="Start the server without opening a browser"
    )
    args = parser.parse_args()
    try:
        args.root = dataset_root(args.scenario, args.root)
    except ValueError as error:
        parser.error(str(error))
    if args.pair is not None:
        numbers = {
            row["number"] for row in json.loads((args.root / "manifest.json").read_text())["pairs"]
        }
        if args.pair not in numbers:
            parser.error(f"Unknown {args.scenario} sample: {args.pair}")
    try:
        ffmpeg_executable()
    except RuntimeError as error:
        parser.error(str(error))
    stores = StoreRegistry(args.scenario, args.root)
    try:
        server = open_server(args.port, args.auto_port)
    except OSError as exc:
        stores.close(wait=False)
        parser.error(
            f"Cannot use port {args.port}: {exc}. Open the existing viewer or choose another --port."
        )
    server.stores = stores
    if args.port and server.server_port != args.port:
        print(
            f"Port {args.port} is in use; selected available port {server.server_port}.", flush=True
        )
    url = f"http://127.0.0.1:{server.server_port}" + (
        f"/?pair={args.pair}" if args.pair is not None else ""
    )
    print(f"G1 dataset viewer: {url}", flush=True)
    print("Ctrl+C to stop. Data is read only; caches and predictions are in outputs/.", flush=True)
    if not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stores.close(wait=False)

if __name__ == "__main__":
    main()
