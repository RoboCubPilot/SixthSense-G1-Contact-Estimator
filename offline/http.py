import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self.respond(False)

    def do_GET(self):
        self.respond(True)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            match = re.fullmatch(r"/api/inference/(\d+)", parsed.path)
            if not match:
                self.send_error(404)
                return
            self.json(self.select_store(parsed).start_inference(int(match[1])))
        except (ValueError, KeyError) as exc:
            self.json({"error": str(exc)}, status=400)
        except Exception as exc:
            self.json({"error": str(exc)}, status=500)

    def json(self, value, body=True, status=200):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if body:
            self.wfile.write(raw)

    def select_store(self, parsed):
        scenario = parse_qs(parsed.query).get("scenario", [None])[0]
        return self.server.stores.get(scenario)

    def file(self, path, mime, body):
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]) if match[2] else end, end)
            else:
                start = max(0, size - int(match[2]))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header(
            "Cache-Control", "no-cache" if mime != "video/mp4" else "private, max-age=86400"
        )
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if body:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(262144, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    def respond(self, body):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            static = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("style.css", "text/css; charset=utf-8"),
            }
            if path in static:
                name, mime = static[path]
                self.file(HERE / "web" / name, mime, body)
                return
            store = self.select_store(parsed)
            if path == "/api/catalog":
                self.json(dict(store.catalog(), scenarios=self.server.stores.available()), body)
                return
            match = re.fullmatch(r"/api/inference/([a-f0-9]{32})", path)
            if match:
                self.json(store.inference_status(match[1]), body)
                return
            match = re.fullmatch(r"/api/(pair|render)/(\d+)", path)
            if match:
                kind, number = match[1], int(match[2])
                mode = parse_qs(parsed.query).get("base", ["fixed"])[0]
                if kind == "pair":
                    value = store.pair(number, mode)
                else:
                    value = store.render_status(
                        number, mode, parse_qs(parsed.query).get("retry") == ["1"]
                    )
                self.json(value, body)
                return
            match = re.fullmatch(r"/video/(\d+)/([12])\.mp4", path)
            if match and int(match[1]) in store.rows:
                self.file(
                    store.root / match[1] / f"camera_{match[2]}.mp4", "video/mp4", body
                )
                return
            match = re.fullmatch(r"/rendered/([0-9]+-fixed-[a-f0-9]{24})\.mp4", path)
            if match and int(match[1].split("-")[0]) in store.rows:
                self.file(store.cache_dir / (match[1] + ".mp4"), "video/mp4", body)
                return
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, KeyError) as exc:
            self.json({"error": str(exc)}, body, 400)
        except Exception as exc:
            print(f"Request failed: {exc}", file=sys.stderr, flush=True)
            self.json({"error": str(exc)}, body, 500)

def open_server(port, auto_port=False):
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as error:
        if not auto_port or error.errno != errno.EADDRINUSE:
            raise

        return ThreadingHTTPServer(("127.0.0.1", 0), Handler)
