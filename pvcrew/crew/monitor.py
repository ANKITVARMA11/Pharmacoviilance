"""The live monitor: a small local web server that shows the crew at work in its own browser window.

Started together with the Streamlit page. It serves one self-contained page (monitor.html) and a JSON feed of the
events from the EventBus. The page polls the feed and draws the group chat and the AutoGen diagram.
Only reachable from this computer (127.0.0.1).
"""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import config

PAGE = Path(__file__).with_name("monitor.html")
_opened = False
_open_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    bus = None  # set per server

    def log_message(self, *args):  # keep the Streamlit console clean
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/events":
            try:
                since = int(parse_qs(url.query).get("since", ["0"])[0])
            except ValueError:
                since = 0
            body = {**self.bus.state(), "events": self.bus.since(since)}
            self._send(200, json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"), "application/json")
        elif url.path == "/api/state":
            self._send(200, json.dumps(self.bus.state()).encode("utf-8"), "application/json")
        else:
            self._send(404, b"not found", "text/plain")


class MonitorServer:
    def __init__(self, bus, host: str = config.MONITOR_HOST, port: int = config.MONITOR_PORT):
        self.bus, self.host, self.port, self._server = bus, host, port, None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> str:
        handler = type("Handler", (_Handler,), {"bus": self.bus})
        last_error = None
        for port in ([self.port] if self.port == 0 else range(self.port, self.port + 20)):
            try:
                self._server = ThreadingHTTPServer((self.host, port), handler)
                break
            except OSError as e:  # port busy: try the next one
                last_error = e
        if self._server is None:
            raise OSError(f"no free port for the monitor near {self.port}: {last_error}")
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, name="crew-monitor", daemon=True).start()
        return self.url

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def open_window(url: str) -> bool:
    """Ask the default browser for a new window, once per process. Browsers decide whether it is a window or a tab."""
    global _opened
    with _open_lock:
        if _opened:
            return False
        _opened = True
    try:
        return bool(webbrowser.open_new(url))
    except Exception:  # noqa: BLE001 - a missing browser must never stop the app
        return False
