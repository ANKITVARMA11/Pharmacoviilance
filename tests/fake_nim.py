"""A throwaway HTTP server on localhost that behaves like NVIDIA NIM's OpenAI-compatible API.

  GET  /v1/models            -> the models you give it
  POST /v1/chat/completions  -> a scripted reply, 410 for retired models, otherwise `reply`

Used with the real `openai` SDK and with the real command line, so no mocks sit between our code and the wire.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GOOD = (
    '{"patient_age": 67, "patient_sex": "female", "suspect_drug": "Amoxicillin", '
    '"reactions": ["rash"], "serious": true, "outcome": "recovering"}'
)


def completion(content):
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # dropped connections are expected in one test
        pass


class FakeNimServer:
    """script: optional list of (status, json) or 'drop' used for chat requests first (the last item repeats)."""

    def __init__(self, script=None, models=(), retired=(), reply=GOOD):
        self.script = list(script) if script else None
        self.models = list(models)
        self.retired = set(retired)
        self.reply = reply
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append({"method": "GET", "path": self.path,
                                       "auth": self.headers.get("authorization"), "body": None})
                self._send(200, {"object": "list", "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": m.split("/")[0]} for m in outer.models]})

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append({"method": "POST", "path": self.path,
                                       "auth": self.headers.get("authorization"), "body": body})
                if outer.script:
                    item = outer.script.pop(0) if len(outer.script) > 1 else outer.script[0]
                    if item == "drop":
                        self.connection.close()
                        return
                    return self._send(*item)
                if body.get("model") in outer.retired:
                    return self._send(410, {"type": "about:blank", "title": "Gone", "status": 410,
                                            "detail": f"The model '{body.get('model')}' has reached its end of life."})
                self._send(200, completion(outer.reply))

            def log_message(self, *a):
                pass

        self._srv = _Server(("127.0.0.1", 0), Handler)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._srv.server_port}/v1"

    @property
    def chat_requests(self):
        return [r for r in self.requests if r["method"] == "POST"]

    def start(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self._srv.shutdown()
        self._srv.server_close()
