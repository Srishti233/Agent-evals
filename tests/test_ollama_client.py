"""Exercises the real OllamaLLM HTTP client against a scripted fake Ollama server."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from common.llm import (LLMBadOutputError, LLMError, LLMTimeoutError, LLMUnavailableError, OllamaLLM, chat_json)


class FakeOllama:
    def __init__(self):
        self.chat_script: list = []     # items: str | ("status", code) | ("sleep", seconds, str)
        self.requests: list[dict] = []
        self.models = ["llama3.2:3b"]
        self.pull_events: list[dict] = [{"status": "pulling"}, {"status": "downloading", "completed": 5, "total": 10},
                                        {"status": "success"}]
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.models]})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, **body})
                if self.path == "/api/pull":
                    self.send_response(200)
                    self.end_headers()
                    for ev in outer.pull_events:
                        self.wfile.write((json.dumps(ev) + "\n").encode())
                        self.wfile.flush()
                    return
                item = outer.chat_script.pop(0) if outer.chat_script else "{}"
                if isinstance(item, tuple) and item[0] == "status":
                    self._send(item[1], {"error": "boom"})
                elif isinstance(item, tuple) and item[0] == "sleep":
                    time.sleep(item[1])
                    self._send(200, {"message": {"role": "assistant", "content": item[2]}})
                else:
                    self._send(200, {"message": {"role": "assistant", "content": item}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def fake():
    f = FakeOllama()
    yield f
    f.close()


def client(url, **kw):
    kw.setdefault("backoff_s", 0.0)
    kw.setdefault("max_retries", 3)
    kw.setdefault("timeout_s", 5)
    return OllamaLLM(url, "llama3.2:3b", **kw)


def test_chat_ok_and_payload(fake):
    fake.chat_script = ["hello"]
    out = client(fake.url, seed=7).chat([{"role": "user", "content": "hi"}], temperature=0.3, json_mode=True, max_tokens=50)
    assert out == "hello"
    sent = fake.requests[0]
    assert sent["format"] == "json" and sent["stream"] is False and sent["model"] == "llama3.2:3b"
    assert sent["options"]["temperature"] == 0.3 and sent["options"]["seed"] == 7 and sent["options"]["num_predict"] == 50


def test_list_and_has_model(fake):
    c = client(fake.url)
    assert c.list_models() == ["llama3.2:3b"]
    assert c.has_model("llama3.2:3b") and not c.has_model("qwen2.5:7b")
    fake.models = ["foo:latest"]
    assert c.has_model("foo")


def test_ollama_down_raises_unavailable():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listening on this port now
    c = client(f"http://127.0.0.1:{port}", max_retries=2)
    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        c.chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMUnavailableError):
        c.list_models()


def test_timeout_retries_then_raises(fake):
    fake.chat_script = [("sleep", 1.0, "late")] * 2
    c = client(fake.url, timeout_s=0.2, max_retries=2)
    t0 = time.time()
    with pytest.raises(LLMTimeoutError):
        c.chat([{"role": "user", "content": "x"}])
    assert len([r for r in fake.requests if r["path"] == "/api/chat"]) == 2
    assert time.time() - t0 < 5


def test_5xx_is_retried_and_recovers(fake):
    fake.chat_script = [("status", 503), ("status", 500), "ok"]
    assert client(fake.url).chat([{"role": "user", "content": "x"}]) == "ok"
    assert len(fake.requests) == 3


def test_404_model_missing_is_not_retried(fake):
    fake.chat_script = [("status", 404)]
    with pytest.raises(LLMError, match="not found"):
        client(fake.url).chat([{"role": "user", "content": "x"}])
    assert len(fake.requests) == 1


def _need_a(obj):
    if "a" not in obj:
        raise ValueError("missing a")
    return obj


def test_chat_json_recovers_from_malformed_then_valid(fake):
    fake.chat_script = ["this is not json", '{"b": 1}', '```json\n{"a": 2,}\n```']
    out = chat_json(client(fake.url), [{"role": "user", "content": "q"}], _need_a, retries=3)
    assert out == {"a": 2}
    last = fake.requests[-1]["messages"]
    assert "invalid" in last[-1]["content"].lower() and last[-2]["role"] == "assistant"


def test_chat_json_gives_up(fake):
    fake.chat_script = ["nope"] * 3
    with pytest.raises(LLMBadOutputError):
        chat_json(client(fake.url), [{"role": "user", "content": "q"}], _need_a, retries=3)
    assert len(fake.requests) == 3


def test_chat_json_does_not_swallow_transport_errors(fake):
    fake.chat_script = [("status", 404)]
    with pytest.raises(LLMError):
        chat_json(client(fake.url), [{"role": "user", "content": "q"}], _need_a)


def test_pull_streams_progress(fake):
    seen = []
    client(fake.url).pull("llama3.2:3b", lambda s, d, t: seen.append((s, d, t)))
    assert seen[-1][0] == "success" and ("downloading", 5, 10) in seen


def test_pull_error_event(fake):
    fake.pull_events = [{"error": "manifest unknown"}]
    with pytest.raises(LLMError, match="manifest unknown"):
        client(fake.url).pull("nope:1b")
