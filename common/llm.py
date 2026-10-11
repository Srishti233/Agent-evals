"""LLM access: Ollama HTTP client (retries/timeouts/health/pull), JSON-mode helper, factory."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Protocol

import requests

from common.jsonparse import JSONParseError, extract_json

log = logging.getLogger(__name__)

Messages = list[dict[str, str]]


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMUnavailableError(LLMError):
    """Ollama is unreachable (not running / wrong host)."""


class LLMTimeoutError(LLMError):
    """The model did not answer in time."""


class LLMBadOutputError(LLMError):
    """The model kept returning output that failed JSON/schema validation."""


class LLM(Protocol):
    model: str

    def chat(self, messages: Messages, *, temperature: float = 0.0,
             json_mode: bool = False, max_tokens: int = 512) -> str: ...


class OllamaLLM:
    """Minimal Ollama client (plain `requests`, no extra SDK)."""

    def __init__(self, host: str, model: str, *, timeout_s: float = 180, connect_timeout_s: float = 5,
                 max_retries: int = 3, backoff_s: float = 1.5, num_ctx: int = 4096,
                 seed: int | None = None) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = (connect_timeout_s, timeout_s)
        self.max_retries = max(1, max_retries)
        self.backoff_s = backoff_s
        self.num_ctx = num_ctx
        self.seed = seed

    # ---- server / model management -------------------------------------------------
    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=(self.timeout[0], 15))
            r.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve` "
                f"(or set OLLAMA_HOST / llm.host to the machine running it)."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMUnavailableError(f"Ollama at {self.host} did not respond to /api/tags") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Ollama /api/tags failed: {exc}") from exc
        return [m.get("name", "") for m in r.json().get("models", [])]

    @staticmethod
    def _has(models: list[str], name: str) -> bool:
        wanted = name if ":" in name else f"{name}:latest"
        return wanted in models or name in models

    def has_model(self, name: str | None = None) -> bool:
        return self._has(self.list_models(), name or self.model)

    def pull(self, name: str | None = None,
             progress_cb: Callable[[str, int | None, int | None], None] | None = None) -> None:
        """Pull a model, streaming progress. Raises LLMError on failure."""
        name = name or self.model
        try:
            with requests.post(f"{self.host}/api/pull", json={"model": name, "stream": True},
                               stream=True, timeout=(self.timeout[0], 600)) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    ev = json.loads(raw)
                    if "error" in ev:
                        raise LLMError(f"Pull of {name} failed: {ev['error']}")
                    if progress_cb:
                        progress_cb(ev.get("status", ""), ev.get("completed"), ev.get("total"))
        except requests.exceptions.ConnectionError as exc:
            raise LLMUnavailableError(f"Cannot reach Ollama at {self.host}") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Pull of {name} failed: {exc}") from exc

    # ---- chat ------------------------------------------------------------------------
    def chat(self, messages: Messages, *, temperature: float = 0.0,
             json_mode: bool = False, max_tokens: int = 512) -> str:
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": self.num_ctx,
                                   "num_predict": max_tokens}
        if self.seed is not None:
            options["seed"] = self.seed
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "stream": False, "options": options}
        if json_mode:
            payload["format"] = "json"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                if r.status_code == 404:
                    raise LLMError(f"Model '{self.model}' not found on {self.host}. "
                                   f"Run `python main.py setup` (or `ollama pull {self.model}`).")
                if r.status_code >= 500:
                    raise requests.exceptions.RequestException(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                body = r.json()
                if "error" in body:
                    raise requests.exceptions.RequestException(str(body["error"]))
                return str(body.get("message", {}).get("content", ""))
            except requests.exceptions.ConnectionError as exc:
                last_exc = LLMUnavailableError(f"Cannot reach Ollama at {self.host}. Start it with `ollama serve` "
                                               f"(or set OLLAMA_HOST / llm.host). Details: {type(exc).__name__}")
            except requests.exceptions.Timeout as exc:
                last_exc = LLMTimeoutError(f"Ollama timed out after {self.timeout[1]}s ({self.model})")
                log.warning("timeout (attempt %d/%d): %s", attempt + 1, self.max_retries, exc)
            except LLMError:
                raise  # non-retryable (e.g. model missing)
            except (requests.exceptions.RequestException, ValueError) as exc:
                last_exc = LLMError(f"Ollama request failed: {exc}")
                log.warning("request error (attempt %d/%d): %s", attempt + 1, self.max_retries, exc)
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff_s * (2 ** attempt))
        assert last_exc is not None
        raise last_exc


def chat_json(llm: LLM, messages: Messages, validator: Callable[[dict[str, Any]], dict[str, Any]], *,
              temperature: float = 0.0, max_tokens: int = 512, retries: int = 3) -> dict[str, Any]:
    """Ask for JSON; on malformed/invalid output, tell the model what was wrong and retry.

    `validator` receives the parsed dict and returns a cleaned dict, or raises ValueError.
    Transport errors (LLMError subclasses) propagate immediately.
    """
    msgs = list(messages)
    last_problem = "unknown"
    for attempt in range(max(1, retries)):
        text = llm.chat(msgs, temperature=temperature, json_mode=True, max_tokens=max_tokens)
        try:
            return validator(extract_json(text))
        except (JSONParseError, ValueError) as exc:
            last_problem = str(exc)
            log.debug("invalid JSON reply (attempt %d/%d): %s", attempt + 1, retries, exc)
            msgs = msgs + [
                {"role": "assistant", "content": text[:1500]},
                {"role": "user", "content": f"That reply was invalid ({last_problem}). "
                                            "Reply again with ONLY one valid JSON object that matches the schema."},
            ]
    raise LLMBadOutputError(f"No valid JSON after {retries} attempts: {last_problem}")


def get_llm(cfg: Any, role: str = "agent") -> LLM:
    """Build the LLM for `role` in {'agent', 'judge'} from config."""
    c = cfg.llm
    if c.backend == "mock":
        from common.mock_llm import MockLLM
        return MockLLM()
    if c.backend != "ollama":
        raise ValueError(f"Unknown llm.backend: {c.backend!r}")
    model = c.judge_model if role == "judge" else c.model
    return OllamaLLM(c.host, model, timeout_s=c.timeout_s, connect_timeout_s=c.get("connect_timeout_s", 5),
                     max_retries=c.max_retries, backoff_s=c.backoff_s, num_ctx=c.get("num_ctx", 4096),
                     seed=c.get("seed"))
