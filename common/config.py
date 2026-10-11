"""Config loading: YAML with `extends:` support, env overrides and path resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class Cfg(dict):
    """A dict with nested attribute access (read-only convenience)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def resolve(self, dotted_path_key: str) -> Path:
        """Resolve e.g. 'paths.db' to an absolute path relative to the config file."""
        node: Any = self
        for part in dotted_path_key.split("."):
            node = node[part]
        p = Path(str(node))
        return p if p.is_absolute() else Path(self["_root"]) / p


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if parent:
        base = _read((path.parent / parent).resolve())
        data = _deep_merge(base, data)
    return data


def load_config(path: str | os.PathLike[str] | None = None) -> Cfg:
    """Load config from `path`, $AGENT_EVALS_CONFIG, or ./config.yaml."""
    chosen = Path(path or os.environ.get("AGENT_EVALS_CONFIG") or DEFAULT_CONFIG).resolve()
    if not chosen.exists():
        raise FileNotFoundError(f"Config file not found: {chosen}")
    data = _read(chosen)
    data["_root"] = str(chosen.parent)
    data["_path"] = str(chosen)
    host = os.environ.get("OLLAMA_HOST")
    if host:
        data.setdefault("llm", {})["host"] = host if "://" in host else f"http://{host}"
    return Cfg(data)
