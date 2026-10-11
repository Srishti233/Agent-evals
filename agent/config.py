"""Versioned agent configuration (system prompt, top_k, temperature, chunk size)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

POINTER = "CURRENT"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentConfig:
    version: int
    system_prompt: str
    top_k: int = 3
    temperature: float = 0.2
    chunk_size: int = 120
    parent: int | None = None
    status: str = "baseline"          # baseline | candidate | accepted | rejected
    notes: str = ""
    created_at: str = field(default_factory=_now)

    @property
    def name(self) -> str:
        return f"v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentConfig":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def parse_version(v: int | str) -> int:
    m = re.fullmatch(r"v?(\d+)", str(v).strip().lower())
    if not m:
        raise ValueError(f"Bad version identifier: {v!r} (expected e.g. 2 or 'v2')")
    return int(m.group(1))


class VersionStore:
    """Reads/writes agent/versions/v{N}.json plus a CURRENT pointer file."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, version: int) -> Path:
        return self.dir / f"v{version}.json"

    def list_versions(self) -> list[int]:
        nums = [int(m.group(1)) for p in self.dir.glob("v*.json") if (m := re.fullmatch(r"v(\d+)\.json", p.name))]
        return sorted(nums)

    def load(self, version: int | str) -> AgentConfig:
        n = parse_version(version)
        p = self._path(n)
        if not p.exists():
            raise FileNotFoundError(f"No such agent version: v{n} ({p})")
        return AgentConfig.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def save(self, cfg: AgentConfig) -> Path:
        p = self._path(cfg.version)
        p.write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return p

    def next_version(self) -> int:
        vs = self.list_versions()
        return (vs[-1] + 1) if vs else 1

    def current_version(self) -> int:
        ptr = self.dir / POINTER
        if ptr.exists():
            return parse_version(ptr.read_text(encoding="utf-8").strip())
        vs = self.list_versions()
        if not vs:
            raise FileNotFoundError(f"No agent versions in {self.dir}")
        return vs[0]

    def current(self) -> AgentConfig:
        return self.load(self.current_version())

    def set_current(self, version: int | str) -> None:
        n = parse_version(version)
        if not self._path(n).exists():
            raise FileNotFoundError(f"Cannot point CURRENT at missing v{n}")
        (self.dir / POINTER).write_text(f"v{n}\n", encoding="utf-8")

    def ensure_baseline(self, baseline: dict[str, Any]) -> AgentConfig:
        """Create v1 from config if the directory is empty; return the current version."""
        if not self.list_versions():
            v1 = AgentConfig(version=1, system_prompt=" ".join(str(baseline["system_prompt"]).split()),
                             top_k=int(baseline.get("top_k", 3)),
                             temperature=float(baseline.get("temperature", 0.2)),
                             chunk_size=int(baseline.get("chunk_size", 120)),
                             status="baseline", notes="Initial baseline prompt")
            self.save(v1)
            self.set_current(1)
        elif not (self.dir / POINTER).exists():
            self.set_current(self.list_versions()[0])
        return self.current()


def store_from_cfg(cfg: Any) -> VersionStore:
    store = VersionStore(cfg.resolve("paths.versions_dir"))
    store.ensure_baseline(cfg.agent.baseline)
    return store
