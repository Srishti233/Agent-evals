"""Console + file logging."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_dir / "agent_evals.log", encoding="utf-8"))
        except OSError:
            pass  # read-only FS: console logging is enough
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy in ("urllib3", "httpx", "transformers", "sentence_transformers", "faiss", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
