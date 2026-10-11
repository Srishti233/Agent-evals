import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import load_config  # noqa: E402


@pytest.fixture()
def offline_cfg(tmp_path):
    """Offline (mock LLM / hash embeddings / lexical NLI) config with all state under tmp_path."""
    override = tmp_path / "test_config.yaml"
    override.write_text(
        f"extends: {ROOT / 'config.offline.yaml'}\n"
        f"paths:\n"
        f"  docs_dir: {ROOT / 'data/docs'}\n"
        f"  eval_set: {ROOT / 'data/eval_set.json'}\n"
        f"  index_dir: {tmp_path / 'index'}\n"
        f"  db: {tmp_path / 'runs.db'}\n"
        f"  versions_dir: {tmp_path / 'versions'}\n"
        f"  changelog: {tmp_path / 'CHANGELOG.md'}\n"
        f"  log_dir: {tmp_path / 'logs'}\n"
        f"eval:\n  workers: 4\n")
    return load_config(override)
