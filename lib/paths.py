"""Project-root resolution, in one place.

Every other module imports ``PROJECT_ROOT`` from here instead of recomputing
``Path(__file__).resolve().parents[N]``, so moving a file between directories
cannot silently break the lookup of ``external_repos/``.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPOS_DIR = PROJECT_ROOT / "external_repos"


def get_external_repo_dir(repo_name: str) -> Path:
    repo_dir = EXTERNAL_REPOS_DIR / repo_name
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"External repository not found: {repo_dir}. "
            f"Clone the repository into {EXTERNAL_REPOS_DIR} first."
        )
    return repo_dir


def ensure_external_repo_on_path(repo_name: str, *relative_parts: str) -> Path:
    repo_dir = get_external_repo_dir(repo_name)
    import_dir = repo_dir.joinpath(*relative_parts) if relative_parts else repo_dir
    if not import_dir.exists():
        raise FileNotFoundError(f"External repository import path not found: {import_dir}")

    import_dir_str = str(import_dir)
    if import_dir_str not in sys.path:
        sys.path.insert(0, import_dir_str)
    return import_dir
