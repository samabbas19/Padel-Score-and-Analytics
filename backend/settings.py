from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BackendSettings:
    repo_root: Path = DEFAULT_REPO_ROOT
    jobs_dir: Path = DEFAULT_REPO_ROOT / "backend_jobs"
    run_script: Path = DEFAULT_REPO_ROOT / "run.ps1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        object.__setattr__(self, "jobs_dir", Path(self.jobs_dir).resolve())
        object.__setattr__(self, "run_script", Path(self.run_script).resolve())

