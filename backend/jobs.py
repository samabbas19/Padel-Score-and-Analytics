from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import threading
import uuid

from backend.pipeline import PipelinePlan, PipelineRequest, build_pipeline_plan
from backend.settings import BackendSettings


TERMINAL_STATUSES = {"succeeded", "failed"}


@dataclass
class JobRecord:
    job_id: str
    status: str
    request: dict
    video_id: str
    output_dir: str
    cache_dir: str
    command: list[str]
    env: dict[str, str]
    log_path: str
    artifacts: list[dict] = field(default_factory=list)
    reports: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: utc_now())
    updated_at: str = field(default_factory=lambda: utc_now())
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None


class JobService:
    def __init__(
        self,
        settings: BackendSettings | None = None,
        runner=None,
        run_inline: bool = False,
    ) -> None:
        self.settings = settings or BackendSettings()
        self.runner = runner or self._default_runner
        self.run_inline = run_inline
        self._lock = threading.Lock()
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, request: PipelineRequest) -> dict:
        plan = build_pipeline_plan(request, self.settings)
        job_id = uuid.uuid4().hex
        log_path = self.settings.jobs_dir / f"{job_id}.log"
        record = JobRecord(
            job_id=job_id,
            status="queued",
            request=serialize_request(plan.request),
            video_id=plan.video_id,
            output_dir=str(plan.output_dir),
            cache_dir=str(plan.cache_dir),
            command=plan.command,
            env=plan.env,
            log_path=str(log_path),
            artifacts=serialize_artifacts(plan),
        )
        self._save(record)

        if self.run_inline:
            self._run(job_id, plan, log_path)
        else:
            thread = threading.Thread(
                target=self._run,
                args=(job_id, plan, log_path),
                name=f"padel-job-{job_id[:8]}",
                daemon=True,
            )
            thread.start()
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        path = self._job_path(job_id)
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        jobs = [json.loads(path.read_text(encoding="utf-8")) for path in self.settings.jobs_dir.glob("*.json")]
        return sorted(jobs, key=lambda job: job["created_at"], reverse=True)

    def refresh_artifacts(self, job_id: str) -> dict:
        record = self.get(job_id)
        plan = plan_from_record(record, self.settings)
        record["artifacts"] = serialize_artifacts(plan)
        record["reports"] = collect_reports(plan)
        record["updated_at"] = utc_now()
        self._save_dict(record)
        return record

    def _run(self, job_id: str, plan: PipelinePlan, log_path: Path) -> None:
        record = self.get(job_id)
        record["status"] = "running"
        record["started_at"] = utc_now()
        record["updated_at"] = utc_now()
        self._save_dict(record)

        try:
            exit_code = int(self.runner(plan, log_path))
            record = self.get(job_id)
            record["exit_code"] = exit_code
            record["status"] = "succeeded" if exit_code == 0 else "failed"
            if exit_code != 0:
                record["error"] = f"Runner exited with code {exit_code}"
        except Exception as exc:
            record = self.get(job_id)
            record["status"] = "failed"
            record["error"] = str(exc)
            record["exit_code"] = -1
        finally:
            record["finished_at"] = utc_now()
            record["updated_at"] = utc_now()
            record["artifacts"] = serialize_artifacts(plan)
            record["reports"] = collect_reports(plan)
            self._save_dict(record)

    def _default_runner(self, plan: PipelinePlan, log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(plan.env)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                plan.command,
                cwd=self.settings.repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return result.returncode

    def _save(self, record: JobRecord) -> None:
        self._save_dict(asdict(record))

    def _save_dict(self, record: dict) -> None:
        with self._lock:
            path = self._job_path(record["job_id"])
            temp_path = path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            temp_path.replace(path)

    def _job_path(self, job_id: str) -> Path:
        safe_job_id = "".join(ch for ch in job_id if ch.isalnum() or ch in {"_", "-"})
        return self.settings.jobs_dir / f"{safe_job_id}.json"


def serialize_request(request: PipelineRequest) -> dict:
    data = asdict(request)
    data["video_path"] = str(data["video_path"])
    return data


def serialize_artifacts(plan: PipelinePlan) -> list[dict]:
    return [
        {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        for name, path in plan.artifacts.items()
    ]


def collect_reports(plan: PipelinePlan) -> dict:
    reports = {}
    for name, path in plan.artifacts.items():
        if not name.endswith("_report") and name != "score_summary":
            continue
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        try:
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reports[name] = {"error": "invalid_json", "path": str(path)}
    return reports


def plan_from_record(record: dict, settings: BackendSettings) -> PipelinePlan:
    request_data = dict(record["request"])
    request_data["video_path"] = Path(request_data["video_path"])
    request = PipelineRequest(**request_data)
    return build_pipeline_plan(request, settings)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
