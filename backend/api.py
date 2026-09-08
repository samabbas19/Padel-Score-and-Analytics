from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from backend.jobs import JobService
from backend.pipeline import PipelineRequest


class SubmitJobRequest(BaseModel):
    video_path: str
    initial_points: str = Field(default="0-0")
    initial_games: str = Field(default="0-0")
    proposal_min_time_seconds: float = Field(default=0.0, ge=0.0)
    score_source: str = Field(default="proposal")
    render_overlay: bool = True
    validate_score_timeline: bool = True
    extract_scoreboard_ground_truth: bool = True
    dry_run: bool = False


def create_app(service: JobService | None = None) -> FastAPI:
    job_service = service or JobService()
    app = FastAPI(
        title="Padel Analytics Backend",
        version="1.0.0",
        description="Local backend for GPU padel scoring jobs.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "padel-analytics-backend"}

    @app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(payload: SubmitJobRequest) -> dict:
        try:
            request = PipelineRequest(
                video_path=Path(payload.video_path),
                initial_points=payload.initial_points,
                initial_games=payload.initial_games,
                proposal_min_time_seconds=payload.proposal_min_time_seconds,
                score_source=payload.score_source,
                render_overlay=payload.render_overlay,
                validate_score_timeline=payload.validate_score_timeline,
                extract_scoreboard_ground_truth=payload.extract_scoreboard_ground_truth,
                dry_run=payload.dry_run,
            )
            return job_service.submit(request)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs")
    def list_jobs() -> dict:
        return {"jobs": job_service.list()}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return job_service.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/v1/jobs/{job_id}/artifacts")
    def get_job_artifacts(job_id: str) -> dict:
        try:
            record = job_service.refresh_artifacts(job_id)
            return {
                "job_id": record["job_id"],
                "status": record["status"],
                "artifacts": record["artifacts"],
                "reports": record["reports"],
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    return app


app = create_app()

