from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

from backend.settings import BackendSettings


ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
ALLOWED_SCORE_SOURCES = {"proposal", "gameplay", "scoreboard"}


@dataclass(frozen=True)
class PipelineRequest:
    video_path: Path
    initial_points: str = "0-0"
    initial_games: str = "0-0"
    proposal_min_time_seconds: float = 0.0
    score_source: str = "proposal"
    render_overlay: bool = True
    validate_score_timeline: bool = True
    extract_scoreboard_ground_truth: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class PipelinePlan:
    request: PipelineRequest
    video_id: str
    output_dir: Path
    cache_dir: Path
    command: list[str]
    env: dict[str, str]
    artifacts: dict[str, Path]


def build_pipeline_plan(request: PipelineRequest, settings: BackendSettings) -> PipelinePlan:
    video_path = normalize_video_path(request.video_path)
    score_source = request.score_source.strip().lower()
    if score_source not in ALLOWED_SCORE_SOURCES:
        raise ValueError(f"Unsupported score source: {request.score_source}")

    normalized = PipelineRequest(
        video_path=video_path,
        initial_points=normalize_score_pair(request.initial_points, "0-0"),
        initial_games=normalize_score_pair(request.initial_games, "0-0"),
        proposal_min_time_seconds=float(request.proposal_min_time_seconds),
        score_source=score_source,
        render_overlay=bool(request.render_overlay),
        validate_score_timeline=bool(request.validate_score_timeline),
        extract_scoreboard_ground_truth=bool(request.extract_scoreboard_ground_truth),
        dry_run=bool(request.dry_run),
    )
    video_id = video_id_for_path(video_path)
    output_dir = settings.repo_root / "outputs" / video_id
    cache_dir = settings.repo_root / "cache" / video_id
    env = {
        "PADEL_SCORE_INITIAL_POINTS": normalized.initial_points,
        "PADEL_SCORE_INITIAL_GAMES": normalized.initial_games,
        "PADEL_PROPOSAL_MIN_TIME_SECONDS": str(normalized.proposal_min_time_seconds),
        "PADEL_SCORE_SOURCE": normalized.score_source,
        "PADEL_RENDER_SCORE_OVERLAY": bool_to_env(normalized.render_overlay),
        "PADEL_VALIDATE_SCORE_TIMELINE": bool_to_env(normalized.validate_score_timeline),
        "PADEL_EXTRACT_SCOREBOARD_GROUND_TRUTH": bool_to_env(
            normalized.extract_scoreboard_ground_truth
        ),
        "PADEL_DRY_RUN": bool_to_env(normalized.dry_run),
    }
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(settings.run_script),
        "-VideoPath",
        str(video_path),
    ]
    return PipelinePlan(
        request=normalized,
        video_id=video_id,
        output_dir=output_dir,
        cache_dir=cache_dir,
        command=command,
        env=env,
        artifacts=artifact_paths(output_dir),
    )


def normalize_video_path(video_path: str | Path) -> Path:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a file: {path}")
    if path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video extension: {path.suffix}")
    return path


def normalize_score_pair(score: str, default: str) -> str:
    value = str(score or default).strip().upper()
    if "-" not in value:
        raise ValueError(f"Score must be formatted as left-right: {score}")
    left, right = [part.strip() for part in value.split("-", 1)]
    if not left or not right:
        raise ValueError(f"Score must be formatted as left-right: {score}")
    return f"{left}-{right}"


def video_id_for_path(path: str | Path) -> str:
    path = Path(path).resolve()
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-") or "video"
    return f"{safe_name}-{digest.hexdigest()[:12]}"


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "score_events": output_dir / "score_events.csv",
        "score_summary": output_dir / "score_summary.json",
        "score_overlay_video": output_dir / "results_score_overlay.mp4",
        "event_candidates": output_dir / "event_candidates.csv",
        "scoreboard_events": output_dir / "scoreboard_events.csv",
        "score_validation_report": output_dir / "score_validation_report.json",
        "gameplay_score_checkpoint_report": output_dir / "gameplay_score_checkpoint_report.json",
        "gameplay_score_sequence_report": output_dir / "gameplay_score_sequence_report.json",
        "scoreboard_transition_expansion_report": (
            output_dir / "scoreboard_transition_expansion_report.json"
        ),
        "winner_attribution_report": output_dir / "gameplay_winner_attribution_report.json",
    }


def bool_to_env(value: bool) -> str:
    return "1" if value else "0"

