"""Diagnostics for connecting gameplay events to score transitions.

This module is validation-only. It compares no-scoreboard event candidates
against scoreboard-derived ground truth to identify whether failures come from
missing point boundaries, timing lag, or winner attribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.scoreboard_ground_truth import (
    clean_score_text,
    format_optional_games,
    infer_winner_from_score_transition,
)


def align_event_candidates_to_scoreboard(
    scoreboard_events_path: str | Path,
    event_candidates_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    tolerance_seconds: float = 2.0,
    search_before_seconds: float = 8.0,
    search_after_seconds: float = 3.0,
) -> dict:
    """Write nearest gameplay event candidate for each scoreboard transition."""

    scoreboard_events_path = Path(scoreboard_events_path)
    event_candidates_path = Path(event_candidates_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)

    scoreboard_events = pd.read_csv(scoreboard_events_path)
    event_candidates = pd.read_csv(event_candidates_path)

    rows = []
    for transition in scoreboard_events.itertuples(index=False):
        candidates = event_candidates[
            (event_candidates["time"] >= float(transition.timestamp) - search_before_seconds)
            & (event_candidates["time"] <= float(transition.timestamp) + search_after_seconds)
        ].copy()
        if candidates.empty:
            rows.append(empty_alignment_row(transition, "no_candidate_in_search_window"))
            continue

        candidates["abs_dt"] = (candidates["time"] - float(transition.timestamp)).abs()
        candidates["before_scoreboard"] = candidates["time"] <= float(transition.timestamp)
        nearest = candidates.sort_values(["abs_dt", "confidence"], ascending=[True, False]).iloc[0]
        nearest_before = candidates[candidates["before_scoreboard"]].copy()
        before = (
            nearest_before.sort_values(["abs_dt", "confidence"], ascending=[True, False]).iloc[0]
            if not nearest_before.empty
            else None
        )

        dt = float(nearest["time"] - float(transition.timestamp))
        before_dt = (
            float(before["time"] - float(transition.timestamp))
            if before is not None
            else np.nan
        )
        rows.append(
            {
                "transition": int(transition.event),
                "scoreboard_frame": int(transition.frame),
                "scoreboard_timestamp": float(transition.timestamp),
                "scoreboard_points": transition.points,
                "scoreboard_games": transition.games,
                "nearest_frame": int(nearest["frame"]),
                "nearest_time": float(nearest["time"]),
                "nearest_dt": dt,
                "nearest_event_type": nearest["event_type"],
                "nearest_confidence": float(nearest["confidence"]),
                "nearest_audio_peak_z": float(nearest["audio_peak_z"]),
                "nearest_ball_x": clean_float(nearest.get("ball_x")),
                "nearest_ball_y": clean_float(nearest.get("ball_y")),
                "nearest_before_frame": int(before["frame"]) if before is not None else "",
                "nearest_before_time": float(before["time"]) if before is not None else "",
                "nearest_before_dt": before_dt,
                "nearest_before_event_type": before["event_type"] if before is not None else "",
                "nearest_before_confidence": (
                    float(before["confidence"]) if before is not None else ""
                ),
                "within_tolerance": abs(dt) <= tolerance_seconds,
                "before_within_tolerance": (
                    bool(before is not None and abs(before_dt) <= tolerance_seconds)
                ),
                "failure_class": classify_alignment_failure(
                    dt=dt,
                    before_dt=before_dt,
                    has_before=before is not None,
                    tolerance=tolerance_seconds,
                ),
            }
        )

    alignment = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(output_path, index=False)

    summary = {
        "mode": "gameplay_event_to_scoreboard_alignment",
        "scoreboard_events_path": str(scoreboard_events_path),
        "event_candidates_path": str(event_candidates_path),
        "alignment_path": str(output_path),
        "scoreboard_transitions": int(len(scoreboard_events)),
        "nearest_within_tolerance": int(alignment["within_tolerance"].sum())
        if not alignment.empty
        else 0,
        "nearest_before_within_tolerance": int(alignment["before_within_tolerance"].sum())
        if not alignment.empty
        else 0,
        "failure_classes": alignment["failure_class"].value_counts().to_dict()
        if not alignment.empty
        else {},
        "tolerance_seconds": tolerance_seconds,
        "search_before_seconds": search_before_seconds,
        "search_after_seconds": search_after_seconds,
        "interpretation": (
            "If nearest_within_tolerance is high but gameplay score accuracy is low, "
            "the event detector is seeing point-ending moments and the remaining "
            "problem is winner attribution, point filtering, or scoreboard display lag."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_gameplay_point_label_dataset(
    alignment_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    candidate_tolerance_seconds: float = 2.0,
) -> dict:
    """Create supervised labels from scoreboard validation for gameplay events.

    This artifact is for training and evaluation. It does not feed production
    gameplay scoring.
    """

    alignment_path = Path(alignment_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    alignment = pd.read_csv(alignment_path)

    rows = []
    previous_points = "0-0"
    previous_games = "0-0"
    for row in alignment.itertuples(index=False):
        after_points = clean_score_text(row.scoreboard_points, "0-0")
        after_games = clean_score_text(row.scoreboard_games, "")
        after_games = after_games if after_games else previous_games
        winner = infer_winner_from_score_transition(
            before_points=previous_points,
            after_points=after_points,
            before_games=previous_games,
            after_games=after_games,
        )

        chosen = choose_label_candidate(row, candidate_tolerance_seconds)
        rows.append(
            {
                "transition": int(row.transition),
                "scoreboard_frame": int(row.scoreboard_frame),
                "scoreboard_timestamp": float(row.scoreboard_timestamp),
                "score_before_points": previous_points,
                "score_after_points": after_points,
                "score_before_games": previous_games,
                "score_after_games": after_games,
                "winner_team": winner,
                "candidate_frame": chosen["frame"],
                "candidate_time": chosen["time"],
                "candidate_dt": chosen["dt"],
                "candidate_event_type": chosen["event_type"],
                "candidate_confidence": chosen["confidence"],
                "candidate_source": chosen["source"],
                "label_quality": chosen["quality"],
            }
        )
        previous_points = after_points
        previous_games = after_games

    labels = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False)

    summary = {
        "mode": "gameplay_point_label_dataset",
        "alignment_path": str(alignment_path),
        "labels_path": str(output_path),
        "labels": int(len(labels)),
        "usable_labels": int((labels["label_quality"] == "usable").sum())
        if not labels.empty
        else 0,
        "quality_counts": labels["label_quality"].value_counts().to_dict()
        if not labels.empty
        else {},
        "candidate_tolerance_seconds": candidate_tolerance_seconds,
        "purpose": (
            "Training/evaluation labels for a gameplay point-ending and winner "
            "classifier. These labels are derived from scoreboard validation and "
            "must not be used as live scoring input."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def empty_alignment_row(transition, failure_class: str) -> dict:
    return {
        "transition": int(transition.event),
        "scoreboard_frame": int(transition.frame),
        "scoreboard_timestamp": float(transition.timestamp),
        "scoreboard_points": transition.points,
        "scoreboard_games": transition.games,
        "nearest_frame": "",
        "nearest_time": "",
        "nearest_dt": "",
        "nearest_event_type": "",
        "nearest_confidence": "",
        "nearest_audio_peak_z": "",
        "nearest_ball_x": "",
        "nearest_ball_y": "",
        "nearest_before_frame": "",
        "nearest_before_time": "",
        "nearest_before_dt": "",
        "nearest_before_event_type": "",
        "nearest_before_confidence": "",
        "within_tolerance": False,
        "before_within_tolerance": False,
        "failure_class": failure_class,
    }


def choose_label_candidate(row, tolerance_seconds: float) -> dict:
    if bool(row.before_within_tolerance):
        return {
            "frame": int(row.nearest_before_frame),
            "time": float(row.nearest_before_time),
            "dt": float(row.nearest_before_dt),
            "event_type": row.nearest_before_event_type,
            "confidence": float(row.nearest_before_confidence),
            "source": "nearest_before_scoreboard",
            "quality": "usable",
        }

    if bool(row.within_tolerance):
        return {
            "frame": int(row.nearest_frame),
            "time": float(row.nearest_time),
            "dt": float(row.nearest_dt),
            "event_type": row.nearest_event_type,
            "confidence": float(row.nearest_confidence),
            "source": "nearest_absolute",
            "quality": "usable_after_scoreboard"
            if float(row.nearest_dt) > 0
            else "usable",
        }

    if clean_score_text(row.nearest_before_frame, ""):
        return {
            "frame": int(row.nearest_before_frame),
            "time": float(row.nearest_before_time),
            "dt": float(row.nearest_before_dt),
            "event_type": row.nearest_before_event_type,
            "confidence": float(row.nearest_before_confidence),
            "source": "nearest_before_scoreboard_outside_tolerance",
            "quality": "review",
        }

    return {
        "frame": "",
        "time": "",
        "dt": "",
        "event_type": "",
        "confidence": "",
        "source": "missing",
        "quality": "missing",
    }


def classify_alignment_failure(
    dt: float,
    before_dt: float,
    has_before: bool,
    tolerance: float,
) -> str:
    if abs(dt) <= tolerance:
        if dt > 0:
            return "candidate_after_scoreboard_update"
        return "candidate_aligned"
    if has_before and abs(before_dt) <= tolerance:
        return "before_candidate_aligned"
    if has_before and before_dt < -tolerance:
        return "candidate_too_early_or_scoreboard_lag"
    return "candidate_too_late_or_missing"


def clean_float(value: object) -> float | str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(parsed):
        return ""
    return parsed
