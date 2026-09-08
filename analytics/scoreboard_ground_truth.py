"""Scoreboard ground-truth extraction and score timeline validation.

This module is intentionally separate from scoring logic. It may read the
broadcast scoreboard to build test data, but the model scorer must not consume
that data as an input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


POINT_VALUES = ("0", "15", "30", "40", "AD")


@dataclass
class ScoreboardTemplateSeed:
    frame: int
    roi_name: str
    label: str


@dataclass
class ScoreboardExtractorConfig:
    """Configuration for the current broadcast scoreboard overlay."""

    sample_stride_frames: int = 15
    min_visible_green_pixels: int = 1_000
    min_template_confidence: float = 0.70
    base_width: int = 1920
    base_height: int = 1080
    point_rois: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: {
            "team_1_points": (330, 82, 64, 45),
            "team_2_points": (330, 128, 64, 45),
        }
    )
    game_rois: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: {
            "team_1_games": (400, 82, 55, 45),
            "team_2_games": (400, 128, 55, 45),
        }
    )
    green_presence_roi: tuple[int, int, int, int] = (330, 82, 64, 91)
    point_template_seeds: list[ScoreboardTemplateSeed] = field(
        default_factory=lambda: [
            ScoreboardTemplateSeed(2040, "team_1_points", "0"),
            ScoreboardTemplateSeed(2040, "team_2_points", "15"),
            ScoreboardTemplateSeed(2700, "team_1_points", "15"),
            ScoreboardTemplateSeed(2700, "team_2_points", "15"),
            ScoreboardTemplateSeed(3300, "team_1_points", "15"),
            ScoreboardTemplateSeed(3300, "team_2_points", "30"),
            ScoreboardTemplateSeed(3600, "team_1_points", "30"),
            ScoreboardTemplateSeed(3600, "team_2_points", "30"),
            ScoreboardTemplateSeed(5100, "team_1_points", "40"),
            ScoreboardTemplateSeed(5100, "team_2_points", "30"),
            ScoreboardTemplateSeed(5400, "team_1_points", "0"),
            ScoreboardTemplateSeed(5400, "team_2_points", "0"),
            ScoreboardTemplateSeed(9300, "team_1_points", "15"),
            ScoreboardTemplateSeed(9300, "team_2_points", "40"),
        ]
    )
    game_template_seeds: list[ScoreboardTemplateSeed] = field(
        default_factory=lambda: [
            ScoreboardTemplateSeed(5400, "team_1_games", "1"),
            ScoreboardTemplateSeed(5400, "team_2_games", "0"),
            ScoreboardTemplateSeed(9300, "team_1_games", "1"),
            ScoreboardTemplateSeed(9300, "team_2_games", "0"),
        ]
    )


def load_scoreboard_template_seeds(
    seeds_path: str | Path,
    config: Optional[ScoreboardExtractorConfig] = None,
) -> ScoreboardExtractorConfig:
    """Override scoreboard template seeds from a per-video JSON file."""

    config = config or ScoreboardExtractorConfig()
    seeds_path = Path(seeds_path)
    data = json.loads(seeds_path.read_text(encoding="utf-8"))

    if "point_template_seeds" in data:
        config.point_template_seeds = [
            ScoreboardTemplateSeed(
                frame=int(seed["frame"]),
                roi_name=str(seed["roi_name"]),
                label=str(seed["label"]),
            )
            for seed in data["point_template_seeds"]
        ]

    if "game_template_seeds" in data:
        config.game_template_seeds = [
            ScoreboardTemplateSeed(
                frame=int(seed["frame"]),
                roi_name=str(seed["roi_name"]),
                label=str(seed["label"]),
            )
            for seed in data["game_template_seeds"]
        ]

    return config


def extract_scoreboard_ground_truth(
    video_path: str | Path,
    states_path: str | Path,
    events_path: str | Path,
    summary_path: str | Path,
    config: Optional[ScoreboardExtractorConfig] = None,
) -> dict:
    """Extract visible scoreboard states and collapsed score transitions."""

    config = config or ScoreboardExtractorConfig()
    video_path = Path(video_path)
    states_path = Path(states_path)
    events_path = Path(events_path)
    summary_path = Path(summary_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    point_templates = build_templates(
        cap=cap,
        config=config,
        seeds=config.point_template_seeds,
        roi_lookup=config.point_rois,
        mask_fn=mask_green_score_text,
        output_size=(96, 54),
    )
    game_templates = build_templates(
        cap=cap,
        config=config,
        seeds=config.game_template_seeds,
        roi_lookup=config.game_rois,
        mask_fn=mask_white_score_text,
        output_size=(66, 54),
    )

    rows = []
    for frame_index in range(0, total_frames, config.sample_stride_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue

        if not scoreboard_visible(frame, config):
            continue

        point_result = read_score_rois(
            frame=frame,
            roi_lookup=config.point_rois,
            templates=point_templates,
            config=config,
            mask_fn=mask_green_score_text,
        )
        if point_result is None:
            continue

        game_result = read_score_rois(
            frame=frame,
            roi_lookup=config.game_rois,
            templates=game_templates,
            config=config,
            mask_fn=mask_white_score_text,
            min_foreground_pixels=80,
            allow_blank=True,
        )
        game_result = game_result or {}

        confidence_values = [
            point_result["team_1_points_confidence"],
            point_result["team_2_points_confidence"],
        ]
        for key in ("team_1_games_confidence", "team_2_games_confidence"):
            if key in game_result and game_result[key] != "":
                confidence_values.append(game_result[key])

        row = {
            "frame": int(frame_index),
            "timestamp": frame_index / fps,
            "team_1_name": "TAPIA / LEBRON",
            "team_2_name": "COELLO / GALAN",
            "team_1_score": point_result["team_1_points"],
            "team_2_score": point_result["team_2_points"],
            "team_1_games": game_result.get("team_1_games", ""),
            "team_2_games": game_result.get("team_2_games", ""),
            "set_context": "",
            "confidence": float(round(min(confidence_values), 4)),
            "source": "scoreboard_template_ocr",
        }
        rows.append(row)

    cap.release()

    states = pd.DataFrame(rows)
    if not states.empty:
        states = collapse_noisy_samples(states)
    events = scoreboard_events_from_states(states)

    states_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(states_path, index=False)
    events.to_csv(events_path, index=False)

    summary = {
        "mode": "scoreboard_ground_truth_extraction",
        "source_video": str(video_path),
        "states_path": str(states_path),
        "events_path": str(events_path),
        "fps": fps,
        "total_frames": total_frames,
        "resolution": [width, height],
        "sample_stride_frames": config.sample_stride_frames,
        "visible_state_samples": int(len(states)),
        "score_change_events": int(len(events)),
        "config": serializable_config(config),
        "notes": (
            "Ground truth is extracted from the broadcast scoreboard for "
            "validation only. It must not be used as scoring input."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_score_timeline(
    predicted_events_path: str | Path,
    ground_truth_states_path: str | Path,
    ground_truth_events_path: str | Path,
    report_path: str | Path,
    mismatches_path: str | Path,
    fps: float,
    event_time_tolerance_seconds: float = 2.0,
) -> dict:
    """Compare predicted score_events.csv against scoreboard ground truth."""

    predicted_events_path = Path(predicted_events_path)
    ground_truth_states_path = Path(ground_truth_states_path)
    ground_truth_events_path = Path(ground_truth_events_path)
    report_path = Path(report_path)
    mismatches_path = Path(mismatches_path)

    gt_states = pd.read_csv(ground_truth_states_path)
    gt_events = pd.read_csv(ground_truth_events_path)
    predicted_events = (
        pd.read_csv(predicted_events_path)
        if predicted_events_path.exists() and predicted_events_path.stat().st_size
        else pd.DataFrame()
    )

    mismatches = []
    correct_frames = 0
    comparable_frames = 0

    for row in gt_states.itertuples(index=False):
        predicted = predicted_state_at_frame(predicted_events, int(row.frame))
        gt_points = f"{row.team_1_score}-{row.team_2_score}"
        gt_games = format_optional_games(row.team_1_games, row.team_2_games)

        points_match = predicted["points"] == gt_points
        games_match = True if gt_games == "" else predicted["games"] == gt_games
        comparable_frames += 1
        if points_match and games_match:
            correct_frames += 1
        else:
            mismatches.append(
                {
                    "frame": int(row.frame),
                    "timestamp": float(row.timestamp),
                    "ground_truth_points": gt_points,
                    "predicted_points": predicted["points"],
                    "ground_truth_games": gt_games,
                    "predicted_games": predicted["games"],
                    "issue": "score_state_mismatch",
                }
            )

    matched_events, missed_events, extra_events = compare_events(
        predicted_events=predicted_events,
        gt_events=gt_events,
        tolerance=event_time_tolerance_seconds,
    )
    rule_violations = rule_violations_for_predicted_events(predicted_events)

    frame_accuracy = (
        correct_frames / comparable_frames if comparable_frames else 0.0
    )
    event_accuracy = (
        matched_events / max(1, matched_events + missed_events + extra_events)
    )

    mismatches_df = pd.DataFrame(mismatches)
    mismatches_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mismatches_df.to_csv(mismatches_path, index=False)

    report = {
        "mode": "score_timeline_validation",
        "predicted_events_path": str(predicted_events_path),
        "ground_truth_states_path": str(ground_truth_states_path),
        "ground_truth_events_path": str(ground_truth_events_path),
        "mismatches_path": str(mismatches_path),
        "frame_accuracy": round(frame_accuracy, 6),
        "event_accuracy": round(event_accuracy, 6),
        "comparable_frames": int(comparable_frames),
        "frame_matches": int(correct_frames),
        "frame_mismatches": int(len(mismatches)),
        "matched_events": int(matched_events),
        "missed_events": int(missed_events),
        "extra_events": int(extra_events),
        "rule_violation_count": int(len(rule_violations)),
        "rule_violations": rule_violations,
        "success": bool(
            frame_accuracy == 1.0
            and event_accuracy == 1.0
            and len(rule_violations) == 0
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_scoreboard_prediction_timeline(
    scoreboard_events_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    fps: float,
) -> dict:
    """Build a score_events.csv-compatible OCR baseline from scoreboard events.

    This is a recognition baseline, not gameplay inference. It exists so the
    validator has a known-good predicted timeline to test against.
    """

    scoreboard_events_path = Path(scoreboard_events_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    events = pd.read_csv(scoreboard_events_path)

    rows = []
    previous_points = "0-0"
    previous_games = "0-0"
    previous_sets = ""
    previous_frame = 0
    previous_time = 0.0

    for row in events.itertuples(index=False):
        after_points = f"{clean_score_text(row.team_1_score)}-{clean_score_text(row.team_2_score)}"
        after_games = format_optional_games(row.team_1_games, row.team_2_games) or previous_games
        winner_team = infer_winner_from_score_transition(
            before_points=previous_points,
            after_points=after_points,
            before_games=previous_games,
            after_games=after_games,
        )
        rows.append(
            {
                "point": int(row.event),
                "start_frame": int(previous_frame),
                "end_frame": int(row.frame),
                "start_time": float(previous_time),
                "end_time": float(row.timestamp),
                "duration": max(0.0, float(row.timestamp) - float(previous_time)),
                "winner_side": "",
                "winner_team": winner_team,
                "winner_name": "Team A" if winner_team == "team_a" else "Team B",
                "rule_id": "scoreboard.baseline_transition",
                "outcome": "scoreboard_recognized_state_change",
                "reason": "template_scoreboard_recognition_baseline",
                "confidence": float(row.confidence),
                "confidence_band": "scoreboard_baseline",
                "review_required": False,
                "review_reason": "",
                "ball_end_x": "",
                "ball_end_y": "",
                "net_crossings": "",
                "valid_coverage": "",
                "out_fraction": "",
                "score_before_points": previous_points,
                "score_before_games": previous_games,
                "score_before_sets": previous_sets,
                "score_after_points": after_points,
                "score_after_games": after_games,
                "score_after_sets": previous_sets,
                "tiebreak_after": False,
            }
        )
        previous_points = after_points
        previous_games = after_games
        previous_frame = int(row.frame)
        previous_time = float(row.timestamp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    rule_violations = rule_violations_for_predicted_events(pd.DataFrame(rows))
    summary = {
        "mode": "scoreboard_recognition_baseline",
        "scoreboard_events_path": str(scoreboard_events_path),
        "events_path": str(output_path),
        "points_detected": len(rows),
        "rule_violation_count": len(rule_violations),
        "rule_violations": rule_violations,
        "limitations": (
            "This timeline is derived from the broadcast scoreboard. It is a "
            "validation/control baseline, not a gameplay-only scorer."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_templates(
    cap: cv2.VideoCapture,
    config: ScoreboardExtractorConfig,
    seeds: list[ScoreboardTemplateSeed],
    roi_lookup: dict[str, tuple[int, int, int, int]],
    mask_fn,
    output_size: tuple[int, int],
) -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {}
    for seed in seeds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seed.frame)
        ok, frame = cap.read()
        if not ok:
            continue
        roi = crop_scaled(frame, roi_lookup[seed.roi_name], config)
        templates.setdefault(seed.label, []).append(mask_fn(roi, output_size))
    return templates


def read_score_rois(
    frame: np.ndarray,
    roi_lookup: dict[str, tuple[int, int, int, int]],
    templates: dict[str, list[np.ndarray]],
    config: ScoreboardExtractorConfig,
    mask_fn,
    min_foreground_pixels: int = 0,
    allow_blank: bool = False,
) -> Optional[dict]:
    result = {}
    for name, roi_spec in roi_lookup.items():
        roi = crop_scaled(frame, roi_spec, config)
        mask = mask_fn(roi)
        if min_foreground_pixels and cv2.countNonZero(mask) < min_foreground_pixels:
            if allow_blank:
                result[name] = ""
                result[f"{name}_confidence"] = ""
                continue
            return None

        label, confidence = classify_mask(mask, templates)
        if confidence < config.min_template_confidence:
            if allow_blank:
                result[name] = ""
                result[f"{name}_confidence"] = ""
                continue
            return None
        result[name] = label
        result[f"{name}_confidence"] = float(round(confidence, 4))
    return result


def scoreboard_visible(frame: np.ndarray, config: ScoreboardExtractorConfig) -> bool:
    roi = crop_scaled(frame, config.green_presence_roi, config)
    b, g, r = cv2.split(roi)
    mask = (g > 80) & (g > r * 1.2) & (g > b * 1.05)
    return int(mask.sum()) >= config.min_visible_green_pixels


def crop_scaled(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    config: ScoreboardExtractorConfig,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    sx = frame_w / config.base_width
    sy = frame_h / config.base_height
    x, y, w, h = roi
    x = int(round(x * sx))
    y = int(round(y * sy))
    w = int(round(w * sx))
    h = int(round(h * sy))
    return frame[y : y + h, x : x + w]


def mask_green_score_text(
    roi: np.ndarray,
    output_size: tuple[int, int] = (96, 54),
) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask[:3, :] = 0
    mask[-3:, :] = 0
    mask[:, :3] = 0
    mask[:, -3:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)


def mask_white_score_text(
    roi: np.ndarray,
    output_size: tuple[int, int] = (66, 54),
) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 150, 255)
    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)


def classify_mask(mask: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    best_label = ""
    best_score = -1.0
    binary = mask > 0
    for label, candidates in templates.items():
        for template in candidates:
            template_binary = template > 0
            intersection = float((binary & template_binary).sum())
            union = float((binary | template_binary).sum()) + 1e-9
            iou = intersection / union
            correlation = float(cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            score = 0.55 * iou + 0.45 * max(0.0, correlation)
            if score > best_score:
                best_label = label
                best_score = score
    return best_label, float(best_score)


def collapse_noisy_samples(states: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows while keeping the sampled timeline readable."""

    columns = ["team_1_score", "team_2_score", "team_1_games", "team_2_games"]
    states = states.copy()
    states["_state"] = states[columns].astype(str).agg("|".join, axis=1)
    return states.drop(columns=["_state"]).reset_index(drop=True)


def scoreboard_events_from_states(states: pd.DataFrame) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()

    events = []
    previous_state = None
    for row in states.itertuples(index=False):
        state = (
            str(row.team_1_score),
            str(row.team_2_score),
            str(row.team_1_games),
            str(row.team_2_games),
        )
        if state == previous_state:
            continue
        events.append(
            {
                "event": len(events) + 1,
                "frame": int(row.frame),
                "timestamp": float(row.timestamp),
                "team_1_score": row.team_1_score,
                "team_2_score": row.team_2_score,
                "team_1_games": row.team_1_games,
                "team_2_games": row.team_2_games,
                "points": f"{row.team_1_score}-{row.team_2_score}",
                "games": format_optional_games(row.team_1_games, row.team_2_games),
                "confidence": float(row.confidence),
            }
        )
        previous_state = state

    return pd.DataFrame(events)


def predicted_state_at_frame(predicted_events: pd.DataFrame, frame: int) -> dict:
    state = {"points": "0-0", "games": "0-0", "sets": ""}
    if predicted_events.empty:
        return state

    past_events = predicted_events[predicted_events["end_frame"].astype(int) <= frame]
    if past_events.empty:
        return state

    last = past_events.iloc[-1]
    state["points"] = clean_score_text(last.get("score_after_points"), "0-0")
    state["games"] = clean_score_text(last.get("score_after_games"), "0-0")
    state["sets"] = clean_score_text(last.get("score_after_sets"), "")
    return state


def compare_events(
    predicted_events: pd.DataFrame,
    gt_events: pd.DataFrame,
    tolerance: float,
) -> tuple[int, int, int]:
    if gt_events.empty:
        return 0, 0, len(predicted_events)

    predicted = []
    for row in predicted_events.itertuples(index=False):
        predicted.append(
            {
                "timestamp": float(row.end_time),
                "points": clean_score_text(row.score_after_points, "0-0"),
                "games": clean_score_text(row.score_after_games, "0-0"),
            }
        )

    matched_indexes = set()
    matched = 0
    for gt in gt_events.itertuples(index=False):
        gt_points = f"{gt.team_1_score}-{gt.team_2_score}"
        gt_games = format_optional_games(gt.team_1_games, gt.team_2_games)
        for index, pred in enumerate(predicted):
            if index in matched_indexes:
                continue
            if pred["points"] != gt_points:
                continue
            if gt_games and pred["games"] != gt_games:
                continue
            if abs(pred["timestamp"] - float(gt.timestamp)) <= tolerance:
                matched_indexes.add(index)
                matched += 1
                break

    missed = max(0, len(gt_events) - matched)
    extra = max(0, len(predicted) - len(matched_indexes))
    return matched, missed, extra


def rule_violations_for_predicted_events(predicted_events: pd.DataFrame) -> list[dict]:
    if predicted_events.empty:
        return []

    violations = []
    previous = {"points": "0-0", "games": "0-0"}
    for row in predicted_events.itertuples(index=False):
        current = {
            "points": clean_score_text(row.score_after_points, "0-0"),
            "games": clean_score_text(row.score_after_games, "0-0"),
        }
        if not is_legal_score_transition(previous, current):
            violations.append(
                {
                    "frame": int(row.end_frame),
                    "timestamp": float(row.end_time),
                    "from": previous,
                    "to": current,
                    "issue": "illegal_padel_score_transition",
                }
            )
        previous = current
    return violations


def infer_winner_from_score_transition(
    before_points: str,
    after_points: str,
    before_games: str,
    after_games: str,
) -> str:
    before_game_score = split_numeric_score(before_games)
    after_game_score = split_numeric_score(after_games)
    if before_game_score is not None and after_game_score is not None:
        if after_game_score[0] > before_game_score[0]:
            return "team_a"
        if after_game_score[1] > before_game_score[1]:
            return "team_b"

    before_point_score = point_score_rank(before_points)
    after_point_score = point_score_rank(after_points)
    if before_point_score is None or after_point_score is None:
        return ""
    if after_point_score[0] > before_point_score[0]:
        return "team_a"
    if after_point_score[1] > before_point_score[1]:
        return "team_b"
    return ""


def is_legal_score_transition(previous: dict, current: dict) -> bool:
    previous_points = split_point_score(previous["points"])
    current_points = split_point_score(current["points"])
    previous_games = split_numeric_score(previous["games"])
    current_games = split_numeric_score(current["games"])
    if previous_points is None or current_points is None:
        return False
    if previous_games is None or current_games is None:
        return False

    if previous["games"] == current["games"]:
        return current_points in legal_next_points(previous_points)

    game_delta = (
        current_games[0] - previous_games[0],
        current_games[1] - previous_games[1],
    )
    if game_delta not in {(1, 0), (0, 1)}:
        return False
    if not is_game_winning_transition(previous_points, game_delta):
        return False
    return current["points"] == "0-0"


def legal_next_points(previous: tuple[str, str]) -> set[tuple[str, str]]:
    a, b = previous
    if previous == ("0", "0"):
        return {("15", "0"), ("0", "15")}
    if previous == ("15", "0"):
        return {("30", "0"), ("15", "15")}
    if previous == ("0", "15"):
        return {("15", "15"), ("0", "30")}
    if previous == ("15", "15"):
        return {("30", "15"), ("15", "30")}
    if previous == ("30", "0"):
        return {("40", "0"), ("30", "15")}
    if previous == ("0", "30"):
        return {("15", "30"), ("0", "40")}
    if previous == ("40", "0"):
        return {("40", "15")}
    if previous == ("0", "40"):
        return {("15", "40")}
    if previous == ("30", "15"):
        return {("40", "15"), ("30", "30")}
    if previous == ("15", "30"):
        return {("30", "30"), ("15", "40")}
    if previous == ("40", "15"):
        return {("40", "30")}
    if previous == ("15", "40"):
        return {("30", "40")}
    if previous == ("30", "30"):
        return {("40", "30"), ("30", "40")}
    if previous == ("40", "30"):
        return {("40", "40")}
    if previous == ("30", "40"):
        return {("40", "40")}
    if previous == ("40", "40"):
        return {("AD", "40"), ("40", "AD")}
    if previous == ("AD", "40"):
        return {("40", "40")}
    if previous == ("40", "AD"):
        return {("40", "40")}
    return set()


def is_game_winning_transition(
    previous_points: tuple[str, str],
    game_delta: tuple[int, int],
) -> bool:
    if game_delta == (1, 0):
        return previous_points in {
            ("40", "0"),
            ("40", "15"),
            ("40", "30"),
            ("AD", "40"),
        }
    if game_delta == (0, 1):
        return previous_points in {
            ("0", "40"),
            ("15", "40"),
            ("30", "40"),
            ("40", "AD"),
        }
    return False


def point_score_rank(score: str) -> Optional[tuple[int, int]]:
    score_pair = split_point_score(score)
    if score_pair is None:
        return None
    rank = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
    left, right = score_pair
    if left not in rank or right not in rank:
        return None
    return rank[left], rank[right]


def split_point_score(score: str) -> Optional[tuple[str, str]]:
    if not isinstance(score, str) or "-" not in score:
        return None
    left, right = score.split("-", 1)
    return left.strip(), right.strip()


def split_numeric_score(score: str) -> Optional[tuple[int, int]]:
    if not isinstance(score, str) or "-" not in score:
        return None
    left, right = score.split("-", 1)
    left = left.strip()
    right = right.strip()
    if not left.isdigit() or not right.isdigit():
        return None
    return int(left), int(right)


def format_optional_games(left: object, right: object) -> str:
    left = clean_score_text(left, "")
    right = clean_score_text(right, "")
    if not left or not right:
        return ""
    return f"{left}-{right}"


def clean_score_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    if text.endswith(".0"):
        return text[:-2]
    return text


def serializable_config(config: ScoreboardExtractorConfig) -> dict:
    data = asdict(config)
    data["point_template_seeds"] = [asdict(seed) for seed in config.point_template_seeds]
    data["game_template_seeds"] = [asdict(seed) for seed in config.game_template_seeds]
    return data
