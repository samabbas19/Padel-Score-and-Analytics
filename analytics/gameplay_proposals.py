"""Gameplay point proposal scoring and diagnostics.

The functions in this module keep a clean boundary:

- scoreboard-derived labels are allowed only for validation/debug artifacts;
- production gameplay scoring must consume event candidates and trajectory
  signals, not the broadcast scoreboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from analytics.scoring import ScoreState, confidence_band, format_sets
from analytics.scoreboard_ground_truth import (
    clean_score_text,
    format_optional_games,
    rule_violations_for_predicted_events,
)


@dataclass
class GameplayProposalConfig:
    """Configuration for lightweight point proposal diagnostics."""

    golden_point: bool = False
    min_time_seconds: float = 20.0
    min_event_confidence: float = 0.48
    min_silence_after_seconds: float = 4.0
    min_proposal_separation_seconds: float = 7.0
    isolated_racket_confidence: float = 0.62
    isolated_racket_gap_before_seconds: float = 3.0
    clip_tail_seconds: float = 5.0
    clip_tail_min_gap_after_seconds: float = 2.0
    dropout_context_window_seconds: float = 4.0
    dropout_min_context_events: int = 2
    dropout_static_motion_px: float = 20.0
    dropout_static_visual_score: float = 0.08
    initial_points: str = "0-0"
    initial_games: str = "0-0"
    team_a_name: str = "Team A"
    team_b_name: str = "Team B"
    auto_apply_confidence: float = 0.92
    provisional_confidence: float = 0.70


def build_event_gap_point_proposals(
    event_candidates_path: str | Path,
    proposals_path: str | Path,
    summary_path: str | Path,
    config: GameplayProposalConfig | None = None,
) -> dict:
    """Write cheap no-scoreboard point-end proposals from event candidates."""

    config = config or GameplayProposalConfig()
    event_candidates_path = Path(event_candidates_path)
    proposals_path = Path(proposals_path)
    summary_path = Path(summary_path)

    candidates = pd.read_csv(event_candidates_path)
    proposals = select_gap_point_proposals(candidates, config)

    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(proposals_path, index=False)

    summary = {
        "mode": "gameplay_gap_point_proposals_no_scoreboard",
        "event_candidates_path": str(event_candidates_path),
        "proposals_path": str(proposals_path),
        "event_candidates": int(len(candidates)),
        "point_proposals": int(len(proposals)),
        "config": asdict(config),
        "limitations": (
            "This is a low-cost proposal detector only. It does not decide "
            "padel legality or point winner; those need hit/bounce/wall/fence "
            "classification and winner attribution."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_proposal_score_timeline(
    proposals_path: str | Path,
    events_path: str | Path,
    summary_path: str | Path,
    fps: float,
    config: GameplayProposalConfig | None = None,
) -> dict:
    """Apply a simple no-scoreboard winner heuristic to point proposals."""

    config = config or GameplayProposalConfig()
    proposals_path = Path(proposals_path)
    events_path = Path(events_path)
    summary_path = Path(summary_path)

    proposals = pd.read_csv(proposals_path)
    rows = score_proposals(proposals, fps, config)
    events = pd.DataFrame(rows)

    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(events_path, index=False)

    rule_violations = rule_violations_for_predicted_events(events)
    summary = {
        "mode": "gameplay_gap_proposal_score_no_scoreboard",
        "proposals_path": str(proposals_path),
        "events_path": str(events_path),
        "point_proposals": int(len(proposals)),
        "points_scored": int(len(rows)),
        "rule_violation_count": int(len(rule_violations)),
        "rule_violations": rule_violations,
        "config": asdict(config),
        "limitations": (
            "Winner attribution is currently a weak side/inside heuristic: "
            "outside terminal ball awards the side where it ended, while an "
            "inside terminal ball awards the opposite side. This is a benchmark "
            "baseline, not commercial-grade scoring."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_proposals_against_labels(
    proposals_path: str | Path,
    labels_path: str | Path,
    report_path: str | Path,
    mismatches_path: str | Path,
    tolerance_seconds: float = 2.0,
) -> dict:
    """Compare no-scoreboard point proposals to validation labels."""

    proposals_path = Path(proposals_path)
    labels_path = Path(labels_path)
    report_path = Path(report_path)
    mismatches_path = Path(mismatches_path)

    proposals = pd.read_csv(proposals_path)
    labels = pd.read_csv(labels_path)
    matched_proposals: set[int] = set()
    mismatch_rows = []
    matched_labels = 0

    for label_index, label in enumerate(labels.itertuples(index=False)):
        label_time = safe_float(getattr(label, "candidate_time", ""), default=float("nan"))
        if pd.isna(label_time):
            mismatch_rows.append(label_mismatch_row(label, "missing_label_candidate"))
            continue

        available = proposals.drop(index=list(matched_proposals), errors="ignore").copy()
        if available.empty:
            mismatch_rows.append(label_mismatch_row(label, "missed_label"))
            continue

        available["abs_dt"] = (available["time"].astype(float) - label_time).abs()
        nearest = available.sort_values("abs_dt").iloc[0]
        if float(nearest["abs_dt"]) <= tolerance_seconds:
            matched_labels += 1
            matched_proposals.add(int(nearest.name))
        else:
            mismatch_rows.append(label_mismatch_row(label, "missed_label"))

    for proposal_index, proposal in proposals.iterrows():
        if int(proposal_index) in matched_proposals:
            continue
        mismatch_rows.append(
            {
                "issue": "extra_proposal",
                "label_transition": "",
                "label_frame": "",
                "label_time": "",
                "proposal": proposal.get("proposal", ""),
                "proposal_frame": int(proposal["frame"]),
                "proposal_time": float(proposal["time"]),
                "dt": "",
            }
        )

    missed_labels = int(len(labels) - matched_labels)
    extra_proposals = int(len(proposals) - len(matched_proposals))
    precision = matched_labels / len(proposals) if len(proposals) else 0.0
    recall = matched_labels / len(labels) if len(labels) else 0.0

    report = {
        "mode": "point_proposal_vs_label_validation",
        "proposals_path": str(proposals_path),
        "labels_path": str(labels_path),
        "mismatches_path": str(mismatches_path),
        "tolerance_seconds": tolerance_seconds,
        "labels": int(len(labels)),
        "proposals": int(len(proposals)),
        "matched_labels": int(matched_labels),
        "missed_labels": missed_labels,
        "extra_proposals": extra_proposals,
        "proposal_precision": round(precision, 6),
        "proposal_recall": round(recall, 6),
        "success": bool(missed_labels == 0 and extra_proposals == 0),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    mismatches_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(mismatch_rows).to_csv(mismatches_path, index=False)
    return report


def evaluate_winners_against_labels(
    predicted_events_path: str | Path,
    labels_path: str | Path,
    report_path: str | Path,
    mismatches_path: str | Path,
    tolerance_seconds: float = 2.0,
) -> dict:
    """Compare predicted point winners to validation labels."""

    predicted_events_path = Path(predicted_events_path)
    labels_path = Path(labels_path)
    report_path = Path(report_path)
    mismatches_path = Path(mismatches_path)

    events = pd.read_csv(predicted_events_path)
    labels = pd.read_csv(labels_path)
    matched_events: set[int] = set()
    mismatch_rows = []
    matched_count = 0
    winner_matches = 0

    for label in labels.itertuples(index=False):
        label_time = safe_float(getattr(label, "candidate_time", ""), default=float("nan"))
        if pd.isna(label_time):
            mismatch_rows.append(winner_mismatch_row(label, None, "missing_label_candidate"))
            continue

        available = events.drop(index=list(matched_events), errors="ignore").copy()
        if available.empty:
            mismatch_rows.append(winner_mismatch_row(label, None, "missed_event"))
            continue

        available["abs_dt"] = (available["end_time"].astype(float) - label_time).abs()
        nearest = available.sort_values("abs_dt").iloc[0]
        if float(nearest["abs_dt"]) > tolerance_seconds:
            mismatch_rows.append(winner_mismatch_row(label, None, "missed_event"))
            continue

        matched_count += 1
        matched_events.add(int(nearest.name))
        if str(nearest["winner_team"]) == str(label.winner_team):
            winner_matches += 1
        else:
            mismatch_rows.append(winner_mismatch_row(label, nearest, "winner_mismatch"))

    for event_index, event in events.iterrows():
        if int(event_index) in matched_events:
            continue
        mismatch_rows.append(
            {
                "issue": "extra_event",
                "label_transition": "",
                "label_frame": "",
                "label_time": "",
                "label_winner": "",
                "event_point": event.get("point", ""),
                "event_frame": int(event["end_frame"]),
                "event_time": float(event["end_time"]),
                "event_winner": event.get("winner_team", ""),
                "dt": "",
            }
        )

    winner_mismatches = matched_count - winner_matches
    winner_accuracy = winner_matches / matched_count if matched_count else 0.0
    report = {
        "mode": "winner_attribution_vs_label_validation",
        "predicted_events_path": str(predicted_events_path),
        "labels_path": str(labels_path),
        "mismatches_path": str(mismatches_path),
        "tolerance_seconds": tolerance_seconds,
        "labels": int(len(labels)),
        "predicted_events": int(len(events)),
        "matched_events": int(matched_count),
        "missed_events": int(len(labels) - matched_count),
        "extra_events": int(len(events) - len(matched_events)),
        "winner_matches": int(winner_matches),
        "winner_mismatches": int(winner_mismatches),
        "winner_accuracy": round(winner_accuracy, 6),
        "success": bool(matched_count == len(labels) and winner_mismatches == 0),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    mismatches_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(mismatch_rows).to_csv(mismatches_path, index=False)
    return report


def evaluate_score_sequence_against_scoreboard(
    predicted_events_path: str | Path,
    scoreboard_events_path: str | Path,
    report_path: str | Path,
    mismatches_path: str | Path,
) -> dict:
    """Compare score transition order while reporting display timing lag."""

    predicted_events_path = Path(predicted_events_path)
    scoreboard_events_path = Path(scoreboard_events_path)
    report_path = Path(report_path)
    mismatches_path = Path(mismatches_path)

    predicted = pd.read_csv(predicted_events_path)
    expected = expected_sequence_from_scoreboard(scoreboard_events_path)
    max_len = max(len(predicted), len(expected))
    score_matches = 0
    timing_deltas = []
    mismatch_rows = []

    for index in range(max_len):
        if index >= len(expected):
            row = predicted.iloc[index]
            mismatch_rows.append(
                {
                    "issue": "extra_prediction",
                    "event_index": index + 1,
                    "expected_points": "",
                    "predicted_points": clean_score_text(row.get("score_after_points"), ""),
                    "expected_games": "",
                    "predicted_games": clean_score_text(row.get("score_after_games"), ""),
                    "expected_time": "",
                    "predicted_time": float(row.get("end_time", 0.0)),
                    "timing_delta_seconds": "",
                }
            )
            continue

        expected_row = expected[index]
        if index >= len(predicted):
            mismatch_rows.append(
                {
                    "issue": "missed_score_transition",
                    "event_index": index + 1,
                    "expected_points": expected_row["points"],
                    "predicted_points": "",
                    "expected_games": expected_row["games"],
                    "predicted_games": "",
                    "expected_time": expected_row["time"],
                    "predicted_time": "",
                    "timing_delta_seconds": "",
                }
            )
            continue

        predicted_row = predicted.iloc[index]
        predicted_points = clean_score_text(predicted_row.get("score_after_points"), "0-0")
        predicted_games = clean_score_text(predicted_row.get("score_after_games"), "0-0")
        timing_delta = float(predicted_row.get("end_time", 0.0)) - expected_row["time"]
        timing_deltas.append(timing_delta)

        if (
            predicted_points == expected_row["points"]
            and predicted_games == expected_row["games"]
        ):
            score_matches += 1
        else:
            mismatch_rows.append(
                {
                    "issue": "score_sequence_mismatch",
                    "event_index": index + 1,
                    "expected_points": expected_row["points"],
                    "predicted_points": predicted_points,
                    "expected_games": expected_row["games"],
                    "predicted_games": predicted_games,
                    "expected_time": expected_row["time"],
                    "predicted_time": float(predicted_row.get("end_time", 0.0)),
                    "timing_delta_seconds": timing_delta,
                }
            )

    accuracy = score_matches / max_len if max_len else 0.0
    timing_summary = summarize_timing_deltas(timing_deltas)
    report = {
        "mode": "score_sequence_validation_ignoring_display_lag",
        "predicted_events_path": str(predicted_events_path),
        "scoreboard_events_path": str(scoreboard_events_path),
        "mismatches_path": str(mismatches_path),
        "expected_transitions": int(len(expected)),
        "predicted_transitions": int(len(predicted)),
        "score_matches": int(score_matches),
        "score_mismatches": int(max_len - score_matches),
        "score_sequence_accuracy": round(accuracy, 6),
        "timing_delta_seconds": timing_summary,
        "success": bool(accuracy == 1.0),
        "interpretation": (
            "This report checks whether the predicted score states occur in the "
            "same order as the scoreboard. Timing deltas are retained separately "
            "because gameplay events usually happen before the broadcast overlay "
            "updates."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    mismatches_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(mismatch_rows).to_csv(mismatches_path, index=False)
    return report


def evaluate_score_checkpoints_against_scoreboard(
    predicted_events_path: str | Path,
    scoreboard_events_path: str | Path,
    report_path: str | Path,
    mismatches_path: str | Path,
) -> dict:
    """Check whether visible scoreboard states appear in gameplay score order.

    Unlike strict sequence validation, this treats predicted intermediate
    gameplay points as allowed hidden transitions. That makes it suitable for
    clips that begin mid-game or where the broadcast overlay only appears at
    sparse moments.
    """

    predicted_events_path = Path(predicted_events_path)
    scoreboard_events_path = Path(scoreboard_events_path)
    report_path = Path(report_path)
    mismatches_path = Path(mismatches_path)

    predicted = pd.read_csv(predicted_events_path)
    predicted_states = predicted_score_states_from_events(predicted)
    expected = expected_sequence_from_scoreboard(scoreboard_events_path)
    mismatch_rows = []
    timing_deltas = []
    cursor = 0
    matched = 0

    for checkpoint_index, expected_row in enumerate(expected, start=1):
        found_index = None
        for state_index in range(cursor, len(predicted_states)):
            candidate = predicted_states[state_index]
            if (
                candidate["points"] == expected_row["points"]
                and candidate["games"] == expected_row["games"]
            ):
                found_index = state_index
                break

        if found_index is None:
            nearest = nearest_score_state(predicted_states, expected_row["time"])
            mismatch_rows.append(
                {
                    "issue": "missing_visible_checkpoint",
                    "checkpoint": checkpoint_index,
                    "expected_points": expected_row["points"],
                    "expected_games": expected_row["games"],
                    "expected_time": expected_row["time"],
                    "expected_frame": expected_row["frame"],
                    "matched_prediction_index": "",
                    "predicted_points": "",
                    "predicted_games": "",
                    "predicted_time": "",
                    "predicted_frame": "",
                    "timing_delta_seconds": "",
                    "nearest_prediction_index": nearest["index"],
                    "nearest_predicted_points": nearest["points"],
                    "nearest_predicted_games": nearest["games"],
                    "nearest_predicted_time": nearest["time"],
                    "nearest_predicted_frame": nearest["frame"],
                    "nearest_time_delta_seconds": nearest["time_delta"],
                }
            )
            continue

        candidate = predicted_states[found_index]
        matched += 1
        cursor = found_index + 1
        timing_delta = candidate["time"] - expected_row["time"]
        timing_deltas.append(timing_delta)

    accuracy = matched / len(expected) if expected else 0.0
    mismatch_columns = [
        "issue",
        "checkpoint",
        "expected_points",
        "expected_games",
        "expected_time",
        "expected_frame",
        "matched_prediction_index",
        "predicted_points",
        "predicted_games",
        "predicted_time",
        "predicted_frame",
        "timing_delta_seconds",
        "nearest_prediction_index",
        "nearest_predicted_points",
        "nearest_predicted_games",
        "nearest_predicted_time",
        "nearest_predicted_frame",
        "nearest_time_delta_seconds",
    ]
    report = {
        "mode": "visible_scoreboard_checkpoint_subsequence_validation",
        "predicted_events_path": str(predicted_events_path),
        "scoreboard_events_path": str(scoreboard_events_path),
        "mismatches_path": str(mismatches_path),
        "visible_checkpoints": int(len(expected)),
        "predicted_states": int(len(predicted_states)),
        "matched_checkpoints": int(matched),
        "missed_checkpoints": int(len(expected) - matched),
        "hidden_or_intermediate_predictions": int(max(0, len(predicted_states) - matched)),
        "checkpoint_accuracy": round(accuracy, 6),
        "timing_delta_seconds": summarize_timing_deltas(timing_deltas),
        "success": bool(accuracy == 1.0),
        "interpretation": (
            "This report is for partial or sparse-scoreboard clips. It checks "
            "whether each visible scoreboard state appears in the predicted "
            "gameplay score order, while allowing extra predicted states between "
            "visible checkpoints."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    mismatches_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(mismatch_rows, columns=mismatch_columns).to_csv(mismatches_path, index=False)
    return report


def predicted_score_states_from_events(events: pd.DataFrame) -> list[dict]:
    states = []
    if events.empty:
        return states

    first = events.iloc[0]
    states.append(
        {
            "points": clean_score_text(first.get("score_before_points"), "0-0"),
            "games": clean_score_text(first.get("score_before_games"), "0-0"),
            "time": safe_float(first.get("start_time"), safe_float(first.get("end_time"), 0.0)),
            "frame": optional_int(first.get("start_frame")),
            "source": "initial",
        }
    )

    for row in events.itertuples(index=False):
        states.append(
            {
                "points": clean_score_text(getattr(row, "score_after_points", ""), "0-0"),
                "games": clean_score_text(getattr(row, "score_after_games", ""), "0-0"),
                "time": safe_float(getattr(row, "end_time", 0.0), 0.0),
                "frame": optional_int(getattr(row, "end_frame", "")),
                "source": "point",
            }
        )
    return states


def nearest_score_state(states: list[dict], expected_time: float) -> dict:
    if not states:
        return {
            "index": "",
            "points": "",
            "games": "",
            "time": "",
            "frame": "",
            "time_delta": "",
        }

    best_index, best_state = min(
        enumerate(states, start=1),
        key=lambda indexed_state: abs(indexed_state[1]["time"] - expected_time),
    )
    return {
        "index": best_index,
        "points": best_state["points"],
        "games": best_state["games"],
        "time": best_state["time"],
        "frame": best_state["frame"],
        "time_delta": round(best_state["time"] - expected_time, 6),
    }


def build_scoreboard_transition_expansion_report(
    scoreboard_events_path: str | Path,
    report_path: str | Path,
    details_path: str | Path,
    initial_points: str = "0-0",
    initial_games: str = "0-0",
    max_depth: int = 10,
    golden_point: bool = False,
) -> dict:
    """Infer the minimum legal point winners between visible score states."""

    scoreboard_events_path = Path(scoreboard_events_path)
    report_path = Path(report_path)
    details_path = Path(details_path)
    expected = expected_sequence_from_scoreboard(scoreboard_events_path)
    rows = []
    previous = {
        "points": clean_score_text(initial_points, "0-0"),
        "games": clean_score_text(initial_games, "0-0"),
        "time": 0.0,
        "frame": 0,
    }
    minimum_point_events = 0
    visible_scored_transitions = 0
    impossible = 0
    ambiguous = 0

    for transition_index, current in enumerate(expected, start=1):
        sequences = infer_minimum_winner_sequences(
            start_points=previous["points"],
            start_games=previous["games"],
            end_points=current["points"],
            end_games=current["games"],
            max_depth=max_depth,
            golden_point=golden_point,
        )
        sequence_count = len(sequences)
        if sequence_count == 0:
            impossible += 1
            minimum_points = ""
            example_sequence = ""
        else:
            minimum_points = len(sequences[0])
            minimum_point_events += minimum_points
            if minimum_points > 0:
                visible_scored_transitions += 1
            example_sequence = " ".join(sequences[0])
            if sequence_count > 1:
                ambiguous += 1

        rows.append(
            {
                "transition": transition_index,
                "from_points": previous["points"],
                "from_games": previous["games"],
                "to_points": current["points"],
                "to_games": current["games"],
                "to_time": current["time"],
                "to_frame": current["frame"],
                "minimum_points": minimum_points,
                "possible_minimum_sequences": sequence_count,
                "ambiguous": sequence_count > 1,
                "impossible": sequence_count == 0,
                "example_winner_sequence": example_sequence,
            }
        )
        previous = current

    visible_transitions = len(expected)
    hidden_point_events = max(0, minimum_point_events - visible_scored_transitions)
    report = {
        "mode": "scoreboard_visible_transition_expansion",
        "scoreboard_events_path": str(scoreboard_events_path),
        "details_path": str(details_path),
        "initial_points": clean_score_text(initial_points, "0-0"),
        "initial_games": clean_score_text(initial_games, "0-0"),
        "visible_transitions": int(visible_transitions),
        "visible_scored_transitions": int(visible_scored_transitions),
        "minimum_point_events": int(minimum_point_events),
        "hidden_point_events": int(hidden_point_events),
        "ambiguous_transitions": int(ambiguous),
        "impossible_transitions": int(impossible),
        "success": bool(impossible == 0),
        "interpretation": (
            "Validation-only report. It expands sparse visible scoreboard "
            "checkpoints into the minimum legal point-winner sequences needed "
            "to move between them. The production scorer must not consume this."
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(details_path, index=False)
    return report


def infer_minimum_winner_sequences(
    start_points: str,
    start_games: str,
    end_points: str,
    end_games: str,
    max_depth: int = 10,
    golden_point: bool = False,
    max_sequences: int = 16,
) -> list[list[str]]:
    """Return shortest winner-team sequences that legally reach a score."""

    start = score_state_from_text(start_points, start_games, golden_point=golden_point)
    target_points = clean_score_text(end_points, "0-0")
    target_games = clean_score_text(end_games, "0-0")
    queue = [(start, [])]

    for depth in range(max_depth + 1):
        solutions = [
            sequence
            for state, sequence in queue
            if state.points_text() == target_points
            and f"{state.games_a}-{state.games_b}" == target_games
        ]
        if solutions:
            return solutions[:max_sequences]
        if depth == max_depth:
            break

        next_queue = []
        for state, sequence in queue:
            for winner in ("team_a", "team_b"):
                child = clone_score_state(state)
                child.add_point(winner)
                next_queue.append((child, sequence + [winner]))
        queue = next_queue

    return []


def score_state_from_text(points: str, games: str, golden_point: bool = False) -> ScoreState:
    state = ScoreState(golden_point=golden_point)
    parsed_points = parse_point_score_for_state(clean_score_text(points, "0-0"))
    parsed_games = parse_numeric_pair(clean_score_text(games, "0-0"))
    if parsed_points is not None:
        state.points_a, state.points_b = parsed_points
    if parsed_games is not None:
        state.games_a, state.games_b = parsed_games
    return state


def clone_score_state(state: ScoreState) -> ScoreState:
    return ScoreState(
        golden_point=state.golden_point,
        sets=list(state.sets),
        games_a=state.games_a,
        games_b=state.games_b,
        points_a=state.points_a,
        points_b=state.points_b,
        tiebreak=state.tiebreak,
    )


def expected_sequence_from_scoreboard(scoreboard_events_path: Path) -> list[dict]:
    events = pd.read_csv(scoreboard_events_path)
    expected = []
    previous_games = "0-0"
    for row in events.itertuples(index=False):
        points = f"{clean_score_text(row.team_1_score)}-{clean_score_text(row.team_2_score)}"
        games = format_optional_games(row.team_1_games, row.team_2_games) or previous_games
        expected.append(
            {
                "points": points,
                "games": games,
                "time": float(row.timestamp),
                "frame": int(row.frame),
            }
        )
        previous_games = games
    return expected


def summarize_timing_deltas(deltas: list[float]) -> dict:
    if not deltas:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(deltas),
        "min": round(min(deltas), 6),
        "max": round(max(deltas), 6),
        "mean": round(sum(deltas) / len(deltas), 6),
    }


def winner_mismatch_row(label, event, issue: str) -> dict:
    if event is None:
        return {
            "issue": issue,
            "label_transition": optional_int(getattr(label, "transition", "")),
            "label_frame": optional_int(getattr(label, "candidate_frame", "")),
            "label_time": optional_float(getattr(label, "candidate_time", "")),
            "label_winner": getattr(label, "winner_team", ""),
            "event_point": "",
            "event_frame": "",
            "event_time": "",
            "event_winner": "",
            "dt": "",
        }

    return {
        "issue": issue,
        "label_transition": optional_int(getattr(label, "transition", "")),
        "label_frame": optional_int(getattr(label, "candidate_frame", "")),
        "label_time": optional_float(getattr(label, "candidate_time", "")),
        "label_winner": getattr(label, "winner_team", ""),
        "event_point": event.get("point", ""),
        "event_frame": int(event["end_frame"]),
        "event_time": float(event["end_time"]),
        "event_winner": event.get("winner_team", ""),
        "dt": float(event["end_time"]) - float(label.candidate_time),
    }


def label_mismatch_row(label, issue: str) -> dict:
    return {
        "issue": issue,
        "label_transition": optional_int(getattr(label, "transition", "")),
        "label_frame": optional_int(getattr(label, "candidate_frame", "")),
        "label_time": optional_float(getattr(label, "candidate_time", "")),
        "proposal": "",
        "proposal_frame": "",
        "proposal_time": "",
        "dt": "",
    }


def optional_int(value: object) -> int | str:
    parsed = safe_float(value, default=float("nan"))
    if pd.isna(parsed):
        return ""
    return int(parsed)


def optional_float(value: object) -> float | str:
    parsed = safe_float(value, default=float("nan"))
    if pd.isna(parsed):
        return ""
    return float(parsed)


def score_proposals(
    proposals: pd.DataFrame,
    fps: float,
    config: GameplayProposalConfig,
) -> list[dict]:
    state = initial_score_state(config)
    rows = []
    previous_end_frame = 0
    previous_end_time = 0.0

    for point_index, row in enumerate(proposals.sort_values("time").itertuples(index=False), start=1):
        winner_team = infer_winner_from_proposal(row)
        if winner_team not in {"team_a", "team_b"}:
            continue

        before = state.snapshot()
        state.add_point(winner_team)
        after = state.snapshot()

        end_frame = int(row.frame)
        end_time = float(row.time)
        confidence = float(getattr(row, "proposal_confidence", getattr(row, "confidence", 0.5)))
        winner_confidence = round(confidence * 0.55, 4)

        rows.append(
            {
                "point": len(rows) + 1,
                "start_frame": int(previous_end_frame),
                "end_frame": end_frame,
                "start_time": float(previous_end_time),
                "end_time": end_time,
                "duration": max(0.0, end_time - previous_end_time),
                "winner_side": winner_team_to_side(winner_team),
                "winner_team": winner_team,
                "winner_name": config.team_a_name
                if winner_team == "team_a"
                else config.team_b_name,
                "rule_id": "gameplay.gap_attribution_heuristic",
                "outcome": "score_from_gap_proposal",
                "reason": "quiet_period_point_end_plus_attribution_context",
                "confidence": winner_confidence,
                "confidence_band": confidence_band(winner_confidence, config),
                "review_required": True,
                "review_reason": (
                    "Winner was inferred from lightweight attribution context; "
                    "requires true bounce/hit/wall/fence classifier before auto-apply."
                ),
                "ball_end_x": getattr(row, "ball_x", ""),
                "ball_end_y": getattr(row, "ball_y", ""),
                "net_crossings": "",
                "valid_coverage": "",
                "out_fraction": "",
                "score_before_points": before.points,
                "score_before_games": f"{before.games[0]}-{before.games[1]}",
                "score_before_sets": format_sets(before.sets),
                "score_after_points": after.points,
                "score_after_games": f"{after.games[0]}-{after.games[1]}",
                "score_after_sets": format_sets(after.sets),
                "tiebreak_after": after.tiebreak,
                "proposal_confidence": confidence,
                "candidate_event_type": getattr(row, "event_type", ""),
                "attribution_side": getattr(row, "attribution_side", ""),
                "attribution_reason": getattr(row, "attribution_reason", ""),
                "attribution_confidence": getattr(row, "attribution_confidence", ""),
            }
        )
        previous_end_frame = end_frame
        previous_end_time = end_time

    return rows


def initial_score_state(config: GameplayProposalConfig) -> ScoreState:
    state = ScoreState(golden_point=config.golden_point)
    points = parse_point_score_for_state(config.initial_points)
    games = parse_numeric_pair(config.initial_games)
    if points is not None:
        state.points_a, state.points_b = points
    if games is not None:
        state.games_a, state.games_b = games
    return state


def parse_point_score_for_state(score: str) -> tuple[int, int] | None:
    if not isinstance(score, str) or "-" not in score:
        return None
    left, right = [part.strip().upper() for part in score.split("-", 1)]
    if left == "AD" and right == "40":
        return 4, 3
    if left == "40" and right == "AD":
        return 3, 4
    values = {"0": 0, "15": 1, "30": 2, "40": 3}
    if left not in values or right not in values:
        return None
    return values[left], values[right]


def parse_numeric_pair(score: str) -> tuple[int, int] | None:
    if not isinstance(score, str) or "-" not in score:
        return None
    left, right = [part.strip() for part in score.split("-", 1)]
    if not left.isdigit() or not right.isdigit():
        return None
    return int(left), int(right)


def infer_winner_from_proposal(row) -> str:
    attribution_side = clean_side(getattr(row, "attribution_side", ""))
    if attribution_side:
        return winner_side_to_team(attribution_side)

    side = str(getattr(row, "ball_side", "")).strip().lower()
    if side not in {"near", "far"}:
        return ""

    inside = parse_bool(getattr(row, "inside_court", False))
    winner_side = opposite_side(side) if inside else side
    return winner_side_to_team(winner_side)


def winner_side_to_team(side: str) -> str:
    return "team_a" if side == "near" else "team_b"


def winner_team_to_side(team: str) -> str:
    return "near" if team == "team_a" else "far"


def opposite_side(side: str) -> str:
    return "far" if side == "near" else "near"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def select_gap_point_proposals(
    candidates: pd.DataFrame,
    config: GameplayProposalConfig,
) -> pd.DataFrame:
    """Select last event before a quiet period as a point-end proposal."""

    required = {"frame", "time", "event_type", "confidence"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Point proposal candidates missing columns: {', '.join(missing)}")

    if candidates.empty:
        return empty_proposals()

    all_candidates = candidates.sort_values("time").copy()
    work = all_candidates[
        (all_candidates["time"].astype(float) >= config.min_time_seconds)
        & (all_candidates["confidence"].astype(float) >= config.min_event_confidence)
    ].copy()
    if work.empty:
        return empty_proposals()

    work["gap_before_seconds"] = work["time"] - work["time"].shift(1)
    work["gap_after_seconds"] = work["time"].shift(-1) - work["time"]
    quiet = work["gap_after_seconds"].ge(config.min_silence_after_seconds)
    isolated_racket = (
        work["event_type"].eq("racket_hit")
        & work["confidence"].astype(float).ge(config.isolated_racket_confidence)
        & work["gap_before_seconds"].ge(config.isolated_racket_gap_before_seconds)
    )
    max_time = float(work["time"].max())
    clip_tail_bounce = (
        work["event_type"].eq("court_bounce")
        & work["time"].astype(float).ge(max_time - config.clip_tail_seconds)
        & work["gap_after_seconds"].ge(config.clip_tail_min_gap_after_seconds)
    )
    selected_mask = quiet | isolated_racket | clip_tail_bounce
    proposals = work[selected_mask].copy()
    if proposals.empty:
        return empty_proposals()

    proposals["proposal_confidence"] = proposals.apply(point_proposal_confidence, axis=1)
    proposals["proposal_reason"] = proposals.apply(assign_proposal_reason, axis=1)
    decisions = [
        classify_point_end_candidate(row, all_candidates, config)
        for row in proposals.itertuples(index=False)
    ]
    proposals["point_end_decision"] = [decision["decision"] for decision in decisions]
    proposals["point_end_reason"] = [decision["reason"] for decision in decisions]
    proposals["point_end_context_events"] = [
        decision["context_events"] for decision in decisions
    ]
    proposals = proposals[proposals["point_end_decision"].eq("accepted")].copy()
    if proposals.empty:
        return empty_proposals()

    contexts = [
        attribution_context_for_candidate(row, all_candidates, config)
        for row in proposals.itertuples(index=False)
    ]
    proposals["attribution_side"] = [context["side"] for context in contexts]
    proposals["attribution_reason"] = [context["reason"] for context in contexts]
    proposals["attribution_confidence"] = [
        context["confidence"] for context in contexts
    ]
    proposals = suppress_nearby_proposals(proposals, config.min_proposal_separation_seconds)
    proposals = proposals.sort_values("time").reset_index(drop=True)
    proposals.insert(0, "proposal", range(1, len(proposals) + 1))
    return proposals


def classify_point_end_candidate(row, candidates: pd.DataFrame, config: GameplayProposalConfig) -> dict:
    context_events = count_rally_context_events(row, candidates, config)
    if is_dropout_candidate(row):
        if context_events < config.dropout_min_context_events:
            return {
                "decision": "rejected",
                "reason": "dead_ball_dropout_without_rally_context",
                "context_events": context_events,
            }
        return {
            "decision": "accepted",
            "reason": "dropout_after_rally_context",
            "context_events": context_events,
        }

    return {
        "decision": "accepted",
        "reason": "standard_point_end_candidate",
        "context_events": context_events,
    }


def count_rally_context_events(row, candidates: pd.DataFrame, config: GameplayProposalConfig) -> int:
    row_time = safe_float(getattr(row, "time", 0.0))
    previous = candidates[
        (candidates["time"].astype(float) < row_time)
        & (
            candidates["time"].astype(float)
            >= row_time - config.dropout_context_window_seconds
        )
    ].copy()
    if previous.empty:
        return 0
    return int(previous.apply(lambda item: is_meaningful_rally_context(item, config), axis=1).sum())


def is_meaningful_rally_context(row, config: GameplayProposalConfig) -> bool:
    if is_static_dropout_series(row, config):
        return False
    if parse_bool(row.get("inside_court", False)):
        return True
    if safe_float(row.get("raw_motion_px", 0.0)) >= config.dropout_static_motion_px:
        return True
    if safe_float(row.get("visual_motion_score", 0.0)) >= config.dropout_static_visual_score:
        return True
    return False


def point_proposal_confidence(row) -> float:
    score = float(row.confidence)
    if row.event_type == "court_bounce":
        score += 0.05
    elif row.event_type == "net_contact":
        score += 0.08
    gap_after = float(row.gap_after_seconds)
    score += min(0.18, max(0.0, gap_after - 3.0) * 0.04)
    return round(max(0.0, min(0.99, score)), 4)


def assign_proposal_reason(row) -> str:
    if float(row.gap_after_seconds) >= 4.0:
        return "quiet_period_after_event"
    if str(row.event_type) == "court_bounce":
        return "clip_tail_bounce"
    return "isolated_strong_racket_hit"


def attribution_context_for_candidate(row, candidates: pd.DataFrame, config: GameplayProposalConfig) -> dict:
    side = clean_side(getattr(row, "ball_side", ""))
    if not side:
        return {"side": "", "reason": "missing_side", "confidence": 0.0}

    if is_dropout_candidate(row):
        previous = previous_meaningful_candidate(row, candidates)
        if previous is not None:
            return {
                "side": clean_side(previous["ball_side"]),
                "reason": "previous_meaningful_event_before_dropout",
                "confidence": 0.72,
            }
        return {"side": side, "reason": "dropout_without_history", "confidence": 0.35}

    event_type = str(getattr(row, "event_type", ""))
    if event_type == "court_bounce":
        return {"side": side, "reason": "terminal_bounce_side", "confidence": 0.70}

    if event_type == "racket_hit":
        if nearest_player_distance(row) <= 2.0:
            return {"side": side, "reason": "terminal_contact_player_side", "confidence": 0.64}
        if parse_bool(getattr(row, "inside_court", False)):
            return {
                "side": opposite_side(side),
                "reason": "inside_terminal_racket_opposite_side",
                "confidence": 0.56,
            }
        return {"side": side, "reason": "outside_terminal_racket_side", "confidence": 0.58}

    if event_type == "net_contact":
        return {"side": side, "reason": "terminal_net_contact_side", "confidence": 0.55}

    return {"side": side, "reason": "terminal_event_side", "confidence": 0.45}


def is_dropout_candidate(row) -> bool:
    return (
        str(getattr(row, "event_type", "")) == "racket_hit"
        and not parse_bool(getattr(row, "inside_court", False))
        and safe_float(getattr(row, "ball_y", 0.0)) <= -14.0
    )


def previous_meaningful_candidate(row, candidates: pd.DataFrame, window_seconds: float = 3.5):
    row_time = safe_float(getattr(row, "time", 0.0))
    previous = candidates[
        (candidates["time"].astype(float) < row_time)
        & (candidates["time"].astype(float) >= row_time - window_seconds)
    ].copy()
    if previous.empty:
        return None

    previous = previous[~previous.apply(is_dropout_series, axis=1)]
    if previous.empty:
        return None
    return previous.sort_values("time").iloc[-1]


def is_dropout_series(row: pd.Series) -> bool:
    return (
        str(row.get("event_type", "")) == "racket_hit"
        and not parse_bool(row.get("inside_court", False))
        and safe_float(row.get("ball_y", 0.0)) <= -14.0
    )


def is_static_dropout_series(row: pd.Series, config: GameplayProposalConfig) -> bool:
    return (
        is_dropout_series(row)
        and safe_float(row.get("raw_motion_px", 0.0)) < config.dropout_static_motion_px
        and safe_float(row.get("visual_motion_score", 0.0)) < config.dropout_static_visual_score
    )


def nearest_player_distance(row) -> float:
    return safe_float(getattr(row, "nearest_player_distance_px", 9999.0), 9999.0)


def clean_side(value: object) -> str:
    side = str(value).strip().lower()
    return side if side in {"near", "far"} else ""


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed


def suppress_nearby_proposals(
    proposals: pd.DataFrame,
    min_separation_seconds: float,
) -> pd.DataFrame:
    selected = []
    for row in proposals.sort_values("proposal_confidence", ascending=False).to_dict("records"):
        if all(abs(float(row["time"]) - float(kept["time"])) >= min_separation_seconds for kept in selected):
            selected.append(row)

    if not selected:
        return empty_proposals()
    return pd.DataFrame(selected)


def empty_proposals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "proposal",
            "frame",
            "time",
            "event_type",
            "confidence",
            "gap_after_seconds",
            "proposal_confidence",
            "proposal_reason",
            "point_end_decision",
            "point_end_reason",
            "point_end_context_events",
        ]
    )


def build_label_replay_score_timeline(
    labels_path: str | Path,
    events_path: str | Path,
    summary_path: str | Path,
    fps: float,
    config: GameplayProposalConfig | None = None,
) -> dict:
    """Replay validated gameplay point labels through the score state machine.

    This is a debugging upper bound. It answers: "If the gameplay detector
    supplied these point boundaries and winners, would scoring be correct?"
    """

    config = config or GameplayProposalConfig()
    labels_path = Path(labels_path)
    events_path = Path(events_path)
    summary_path = Path(summary_path)

    labels = pd.read_csv(labels_path)
    if labels.empty:
        rows = []
    else:
        labels = labels.sort_values(
            ["scoreboard_frame", "candidate_frame"],
            ascending=[True, True],
        )
        rows = replay_winners_as_score_events(labels, fps, config)

    events = pd.DataFrame(rows)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(events_path, index=False)

    rule_violations = rule_violations_for_predicted_events(events)
    summary = {
        "mode": "gameplay_label_replay_debug_upper_bound",
        "labels_path": str(labels_path),
        "events_path": str(events_path),
        "points_detected": int(len(rows)),
        "rule_violation_count": int(len(rule_violations)),
        "rule_violations": rule_violations,
        "config": asdict(config),
        "limitations": (
            "This timeline replays labels derived from scoreboard validation. "
            "It verifies the score engine and label consistency, but it is not "
            "a live no-scoreboard scorer."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def replay_winners_as_score_events(
    labels: pd.DataFrame,
    fps: float,
    config: GameplayProposalConfig,
) -> list[dict]:
    state = ScoreState(golden_point=config.golden_point)
    rows = []
    previous_end_frame = 0
    previous_end_time = 0.0

    for point_index, row in enumerate(labels.itertuples(index=False), start=1):
        winner_team = str(row.winner_team)
        if winner_team not in {"team_a", "team_b"}:
            continue

        before = state.snapshot()
        state.add_point(winner_team)
        after = state.snapshot()

        end_frame = int(getattr(row, "scoreboard_frame", getattr(row, "candidate_frame")))
        end_time = float(getattr(row, "scoreboard_timestamp", end_frame / fps))
        candidate_frame = getattr(row, "candidate_frame", "")
        candidate_time = getattr(row, "candidate_time", "")
        confidence = float(getattr(row, "candidate_confidence", 1.0))
        event_type = getattr(row, "candidate_event_type", "")

        rows.append(
            {
                "point": point_index,
                "start_frame": int(previous_end_frame),
                "end_frame": end_frame,
                "start_time": float(previous_end_time),
                "end_time": end_time,
                "duration": max(0.0, end_time - previous_end_time),
                "winner_side": "",
                "winner_team": winner_team,
                "winner_name": config.team_a_name
                if winner_team == "team_a"
                else config.team_b_name,
                "rule_id": "gameplay.label_replay_transition",
                "outcome": "score_from_validated_gameplay_label",
                "reason": "debug_replay_of_scoreboard_aligned_gameplay_label",
                "confidence": confidence,
                "confidence_band": confidence_band(confidence, config),
                "review_required": False,
                "review_reason": "",
                "ball_end_x": "",
                "ball_end_y": "",
                "net_crossings": "",
                "valid_coverage": "",
                "out_fraction": "",
                "score_before_points": before.points,
                "score_before_games": f"{before.games[0]}-{before.games[1]}",
                "score_before_sets": format_sets(before.sets),
                "score_after_points": after.points,
                "score_after_games": f"{after.games[0]}-{after.games[1]}",
                "score_after_sets": format_sets(after.sets),
                "tiebreak_after": after.tiebreak,
                "candidate_frame": candidate_frame,
                "candidate_time": candidate_time,
                "candidate_event_type": event_type,
            }
        )
        previous_end_frame = end_frame
        previous_end_time = end_time

    return rows
