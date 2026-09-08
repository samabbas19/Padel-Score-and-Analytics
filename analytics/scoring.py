"""Pure trajectory-based padel score calculation.

This module intentionally ignores broadcast scoreboards. It estimates point
winners from the tracked ball/player trajectories, then applies padel scoring
rules to the inferred point stream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


Side = Literal["near", "far"]
Team = Literal["team_a", "team_b"]


@dataclass
class ScoreConfig:
    """Configuration for point detection and padel scoring."""

    golden_point: bool = False
    point_break_seconds: float = 6.0
    min_rally_seconds: float = 1.0
    active_window_seconds: float = 0.45
    active_window_distance_m: float = 0.75
    max_projected_x_m: float = 14.0
    max_projected_y_m: float = 25.0
    court_x_limit_m: float = 5.0
    court_y_limit_m: float = 10.0
    line_tolerance_m: float = 0.6
    dead_ball_window_seconds: float = 0.8
    auto_apply_confidence: float = 0.92
    provisional_confidence: float = 0.70
    team_a_name: str = "Team A"
    team_b_name: str = "Team B"


@dataclass
class ScoreSnapshot:
    sets: list[tuple[int, int]]
    games: tuple[int, int]
    points: str
    tiebreak: bool

    def serialize(self) -> dict:
        return {
            "sets": [list(score) for score in self.sets],
            "games": list(self.games),
            "points": self.points,
            "tiebreak": self.tiebreak,
        }


@dataclass
class ScoreState:
    """Padel point/game/set state machine."""

    golden_point: bool = False
    sets: list[tuple[int, int]] = field(default_factory=list)
    games_a: int = 0
    games_b: int = 0
    points_a: int = 0
    points_b: int = 0
    tiebreak: bool = False

    def snapshot(self) -> ScoreSnapshot:
        return ScoreSnapshot(
            sets=list(self.sets),
            games=(self.games_a, self.games_b),
            points=self.points_text(),
            tiebreak=self.tiebreak,
        )

    def points_text(self) -> str:
        if self.tiebreak:
            return f"{self.points_a}-{self.points_b}"

        labels = ["0", "15", "30", "40"]
        if self.points_a >= 3 and self.points_b >= 3:
            if self.points_a == self.points_b:
                return "40-40"
            if self.points_a == self.points_b + 1:
                return "AD-40"
            if self.points_b == self.points_a + 1:
                return "40-AD"

        return f"{labels[min(self.points_a, 3)]}-{labels[min(self.points_b, 3)]}"

    def add_point(self, winner: Team) -> None:
        if winner == "team_a":
            self.points_a += 1
        else:
            self.points_b += 1

        if self.tiebreak:
            self._maybe_finish_tiebreak()
        else:
            self._maybe_finish_game()

    def _maybe_finish_game(self) -> None:
        if self.golden_point and self.points_a >= 3 and self.points_b >= 3:
            if self.points_a != self.points_b:
                self._award_game("team_a" if self.points_a > self.points_b else "team_b")
            return

        if self.points_a >= 4 and self.points_a - self.points_b >= 2:
            self._award_game("team_a")
        elif self.points_b >= 4 and self.points_b - self.points_a >= 2:
            self._award_game("team_b")

    def _maybe_finish_tiebreak(self) -> None:
        if self.points_a >= 7 and self.points_a - self.points_b >= 2:
            self._award_game("team_a")
        elif self.points_b >= 7 and self.points_b - self.points_a >= 2:
            self._award_game("team_b")

    def _award_game(self, winner: Team) -> None:
        if winner == "team_a":
            self.games_a += 1
        else:
            self.games_b += 1

        self.points_a = 0
        self.points_b = 0
        self._maybe_finish_set()

    def _maybe_finish_set(self) -> None:
        if self.tiebreak:
            self.sets.append((self.games_a, self.games_b))
            self.games_a = 0
            self.games_b = 0
            self.tiebreak = False
            return

        if self.games_a == 6 and self.games_b == 6:
            self.tiebreak = True
            return

        if self.games_a >= 6 and self.games_a - self.games_b >= 2:
            self.sets.append((self.games_a, self.games_b))
            self.games_a = 0
            self.games_b = 0
        elif self.games_b >= 6 and self.games_b - self.games_a >= 2:
            self.sets.append((self.games_a, self.games_b))
            self.games_a = 0
            self.games_b = 0


@dataclass
class RawPoint:
    start_frame: int
    end_frame: int
    winner_side: Side
    rule_id: str
    outcome: str
    reason: str
    confidence: float
    review_required: bool
    review_reason: str
    ball_end_x: Optional[float]
    ball_end_y: Optional[float]
    net_crossings: int
    valid_coverage: float
    out_fraction: float


def calculate_scores(
    data_path: str | Path,
    events_path: str | Path,
    summary_path: str | Path,
    fps: float,
    config: Optional[ScoreConfig] = None,
) -> dict:
    """Infer point winners from trajectory data and write score artifacts."""

    config = config or ScoreConfig()
    data_path = Path(data_path)
    events_path = Path(events_path)
    summary_path = Path(summary_path)

    df = pd.read_csv(data_path)
    raw_points, team_info = infer_points(df=df, fps=fps, config=config)

    state = ScoreState(golden_point=config.golden_point)
    rows = []
    for point_index, point in enumerate(raw_points, start=1):
        winner_team = side_to_team(df, point.winner_side, point.end_frame, team_info)
        before = state.snapshot()
        state.add_point(winner_team)
        after = state.snapshot()
        rows.append(
            {
                "point": point_index,
                "start_frame": point.start_frame,
                "end_frame": point.end_frame,
                "start_time": point.start_frame / fps,
                "end_time": point.end_frame / fps,
                "duration": (point.end_frame - point.start_frame + 1) / fps,
                "winner_side": point.winner_side,
                "winner_team": winner_team,
                "winner_name": config.team_a_name
                if winner_team == "team_a"
                else config.team_b_name,
                "rule_id": point.rule_id,
                "outcome": point.outcome,
                "reason": point.reason,
                "confidence": point.confidence,
                "confidence_band": confidence_band(point.confidence, config),
                "review_required": point.review_required,
                "review_reason": point.review_reason,
                "ball_end_x": point.ball_end_x,
                "ball_end_y": point.ball_end_y,
                "net_crossings": point.net_crossings,
                "valid_coverage": point.valid_coverage,
                "out_fraction": point.out_fraction,
                "score_before_points": before.points,
                "score_before_games": f"{before.games[0]}-{before.games[1]}",
                "score_before_sets": format_sets(before.sets),
                "score_after_points": after.points,
                "score_after_games": f"{after.games[0]}-{after.games[1]}",
                "score_after_sets": format_sets(after.sets),
                "tiebreak_after": after.tiebreak,
            }
        )

    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(events_path, index=False)

    final_snapshot = state.snapshot()
    summary = {
        "mode": "pure_trajectory_no_ocr",
        "source_data": str(data_path),
        "events_path": str(events_path),
        "points_detected": len(rows),
        "review_required_count": sum(1 for row in rows if row["review_required"]),
        "final_score": final_snapshot.serialize(),
        "teams": team_info,
        "config": asdict(config),
        "limitations": (
            "No OCR or scoreboard data was used. Winners are inferred from ball "
            "trajectory, court side, rally reset gaps, and player-side mapping; "
            "single-camera video can still hide lets, touches, wall legality, "
            "and double-bounce calls."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def infer_points(df: pd.DataFrame, fps: float, config: ScoreConfig) -> tuple[list[RawPoint], dict]:
    require_columns(df, ["frame", "ball_x", "ball_y", "ball_visible"])
    team_info = infer_initial_teams(df)

    trajectory = clean_ball_trajectory(df, config)
    active = detect_active_frames(trajectory, fps, config)
    active_runs = contiguous_runs(active)
    min_frames = max(1, int(config.min_rally_seconds * fps))
    active_runs = [(start, end) for start, end in active_runs if end - start + 1 >= min_frames]
    rally_runs = merge_runs_by_gap(active_runs, max_gap_frames=int(config.point_break_seconds * fps))

    points = []
    for start, end in rally_runs:
        raw_point = infer_point_from_run(trajectory, start, end, fps, config)
        if raw_point is not None:
            points.append(raw_point)

    return points, team_info


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Score calculation needs collected ball trajectory columns. "
            f"Missing: {', '.join(missing)}"
        )


def clean_ball_trajectory(df: pd.DataFrame, config: ScoreConfig) -> pd.DataFrame:
    trajectory = df[["frame", "ball_x", "ball_y", "ball_visible"]].copy()
    trajectory["ball_visible"] = trajectory["ball_visible"].fillna(0).astype(int)

    finite = np.isfinite(trajectory["ball_x"]) & np.isfinite(trajectory["ball_y"])
    plausible = (
        trajectory["ball_visible"].eq(1)
        & finite
        & trajectory["ball_x"].abs().le(config.max_projected_x_m)
        & trajectory["ball_y"].abs().le(config.max_projected_y_m)
    )
    trajectory["valid"] = plausible
    trajectory.loc[~trajectory["valid"], ["ball_x", "ball_y"]] = np.nan

    trajectory["dx"] = trajectory["ball_x"].diff()
    trajectory["dy"] = trajectory["ball_y"].diff()
    invalid_step = ~(trajectory["valid"] & trajectory["valid"].shift(fill_value=False))
    trajectory["step_m"] = np.sqrt(trajectory["dx"] ** 2 + trajectory["dy"] ** 2)
    trajectory.loc[invalid_step, "step_m"] = 0.0

    return trajectory


def detect_active_frames(
    trajectory: pd.DataFrame,
    fps: float,
    config: ScoreConfig,
) -> np.ndarray:
    window = max(3, int(config.active_window_seconds * fps))
    rolling_distance = (
        trajectory["step_m"]
        .fillna(0)
        .rolling(window=window, center=True, min_periods=1)
        .sum()
    )
    active = trajectory["valid"].to_numpy() & rolling_distance.gt(
        config.active_window_distance_m
    ).to_numpy()
    return close_short_gaps(active, max_gap=max(1, int(0.35 * fps)))


def close_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    output = mask.copy()
    runs = contiguous_runs(mask)
    if len(runs) < 2:
        return output

    for (_, previous_end), (next_start, _) in zip(runs, runs[1:]):
        gap_start = previous_end + 1
        gap_end = next_start - 1
        if gap_end >= gap_start and gap_end - gap_start + 1 <= max_gap:
            output[gap_start : gap_end + 1] = True

    return output


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    while i < len(mask):
        while i < len(mask) and not mask[i]:
            i += 1
        if i >= len(mask):
            break
        start = i
        while i < len(mask) and mask[i]:
            i += 1
        runs.append((start, i - 1))
    return runs


def merge_runs_by_gap(
    runs: list[tuple[int, int]],
    max_gap_frames: int,
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in runs:
        if not merged or start - merged[-1][1] - 1 > max_gap_frames:
            merged.append([start, end])
        else:
            merged[-1][1] = end

    return [(start, end) for start, end in merged]


def infer_point_from_run(
    trajectory: pd.DataFrame,
    start: int,
    end: int,
    fps: float,
    config: ScoreConfig,
) -> Optional[RawPoint]:
    segment = trajectory.iloc[start : end + 1]
    valid_segment = segment[segment["valid"]]
    if valid_segment.empty:
        return None

    last_window = valid_segment.tail(max(1, int(config.dead_ball_window_seconds * fps)))
    ball_end_x = float(last_window["ball_x"].median())
    ball_end_y = float(last_window["ball_y"].median())
    last_side: Side = "near" if ball_end_y >= 0 else "far"

    outside = (
        valid_segment["ball_x"].abs().gt(config.court_x_limit_m + config.line_tolerance_m)
        | valid_segment["ball_y"].abs().gt(config.court_y_limit_m + config.line_tolerance_m)
    )
    last_outside = bool(outside.tail(max(1, int(0.35 * fps))).mean() >= 0.5)
    out_fraction = float(outside.mean())

    y_signs = np.sign(valid_segment["ball_y"].to_numpy())
    y_signs = y_signs[y_signs != 0]
    net_crossings = int(np.sum(y_signs[1:] != y_signs[:-1])) if len(y_signs) > 1 else 0

    if last_outside:
        winner_side = last_side
        reason = "opponent_hit_out"
        rule_id = "rally.out_after_good_bounce"
        outcome = "point_to_hitter"
        review_reason = (
            "Endpoint suggests an out/winner, but commercial certainty needs "
            "ground-bounce and wall/fence contact classification."
        )
    else:
        winner_side = opposite_side(last_side)
        reason = "dead_ball_on_opponent_side"
        rule_id = "rally.double_bounce"
        outcome = "point_to_hitter_after_not_up"
        review_reason = (
            "Endpoint suggests the ball died on the defending side, but "
            "commercial certainty needs bounce and hit detection."
        )

    valid_coverage = len(valid_segment) / max(1, end - start + 1)
    confidence = confidence_score(
        valid_coverage=valid_coverage,
        net_crossings=net_crossings,
        last_side_margin=abs(ball_end_y),
        last_outside=last_outside,
        out_fraction=out_fraction,
    )
    review_required = confidence < config.auto_apply_confidence

    return RawPoint(
        start_frame=int(trajectory.iloc[start]["frame"]),
        end_frame=int(trajectory.iloc[end]["frame"]),
        winner_side=winner_side,
        rule_id=rule_id,
        outcome=outcome,
        reason=reason,
        confidence=confidence,
        review_required=review_required,
        review_reason=review_reason if review_required else "",
        ball_end_x=ball_end_x,
        ball_end_y=ball_end_y,
        net_crossings=net_crossings,
        valid_coverage=float(valid_coverage),
        out_fraction=out_fraction,
    )


def confidence_score(
    valid_coverage: float,
    net_crossings: int,
    last_side_margin: float,
    last_outside: bool,
    out_fraction: float,
) -> float:
    score = 0.35
    score += min(0.25, valid_coverage * 0.25)
    score += 0.15 if net_crossings > 0 else 0.0
    score += 0.15 if last_side_margin >= 1.0 else 0.0
    score += 0.10 if last_outside else 0.0
    score -= 0.20 if out_fraction > 0.45 else 0.0
    return float(max(0.05, min(0.95, score)))


def confidence_band(confidence: float, config: ScoreConfig) -> str:
    if confidence >= config.auto_apply_confidence:
        return "auto"
    if confidence >= config.provisional_confidence:
        return "provisional_review"
    return "hold_for_review"


def opposite_side(side: Side) -> Side:
    return "far" if side == "near" else "near"


def infer_initial_teams(df: pd.DataFrame) -> dict:
    player_sides: dict[str, str] = {}
    for player_id in (1, 2, 3, 4):
        column = f"player{player_id}_y"
        if column not in df.columns:
            continue
        median_y = df[column].dropna().head(600).median()
        if np.isfinite(median_y):
            player_sides[str(player_id)] = "near" if median_y >= 0 else "far"

    near_players = [player for player, side in player_sides.items() if side == "near"]
    far_players = [player for player, side in player_sides.items() if side == "far"]

    return {
        "team_a": {
            "initial_side": "near",
            "players": near_players,
        },
        "team_b": {
            "initial_side": "far",
            "players": far_players,
        },
    }


def side_to_team(
    df: pd.DataFrame,
    side: Side,
    frame: int,
    team_info: dict,
) -> Team:
    team_a_side = team_side_at_frame(df, team_info["team_a"]["players"], frame)
    team_b_side = team_side_at_frame(df, team_info["team_b"]["players"], frame)

    if team_a_side != team_b_side and side == team_a_side:
        return "team_a"
    if team_a_side != team_b_side and side == team_b_side:
        return "team_b"

    return "team_a" if side == team_info["team_a"]["initial_side"] else "team_b"


def team_side_at_frame(df: pd.DataFrame, players: list[str], frame: int) -> Optional[Side]:
    values = []
    window = df[(df["frame"] >= frame - 90) & (df["frame"] <= frame + 30)]
    for player in players:
        column = f"player{player}_y"
        if column in window:
            values.extend(window[column].dropna().tolist())

    if not values:
        return None

    return "near" if float(np.median(values)) >= 0 else "far"


def format_sets(sets: list[tuple[int, int]]) -> str:
    return " ".join(f"{a}-{b}" for a, b in sets)
