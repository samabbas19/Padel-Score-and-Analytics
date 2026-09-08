"""Lightweight score overlay rendering.

The overlay is built with basic OpenCV drawing operations so it can be reused
in live scoring loops without adding a UI framework or GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


@dataclass
class ScoreOverlayState:
    team_a_name: str = "Team A"
    team_b_name: str = "Team B"
    sets: str = ""
    games: str = "0-0"
    points: str = "0-0"
    tiebreak: bool = False
    last_winner: str = ""
    confidence: Optional[float] = None


class ScoreOverlay:
    """Broadcast-style scoreboard overlay using only cheap OpenCV primitives."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    PANEL_ALPHA = 0.76
    TEAM_A_COLOR = (72, 226, 174)
    TEAM_B_COLOR = (78, 171, 255)
    TEXT = (245, 248, 252)
    MUTED = (164, 178, 196)
    DARK = (12, 20, 31)
    DARK_2 = (22, 33, 49)
    ACCENT = (34, 220, 255)

    def draw(self, frame: np.ndarray, state: ScoreOverlayState) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = max(0.72, min(1.22, w / 1920))
        x = int(28 * scale)
        y = int(24 * scale)
        panel_w = int(520 * scale)
        panel_h = int(176 * scale)

        output = frame.copy()
        self._draw_panel(output, x, y, panel_w, panel_h)

        pad = int(16 * scale)
        header_h = int(30 * scale)
        row_h = int(40 * scale)
        name_w = int(225 * scale)
        score_w = int(74 * scale)

        cv2.putText(
            output,
            "PADEL LIVE SCORE",
            (x + pad, y + int(21 * scale)),
            self.FONT,
            0.48 * scale,
            self.ACCENT,
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )

        if state.tiebreak:
            cv2.putText(
                output,
                "TIEBREAK",
                (x + panel_w - int(105 * scale), y + int(21 * scale)),
                self.FONT,
                0.42 * scale,
                (255, 232, 126),
                max(1, int(1.4 * scale)),
                cv2.LINE_AA,
            )

        labels_y = y + header_h + int(18 * scale)
        cv2.putText(
            output,
            "SETS",
            (x + pad + name_w, labels_y),
            self.FONT,
            0.38 * scale,
            self.MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            "GAMES",
            (x + pad + name_w + score_w, labels_y),
            self.FONT,
            0.38 * scale,
            self.MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            "PTS",
            (x + pad + name_w + score_w * 2 + int(12 * scale), labels_y),
            self.FONT,
            0.38 * scale,
            self.MUTED,
            1,
            cv2.LINE_AA,
        )

        sets_a, sets_b = self._split_score(state.sets, default=("0", "0"))
        games_a, games_b = self._split_score(state.games, default=("0", "0"))
        points_a, points_b = self._split_score(state.points, default=("0", "0"))

        row1_y = y + header_h + int(28 * scale)
        row2_y = row1_y + row_h
        self._draw_team_row(
            output,
            x + pad,
            row1_y,
            panel_w - pad * 2,
            row_h,
            state.team_a_name,
            sets_a,
            games_a,
            points_a,
            self.TEAM_A_COLOR,
            scale,
        )
        self._draw_team_row(
            output,
            x + pad,
            row2_y,
            panel_w - pad * 2,
            row_h,
            state.team_b_name,
            sets_b,
            games_b,
            points_b,
            self.TEAM_B_COLOR,
            scale,
        )

        if state.last_winner:
            footer = f"Last point: {state.last_winner}"
            if state.confidence is not None:
                footer += f"  {state.confidence:.0%}"
            cv2.putText(
                output,
                footer,
                (x + pad, y + panel_h - int(10 * scale)),
                self.FONT,
                0.42 * scale,
                self.MUTED,
                1,
                cv2.LINE_AA,
            )

        frame[:] = output
        return frame

    def _draw_panel(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return

        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), self.DARK, -1)
        cv2.rectangle(overlay, (0, 0), (w, int(h * 0.24)), self.DARK_2, -1)
        cv2.addWeighted(overlay, self.PANEL_ALPHA, roi, 1 - self.PANEL_ALPHA, 0, roi)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (68, 84, 105), 1)
        cv2.line(frame, (x, y), (x + w, y), self.ACCENT, 3)
        slant = np.array(
            [
                [x + w - 80, y],
                [x + w, y],
                [x + w - 34, y + int(h * 0.24)],
                [x + w - 114, y + int(h * 0.24)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [slant], (30, 103, 136))

    def _draw_team_row(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        team_name: str,
        sets: str,
        games: str,
        points: str,
        color: tuple[int, int, int],
        scale: float,
    ) -> None:
        cv2.rectangle(frame, (x, y), (x + w, y + h - 4), (17, 27, 41), -1)
        cv2.rectangle(frame, (x, y), (x + int(5 * scale), y + h - 4), color, -1)

        baseline = y + int(27 * scale)
        cv2.putText(
            frame,
            self._clip(team_name.upper(), 20),
            (x + int(14 * scale), baseline),
            self.FONT,
            0.52 * scale,
            self.TEXT,
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )

        score_x = x + int(230 * scale)
        self._draw_score_cell(frame, score_x, y + int(5 * scale), sets, scale, small=True)
        self._draw_score_cell(
            frame,
            score_x + int(74 * scale),
            y + int(5 * scale),
            games,
            scale,
            small=True,
        )
        self._draw_score_cell(
            frame,
            score_x + int(152 * scale),
            y + int(3 * scale),
            points,
            scale,
            small=False,
            accent=color,
        )

    def _draw_score_cell(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        text: str,
        scale: float,
        small: bool,
        accent: Optional[tuple[int, int, int]] = None,
    ) -> None:
        cell_w = int(56 * scale) if small else int(82 * scale)
        cell_h = int(28 * scale) if small else int(32 * scale)
        cv2.rectangle(frame, (x, y), (x + cell_w, y + cell_h), (31, 45, 63), -1)
        if accent:
            cv2.rectangle(frame, (x, y), (x + cell_w, y + cell_h), accent, 1)

        font_scale = 0.52 * scale if small else 0.68 * scale
        thickness = max(1, int(1.7 * scale))
        (tw, th), _ = cv2.getTextSize(text, self.FONT, font_scale, thickness)
        tx = x + max(2, int((cell_w - tw) / 2))
        ty = y + int((cell_h + th) / 2) - 1
        cv2.putText(
            frame,
            text,
            (tx, ty),
            self.FONT,
            font_scale,
            self.TEXT,
            thickness,
            cv2.LINE_AA,
        )

    def _split_score(self, value: object, default: tuple[str, str]) -> tuple[str, str]:
        if value is None:
            return default

        text = str(value).strip()
        if not text or text.lower() == "nan":
            return default

        if " " in text:
            text = text.split()[-1]

        if "-" not in text:
            return text, ""

        left, right = text.split("-", 1)
        return left.strip(), right.strip()

    def _clip(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "."


class ScoreTimeline:
    """Frame-indexed score state from score_events.csv."""

    def __init__(
        self,
        events_path: str | Path,
        summary_path: str | Path,
        last_point_hold_frames: int,
    ):
        self.events_path = Path(events_path)
        self.summary_path = Path(summary_path)
        self.events = self._load_events()
        self.summary = self._load_summary()
        self.last_point_hold_frames = last_point_hold_frames
        self.index = 0
        self.state = self._initial_state()
        self.last_event: Optional[pd.Series] = None

    def state_for_frame(self, frame_index: int) -> ScoreOverlayState:
        while self.index < len(self.events):
            row = self.events.iloc[self.index]
            if frame_index < int(row["end_frame"]):
                break
            self.last_event = row
            self.state = self._state_after(row)
            self.index += 1

        if self.last_event is not None:
            frames_since = frame_index - int(self.last_event["end_frame"])
            if frames_since <= self.last_point_hold_frames:
                self.state.last_winner = self._clean_text(self.last_event.get("winner_name"))
                self.state.confidence = self._clean_float(self.last_event.get("confidence"))
            else:
                self.state.last_winner = ""
                self.state.confidence = None

        return self.state

    def _load_events(self) -> pd.DataFrame:
        if not self.events_path.exists() or self.events_path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(self.events_path)

    def _load_summary(self) -> dict:
        if not self.summary_path.exists():
            return {}
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    def _initial_state(self) -> ScoreOverlayState:
        config = self.summary.get("config", {})
        state = ScoreOverlayState(
            team_a_name=config.get("team_a_name", "Team A"),
            team_b_name=config.get("team_b_name", "Team B"),
        )

        if not self.events.empty:
            first = self.events.iloc[0]
            state.sets = self._clean_text(first.get("score_before_sets"))
            state.games = self._clean_text(first.get("score_before_games"), "0-0")
            state.points = self._clean_text(first.get("score_before_points"), "0-0")

        return state

    def _state_after(self, row: pd.Series) -> ScoreOverlayState:
        config = self.summary.get("config", {})
        return ScoreOverlayState(
            team_a_name=config.get("team_a_name", "Team A"),
            team_b_name=config.get("team_b_name", "Team B"),
            sets=self._clean_text(row.get("score_after_sets")),
            games=self._clean_text(row.get("score_after_games"), "0-0"),
            points=self._clean_text(row.get("score_after_points"), "0-0"),
            tiebreak=bool(row.get("tiebreak_after", False)),
        )

    def _clean_text(self, value: object, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return default
        return text

    def _clean_float(self, value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(parsed):
            return None
        return parsed


def render_score_overlay_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    events_path: str | Path,
    summary_path: str | Path,
    last_point_hold_seconds: float = 3.0,
) -> None:
    """Render score overlay onto an existing inference video."""

    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not write video: {output_video_path}")

    overlay = ScoreOverlay()
    timeline = ScoreTimeline(
        events_path=events_path,
        summary_path=summary_path,
        last_point_hold_frames=int(last_point_hold_seconds * fps),
    )

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        state = timeline.state_for_frame(frame_index)
        overlay.draw(frame, state)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    if total_frames and frame_index != total_frames:
        print(
            "score_overlay: WARNING "
            f"rendered {frame_index}/{total_frames} frames"
        )
