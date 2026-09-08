"""No-training audio + vision event candidates for padel scoring.

The scorer needs reliable impact events before it can become rule-accurate.
This module does not use OCR and does not train a model. It fuses fast audio
transients with the existing ball/player detections to produce candidate
racket hits, court bounces, net contacts, and wall/fence contacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AudioVisualEventConfig:
    """Tunable thresholds for no-training event spotting."""

    audio_sample_rate: int = 48_000
    audio_window_seconds: float = 0.008
    audio_hop_seconds: float = 0.004
    audio_peak_z: float = 6.5
    audio_min_gap_seconds: float = 0.075
    spectral_window_seconds: float = 0.016
    spectral_high_hz: float = 3_000.0
    candidate_min_confidence: float = 0.45
    auto_confidence: float = 0.82
    player_contact_px: float = 70.0
    player_near_px: float = 190.0
    trajectory_window_frames: int = 4
    court_half_width_m: float = 5.0
    court_half_length_m: float = 10.0
    court_margin_m: float = 0.75
    net_tolerance_m: float = 0.75
    boundary_tolerance_m: float = 1.25


def detect_audio_visual_events(
    video_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    fps: float,
    ball_detections_path: Optional[str | Path] = None,
    players_detections_path: Optional[str | Path] = None,
    config: Optional[AudioVisualEventConfig] = None,
) -> dict:
    """Write event candidate CSV/JSON summary for an analysed video."""

    config = config or AudioVisualEventConfig()
    video_path = Path(video_path)
    data_path = Path(data_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)

    audio = load_audio_mono(video_path, config.audio_sample_rate)
    envelope = audio_transient_envelope(audio, config)
    peaks = pick_audio_peaks(envelope, config)

    df = pd.read_csv(data_path)
    ball_detections = load_json_list(ball_detections_path)
    players_detections = load_json_list(players_detections_path)

    rows = []
    for peak_index in peaks:
        audio_features = audio_features_for_peak(audio, envelope, peak_index, config)
        frame = int(round(audio_features["time"] * fps))
        if frame < 0 or frame >= len(df):
            continue

        visual_features = visual_features_for_frame(
            frame=frame,
            df=df,
            fps=fps,
            ball_detections=ball_detections,
            players_detections=players_detections,
            config=config,
        )
        row = classify_event(
            frame=frame,
            audio_features=audio_features,
            visual_features=visual_features,
            config=config,
        )
        if row["confidence"] >= config.candidate_min_confidence:
            rows.append(row)

    rows = suppress_duplicate_candidates(rows, fps=fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    events_df = pd.DataFrame(rows)
    events_df.to_csv(output_path, index=False)

    counts = (
        events_df["event_type"].value_counts().to_dict()
        if not events_df.empty and "event_type" in events_df
        else {}
    )
    summary = {
        "mode": "no_training_audio_visual_event_spotting",
        "source_video": str(video_path),
        "source_data": str(data_path),
        "events_path": str(output_path),
        "total_audio_peaks": int(len(peaks)),
        "candidates_kept": int(len(rows)),
        "counts_by_event_type": counts,
        "review_required_count": int(
            events_df["review_required"].sum()
            if not events_df.empty and "review_required" in events_df
            else 0
        ),
        "config": asdict(config),
        "live_latency_seconds": round(
            max(
                config.spectral_window_seconds,
                config.trajectory_window_frames / max(fps, 1.0),
            ),
            4,
        ),
        "limitations": (
            "This layer ignores the scoreboard and uses no model training. It is "
            "suitable for live candidate generation, but ambiguous wall/fence/net, "
            "body touch, double hit, serve legality, and obstructed ball events "
            "still need either a trained event model, extra camera/microphone "
            "coverage, official feed data, or human review."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_audio_mono(video_path: Path, sample_rate: int) -> np.ndarray:
    """Decode video audio to mono float32 using ffmpeg."""

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    raw = subprocess.check_output(command)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        raise ValueError(f"No audio could be decoded from {video_path}")
    audio /= np.iinfo(np.int16).max
    audio -= float(np.median(audio))
    return audio


def audio_transient_envelope(
    audio: np.ndarray,
    config: AudioVisualEventConfig,
) -> dict:
    """Return robust-z scored transient envelope from high-emphasis audio."""

    sample_rate = config.audio_sample_rate
    window = max(8, int(config.audio_window_seconds * sample_rate))
    hop = max(1, int(config.audio_hop_seconds * sample_rate))
    if len(audio) < window:
        raise ValueError("Audio is shorter than one analysis window")

    emphasized = np.diff(audio, prepend=audio[0])
    frame_count = 1 + (len(emphasized) - window) // hop
    frames = np.lib.stride_tricks.as_strided(
        emphasized,
        shape=(frame_count, window),
        strides=(emphasized.strides[0] * hop, emphasized.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    median = float(np.median(rms))
    mad = float(np.median(np.abs(rms - median))) + 1e-9
    robust_z = (rms - median) / (1.4826 * mad)
    return {
        "rms": rms,
        "z": robust_z,
        "hop": hop,
        "sample_rate": sample_rate,
    }


def pick_audio_peaks(
    envelope: dict,
    config: AudioVisualEventConfig,
) -> list[int]:
    """Simple local-max peak picking without scipy."""

    z = envelope["z"]
    hop = envelope["hop"]
    sample_rate = envelope["sample_rate"]
    min_gap = max(1, int(config.audio_min_gap_seconds * sample_rate / hop))

    candidates = np.flatnonzero(
        (z >= config.audio_peak_z)
        & (z >= np.r_[z[0], z[:-1]])
        & (z > np.r_[z[1:], z[-1]])
    )
    if len(candidates) == 0:
        return []

    ordered = sorted(candidates.tolist(), key=lambda index: float(z[index]), reverse=True)
    selected: list[int] = []
    occupied = np.zeros(len(z), dtype=bool)
    for index in ordered:
        start = max(0, index - min_gap)
        end = min(len(z), index + min_gap + 1)
        if occupied[start:end].any():
            continue
        selected.append(index)
        occupied[start:end] = True

    return sorted(selected)


def audio_features_for_peak(
    audio: np.ndarray,
    envelope: dict,
    peak_index: int,
    config: AudioVisualEventConfig,
) -> dict:
    sample_rate = config.audio_sample_rate
    hop = envelope["hop"]
    center_sample = int(peak_index * hop)
    half = max(16, int(config.spectral_window_seconds * sample_rate / 2))
    start = max(0, center_sample - half)
    end = min(len(audio), center_sample + half)
    windowed = audio[start:end]
    if len(windowed) < 8:
        windowed = np.pad(windowed, (0, 8 - len(windowed)))

    windowed = windowed * np.hanning(len(windowed))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), 1 / sample_rate)
    power = spectrum * spectrum
    total_power = float(power.sum()) + 1e-12
    high_ratio = float(power[freqs >= config.spectral_high_hz].sum() / total_power)
    centroid = float((freqs * power).sum() / total_power)
    peak_rms = float(envelope["rms"][peak_index])
    peak_z = float(envelope["z"][peak_index])

    return {
        "time": center_sample / sample_rate,
        "audio_peak_z": peak_z,
        "audio_rms": peak_rms,
        "audio_centroid_hz": centroid,
        "audio_high_ratio": high_ratio,
    }


def visual_features_for_frame(
    frame: int,
    df: pd.DataFrame,
    fps: float,
    ball_detections: list,
    players_detections: list,
    config: AudioVisualEventConfig,
) -> dict:
    row = df.iloc[frame]
    ball_court = get_court_ball(row)
    ball_image = get_image_ball(frame, ball_detections)
    trajectory = trajectory_features(frame, ball_detections, df, config)
    player = nearest_player(frame, ball_image, players_detections, config)

    court_x = ball_court[0] if ball_court is not None else np.nan
    court_y = ball_court[1] if ball_court is not None else np.nan
    inside_court = bool(
        ball_court is not None
        and abs(court_x) <= config.court_half_width_m + config.court_margin_m
        and abs(court_y) <= config.court_half_length_m + config.court_margin_m
    )
    side = "near" if ball_court is not None and court_y >= 0 else "far"

    if ball_court is not None:
        boundary_distance = min(
            abs(config.court_half_width_m - abs(court_x)),
            abs(config.court_half_length_m - abs(court_y)),
        )
        boundary_score = clamp01(
            (config.boundary_tolerance_m - boundary_distance)
            / max(config.boundary_tolerance_m, 1e-6)
        )
        net_score = clamp01(
            (config.net_tolerance_m - abs(court_y))
            / max(config.net_tolerance_m, 1e-6)
        )
    else:
        boundary_score = 0.0
        net_score = 0.0

    return {
        "ball_img_x": ball_image[0] if ball_image is not None else np.nan,
        "ball_img_y": ball_image[1] if ball_image is not None else np.nan,
        "ball_x": court_x,
        "ball_y": court_y,
        "ball_side": side,
        "inside_court": inside_court,
        "boundary_score": float(boundary_score),
        "net_score": float(net_score),
        "nearest_player_id": player["id"],
        "nearest_player_distance_px": player["distance_px"],
        "nearest_player_score": player["score"],
        "raw_motion_px": trajectory["raw_motion_px"],
        "raw_acceleration_px": trajectory["raw_acceleration_px"],
        "raw_angle_change_deg": trajectory["raw_angle_change_deg"],
        "visual_motion_score": trajectory["visual_motion_score"],
        "fps": fps,
    }


def get_court_ball(row: pd.Series) -> Optional[tuple[float, float]]:
    try:
        visible = int(row.get("ball_visible", 0)) == 1
        x = float(row.get("ball_x"))
        y = float(row.get("ball_y"))
    except (TypeError, ValueError):
        return None
    if not visible or not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def get_image_ball(frame: int, ball_detections: list) -> Optional[tuple[float, float]]:
    if frame < 0 or frame >= len(ball_detections):
        return None
    detection = ball_detections[frame]
    if not detection or int(detection.get("visibility", 0)) != 1:
        return None
    xy = detection.get("xy")
    if xy is None or len(xy) != 2:
        return None
    x, y = float(xy[0]), float(xy[1])
    if not np.isfinite(x) or not np.isfinite(y) or (x == 0 and y == 0):
        return None
    return x, y


def trajectory_features(
    frame: int,
    ball_detections: list,
    df: pd.DataFrame,
    config: AudioVisualEventConfig,
) -> dict:
    window = config.trajectory_window_frames
    raw_positions = []
    for index in range(frame - window, frame + window + 1):
        raw_positions.append(get_image_ball(index, ball_detections))

    raw = interpolate_positions(raw_positions)
    if raw is None:
        court_positions = []
        for index in range(frame - window, frame + window + 1):
            if index < 0 or index >= len(df):
                court_positions.append(None)
            else:
                court_positions.append(get_court_ball(df.iloc[index]))
        raw = interpolate_positions(court_positions)

    if raw is None or len(raw) < 5:
        return {
            "raw_motion_px": 0.0,
            "raw_acceleration_px": 0.0,
            "raw_angle_change_deg": 0.0,
            "visual_motion_score": 0.0,
        }

    center = len(raw) // 2
    pre = raw[center] - raw[max(0, center - 2)]
    post = raw[min(len(raw) - 1, center + 2)] - raw[center]
    acceleration = np.diff(np.diff(raw, axis=0), axis=0)
    max_acceleration = float(np.nanmax(np.linalg.norm(acceleration, axis=1)))
    motion = float(np.nansum(np.linalg.norm(np.diff(raw, axis=0), axis=1)))
    angle_change = vector_angle_degrees(pre, post)
    visual_score = clamp01(max_acceleration / 120.0) * 0.60
    visual_score += clamp01(angle_change / 90.0) * 0.25
    visual_score += clamp01(motion / 160.0) * 0.15

    return {
        "raw_motion_px": motion,
        "raw_acceleration_px": max_acceleration,
        "raw_angle_change_deg": angle_change,
        "visual_motion_score": float(clamp01(visual_score)),
    }


def interpolate_positions(
    positions: list[Optional[tuple[float, float]]],
) -> Optional[np.ndarray]:
    values = np.array(
        [
            [np.nan, np.nan] if position is None else [position[0], position[1]]
            for position in positions
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(values[:, 0]) & np.isfinite(values[:, 1])
    if valid.sum() < 3:
        return None
    indexes = np.arange(len(values))
    for column in (0, 1):
        values[:, column] = np.interp(
            indexes,
            indexes[valid],
            values[valid, column],
        )
    return values


def nearest_player(
    frame: int,
    ball_image: Optional[tuple[float, float]],
    players_detections: list,
    config: AudioVisualEventConfig,
) -> dict:
    if ball_image is None or not players_detections:
        return {"id": "", "distance_px": np.nan, "score": 0.0}

    best = {"id": "", "distance_px": float("inf"), "score": 0.0}
    for index in range(max(0, frame - 2), min(len(players_detections), frame + 3)):
        for detection in players_detections[index] or []:
            xyxy = detection.get("xyxy")
            if xyxy is None or len(xyxy) != 4:
                continue
            distance = distance_to_box(ball_image, xyxy)
            if distance < best["distance_px"]:
                best = {
                    "id": str(detection.get("id", "")),
                    "distance_px": float(distance),
                    "score": float(
                        clamp01(
                            (config.player_near_px - distance)
                            / max(config.player_near_px, 1e-6)
                        )
                    ),
                }

    if not np.isfinite(best["distance_px"]):
        return {"id": "", "distance_px": np.nan, "score": 0.0}
    return best


def distance_to_box(point: tuple[float, float], xyxy: list[float]) -> float:
    x, y = point
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    nearest_x = min(max(x, x1), x2)
    nearest_y = min(max(y, y1), y2)
    return float(np.hypot(x - nearest_x, y - nearest_y))


def classify_event(
    frame: int,
    audio_features: dict,
    visual_features: dict,
    config: AudioVisualEventConfig,
) -> dict:
    audio_score = clamp01((audio_features["audio_peak_z"] - config.audio_peak_z) / 16.0)
    high_score = clamp01(audio_features["audio_high_ratio"] / 0.38)
    motion_score = visual_features["visual_motion_score"]
    player_score = visual_features["nearest_player_score"]
    net_score = visual_features["net_score"]
    boundary_score = visual_features["boundary_score"]
    non_player_score = 1.0 - player_score

    racket_confidence = (
        0.34 * audio_score + 0.34 * player_score + 0.20 * motion_score + 0.12 * high_score
    )
    net_confidence = (
        0.35 * audio_score + 0.30 * net_score + 0.20 * motion_score + 0.15 * non_player_score
    )
    wall_confidence = (
        0.34 * audio_score
        + 0.30 * boundary_score
        + 0.20 * motion_score
        + 0.16 * non_player_score
    )
    bounce_confidence = (
        0.34 * audio_score
        + 0.28 * motion_score
        + 0.20 * non_player_score
        + 0.18 * float(visual_features["inside_court"])
    )

    scores = {
        "racket_hit": racket_confidence,
        "net_contact": net_confidence,
        "wall_or_fence_contact": wall_confidence,
        "court_bounce": bounce_confidence,
    }
    event_type, confidence = max(scores.items(), key=lambda item: item[1])

    if event_type == "racket_hit" and player_score < 0.35:
        event_type = "unknown_impact"
        confidence *= 0.70
    elif event_type == "net_contact" and net_score < 0.55:
        event_type = "unknown_impact"
        confidence *= 0.70
    elif event_type == "wall_or_fence_contact" and boundary_score < 0.45:
        event_type = "unknown_impact"
        confidence *= 0.75
    elif event_type == "court_bounce" and not visual_features["inside_court"]:
        event_type = "unknown_impact"
        confidence *= 0.70

    review_required = confidence < config.auto_confidence
    review_reason = ""
    if review_required:
        review_reason = review_reason_for(event_type, visual_features)

    row = {
        "frame": frame,
        "time": audio_features["time"],
        "event_type": event_type,
        "confidence": float(round(clamp01(confidence), 4)),
        "review_required": bool(review_required),
        "review_reason": review_reason,
    }
    row.update(audio_features)
    row.update(visual_features)
    return row


def review_reason_for(event_type: str, visual_features: dict) -> str:
    if event_type == "unknown_impact":
        return "Audio transient did not align cleanly with player, net, boundary, or court-bounce geometry."
    if event_type == "racket_hit":
        return "Racket hit candidate needs confirmation when the ball is close to player boxes but contact is visually occluded."
    if event_type == "court_bounce":
        return "Single-camera floor projection cannot prove vertical bounce height in every frame."
    if event_type == "net_contact":
        return "Net contact needs visual confirmation because audio and projected court position can be ambiguous."
    if event_type == "wall_or_fence_contact":
        return "Wall/fence contact needs confirmation because projected ball position may represent an airborne ball."
    return "Low-confidence event candidate."


def suppress_duplicate_candidates(rows: list[dict], fps: float) -> list[dict]:
    """Keep the strongest candidate in very short event bursts."""

    if not rows:
        return []

    rows = sorted(rows, key=lambda row: row["time"])
    min_gap = max(0.050, 3.0 / max(fps, 1.0))
    kept: list[dict] = []
    for row in rows:
        if not kept or row["time"] - kept[-1]["time"] > min_gap:
            kept.append(row)
            continue

        previous = kept[-1]
        if row["confidence"] > previous["confidence"]:
            kept[-1] = row
    return kept


def load_json_list(path: Optional[str | Path]) -> list:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def vector_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-6 or b_norm < 1e-6:
        return 0.0
    cosine = float(np.dot(a, b) / (a_norm * b_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
