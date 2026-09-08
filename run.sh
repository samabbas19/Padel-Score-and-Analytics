#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR="${PADEL_VENV_DIR:-.venv}"
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # Windows venv used from Git Bash.
  source "$VENV_DIR/Scripts/activate"
  PYTHON_BIN="$VENV_DIR/Scripts/python.exe"
elif [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
else
  echo "Virtual environment not found at $VENV_DIR"
  echo "Create it with: py -3.12 -m venv .venv"
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, got {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required, but torch.cuda.is_available() is False")

print(f"GPU OK: torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
PY

missing_weight=0
for weight in \
  "weights/players_detection/yolov8m.pt" \
  "weights/players_keypoints_detection/best.pt" \
  "weights/ball_detection/TrackNet_best.pt" \
  "weights/ball_detection/InpaintNet_best.pt" \
  "weights/court_keypoints_detection/best.pt"
do
  if [[ ! -f "$weight" ]]; then
    echo "Missing weight: $weight"
    missing_weight=1
  fi
done
if [[ "$missing_weight" -ne 0 ]]; then
  echo "Download the repo weights before running inference."
  exit 1
fi

DEFAULT_VIDEO="../padel_match.mp4"
if [[ ! -f "$DEFAULT_VIDEO" ]]; then
  DEFAULT_VIDEO="./examples/videos/rally.mp4"
fi

VIDEO_PATH="${1:-${PADEL_INPUT_VIDEO_PATH:-$DEFAULT_VIDEO}}"
VIDEO_PATH_FOR_PY="$VIDEO_PATH"
if command -v cygpath >/dev/null 2>&1; then
  VIDEO_PATH_FOR_PY="$(cygpath -w "$VIDEO_PATH")"
elif command -v wslpath >/dev/null 2>&1; then
  VIDEO_PATH_FOR_PY="$(wslpath -w "$VIDEO_PATH")"
fi

VIDEO_ABS="$("$PYTHON_BIN" - "$VIDEO_PATH_FOR_PY" <<'PY' | tr -d '\r'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
if not path.exists():
    raise SystemExit(f"Video not found: {path}")
print(path.resolve().as_posix())
PY
)"

VIDEO_ID="$("$PYTHON_BIN" - "$VIDEO_ABS" <<'PY' | tr -d '\r'
from pathlib import Path
import hashlib
import re
import sys

path = Path(sys.argv[1])
digest = hashlib.sha1()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)

safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-") or "video"
print(f"{safe_name}-{digest.hexdigest()[:12]}")
PY
)"

CACHE_DIR="cache/$VIDEO_ID"
OUTPUT_DIR="outputs/$VIDEO_ID"
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

set_load_path() {
  local name="$1"
  local path="$2"
  export "$name=$path"
}

export PADEL_STRICT_GPU=1
export PADEL_INPUT_VIDEO_PATH="$VIDEO_ABS"
export PADEL_SCORE_SOURCE="${PADEL_SCORE_SOURCE:-scoreboard}"
export PADEL_OUTPUT_VIDEO_PATH="$OUTPUT_DIR/results.mp4"
export PADEL_COLLECT_DATA="${PADEL_COLLECT_DATA:-1}"
export PADEL_COLLECT_DATA_PATH="$OUTPUT_DIR/data.csv"
export PADEL_AUTO_SCORE="${PADEL_AUTO_SCORE:-1}"
export PADEL_SCORE_EVENTS_PATH="$OUTPUT_DIR/score_events.csv"
export PADEL_SCORE_SUMMARY_PATH="$OUTPUT_DIR/score_summary.json"
export PADEL_GAMEPLAY_SCORE_EVENTS_PATH="$OUTPUT_DIR/gameplay_score_events.csv"
export PADEL_GAMEPLAY_SCORE_SUMMARY_PATH="$OUTPUT_DIR/gameplay_score_summary.json"
export PADEL_RENDER_SCORE_OVERLAY="${PADEL_RENDER_SCORE_OVERLAY:-1}"
export PADEL_SCORE_OVERLAY_VIDEO_PATH="$OUTPUT_DIR/results_score_overlay.mp4"
export PADEL_DETECT_AUDIO_VISUAL_EVENTS="${PADEL_DETECT_AUDIO_VISUAL_EVENTS:-1}"
export PADEL_EVENT_CANDIDATES_PATH="$OUTPUT_DIR/event_candidates.csv"
export PADEL_EVENT_SUMMARY_PATH="$OUTPUT_DIR/event_summary.json"
export PADEL_EXTRACT_SCOREBOARD_GROUND_TRUTH="${PADEL_EXTRACT_SCOREBOARD_GROUND_TRUTH:-1}"
export PADEL_SCOREBOARD_STATES_PATH="$OUTPUT_DIR/scoreboard_states.csv"
export PADEL_SCOREBOARD_EVENTS_PATH="$OUTPUT_DIR/scoreboard_events.csv"
export PADEL_SCOREBOARD_SUMMARY_PATH="$OUTPUT_DIR/scoreboard_summary.json"
export PADEL_VALIDATE_SCORE_TIMELINE="${PADEL_VALIDATE_SCORE_TIMELINE:-1}"
export PADEL_SCORE_VALIDATION_REPORT_PATH="$OUTPUT_DIR/score_validation_report.json"
export PADEL_SCORE_VALIDATION_MISMATCHES_PATH="$OUTPUT_DIR/score_validation_mismatches.csv"
export PADEL_SCOREBOARD_BASELINE_EVENTS_PATH="$OUTPUT_DIR/scoreboard_baseline_events.csv"
export PADEL_SCOREBOARD_BASELINE_SUMMARY_PATH="$OUTPUT_DIR/scoreboard_baseline_summary.json"
export PADEL_SCOREBOARD_BASELINE_VALIDATION_REPORT_PATH="$OUTPUT_DIR/scoreboard_baseline_validation_report.json"
export PADEL_SCOREBOARD_BASELINE_VALIDATION_MISMATCHES_PATH="$OUTPUT_DIR/scoreboard_baseline_validation_mismatches.csv"
export PADEL_GAMEPLAY_SCORE_VALIDATION_REPORT_PATH="$OUTPUT_DIR/gameplay_score_validation_report.json"
export PADEL_GAMEPLAY_SCORE_VALIDATION_MISMATCHES_PATH="$OUTPUT_DIR/gameplay_score_validation_mismatches.csv"
export PADEL_GAMEPLAY_ALIGNMENT_PATH="$OUTPUT_DIR/gameplay_event_alignment.csv"
export PADEL_GAMEPLAY_ALIGNMENT_SUMMARY_PATH="$OUTPUT_DIR/gameplay_event_alignment_summary.json"
export PADEL_GAMEPLAY_POINT_LABELS_PATH="$OUTPUT_DIR/gameplay_point_labels.csv"
export PADEL_GAMEPLAY_POINT_LABELS_SUMMARY_PATH="$OUTPUT_DIR/gameplay_point_labels_summary.json"

KEYPOINTS_FILE="$CACHE_DIR/fixed_keypoints_detection.json"
export PADEL_FIXED_COURT_KEYPOINTS_SAVE_PATH="$KEYPOINTS_FILE"
set_load_path PADEL_FIXED_COURT_KEYPOINTS_LOAD_PATH "$KEYPOINTS_FILE"

export PADEL_PLAYERS_TRACKER_SAVE_PATH="$CACHE_DIR/players_detections.json"
set_load_path PADEL_PLAYERS_TRACKER_LOAD_PATH "$PADEL_PLAYERS_TRACKER_SAVE_PATH"

export PADEL_PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH="$CACHE_DIR/players_keypoints_detections.json"
set_load_path PADEL_PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH "$PADEL_PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH"

export PADEL_BALL_TRACKER_SAVE_PATH="$CACHE_DIR/ball_detections.json"
set_load_path PADEL_BALL_TRACKER_LOAD_PATH "$PADEL_BALL_TRACKER_SAVE_PATH"

echo "Video: $VIDEO_ABS"
echo "Video cache: $CACHE_DIR"
echo "Score source: $PADEL_SCORE_SOURCE"
echo "Output video: $PADEL_OUTPUT_VIDEO_PATH"
echo "Score events: $PADEL_SCORE_EVENTS_PATH"
echo "Score summary: $PADEL_SCORE_SUMMARY_PATH"
echo "Score overlay video: $PADEL_SCORE_OVERLAY_VIDEO_PATH"
echo "Event candidates: $PADEL_EVENT_CANDIDATES_PATH"
echo "Scoreboard states: $PADEL_SCOREBOARD_STATES_PATH"
echo "Score validation: $PADEL_SCORE_VALIDATION_REPORT_PATH"
echo "Gameplay score validation: $PADEL_GAMEPLAY_SCORE_VALIDATION_REPORT_PATH"
echo "Gameplay event alignment: $PADEL_GAMEPLAY_ALIGNMENT_PATH"
echo "Gameplay point labels: $PADEL_GAMEPLAY_POINT_LABELS_PATH"
echo "Scoreboard baseline validation: $PADEL_SCOREBOARD_BASELINE_VALIDATION_REPORT_PATH"

if [[ -f "$KEYPOINTS_FILE" ]]; then
  echo "Using saved court keypoints: $KEYPOINTS_FILE"
else
  echo "Court keypoint selection will open now."
  echo "Click exactly 12 keypoints in order k1 through k12, then press any key in the OpenCV window."
fi

if [[ "${PADEL_DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run complete. Set PADEL_DRY_RUN=0 or unset it to run inference."
  exit 0
fi

"$PYTHON_BIN" main.py
