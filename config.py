"""General configurations for main.py."""

import os


def _env_path(name: str, default: str | None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or None


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


# Input video path
INPUT_VIDEO_PATH = _env_path("PADEL_INPUT_VIDEO_PATH", "./examples/videos/rally.mp4")

# Inference video path
OUTPUT_VIDEO_PATH = _env_path("PADEL_OUTPUT_VIDEO_PATH", "results.mp4")

# True to collect 2d projection data
COLLECT_DATA = _env_bool("PADEL_COLLECT_DATA", True)
# Collected data path
COLLECT_DATA_PATH = _env_path("PADEL_COLLECT_DATA_PATH", "data.csv")

# Pure trajectory score calculation
AUTO_SCORE = _env_bool("PADEL_AUTO_SCORE", True)
SCORE_SOURCE = os.environ.get("PADEL_SCORE_SOURCE", "proposal").strip().lower()
SCORE_EVENTS_PATH = _env_path("PADEL_SCORE_EVENTS_PATH", "score_events.csv")
SCORE_SUMMARY_PATH = _env_path("PADEL_SCORE_SUMMARY_PATH", "score_summary.json")
GAMEPLAY_SCORE_EVENTS_PATH = _env_path(
    "PADEL_GAMEPLAY_SCORE_EVENTS_PATH",
    "gameplay_score_events.csv",
)
GAMEPLAY_SCORE_SUMMARY_PATH = _env_path(
    "PADEL_GAMEPLAY_SCORE_SUMMARY_PATH",
    "gameplay_score_summary.json",
)
RENDER_SCORE_OVERLAY = _env_bool("PADEL_RENDER_SCORE_OVERLAY", True)
SCORE_OVERLAY_VIDEO_PATH = _env_path(
    "PADEL_SCORE_OVERLAY_VIDEO_PATH",
    "results_score_overlay.mp4",
)
SCORE_GOLDEN_POINT = _env_bool("PADEL_SCORE_GOLDEN_POINT", False)
SCORE_POINT_BREAK_SECONDS = _env_float("PADEL_SCORE_POINT_BREAK_SECONDS", 6.0)
SCORE_AUTO_APPLY_CONFIDENCE = _env_float("PADEL_SCORE_AUTO_APPLY_CONFIDENCE", 0.92)
SCORE_PROVISIONAL_CONFIDENCE = _env_float("PADEL_SCORE_PROVISIONAL_CONFIDENCE", 0.70)
SCORE_INITIAL_POINTS = os.environ.get("PADEL_SCORE_INITIAL_POINTS", "0-0").strip()
SCORE_INITIAL_GAMES = os.environ.get("PADEL_SCORE_INITIAL_GAMES", "0-0").strip()
PROPOSAL_MIN_TIME_SECONDS = _env_float("PADEL_PROPOSAL_MIN_TIME_SECONDS", 20.0)

# No-training audio + vision event candidates
DETECT_AUDIO_VISUAL_EVENTS = _env_bool("PADEL_DETECT_AUDIO_VISUAL_EVENTS", True)
EVENT_CANDIDATES_PATH = _env_path("PADEL_EVENT_CANDIDATES_PATH", "event_candidates.csv")
EVENT_SUMMARY_PATH = _env_path("PADEL_EVENT_SUMMARY_PATH", "event_summary.json")
AUDIO_EVENT_PEAK_Z = _env_float("PADEL_AUDIO_EVENT_PEAK_Z", 6.5)
AUDIO_EVENT_MIN_CONFIDENCE = _env_float("PADEL_AUDIO_EVENT_MIN_CONFIDENCE", 0.45)
AUDIO_EVENT_AUTO_CONFIDENCE = _env_float("PADEL_AUDIO_EVENT_AUTO_CONFIDENCE", 0.82)

# Scoreboard extraction is validation-only ground truth. It must not feed scorer.
EXTRACT_SCOREBOARD_GROUND_TRUTH = _env_bool(
    "PADEL_EXTRACT_SCOREBOARD_GROUND_TRUTH",
    True,
)
SCOREBOARD_STATES_PATH = _env_path(
    "PADEL_SCOREBOARD_STATES_PATH",
    "scoreboard_states.csv",
)
SCOREBOARD_EVENTS_PATH = _env_path(
    "PADEL_SCOREBOARD_EVENTS_PATH",
    "scoreboard_events.csv",
)
SCOREBOARD_SUMMARY_PATH = _env_path(
    "PADEL_SCOREBOARD_SUMMARY_PATH",
    "scoreboard_summary.json",
)
SCOREBOARD_TRANSITION_EXPANSION_REPORT_PATH = _env_path(
    "PADEL_SCOREBOARD_TRANSITION_EXPANSION_REPORT_PATH",
    "scoreboard_transition_expansion_report.json",
)
SCOREBOARD_TRANSITION_EXPANSION_DETAILS_PATH = _env_path(
    "PADEL_SCOREBOARD_TRANSITION_EXPANSION_DETAILS_PATH",
    "scoreboard_transition_expansion_details.csv",
)
SCOREBOARD_SAMPLE_STRIDE_FRAMES = _env_int(
    "PADEL_SCOREBOARD_SAMPLE_STRIDE_FRAMES",
    15,
)
SCOREBOARD_TEMPLATE_SEEDS_PATH = _env_path(
    "PADEL_SCOREBOARD_TEMPLATE_SEEDS_PATH",
    None,
)
VALIDATE_SCORE_TIMELINE = _env_bool("PADEL_VALIDATE_SCORE_TIMELINE", True)
SCORE_VALIDATION_REPORT_PATH = _env_path(
    "PADEL_SCORE_VALIDATION_REPORT_PATH",
    "score_validation_report.json",
)
SCORE_VALIDATION_MISMATCHES_PATH = _env_path(
    "PADEL_SCORE_VALIDATION_MISMATCHES_PATH",
    "score_validation_mismatches.csv",
)
SCOREBOARD_BASELINE_EVENTS_PATH = _env_path(
    "PADEL_SCOREBOARD_BASELINE_EVENTS_PATH",
    "scoreboard_baseline_events.csv",
)
SCOREBOARD_BASELINE_SUMMARY_PATH = _env_path(
    "PADEL_SCOREBOARD_BASELINE_SUMMARY_PATH",
    "scoreboard_baseline_summary.json",
)
SCOREBOARD_BASELINE_VALIDATION_REPORT_PATH = _env_path(
    "PADEL_SCOREBOARD_BASELINE_VALIDATION_REPORT_PATH",
    "scoreboard_baseline_validation_report.json",
)
SCOREBOARD_BASELINE_VALIDATION_MISMATCHES_PATH = _env_path(
    "PADEL_SCOREBOARD_BASELINE_VALIDATION_MISMATCHES_PATH",
    "scoreboard_baseline_validation_mismatches.csv",
)
GAMEPLAY_SCORE_VALIDATION_REPORT_PATH = _env_path(
    "PADEL_GAMEPLAY_SCORE_VALIDATION_REPORT_PATH",
    "gameplay_score_validation_report.json",
)
GAMEPLAY_SCORE_VALIDATION_MISMATCHES_PATH = _env_path(
    "PADEL_GAMEPLAY_SCORE_VALIDATION_MISMATCHES_PATH",
    "gameplay_score_validation_mismatches.csv",
)
GAMEPLAY_ALIGNMENT_PATH = _env_path(
    "PADEL_GAMEPLAY_ALIGNMENT_PATH",
    "gameplay_event_alignment.csv",
)
GAMEPLAY_ALIGNMENT_SUMMARY_PATH = _env_path(
    "PADEL_GAMEPLAY_ALIGNMENT_SUMMARY_PATH",
    "gameplay_event_alignment_summary.json",
)
GAMEPLAY_POINT_LABELS_PATH = _env_path(
    "PADEL_GAMEPLAY_POINT_LABELS_PATH",
    "gameplay_point_labels.csv",
)
GAMEPLAY_POINT_LABELS_SUMMARY_PATH = _env_path(
    "PADEL_GAMEPLAY_POINT_LABELS_SUMMARY_PATH",
    "gameplay_point_labels_summary.json",
)
GAMEPLAY_POINT_PROPOSALS_PATH = _env_path(
    "PADEL_GAMEPLAY_POINT_PROPOSALS_PATH",
    "gameplay_point_proposals.csv",
)
GAMEPLAY_POINT_PROPOSALS_SUMMARY_PATH = _env_path(
    "PADEL_GAMEPLAY_POINT_PROPOSALS_SUMMARY_PATH",
    "gameplay_point_proposals_summary.json",
)
GAMEPLAY_PROPOSAL_SCORE_EVENTS_PATH = _env_path(
    "PADEL_GAMEPLAY_PROPOSAL_SCORE_EVENTS_PATH",
    "gameplay_proposal_score_events.csv",
)
GAMEPLAY_PROPOSAL_SCORE_SUMMARY_PATH = _env_path(
    "PADEL_GAMEPLAY_PROPOSAL_SCORE_SUMMARY_PATH",
    "gameplay_proposal_score_summary.json",
)
GAMEPLAY_PROPOSAL_VALIDATION_REPORT_PATH = _env_path(
    "PADEL_GAMEPLAY_PROPOSAL_VALIDATION_REPORT_PATH",
    "gameplay_proposal_validation_report.json",
)
GAMEPLAY_PROPOSAL_VALIDATION_MISMATCHES_PATH = _env_path(
    "PADEL_GAMEPLAY_PROPOSAL_VALIDATION_MISMATCHES_PATH",
    "gameplay_proposal_validation_mismatches.csv",
)
GAMEPLAY_SCORE_SEQUENCE_REPORT_PATH = _env_path(
    "PADEL_GAMEPLAY_SCORE_SEQUENCE_REPORT_PATH",
    "gameplay_score_sequence_report.json",
)
GAMEPLAY_SCORE_SEQUENCE_MISMATCHES_PATH = _env_path(
    "PADEL_GAMEPLAY_SCORE_SEQUENCE_MISMATCHES_PATH",
    "gameplay_score_sequence_mismatches.csv",
)
GAMEPLAY_SCORE_CHECKPOINT_REPORT_PATH = _env_path(
    "PADEL_GAMEPLAY_SCORE_CHECKPOINT_REPORT_PATH",
    "gameplay_score_checkpoint_report.json",
)
GAMEPLAY_SCORE_CHECKPOINT_MISMATCHES_PATH = _env_path(
    "PADEL_GAMEPLAY_SCORE_CHECKPOINT_MISMATCHES_PATH",
    "gameplay_score_checkpoint_mismatches.csv",
)
GAMEPLAY_POINT_PROPOSAL_LABEL_REPORT_PATH = _env_path(
    "PADEL_GAMEPLAY_POINT_PROPOSAL_LABEL_REPORT_PATH",
    "gameplay_point_proposal_label_report.json",
)
GAMEPLAY_POINT_PROPOSAL_LABEL_MISMATCHES_PATH = _env_path(
    "PADEL_GAMEPLAY_POINT_PROPOSAL_LABEL_MISMATCHES_PATH",
    "gameplay_point_proposal_label_mismatches.csv",
)
GAMEPLAY_WINNER_ATTRIBUTION_REPORT_PATH = _env_path(
    "PADEL_GAMEPLAY_WINNER_ATTRIBUTION_REPORT_PATH",
    "gameplay_winner_attribution_report.json",
)
GAMEPLAY_WINNER_ATTRIBUTION_MISMATCHES_PATH = _env_path(
    "PADEL_GAMEPLAY_WINNER_ATTRIBUTION_MISMATCHES_PATH",
    "gameplay_winner_attribution_mismatches.csv",
)
GAMEPLAY_LABEL_REPLAY_EVENTS_PATH = _env_path(
    "PADEL_GAMEPLAY_LABEL_REPLAY_EVENTS_PATH",
    "gameplay_label_replay_score_events.csv",
)
GAMEPLAY_LABEL_REPLAY_SUMMARY_PATH = _env_path(
    "PADEL_GAMEPLAY_LABEL_REPLAY_SUMMARY_PATH",
    "gameplay_label_replay_score_summary.json",
)
GAMEPLAY_LABEL_REPLAY_VALIDATION_REPORT_PATH = _env_path(
    "PADEL_GAMEPLAY_LABEL_REPLAY_VALIDATION_REPORT_PATH",
    "gameplay_label_replay_validation_report.json",
)
GAMEPLAY_LABEL_REPLAY_VALIDATION_MISMATCHES_PATH = _env_path(
    "PADEL_GAMEPLAY_LABEL_REPLAY_VALIDATION_MISMATCHES_PATH",
    "gameplay_label_replay_validation_mismatches.csv",
)

# Maximum number of frames to be analysed
MAX_FRAMES = _env_int("PADEL_MAX_FRAMES", None)

# Fixed court keypoints
FIXED_COURT_KEYPOINTS_LOAD_PATH = _env_path(
    "PADEL_FIXED_COURT_KEYPOINTS_LOAD_PATH",
    "./cache/fixed_keypoints_detection.json",
)
FIXED_COURT_KEYPOINTS_SAVE_PATH = _env_path(
    "PADEL_FIXED_COURT_KEYPOINTS_SAVE_PATH",
    None,
)

# Players tracker
PLAYERS_TRACKER_MODEL = _env_path(
    "PADEL_PLAYERS_TRACKER_MODEL",
    "./weights/players_detection/yolov8m.pt",
)
PLAYERS_TRACKER_BATCH_SIZE = _env_int("PADEL_PLAYERS_TRACKER_BATCH_SIZE", 8)
PLAYERS_TRACKER_ANNOTATOR = os.environ.get(
    "PADEL_PLAYERS_TRACKER_ANNOTATOR",
    "rectangle_bounding_box",
)
PLAYERS_TRACKER_LOAD_PATH = _env_path(
    "PADEL_PLAYERS_TRACKER_LOAD_PATH",
    "./cache/players_detections.json",
)
PLAYERS_TRACKER_SAVE_PATH = _env_path(
    "PADEL_PLAYERS_TRACKER_SAVE_PATH",
    "./cache/players_detections.json",
)

# Players keypoints tracker
PLAYERS_KEYPOINTS_TRACKER_MODEL = _env_path(
    "PADEL_PLAYERS_KEYPOINTS_TRACKER_MODEL",
    "./weights/players_keypoints_detection/best.pt",
)
PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE = _env_int(
    "PADEL_PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE",
    1280,
)
PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE = _env_int(
    "PADEL_PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE",
    8,
)
PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH = _env_path(
    "PADEL_PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH",
    "./cache/players_keypoints_detections.json",
)
PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH = _env_path(
    "PADEL_PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH",
    "./cache/players_keypoints_detections.json",
)

# Ball tracker
BALL_TRACKER_MODEL = _env_path(
    "PADEL_BALL_TRACKER_MODEL",
    "./weights/ball_detection/TrackNet_best.pt",
)
BALL_TRACKER_INPAINT_MODEL = _env_path(
    "PADEL_BALL_TRACKER_INPAINT_MODEL",
    "./weights/ball_detection/InpaintNet_best.pt",
)
BALL_TRACKER_BATCH_SIZE = _env_int("PADEL_BALL_TRACKER_BATCH_SIZE", 8)
BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM = _env_int(
    "PADEL_BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM",
    400,
)
BALL_TRACKER_LOAD_PATH = _env_path(
    "PADEL_BALL_TRACKER_LOAD_PATH",
    "./cache/ball_detections.json",
)
BALL_TRACKER_SAVE_PATH = _env_path(
    "PADEL_BALL_TRACKER_SAVE_PATH",
    "./cache/ball_detections.json",
)

# Court keypoints tracker
KEYPOINTS_TRACKER_MODEL = _env_path(
    "PADEL_KEYPOINTS_TRACKER_MODEL",
    "./weights/court_keypoints_detection/best.pt",
)
KEYPOINTS_TRACKER_BATCH_SIZE = _env_int("PADEL_KEYPOINTS_TRACKER_BATCH_SIZE", 8)
KEYPOINTS_TRACKER_MODEL_TYPE = os.environ.get(
    "PADEL_KEYPOINTS_TRACKER_MODEL_TYPE",
    "yolo",
)
KEYPOINTS_TRACKER_LOAD_PATH = _env_path("PADEL_KEYPOINTS_TRACKER_LOAD_PATH", None)
KEYPOINTS_TRACKER_SAVE_PATH = _env_path("PADEL_KEYPOINTS_TRACKER_SAVE_PATH", None)
