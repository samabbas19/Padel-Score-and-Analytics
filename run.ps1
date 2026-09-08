param(
    [string]$VideoPath
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$venvDir = if ($env:PADEL_VENV_DIR) { $env:PADEL_VENV_DIR } else { ".venv" }
$pythonBin = Join-Path $venvDir "Scripts\python.exe"
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $pythonBin)) {
    Write-Error "Virtual environment not found at $venvDir. Create it with: py -3.12 -m venv .venv"
}

if (Test-Path -LiteralPath $activateScript) {
    . $activateScript
}

$gpuCheck = @'
import sys
import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, got {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required, but torch.cuda.is_available() is False")

print(f"GPU OK: torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
'@
$gpuCheck | & $pythonBin -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$weights = @(
    "weights\players_detection\yolov8m.pt",
    "weights\players_keypoints_detection\best.pt",
    "weights\ball_detection\TrackNet_best.pt",
    "weights\ball_detection\InpaintNet_best.pt",
    "weights\court_keypoints_detection\best.pt"
)

$missingWeights = @()
foreach ($weight in $weights) {
    if (-not (Test-Path -LiteralPath $weight)) {
        $missingWeights += $weight
    }
}

if ($missingWeights.Count -gt 0) {
    foreach ($weight in $missingWeights) {
        Write-Host "Missing weight: $weight"
    }
    Write-Error "Download the repo weights before running inference."
}

$defaultVideo = "..\padel_match.mp4"
if (-not (Test-Path -LiteralPath $defaultVideo)) {
    $defaultVideo = ".\examples\videos\rally.mp4"
}

$videoPath = if ($VideoPath) {
    $VideoPath
} elseif ($args.Count -gt 0) {
    $args[0]
} elseif ($env:PADEL_INPUT_VIDEO_PATH) {
    $env:PADEL_INPUT_VIDEO_PATH
} else {
    $defaultVideo
}

$resolveVideo = @'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser()
if not path.exists():
    raise SystemExit(f"Video not found: {path}")
print(path.resolve().as_posix())
'@
$videoAbs = ($resolveVideo | & $pythonBin - $videoPath).Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$makeVideoId = @'
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
'@
$videoId = ($makeVideoId | & $pythonBin - $videoAbs).Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$cacheDir = Join-Path "cache" $videoId
$outputDir = Join-Path "outputs" $videoId
New-Item -ItemType Directory -Force -Path $cacheDir, $outputDir | Out-Null

function Set-LoadPath {
    param(
        [string] $Name,
        [string] $Path
    )

    Set-Item -Path "env:$Name" -Value $Path
}

$env:PADEL_STRICT_GPU = "1"
$env:PADEL_INPUT_VIDEO_PATH = $videoAbs
$env:PADEL_SCORE_SOURCE = if ($env:PADEL_SCORE_SOURCE) { $env:PADEL_SCORE_SOURCE } else { "proposal" }
$env:PADEL_OUTPUT_VIDEO_PATH = (Join-Path $outputDir "results.mp4")
$env:PADEL_COLLECT_DATA = if ($env:PADEL_COLLECT_DATA) { $env:PADEL_COLLECT_DATA } else { "1" }
$env:PADEL_COLLECT_DATA_PATH = (Join-Path $outputDir "data.csv")
$env:PADEL_AUTO_SCORE = if ($env:PADEL_AUTO_SCORE) { $env:PADEL_AUTO_SCORE } else { "1" }
$env:PADEL_SCORE_EVENTS_PATH = (Join-Path $outputDir "score_events.csv")
$env:PADEL_SCORE_SUMMARY_PATH = (Join-Path $outputDir "score_summary.json")
$env:PADEL_GAMEPLAY_SCORE_EVENTS_PATH = (Join-Path $outputDir "gameplay_score_events.csv")
$env:PADEL_GAMEPLAY_SCORE_SUMMARY_PATH = (Join-Path $outputDir "gameplay_score_summary.json")
$env:PADEL_RENDER_SCORE_OVERLAY = if ($env:PADEL_RENDER_SCORE_OVERLAY) { $env:PADEL_RENDER_SCORE_OVERLAY } else { "1" }
$env:PADEL_SCORE_OVERLAY_VIDEO_PATH = (Join-Path $outputDir "results_score_overlay.mp4")
$env:PADEL_DETECT_AUDIO_VISUAL_EVENTS = if ($env:PADEL_DETECT_AUDIO_VISUAL_EVENTS) { $env:PADEL_DETECT_AUDIO_VISUAL_EVENTS } else { "1" }
$env:PADEL_EVENT_CANDIDATES_PATH = (Join-Path $outputDir "event_candidates.csv")
$env:PADEL_EVENT_SUMMARY_PATH = (Join-Path $outputDir "event_summary.json")
$env:PADEL_EXTRACT_SCOREBOARD_GROUND_TRUTH = if ($env:PADEL_EXTRACT_SCOREBOARD_GROUND_TRUTH) { $env:PADEL_EXTRACT_SCOREBOARD_GROUND_TRUTH } else { "1" }
$env:PADEL_SCOREBOARD_STATES_PATH = (Join-Path $outputDir "scoreboard_states.csv")
$env:PADEL_SCOREBOARD_EVENTS_PATH = (Join-Path $outputDir "scoreboard_events.csv")
$env:PADEL_SCOREBOARD_SUMMARY_PATH = (Join-Path $outputDir "scoreboard_summary.json")
$env:PADEL_SCOREBOARD_TRANSITION_EXPANSION_REPORT_PATH = (Join-Path $outputDir "scoreboard_transition_expansion_report.json")
$env:PADEL_SCOREBOARD_TRANSITION_EXPANSION_DETAILS_PATH = (Join-Path $outputDir "scoreboard_transition_expansion_details.csv")
$scoreboardTemplateSeedsPath = Join-Path $cacheDir "scoreboard_template_seeds.json"
if ((-not $env:PADEL_SCOREBOARD_TEMPLATE_SEEDS_PATH) -and (Test-Path -LiteralPath $scoreboardTemplateSeedsPath)) {
    $env:PADEL_SCOREBOARD_TEMPLATE_SEEDS_PATH = $scoreboardTemplateSeedsPath
}
$env:PADEL_VALIDATE_SCORE_TIMELINE = if ($env:PADEL_VALIDATE_SCORE_TIMELINE) { $env:PADEL_VALIDATE_SCORE_TIMELINE } else { "1" }
$env:PADEL_SCORE_VALIDATION_REPORT_PATH = (Join-Path $outputDir "score_validation_report.json")
$env:PADEL_SCORE_VALIDATION_MISMATCHES_PATH = (Join-Path $outputDir "score_validation_mismatches.csv")
$env:PADEL_SCOREBOARD_BASELINE_EVENTS_PATH = (Join-Path $outputDir "scoreboard_baseline_events.csv")
$env:PADEL_SCOREBOARD_BASELINE_SUMMARY_PATH = (Join-Path $outputDir "scoreboard_baseline_summary.json")
$env:PADEL_SCOREBOARD_BASELINE_VALIDATION_REPORT_PATH = (Join-Path $outputDir "scoreboard_baseline_validation_report.json")
$env:PADEL_SCOREBOARD_BASELINE_VALIDATION_MISMATCHES_PATH = (Join-Path $outputDir "scoreboard_baseline_validation_mismatches.csv")
$env:PADEL_GAMEPLAY_SCORE_VALIDATION_REPORT_PATH = (Join-Path $outputDir "gameplay_score_validation_report.json")
$env:PADEL_GAMEPLAY_SCORE_VALIDATION_MISMATCHES_PATH = (Join-Path $outputDir "gameplay_score_validation_mismatches.csv")
$env:PADEL_GAMEPLAY_ALIGNMENT_PATH = (Join-Path $outputDir "gameplay_event_alignment.csv")
$env:PADEL_GAMEPLAY_ALIGNMENT_SUMMARY_PATH = (Join-Path $outputDir "gameplay_event_alignment_summary.json")
$env:PADEL_GAMEPLAY_POINT_LABELS_PATH = (Join-Path $outputDir "gameplay_point_labels.csv")
$env:PADEL_GAMEPLAY_POINT_LABELS_SUMMARY_PATH = (Join-Path $outputDir "gameplay_point_labels_summary.json")
$env:PADEL_GAMEPLAY_POINT_PROPOSALS_PATH = (Join-Path $outputDir "gameplay_point_proposals.csv")
$env:PADEL_GAMEPLAY_POINT_PROPOSALS_SUMMARY_PATH = (Join-Path $outputDir "gameplay_point_proposals_summary.json")
$env:PADEL_GAMEPLAY_PROPOSAL_SCORE_EVENTS_PATH = (Join-Path $outputDir "gameplay_proposal_score_events.csv")
$env:PADEL_GAMEPLAY_PROPOSAL_SCORE_SUMMARY_PATH = (Join-Path $outputDir "gameplay_proposal_score_summary.json")
$env:PADEL_GAMEPLAY_PROPOSAL_VALIDATION_REPORT_PATH = (Join-Path $outputDir "gameplay_proposal_validation_report.json")
$env:PADEL_GAMEPLAY_PROPOSAL_VALIDATION_MISMATCHES_PATH = (Join-Path $outputDir "gameplay_proposal_validation_mismatches.csv")
$env:PADEL_GAMEPLAY_SCORE_SEQUENCE_REPORT_PATH = (Join-Path $outputDir "gameplay_score_sequence_report.json")
$env:PADEL_GAMEPLAY_SCORE_SEQUENCE_MISMATCHES_PATH = (Join-Path $outputDir "gameplay_score_sequence_mismatches.csv")
$env:PADEL_GAMEPLAY_SCORE_CHECKPOINT_REPORT_PATH = (Join-Path $outputDir "gameplay_score_checkpoint_report.json")
$env:PADEL_GAMEPLAY_SCORE_CHECKPOINT_MISMATCHES_PATH = (Join-Path $outputDir "gameplay_score_checkpoint_mismatches.csv")
$env:PADEL_GAMEPLAY_POINT_PROPOSAL_LABEL_REPORT_PATH = (Join-Path $outputDir "gameplay_point_proposal_label_report.json")
$env:PADEL_GAMEPLAY_POINT_PROPOSAL_LABEL_MISMATCHES_PATH = (Join-Path $outputDir "gameplay_point_proposal_label_mismatches.csv")
$env:PADEL_GAMEPLAY_WINNER_ATTRIBUTION_REPORT_PATH = (Join-Path $outputDir "gameplay_winner_attribution_report.json")
$env:PADEL_GAMEPLAY_WINNER_ATTRIBUTION_MISMATCHES_PATH = (Join-Path $outputDir "gameplay_winner_attribution_mismatches.csv")
$env:PADEL_GAMEPLAY_LABEL_REPLAY_EVENTS_PATH = (Join-Path $outputDir "gameplay_label_replay_score_events.csv")
$env:PADEL_GAMEPLAY_LABEL_REPLAY_SUMMARY_PATH = (Join-Path $outputDir "gameplay_label_replay_score_summary.json")
$env:PADEL_GAMEPLAY_LABEL_REPLAY_VALIDATION_REPORT_PATH = (Join-Path $outputDir "gameplay_label_replay_validation_report.json")
$env:PADEL_GAMEPLAY_LABEL_REPLAY_VALIDATION_MISMATCHES_PATH = (Join-Path $outputDir "gameplay_label_replay_validation_mismatches.csv")

$keypointsFile = Join-Path $cacheDir "fixed_keypoints_detection.json"
$env:PADEL_FIXED_COURT_KEYPOINTS_SAVE_PATH = $keypointsFile
Set-LoadPath "PADEL_FIXED_COURT_KEYPOINTS_LOAD_PATH" $keypointsFile

$env:PADEL_PLAYERS_TRACKER_SAVE_PATH = Join-Path $cacheDir "players_detections.json"
Set-LoadPath "PADEL_PLAYERS_TRACKER_LOAD_PATH" $env:PADEL_PLAYERS_TRACKER_SAVE_PATH

$env:PADEL_PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH = Join-Path $cacheDir "players_keypoints_detections.json"
Set-LoadPath "PADEL_PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH" $env:PADEL_PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH

$env:PADEL_BALL_TRACKER_SAVE_PATH = Join-Path $cacheDir "ball_detections.json"
Set-LoadPath "PADEL_BALL_TRACKER_LOAD_PATH" $env:PADEL_BALL_TRACKER_SAVE_PATH

Write-Host "Video: $videoAbs"
Write-Host "Video cache: $cacheDir"
Write-Host "Score source: $env:PADEL_SCORE_SOURCE"
if ($env:PADEL_SCORE_INITIAL_POINTS) {
    Write-Host "Initial points: $env:PADEL_SCORE_INITIAL_POINTS"
}
if ($env:PADEL_SCORE_INITIAL_GAMES) {
    Write-Host "Initial games: $env:PADEL_SCORE_INITIAL_GAMES"
}
if ($env:PADEL_PROPOSAL_MIN_TIME_SECONDS) {
    Write-Host "Proposal min time seconds: $env:PADEL_PROPOSAL_MIN_TIME_SECONDS"
}
Write-Host "Output video: $env:PADEL_OUTPUT_VIDEO_PATH"
Write-Host "Score events: $env:PADEL_SCORE_EVENTS_PATH"
Write-Host "Score summary: $env:PADEL_SCORE_SUMMARY_PATH"
Write-Host "Score overlay video: $env:PADEL_SCORE_OVERLAY_VIDEO_PATH"
Write-Host "Event candidates: $env:PADEL_EVENT_CANDIDATES_PATH"
Write-Host "Scoreboard states: $env:PADEL_SCOREBOARD_STATES_PATH"
Write-Host "Scoreboard transition expansion: $env:PADEL_SCOREBOARD_TRANSITION_EXPANSION_REPORT_PATH"
if ($env:PADEL_SCOREBOARD_TEMPLATE_SEEDS_PATH) {
    Write-Host "Scoreboard template seeds: $env:PADEL_SCOREBOARD_TEMPLATE_SEEDS_PATH"
}
Write-Host "Score validation: $env:PADEL_SCORE_VALIDATION_REPORT_PATH"
Write-Host "Gameplay score validation: $env:PADEL_GAMEPLAY_SCORE_VALIDATION_REPORT_PATH"
Write-Host "Gameplay event alignment: $env:PADEL_GAMEPLAY_ALIGNMENT_PATH"
Write-Host "Gameplay point labels: $env:PADEL_GAMEPLAY_POINT_LABELS_PATH"
Write-Host "Gameplay point proposals: $env:PADEL_GAMEPLAY_POINT_PROPOSALS_PATH"
Write-Host "Gameplay proposal validation: $env:PADEL_GAMEPLAY_PROPOSAL_VALIDATION_REPORT_PATH"
Write-Host "Gameplay score sequence validation: $env:PADEL_GAMEPLAY_SCORE_SEQUENCE_REPORT_PATH"
Write-Host "Gameplay score checkpoint validation: $env:PADEL_GAMEPLAY_SCORE_CHECKPOINT_REPORT_PATH"
Write-Host "Winner attribution validation: $env:PADEL_GAMEPLAY_WINNER_ATTRIBUTION_REPORT_PATH"
Write-Host "Gameplay label replay validation: $env:PADEL_GAMEPLAY_LABEL_REPLAY_VALIDATION_REPORT_PATH"
Write-Host "Scoreboard baseline validation: $env:PADEL_SCOREBOARD_BASELINE_VALIDATION_REPORT_PATH"

if (Test-Path -LiteralPath $keypointsFile) {
    Write-Host "Using saved court keypoints: $keypointsFile"
} else {
    Write-Host "Court keypoint selection will open now."
    Write-Host "Click exactly 12 keypoints in order k1 through k12, then press any key in the OpenCV window."
}

if ($env:PADEL_DRY_RUN -eq "1") {
    Write-Host "Dry run complete. Set PADEL_DRY_RUN=0 or unset it to run inference."
    exit 0
}

& $pythonBin main.py
exit $LASTEXITCODE
