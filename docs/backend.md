# Padel Analytics Backend

The backend is a local FastAPI service that runs the existing Windows GPU
pipeline as durable jobs. It does not change scoring logic; it sets explicit
job configuration, launches `run.ps1`, persists status, and exposes generated
artifacts and validation reports.

## Start the Backend

```powershell
.\run_backend.ps1 -HostAddress 127.0.0.1 -Port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Submit a Zero-Start Video

```powershell
$body = @{
  video_path = "C:\datasets\padel\match.mp4"
  initial_points = "0-0"
  initial_games = "0-0"
  proposal_min_time_seconds = 0.0
  score_source = "proposal"
  render_overlay = $true
  validate_score_timeline = $true
  extract_scoreboard_ground_truth = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/jobs `
  -ContentType "application/json" `
  -Body $body
```

Job status:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/jobs/<job_id>
```

Artifacts and reports:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/jobs/<job_id>/artifacts
```

## Production Boundaries

- The backend uses the no-OCR scorer by default: `score_source = "proposal"`.
- Broadcast scoreboard extraction remains validation-only.
- Per-video scoreboard seed files are allowed for validation calibration under
  `cache/<video-id>/scoreboard_template_seeds.json`.
- Job manifests and logs are stored in `backend_jobs/`.
- Pipeline outputs are stored in `outputs/<video-id>/`.

For `padel_start.mov`, the video ID is `padel_start-a542be0a48c5`.
The calibrated validation target has 8 visible scoreboard checkpoints.

Current `padel_start.mov` scoring status after the dead-ball point-end filter:

- point proposals: `9`
- rule violations: `0`
- visible checkpoint accuracy: `0.500`
- visible checkpoints matched: `4 / 8`
- proposal precision: `0.555556`
- proposal recall: `0.625`
- winner accuracy on matched visible labels: `0.800`

The filter removes startup/pre-serve dead-ball proposals at `1.996s` and
`14.696s` without using scoreboard state as scorer input. Remaining scorer
failures are short point-boundary recall and winner attribution around hidden
scoreboard transitions.
