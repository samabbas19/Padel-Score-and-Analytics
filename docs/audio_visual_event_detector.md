# Audio + Vision Event Detector

This repo now has a no-training event candidate layer for padel scoring:

```text
video audio transient
  + existing ball detections
  + existing player boxes
  + projected court position
  -> event_candidates.csv
```

It deliberately ignores the broadcast scoreboard. The scoreboard may only be
used later as a validation target.

## Current implementation

Implemented in `analytics/audio_visual_events.py` and enabled by default in
`run.ps1` / `run.sh`.

Outputs per analysed video:

- `event_candidates.csv`
- `event_summary.json`
- `gameplay_point_proposals.csv`
- `gameplay_point_proposal_label_report.json`
- `gameplay_winner_attribution_report.json`
- `gameplay_score_checkpoint_report.json`
- `scoreboard_transition_expansion_report.json`
- `gameplay_label_replay_score_events.csv`
- `gameplay_label_replay_validation_report.json`

Each candidate contains:

- `event_type`: `racket_hit`, `court_bounce`, `net_contact`,
  `wall_or_fence_contact`, or `unknown_impact`
- `confidence`
- `audio_peak_z`, `audio_centroid_hz`, `audio_high_ratio`
- `ball_img_x`, `ball_img_y`, `ball_x`, `ball_y`
- `nearest_player_id`, `nearest_player_distance_px`
- `raw_motion_px`, `raw_acceleration_px`, `raw_angle_change_deg`
- `review_required` and `review_reason`

The detector uses cheap operations only: ffmpeg audio decode, short-window RMS,
robust z-score peak picking, FFT features around the peak, existing tracker JSON,
and geometry heuristics. It is suitable for live scoring loops because the
current look-ahead is about 0.07 seconds at 60 FPS.

## Why this is necessary

Trajectory-only scoring merged almost the whole sample into one long rally
because the inpainted ball track stays active through broadcast cuts and replays.
Impact events provide the missing timeline: racket hit, bounce, net, wall/fence,
then rally pauses.

## Research notes

- TrackNet-style ball tracking is the right family for fast, tiny sports balls,
  but tracking alone does not solve bounce/hit semantics.
- TrackNetV4 specifically improves tracking by adding motion attention, which is
  relevant for tiny fast-object localization.
- TTNet is the closest scoring architecture pattern: ball detection, semantic
  context, and short-window event spotting for bounces/net hits.
- Audio impact research in racket sports supports using very short transient and
  spectral windows to distinguish surface/contact classes such as racket, floor,
  wall, and glass.

## Commercial accuracy limits

No single-camera, no-training system can honestly guarantee 100 percent
rule-accurate padel scoring. These cases can remain ambiguous:

- ball grazes or rolls on the net
- ball touches player body/clothing instead of racket
- double hit or carry
- serve contact height / foot fault
- ball first contacts wall/fence before the court
- out-of-court recovery legality
- obstruction or deliberate/involuntary interference
- ball hidden behind players, net tape, glass frame, or broadcast overlays
- broadcast replay/cutaway mixed into live play
- audio contamination from crowd, commentary, music, shoe squeaks, and edit cuts

For a commercial app, these need at least one of:

- trained event spotting model using labelled padel events
- second camera angle or fixed court camera
- calibrated court/net/wall segmentation
- close microphone or court audio feed
- referee/operator review queue for low-confidence points
- optional scoreboard/OCR validation lane that never feeds the scorer

## Next scoring step

Do not make the scorer match the scoreboard by leaking scoreboard state into
logic. Instead:

1. Use `event_candidates.csv` to segment rallies.
2. Promote only high-confidence event sequences to automatic score changes.
3. Hold ambiguous rules for review with the exact frame/time and reason.
4. Compare the resulting score timeline against the scoreboard as a validation
   report, not as scoring input.

## Current validation split

For `padel_match.mp4`, the latest no-scoreboard point-boundary proposal layer
matches the scoreboard-derived gameplay labels exactly:

- point proposal precision: `1.000`
- point proposal recall: `1.000`
- proposal label misses: `0`
- proposal label extras: `0`

The full gameplay score is still not correct because winner attribution is
now handled by an attribution context rule:

- winner attribution accuracy: `1.000`
- winner matches: `11 / 11`
- winner mismatches: `0 / 11`

The gameplay score sequence also matches the scoreboard sequence exactly:

- expected transitions: `11`
- predicted transitions: `11`
- score sequence accuracy: `1.000`
- score sequence mismatches: `0`
- visible checkpoint accuracy: `1.000`

The strict frame/event scoreboard validator is still lower because gameplay
events happen before the broadcast overlay updates:

- strict frame accuracy: `0.962745`
- strict event accuracy: `0.833333`
- mean gameplay-to-scoreboard display delta: `-0.813455s`
- earliest display delta: `-5.296s`

That means the next commercial blocker is not the score sequence for this clip;
it is making the attribution rules general across more videos and separating
live gameplay timing from broadcast display timing. The scoreboard can keep
generating validation labels, but production scoring needs one of:

- a true short-window hit/bounce/wall/fence event classifier
- a trained winner attribution model from the generated label dataset
- a human review path for low-confidence winner calls

## Second-video validation status

`padel_match_2.mp4` is a partial/mid-game clip. The first visible scoreboard
state is already `15-15`, and the broadcast hides intermediate score states.
The run now supports per-video scoreboard calibration through
`cache/<video-id>/scoreboard_template_seeds.json`; for this clip that reduced
OCR noise from `40` visible transitions with `33` illegal jumps down to `4`
visible states.

Current video-2 findings:

- selected score source defaults to `proposal`, not scoreboard
- proposal scorer writes selected `score_events.csv`
- visible scoreboard states: `15-15`, `15-30`, `40-30`, `15-30` with games `1-0`
- no-scoreboard proposal scorer emits `9` point events
- strict default checkpoint accuracy: `0.000`
- mid-game seeded checkpoint accuracy: `0.750`
- mid-game seeded visible checkpoints matched: `3 / 4`
- mid-game seeded visible-scoreboard expansion requires at least `7` legal
  point events, including `4` hidden point events between visible states
- the final visible transition `40-30 / 0-0 -> 15-30 / 1-0` has `3`
  equally short legal winner patterns, so the scoreboard alone cannot identify
  the exact hidden winner sequence without gameplay evidence
- matched visible proposal labels: `2 / 4`
- visible-label proposal precision: `0.222222`
- visible-label proposal recall: `0.5`
- winner attribution on matched labels: `2 / 2`

This is not a solved clip yet. It needs one or more of:

- initial score/start-time configuration for mid-match footage
- stronger point-end filtering to remove false proposals
- better detection of hidden game-boundary points
- a confirmed true hit/bounce/wall classifier around the game-boundary interval
  where the current no-training heuristic misses the final `15-30 / 1-0`
  visible checkpoint

## Zero-start video status

`padel_start.mov` starts from `0-0`. The no-scoreboard scorer is run with
`PADEL_SCORE_INITIAL_POINTS=0-0`, `PADEL_SCORE_INITIAL_GAMES=0-0`, and
`PADEL_PROPOSAL_MIN_TIME_SECONDS=0.0`.

The current point-end filter rejects dead-ball dropout candidates when they do
not have enough preceding rally context. On this clip, that removes startup
pre-serve score events at `1.996s` and `14.696s`.

Current metrics:

- point proposals: `9`
- rule violations: `0`
- visible checkpoint accuracy: `0.500`
- visible checkpoints matched: `4 / 8`
- proposal precision: `0.555556`
- proposal recall: `0.625`
- winner accuracy on matched visible labels: `0.800`

Remaining failures are short point-boundary recall and winner attribution around
hidden scoreboard transitions.
