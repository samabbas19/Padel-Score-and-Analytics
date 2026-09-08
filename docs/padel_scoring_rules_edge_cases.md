# Padel Scoring Rules, Edge Cases, and Automation Plan

Source baseline: FIP Rules of Padel, review of application 2026-01-01, plus the 2025 FIP Beyond Rulebook umpire guidance. This document converts the rules into product requirements for a commercial automatic score calculator that ignores scoreboards.

Important product principle: the score engine must be deterministic, but the event detector cannot pretend to be certain when the camera cannot see a legal fact. Every point-changing event must carry `rule_id`, `confidence`, `evidence_frames`, and `review_required`.

## Direct Answer: Ball Touches The Net

There are four different "net" cases:

| Situation | Result | Score effect | Automation requirement |
| --- | --- | --- | --- |
| Serve touches net/post, then lands in the correct receiver service box, and does not hit the metallic fence before the second bounce | Net/let serve | No point; repeat same serve. If first serve, repeat full first serve. If second serve, repeat second serve. | Needs serve phase, net-contact detector, service-box bounce detector, fence-contact detector |
| Serve touches net/post, then is not otherwise valid | Service fault | Receiver only gets point if this creates a double fault | Same as above |
| Rally ball touches net/post, then lands in opponent court | Correct return | Ball stays live. If opponent fails to return, hitter wins | Needs net-contact and bounce/contact sequence |
| Rally ball hits net and comes back or stays on hitter side | Bad return | Opponent wins point | Mostly feasible from trajectory if the hitter side is known |

Separate rule: if a player, racket, clothing, or carried object touches the net, posts, tension cable, or opponent court while the ball is live, that player's team loses the point. This is not the same thing as the ball touching the net.

## Scoring State Rules

| Rule | Result | Current support |
| --- | --- | --- |
| Standard game scoring | 0, 15, 30, 40, deuce, advantage, game | Implemented in `ScoreState` |
| Game win | Win at least 4 points with 2-point margin, unless no-ad format configured | Implemented |
| Golden point/no-adv | At deuce, next point wins game | Implemented by config |
| Star Point | After repeated deuce sequence, deciding Star Point wins game | Not implemented; needs format config |
| Set | First to 6 games with 2-game margin | Implemented |
| Tie-break at 6-6 | First to 7 with 2-point margin wins set | Implemented |
| Match | Best of 3 sets by default | Not fully implemented; easy state-machine extension |
| Mini-set, match tie-break, super tie-break | Alternative formats set by competition | Not implemented; needs competition format config |
| Side changes | After odd games; every 6 tie-break points | Not needed for score arithmetic, useful for team-side mapping |
| Wrong side / wrong server / wrong receiver discovered later | Previous points usually remain valid; correct when discovered | Human/referee or operator input required |

## Serve Rules And Edge Cases

| Event | Result | Who gets point | Can automate now? | Required upgrades |
| --- | --- | --- | --- | --- |
| First serve fault | Server gets second serve | Nobody yet | No | Serve-state detector |
| Second serve fault | Double fault | Receiver team | No | Serve fault classifier |
| Server foot touches service line or imaginary center extension before strike | Fault | Receiver only if second fault | No | Player foot pose + court line calibration |
| Server bounces ball in wrong area or ball crosses service/center line before strike | Fault | Receiver only if second fault | No | Ball bounce detector + serve box geometry |
| Serve struck above waist | Fault | Receiver only if second fault | No | Player pose + racket/ball contact height |
| Server misses ball in attempted serve | Fault | Receiver only if second fault | No | Serve attempt classifier |
| Serve lands outside correct receiver service box | Fault; lines count as good | Receiver only if second fault | Partly | 3D bounce detector and service box line tolerance |
| Serve lands in box then touches metallic fence before second bounce | Fault | Receiver only if second fault | No | Fence contact classifier |
| Serve lands in box then exits through gate on a court without authorized out-of-court play | Fault | Receiver only if second fault | No | Venue config + exit path detector |
| Serve hits server, partner, or their equipment | Fault | Receiver only if second fault | No | Ball-player/equipment contact detector |
| Serve hits receiver or receiver racket before bounce | Server wins point | Server team | No | Ball-player/racket contact detector |
| Receiver not ready and serve happens | Let | No point | Human review/input | Readiness is not reliable from video alone |
| Receiver attempts return, then claims not ready | Claim invalid under rules | Continue from result | Human review/input | Intent and attempt need referee/operator |
| Wrong server / wrong side / wrong receiver | Correct when discovered; prior points generally stand | Depends | Human input | Match administration state |

## Rally Ball-In-Play Rules

| Event | Result | Who gets point | Can automate now? | Required upgrades |
| --- | --- | --- | --- | --- |
| Ball bounces once in opponent court and is returned before second bounce | Legal | Nobody yet | Partly | Bounce + hit detector |
| Ball bounces twice before return | Point lost by defending side | Hitter team | No | High-confidence bounce detector |
| Ball touches net/post then lands in opponent court | Legal | Nobody yet | No | Net-contact detector + bounce detector |
| Ball touches net/post then returns to hitter side | Point lost by hitter | Opponent team | Partly | Hitter-side and ball-path inference |
| Ball touches net/post then directly hits opponent wall/fence before ground bounce | Point lost by hitter | Opponent team | No | Net, wall/fence, and ground contact sequence |
| Shot directly hits opponent wall/fence/foreign object before ground bounce | Point lost by hitter | Opponent team | No | 3D ball trajectory + contact classifier |
| Shot hits player's own court, fence, or ground object before crossing net | Point lost by hitter | Opponent team | No | Side-specific contact classifier |
| Ball hits a player/body/clothing/equipment other than legal racket strike | Point lost by contacted player's team | Opponent team | No | Ball-player/equipment contact detector |
| Ball is volleyed by opponent before bounce | Legal return/contact | Ball stays live or hitter wins if contact invalid | No | Racket/body distinction |
| Player hits ball before it crosses to their side | Point lost by that player | Opponent team | No | 3D net plane + racket contact |
| Legal reach-over after ball bounces in opponent court and spins/returns back over net | Legal if no net/opponent-court touch | Nobody yet | No | 3D ball/racket/net-plane tracking |
| Same player hits ball twice | Usually point lost; same-motion double contact can be legal if direction not substantially changed | Opponent team or play continues | Human review at low confidence | Racket contact timing + direction-change analysis |
| Both partners hit ball simultaneously or consecutively | Point lost | Opponent team | No | Multi-player racket contact detector |
| Player jumps over the net while point live | Point lost | Opponent team | No | Player pose and court-plane detector |
| Player/racket/clothing/cap touches net, post, cable, or opponent court | Point lost | Opponent team | No | Multi-camera or sensor; video review |
| Racket thrown at ball | Point lost | Opponent team | Human review | Racket tracking + intent/context |
| Racket dropped or safety cord breaks | Immediate point lost under 2026 rules | Opponent team | Human review | Equipment state often invisible |

## Out-Of-Court, Walls, Fence, And Stuck Ball

| Event | Result | Who gets point | Required config/model |
| --- | --- | --- | --- |
| Ball bounces in opponent court, then hits wall/fence | Legal; must be returned before second bounce | Nobody yet | Wall/fence contact and bounce count |
| Ball bounces in opponent court, then goes out through authorized area | Defending team may play it out-of-court if venue allows | Nobody yet | Venue `out_of_court_authorized`, player/ball tracking outside court |
| No out-of-court play authorized and ball exits after bounce over court limit or gate | Defending side loses | Hitter team | Venue config + exit path |
| Authorized out-of-court play, ball exits over end wall after good bounce | Defending side loses | Hitter team | Exit-side classifier |
| Authorized out-of-court play, ball exits over side wall or door | Point ends only at second bounce or external-object contact | Usually hitter team if not returned | Bounce/contact outside-court tracking |
| Ball bounces in opponent court then hits ceiling, lights, or unrelated object | Correct return by hitter; point normally to hitter if opponent cannot return | Hitter team | Ceiling/object contact detector |
| Ball goes through a hole in the metallic fence after good bounce | Hitter wins | Hitter team | Fence-hole/stuck detector or human review |
| Ball gets stuck in fence hole or top flat surface of wall after good bounce | Hitter wins | Hitter team | Stuck-ball detector or human review |
| Corner where wall joins ground | Good if ball bounces in the corner | Nobody yet | 3D corner/ground contact classifier |

## Let, Interference, And Admin Events

| Event | Result | Human in loop? |
| --- | --- | --- |
| Ball splits during play | Let/replay | Usually yes unless ball-break detector is added |
| Foreign object/person/ball invades court and affects play | Let/replay | Yes; impact on play is judgment |
| Unexpected external interruption | Let/replay | Yes |
| Player immediately requests let and umpire agrees | Let/replay | Yes |
| Player requests let, umpire rejects, and play had stopped | Requesting player loses point | Yes |
| Deliberate interference by player | Opponent wins point | Yes; intent cannot be reliably inferred |
| First involuntary interference by a pair | Let/replay | Yes |
| Second involuntary interference by same pair | Opponent wins point | Yes; requires incident history |
| Time violation | First warning; repeat may lose first serve or point; later penalties escalate | Yes |
| Conduct/code violation | Warning, point, game, or disqualification depending on penalty table | Yes |
| Medical timeout, suspension, lack of light, accident recovery | Preserve score and resume exact state | Yes |

## Commercial Detection Architecture

The production app should separate three layers:

1. `ScoreState`: deterministic scoreboard arithmetic.
2. `RuleEventEngine`: converts gameplay events into point, fault, let, replay, or penalty events.
3. `Perception`: models that produce observations with confidence.

Required event schema:

```json
{
  "event_id": "uuid",
  "rule_id": "serve.net_good",
  "frame_start": 1234,
  "frame_end": 1268,
  "side": "near",
  "team": "team_a",
  "outcome": "replay_same_serve",
  "confidence": 0.91,
  "evidence": {
    "ball_track": [1234, 1268],
    "bounce_frame": 1251,
    "net_contact_frame": 1240
  },
  "review_required": false
}
```

## Model And System Upgrades Required

| Component | Why needed | Priority |
| --- | --- | --- |
| Serve phase detector | Distinguish serve vs rally; track first/second serve; detect server/receiver | P0 |
| Bounce detector | Double bounce, service-box validity, rally endpoint, out-of-court legality | P0 |
| Hit/contact detector | Identify hitter side/team and legal racket contact | P0 |
| 3D court calibration | Homography floor projection is not enough for net height, walls, and over-net calls | P0 |
| Ball trajectory smoother with uncertainty | Commercial scoring needs stable tracks and confidence, not frame-to-frame jitter | P0 |
| Net-contact detector | Required for serve lets and rally net edge cases | P1 |
| Wall/fence/contact classifier | Distinguish ground, glass, mesh, net, gate, ceiling, lights | P1 |
| Player/racket pose tracker | Needed for foot faults, waist-height serves, hit-before-crossing, double hits | P1 |
| Net-touch/equipment detector | Needed for player/racket/clothing net faults | P2, likely multi-camera/sensor |
| Out-of-court tracking | Required only for venues allowing out-of-court play | P2 |
| Ref/operator event console | Required for readiness, interference, conduct, medical/admin, disputed low-confidence calls | P0 |
| Review UI with frame clips | Required for commercial trust and corrections | P0 |
| Labeled evaluation dataset | Required to measure commercial accuracy by rule type | P0 |

## What Can Be Automatic Vs Human-Reviewed

Fully automatic target after model upgrades:

- Standard scoring arithmetic.
- Point/game/set/tie-break state.
- Most normal rally winners.
- Double faults after service classifiers exist.
- Serve box in/out and line tolerance.
- Double-bounce calls when visible.
- Shot into net and same-side failure.
- Most out calls after a visible bounce.

Human-in-the-loop required:

- Receiver not ready.
- Deliberate vs involuntary interference.
- Let request validity.
- Code of conduct and time penalties.
- Medical/admin suspensions.
- Wrong server/receiver corrections unless operator entered lineups/order.
- Equipment/safety cord break if not visible.
- Net touch by clothing/racket/body when camera is occluded.
- Very close double-bounce or line calls below confidence threshold.
- Any event where perception confidence falls below commercial threshold.

Recommended thresholds:

- `confidence >= 0.92`: auto-apply.
- `0.70 <= confidence < 0.92`: provisional apply, mark for review.
- `confidence < 0.70`: hold score update or request operator confirmation.

## Implementation Roadmap For This Repo

P0:

- Keep `ScoreState` as the deterministic core.
- Add normalized `RuleEvent` objects and make `score_events.csv` rule-driven.
- Add a manual review CSV/JSON lane: `review_required`, `review_reason`, `evidence_clip_start`, `evidence_clip_end`.
- Add serve/rally segmentation from current ball motion and player setup.
- Add a simple bounce detector from ball trajectory velocity reversals as a baseline.
- Add tests for all score-state formats and event outcomes.

P1:

- Train or integrate bounce/hit/net-contact models.
- Add 3D court calibration from fixed camera geometry.
- Add rule-specific confidence scores.
- Add overlay badges for `FAULT`, `LET`, `REVIEW`, and `POINT`.

P2:

- Multi-camera support or optional near-net sensors for net touches and occluded contacts.
- Operator/reviewer UI for commercial use.
- Dataset labeling workflow and per-rule accuracy dashboard.

## Sources

- Official FIP Rules of Padel, review of application 2026-01-01: https://www.padelfip.com/wp-content/uploads/2025/12/FIP_Rules-of-Padel.pdf
- FIP Beyond Rulebook, umpire guidance: https://www.padelfip.com/wp-content/uploads/2025/12/FIP-Beyond-Rulebook-EN.pdf
- Powerleague padel rules overview supplied by user: https://www.powerleague.com/blog/padel-rules
