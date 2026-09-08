"""Rule catalogue for automatic padel scoring.

The scorer should emit normalized rule events and let ScoreState handle the
score arithmetic. This catalogue is intentionally explicit about what can be
automated from video and what needs review.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


AutomationLevel = Literal[
    "implemented",
    "feasible_with_current_video",
    "requires_new_model",
    "requires_multicamera_or_sensor",
    "human_review_required",
]


@dataclass(frozen=True)
class ScoringRule:
    rule_id: str
    category: str
    event: str
    outcome: str
    award_to: str
    automation_level: AutomationLevel
    required_signals: tuple[str, ...]
    human_review_reason: str = ""

    def serialize(self) -> dict:
        return asdict(self)


SCORING_RULES: tuple[ScoringRule, ...] = (
    ScoringRule(
        rule_id="score.standard_game",
        category="score_state",
        event="Standard 15/30/40/deuce/advantage scoring",
        outcome="Increment point, game, set, match",
        award_to="point_winner",
        automation_level="implemented",
        required_signals=("point_winner", "scoring_format"),
    ),
    ScoringRule(
        rule_id="score.golden_point",
        category="score_state",
        event="No-advantage deciding point at deuce",
        outcome="Deciding point winner wins the game",
        award_to="point_winner",
        automation_level="implemented",
        required_signals=("point_winner", "scoring_format"),
    ),
    ScoringRule(
        rule_id="score.tie_break",
        category="score_state",
        event="Tie-break at 6-6 unless format says otherwise",
        outcome="First to 7 with 2-point margin wins set",
        award_to="point_winner",
        automation_level="implemented",
        required_signals=("point_winner", "game_score", "set_format"),
    ),
    ScoringRule(
        rule_id="serve.net_good",
        category="serve",
        event="Serve touches net or post, then lands in correct service box without fence before second bounce",
        outcome="Replay the same serve",
        award_to="none",
        automation_level="requires_new_model",
        required_signals=(
            "serve_phase",
            "net_contact",
            "service_box_bounce",
            "fence_contact_before_second_bounce",
        ),
    ),
    ScoringRule(
        rule_id="serve.net_bad",
        category="serve",
        event="Serve touches net or post, then is otherwise not valid",
        outcome="Service fault",
        award_to="receiver_on_second_fault",
        automation_level="requires_new_model",
        required_signals=("serve_phase", "net_contact", "service_box_bounce", "serve_count"),
    ),
    ScoringRule(
        rule_id="serve.receiver_hit_before_bounce",
        category="serve",
        event="Serve hits receiving player or their racket before first bounce",
        outcome="Server wins point",
        award_to="server_team",
        automation_level="requires_new_model",
        required_signals=("serve_phase", "ball_player_contact", "first_bounce_status"),
    ),
    ScoringRule(
        rule_id="serve.double_fault",
        category="serve",
        event="Server commits two consecutive service faults",
        outcome="Receiver wins point",
        award_to="receiver_team",
        automation_level="requires_new_model",
        required_signals=("serve_fault", "serve_count"),
    ),
    ScoringRule(
        rule_id="serve.foot_or_waist_fault",
        category="serve",
        event="Illegal serve motion, line foot fault, wrong bounce area, or above-waist strike",
        outcome="Service fault",
        award_to="receiver_on_second_fault",
        automation_level="requires_new_model",
        required_signals=("server_pose", "court_lines", "racket_ball_contact", "serve_count"),
    ),
    ScoringRule(
        rule_id="rally.ball_net_good",
        category="rally",
        event="Rally ball touches net or post, then lands in opponent court",
        outcome="Ball remains in play",
        award_to="none",
        automation_level="requires_new_model",
        required_signals=("rally_phase", "net_contact", "opponent_court_bounce"),
    ),
    ScoringRule(
        rule_id="rally.ball_net_same_side",
        category="rally",
        event="Shot hits net and returns to hitter side",
        outcome="Opponent wins point",
        award_to="opponent_team",
        automation_level="feasible_with_current_video",
        required_signals=("hitter_side", "ball_trajectory", "rally_end_side"),
    ),
    ScoringRule(
        rule_id="rally.net_then_wall_no_bounce",
        category="rally",
        event="Shot touches net or post, then directly hits opponent wall, fence, or off-court element",
        outcome="Opponent wins point",
        award_to="opponent_team",
        automation_level="requires_new_model",
        required_signals=("net_contact", "wall_or_fence_contact", "ground_bounce_before_contact"),
    ),
    ScoringRule(
        rule_id="rally.player_touches_net",
        category="rally",
        event="Player, racket, clothing, or carried object touches net, post, cable, or opponent court",
        outcome="Opponent wins point",
        award_to="opponent_team",
        automation_level="requires_multicamera_or_sensor",
        required_signals=("player_pose", "racket_pose", "net_geometry", "contact_detection"),
        human_review_reason="Single broadcast views often hide light net/equipment touches.",
    ),
    ScoringRule(
        rule_id="rally.double_bounce",
        category="rally",
        event="Ball bounces twice before the defending team returns it",
        outcome="Hitter wins point",
        award_to="hitter_team",
        automation_level="requires_new_model",
        required_signals=("bounce_detector", "hit_detector", "side_assignment"),
    ),
    ScoringRule(
        rule_id="rally.direct_opponent_wall",
        category="rally",
        event="Shot reaches opponent wall, fence, or off-court element before bouncing on opponent ground",
        outcome="Opponent wins point",
        award_to="opponent_team",
        automation_level="requires_new_model",
        required_signals=("3d_ball_trajectory", "ground_bounce", "wall_fence_contact"),
    ),
    ScoringRule(
        rule_id="rally.own_side_first",
        category="rally",
        event="Shot touches hitter side ground, fence, or off-court element before crossing net",
        outcome="Opponent wins point",
        award_to="opponent_team",
        automation_level="requires_new_model",
        required_signals=("hitter_side", "ground_bounce", "fence_contact", "net_crossing"),
    ),
    ScoringRule(
        rule_id="rally.body_or_equipment_touch",
        category="rally",
        event="Ball touches player body, clothing, equipment, or partner after a shot",
        outcome="Opposing team wins point",
        award_to="opponent_team",
        automation_level="requires_multicamera_or_sensor",
        required_signals=("ball_player_contact", "team_assignment", "racket_contact"),
        human_review_reason="Small deflections on clothing/racket/body are hard in occlusion.",
    ),
    ScoringRule(
        rule_id="rally.double_hit_or_two_players",
        category="rally",
        event="Illegal double hit or both team members hit the ball",
        outcome="Opponent wins point",
        award_to="opponent_team",
        automation_level="requires_new_model",
        required_signals=("racket_contact_events", "player_identity", "stroke_timing"),
        human_review_reason="Same-motion double contact can be legal, so event timing needs review at low confidence.",
    ),
    ScoringRule(
        rule_id="rally.thrown_or_dropped_racket",
        category="rally",
        event="Ball hit with thrown racket, racket dropped, or safety cord breaks",
        outcome="Opponent wins point",
        award_to="opponent_team",
        automation_level="human_review_required",
        required_signals=("racket_tracking", "safety_cord_state"),
        human_review_reason="Safety cord break and deliberate thrown-racket context are referee facts.",
    ),
    ScoringRule(
        rule_id="rally.out_after_good_bounce",
        category="rally",
        event="Ball bounces in opponent court and goes out, hits ceiling/lights, or cannot be returned legally",
        outcome="Hitter wins point unless authorized out-of-court play keeps it live",
        award_to="hitter_team",
        automation_level="requires_new_model",
        required_signals=("ground_bounce", "out_of_court_path", "venue_out_of_court_config"),
    ),
    ScoringRule(
        rule_id="rally.authorized_out_of_court",
        category="rally",
        event="Authorized player leaves court and legally returns ball before second bounce",
        outcome="Ball remains in play",
        award_to="none",
        automation_level="requires_multicamera_or_sensor",
        required_signals=("venue_out_of_court_config", "player_position_3d", "ball_hit", "bounce_count"),
    ),
    ScoringRule(
        rule_id="let.external_or_ball_break",
        category="let",
        event="Ball breaks, foreign object/person enters, or external interruption affects point",
        outcome="Replay point",
        award_to="none",
        automation_level="human_review_required",
        required_signals=("referee_input", "object_detection"),
        human_review_reason="Determining interference impact is a referee judgment.",
    ),
    ScoringRule(
        rule_id="let.player_interference",
        category="let",
        event="Player hindrance/interference",
        outcome="First involuntary case replayed; deliberate or repeated involuntary case loses point",
        award_to="depends",
        automation_level="human_review_required",
        required_signals=("referee_input", "incident_history"),
        human_review_reason="Intentional vs involuntary interference is not a reliable video-only decision.",
    ),
    ScoringRule(
        rule_id="conduct.time_or_code_penalty",
        category="conduct",
        event="Time violation, abuse, unsportsmanlike conduct, medical/admin penalty",
        outcome="Warning, first-serve loss, point loss, game loss, or disqualification depending on rule",
        award_to="depends",
        automation_level="human_review_required",
        required_signals=("referee_input", "match_admin_state"),
        human_review_reason="Conduct penalties require official/referee authority.",
    ),
)


def rule_catalogue() -> list[dict]:
    return [rule.serialize() for rule in SCORING_RULES]
