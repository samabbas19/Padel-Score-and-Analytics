import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analytics.gameplay_proposals import (
    GameplayProposalConfig,
    build_label_replay_score_timeline,
    build_proposal_score_timeline,
    build_scoreboard_transition_expansion_report,
    evaluate_score_checkpoints_against_scoreboard,
    evaluate_score_sequence_against_scoreboard,
    evaluate_proposals_against_labels,
    evaluate_winners_against_labels,
    infer_minimum_winner_sequences,
    select_gap_point_proposals,
)


class GameplayProposalTests(unittest.TestCase):
    def test_gap_proposals_select_last_candidate_before_quiet_period(self):
        candidates = pd.DataFrame(
            [
                {"frame": 100, "time": 10.0, "event_type": "racket_hit", "confidence": 0.55},
                {"frame": 120, "time": 12.0, "event_type": "racket_hit", "confidence": 0.56},
                {"frame": 500, "time": 20.0, "event_type": "racket_hit", "confidence": 0.54},
                {"frame": 540, "time": 21.0, "event_type": "court_bounce", "confidence": 0.60},
                {"frame": 900, "time": 28.0, "event_type": "racket_hit", "confidence": 0.58},
            ]
        )

        proposals = select_gap_point_proposals(
            candidates,
            GameplayProposalConfig(
                min_time_seconds=0.0,
                min_silence_after_seconds=4.0,
                min_event_confidence=0.50,
            ),
        )

        self.assertEqual(proposals["frame"].tolist(), [120, 540])
        self.assertEqual(proposals["proposal_reason"].tolist(), [
            "quiet_period_after_event",
            "quiet_period_after_event",
        ])

    def test_gap_proposals_add_previous_meaningful_side_for_dropout(self):
        candidates = pd.DataFrame(
            [
                {
                    "frame": 100,
                    "time": 10.0,
                    "event_type": "racket_hit",
                    "confidence": 0.55,
                    "ball_side": "near",
                    "inside_court": True,
                    "ball_y": 5.0,
                    "nearest_player_distance_px": 20.0,
                },
                {
                    "frame": 108,
                    "time": 10.4,
                    "event_type": "court_bounce",
                    "confidence": 0.57,
                    "ball_side": "near",
                    "inside_court": True,
                    "ball_y": 4.0,
                    "nearest_player_distance_px": 60.0,
                },
                {
                    "frame": 120,
                    "time": 10.8,
                    "event_type": "racket_hit",
                    "confidence": 0.56,
                    "ball_side": "far",
                    "inside_court": False,
                    "ball_y": -16.8,
                    "nearest_player_distance_px": 8.0,
                },
                {
                    "frame": 500,
                    "time": 20.0,
                    "event_type": "racket_hit",
                    "confidence": 0.54,
                    "ball_side": "far",
                    "inside_court": True,
                    "ball_y": -3.0,
                    "nearest_player_distance_px": 0.0,
                },
            ]
        )

        proposals = select_gap_point_proposals(
            candidates,
            GameplayProposalConfig(
                min_time_seconds=0.0,
                min_silence_after_seconds=4.0,
                min_event_confidence=0.50,
            ),
        )

        self.assertEqual(proposals.iloc[0]["attribution_side"], "near")
        self.assertEqual(
            proposals.iloc[0]["attribution_reason"],
            "previous_meaningful_event_before_dropout",
        )

    def test_gap_proposals_reject_dead_ball_dropout_without_rally_context(self):
        candidates = pd.DataFrame(
            [
                {
                    "frame": 60,
                    "time": 1.0,
                    "event_type": "court_bounce",
                    "confidence": 0.57,
                    "ball_side": "far",
                    "inside_court": True,
                    "ball_y": -2.0,
                    "raw_motion_px": 180.0,
                    "visual_motion_score": 0.40,
                    "nearest_player_distance_px": 80.0,
                },
                {
                    "frame": 120,
                    "time": 2.0,
                    "event_type": "racket_hit",
                    "confidence": 0.62,
                    "ball_side": "far",
                    "inside_court": False,
                    "ball_y": -16.0,
                    "raw_motion_px": 0.0,
                    "visual_motion_score": 0.0,
                    "nearest_player_distance_px": 0.0,
                },
                {
                    "frame": 1200,
                    "time": 20.0,
                    "event_type": "racket_hit",
                    "confidence": 0.58,
                    "ball_side": "near",
                    "inside_court": True,
                    "ball_y": 1.0,
                    "raw_motion_px": 65.0,
                    "visual_motion_score": 0.2,
                    "nearest_player_distance_px": 25.0,
                },
                {
                    "frame": 1230,
                    "time": 20.5,
                    "event_type": "racket_hit",
                    "confidence": 0.59,
                    "ball_side": "near",
                    "inside_court": True,
                    "ball_y": 2.0,
                    "raw_motion_px": 70.0,
                    "visual_motion_score": 0.25,
                    "nearest_player_distance_px": 10.0,
                },
                {
                    "frame": 1260,
                    "time": 21.0,
                    "event_type": "racket_hit",
                    "confidence": 0.61,
                    "ball_side": "far",
                    "inside_court": False,
                    "ball_y": -16.0,
                    "raw_motion_px": 0.0,
                    "visual_motion_score": 0.0,
                    "nearest_player_distance_px": 0.0,
                },
                {
                    "frame": 1560,
                    "time": 26.0,
                    "event_type": "racket_hit",
                    "confidence": 0.55,
                    "ball_side": "far",
                    "inside_court": True,
                    "ball_y": -1.0,
                    "raw_motion_px": 45.0,
                    "visual_motion_score": 0.18,
                    "nearest_player_distance_px": 40.0,
                },
            ]
        )

        proposals = select_gap_point_proposals(
            candidates,
            GameplayProposalConfig(
                min_time_seconds=0.0,
                min_silence_after_seconds=4.0,
                min_event_confidence=0.50,
            ),
        )

        self.assertEqual(proposals["frame"].tolist(), [1260])
        self.assertEqual(proposals.iloc[0]["point_end_decision"], "accepted")
        self.assertEqual(
            proposals.iloc[0]["point_end_reason"],
            "dropout_after_rally_context",
        )
        self.assertEqual(proposals.iloc[0]["point_end_context_events"], 2)

    def test_label_replay_builds_legal_score_timeline(self):
        rows = [
            {
                "transition": 1,
                "scoreboard_frame": 2040,
                "scoreboard_timestamp": 34.0,
                "candidate_frame": 1722,
                "candidate_time": 28.704,
                "candidate_event_type": "racket_hit",
                "candidate_confidence": 0.61,
                "winner_team": "team_b",
            },
            {
                "transition": 2,
                "scoreboard_frame": 2475,
                "scoreboard_timestamp": 41.25,
                "candidate_frame": 2416,
                "candidate_time": 40.272,
                "candidate_event_type": "racket_hit",
                "candidate_confidence": 0.53,
                "winner_team": "team_a",
            },
            {
                "transition": 3,
                "scoreboard_frame": 2925,
                "scoreboard_timestamp": 48.75,
                "candidate_frame": 2923,
                "candidate_time": 48.716,
                "candidate_event_type": "racket_hit",
                "candidate_confidence": 0.63,
                "winner_team": "team_b",
            },
            {
                "transition": 4,
                "scoreboard_frame": 3360,
                "scoreboard_timestamp": 56.0,
                "candidate_frame": 3359,
                "candidate_time": 55.988,
                "candidate_event_type": "racket_hit",
                "candidate_confidence": 0.60,
                "winner_team": "team_a",
            },
            {
                "transition": 5,
                "scoreboard_frame": 4125,
                "scoreboard_timestamp": 68.75,
                "candidate_frame": 4110,
                "candidate_time": 68.496,
                "candidate_event_type": "court_bounce",
                "candidate_confidence": 0.77,
                "winner_team": "team_a",
            },
            {
                "transition": 6,
                "scoreboard_frame": 5280,
                "scoreboard_timestamp": 88.0,
                "candidate_frame": 5340,
                "candidate_time": 88.992,
                "candidate_event_type": "racket_hit",
                "candidate_confidence": 0.57,
                "winner_team": "team_a",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.csv"
            events_path = root / "events.csv"
            summary_path = root / "summary.json"
            pd.DataFrame(rows).to_csv(labels_path, index=False)

            summary = build_label_replay_score_timeline(
                labels_path=labels_path,
                events_path=events_path,
                summary_path=summary_path,
                fps=60.0,
            )

            events = pd.read_csv(events_path)
            self.assertEqual(summary["points_detected"], 6)
            self.assertEqual(events["score_after_points"].tolist(), [
                "0-15",
                "15-15",
                "15-30",
                "30-30",
                "40-30",
                "0-0",
            ])
            self.assertEqual(events.iloc[-1]["score_after_games"], "1-0")
            self.assertEqual(summary["rule_violation_count"], 0)

    def test_proposal_score_timeline_uses_side_heuristic(self):
        proposals = pd.DataFrame(
            [
                {
                    "proposal": 1,
                    "frame": 100,
                    "time": 10.0,
                    "event_type": "racket_hit",
                    "confidence": 0.7,
                    "proposal_confidence": 0.8,
                    "ball_side": "far",
                    "inside_court": False,
                },
                {
                    "proposal": 2,
                    "frame": 200,
                    "time": 20.0,
                    "event_type": "court_bounce",
                    "confidence": 0.7,
                    "proposal_confidence": 0.8,
                    "ball_side": "near",
                    "inside_court": False,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposals_path = root / "proposals.csv"
            events_path = root / "events.csv"
            summary_path = root / "summary.json"
            proposals.to_csv(proposals_path, index=False)

            summary = build_proposal_score_timeline(
                proposals_path=proposals_path,
                events_path=events_path,
                summary_path=summary_path,
                fps=10.0,
            )

            events = pd.read_csv(events_path)
            self.assertEqual(summary["points_scored"], 2)
            self.assertEqual(events["winner_team"].tolist(), ["team_b", "team_a"])
            self.assertEqual(events["score_after_points"].tolist(), ["0-15", "15-15"])

    def test_proposal_score_timeline_can_start_from_initial_score(self):
        proposals = pd.DataFrame(
            [
                {
                    "proposal": 1,
                    "frame": 100,
                    "time": 10.0,
                    "event_type": "racket_hit",
                    "confidence": 0.7,
                    "proposal_confidence": 0.8,
                    "attribution_side": "far",
                    "inside_court": False,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposals_path = root / "proposals.csv"
            events_path = root / "events.csv"
            summary_path = root / "summary.json"
            proposals.to_csv(proposals_path, index=False)

            build_proposal_score_timeline(
                proposals_path=proposals_path,
                events_path=events_path,
                summary_path=summary_path,
                fps=10.0,
                config=GameplayProposalConfig(
                    initial_points="15-15",
                    initial_games="1-0",
                ),
            )

            events = pd.read_csv(events_path)
            self.assertEqual(events.iloc[0]["score_before_points"], "15-15")
            self.assertEqual(events.iloc[0]["score_before_games"], "1-0")
            self.assertEqual(events.iloc[0]["score_after_points"], "15-30")
            self.assertEqual(events.iloc[0]["score_after_games"], "1-0")

    def test_evaluate_proposals_against_labels_reports_misses_and_extras(self):
        labels = pd.DataFrame(
            [
                {"transition": 1, "candidate_time": 10.0, "candidate_frame": 100},
                {"transition": 2, "candidate_time": 20.0, "candidate_frame": 200},
            ]
        )
        proposals = pd.DataFrame(
            [
                {"proposal": 1, "time": 10.5, "frame": 105},
                {"proposal": 2, "time": 30.0, "frame": 300},
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.csv"
            proposals_path = root / "proposals.csv"
            report_path = root / "report.json"
            mismatches_path = root / "mismatches.csv"
            labels.to_csv(labels_path, index=False)
            proposals.to_csv(proposals_path, index=False)

            report = evaluate_proposals_against_labels(
                proposals_path=proposals_path,
                labels_path=labels_path,
                report_path=report_path,
                mismatches_path=mismatches_path,
                tolerance_seconds=1.0,
            )

            mismatches = pd.read_csv(mismatches_path)
            self.assertEqual(report["matched_labels"], 1)
            self.assertEqual(report["missed_labels"], 1)
            self.assertEqual(report["extra_proposals"], 1)
            self.assertEqual(set(mismatches["issue"]), {"missed_label", "extra_proposal"})

    def test_evaluate_proposals_handles_missing_label_candidate(self):
        labels = pd.DataFrame(
            [
                {"transition": 1, "candidate_time": "", "candidate_frame": ""},
            ]
        )
        proposals = pd.DataFrame(
            [
                {"proposal": 1, "time": 10.0, "frame": 100},
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.csv"
            proposals_path = root / "proposals.csv"
            report_path = root / "report.json"
            mismatches_path = root / "mismatches.csv"
            labels.to_csv(labels_path, index=False)
            proposals.to_csv(proposals_path, index=False)

            report = evaluate_proposals_against_labels(
                proposals_path=proposals_path,
                labels_path=labels_path,
                report_path=report_path,
                mismatches_path=mismatches_path,
            )

            mismatches = pd.read_csv(mismatches_path)
            self.assertEqual(report["matched_labels"], 0)
            self.assertEqual(mismatches.iloc[0]["issue"], "missing_label_candidate")

    def test_evaluate_winners_against_labels_reports_wrong_winner(self):
        labels = pd.DataFrame(
            [
                {"transition": 1, "candidate_time": 10.0, "candidate_frame": 100, "winner_team": "team_a"},
                {"transition": 2, "candidate_time": 20.0, "candidate_frame": 200, "winner_team": "team_b"},
            ]
        )
        events = pd.DataFrame(
            [
                {"point": 1, "end_time": 10.5, "end_frame": 105, "winner_team": "team_a"},
                {"point": 2, "end_time": 20.5, "end_frame": 205, "winner_team": "team_a"},
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.csv"
            events_path = root / "events.csv"
            report_path = root / "report.json"
            mismatches_path = root / "mismatches.csv"
            labels.to_csv(labels_path, index=False)
            events.to_csv(events_path, index=False)

            report = evaluate_winners_against_labels(
                predicted_events_path=events_path,
                labels_path=labels_path,
                report_path=report_path,
                mismatches_path=mismatches_path,
                tolerance_seconds=1.0,
            )

            mismatches = pd.read_csv(mismatches_path)
            self.assertEqual(report["matched_events"], 2)
            self.assertEqual(report["winner_matches"], 1)
            self.assertEqual(report["winner_mismatches"], 1)
            self.assertEqual(mismatches.iloc[0]["issue"], "winner_mismatch")

    def test_evaluate_score_sequence_ignores_display_lag_but_reports_it(self):
        events = pd.DataFrame(
            [
                {
                    "point": 1,
                    "end_time": 8.0,
                    "end_frame": 80,
                    "score_after_points": "0-15",
                    "score_after_games": "0-0",
                },
                {
                    "point": 2,
                    "end_time": 18.0,
                    "end_frame": 180,
                    "score_after_points": "15-15",
                    "score_after_games": "0-0",
                },
            ]
        )
        scoreboard = pd.DataFrame(
            [
                {
                    "event": 1,
                    "frame": 100,
                    "timestamp": 10.0,
                    "team_1_score": "0",
                    "team_2_score": "15",
                    "team_1_games": "",
                    "team_2_games": "",
                },
                {
                    "event": 2,
                    "frame": 200,
                    "timestamp": 20.0,
                    "team_1_score": "15",
                    "team_2_score": "15",
                    "team_1_games": "",
                    "team_2_games": "",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events_path = root / "events.csv"
            scoreboard_path = root / "scoreboard.csv"
            report_path = root / "report.json"
            mismatches_path = root / "mismatches.csv"
            events.to_csv(events_path, index=False)
            scoreboard.to_csv(scoreboard_path, index=False)

            report = evaluate_score_sequence_against_scoreboard(
                predicted_events_path=events_path,
                scoreboard_events_path=scoreboard_path,
                report_path=report_path,
                mismatches_path=mismatches_path,
            )

            self.assertEqual(report["score_sequence_accuracy"], 1.0)
            self.assertEqual(report["score_mismatches"], 0)
            self.assertEqual(report["timing_delta_seconds"]["mean"], -2.0)

    def test_evaluate_score_checkpoints_allows_hidden_transitions_in_partial_clip(self):
        events = pd.DataFrame(
            [
                {
                    "point": 1,
                    "start_time": 45.0,
                    "start_frame": 2700,
                    "end_time": 60.0,
                    "end_frame": 3600,
                    "score_before_points": "15-15",
                    "score_before_games": "0-0",
                    "score_after_points": "15-30",
                    "score_after_games": "0-0",
                },
                {
                    "point": 2,
                    "start_time": 60.0,
                    "start_frame": 3600,
                    "end_time": 79.0,
                    "end_frame": 4740,
                    "score_before_points": "15-30",
                    "score_before_games": "0-0",
                    "score_after_points": "30-30",
                    "score_after_games": "0-0",
                },
                {
                    "point": 3,
                    "start_time": 79.0,
                    "start_frame": 4740,
                    "end_time": 103.0,
                    "end_frame": 6180,
                    "score_before_points": "30-30",
                    "score_before_games": "0-0",
                    "score_after_points": "40-30",
                    "score_after_games": "0-0",
                },
                {
                    "point": 4,
                    "start_time": 103.0,
                    "start_frame": 6180,
                    "end_time": 150.0,
                    "end_frame": 9000,
                    "score_before_points": "40-30",
                    "score_before_games": "0-0",
                    "score_after_points": "0-0",
                    "score_after_games": "1-0",
                },
            ]
        )
        scoreboard = pd.DataFrame(
            [
                {
                    "event": 1,
                    "frame": 2700,
                    "timestamp": 45.0,
                    "team_1_score": "15",
                    "team_2_score": "15",
                    "team_1_games": "",
                    "team_2_games": "",
                },
                {
                    "event": 2,
                    "frame": 3150,
                    "timestamp": 52.5,
                    "team_1_score": "15",
                    "team_2_score": "30",
                    "team_1_games": "",
                    "team_2_games": "",
                },
                {
                    "event": 3,
                    "frame": 4770,
                    "timestamp": 79.5,
                    "team_1_score": "40",
                    "team_2_score": "30",
                    "team_1_games": "",
                    "team_2_games": "",
                },
                {
                    "event": 4,
                    "frame": 8970,
                    "timestamp": 149.5,
                    "team_1_score": "15",
                    "team_2_score": "30",
                    "team_1_games": "1",
                    "team_2_games": "0",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events_path = root / "events.csv"
            scoreboard_path = root / "scoreboard.csv"
            report_path = root / "report.json"
            mismatches_path = root / "mismatches.csv"
            events.to_csv(events_path, index=False)
            scoreboard.to_csv(scoreboard_path, index=False)

            report = evaluate_score_checkpoints_against_scoreboard(
                predicted_events_path=events_path,
                scoreboard_events_path=scoreboard_path,
                report_path=report_path,
                mismatches_path=mismatches_path,
            )

            mismatches = pd.read_csv(mismatches_path)
            self.assertEqual(report["visible_checkpoints"], 4)
            self.assertEqual(report["matched_checkpoints"], 3)
            self.assertEqual(report["missed_checkpoints"], 1)
            self.assertEqual(report["checkpoint_accuracy"], 0.75)
            self.assertEqual(mismatches.iloc[0]["issue"], "missing_visible_checkpoint")
            self.assertEqual(mismatches.iloc[0]["expected_games"], "1-0")
            self.assertEqual(mismatches.iloc[0]["nearest_predicted_points"], "0-0")
            self.assertEqual(mismatches.iloc[0]["nearest_predicted_games"], "1-0")

    def test_infer_minimum_hidden_winners_for_game_boundary_checkpoint(self):
        sequences = infer_minimum_winner_sequences(
            start_points="40-30",
            start_games="0-0",
            end_points="15-30",
            end_games="1-0",
        )

        self.assertIn(["team_a", "team_a", "team_b", "team_b"], sequences)
        self.assertEqual(len(sequences), 3)

    def test_build_scoreboard_transition_expansion_report_counts_hidden_points(self):
        scoreboard = pd.DataFrame(
            [
                {
                    "event": 1,
                    "frame": 2700,
                    "timestamp": 45.0,
                    "team_1_score": "15",
                    "team_2_score": "15",
                    "team_1_games": "",
                    "team_2_games": "",
                },
                {
                    "event": 2,
                    "frame": 3150,
                    "timestamp": 52.5,
                    "team_1_score": "15",
                    "team_2_score": "30",
                    "team_1_games": "",
                    "team_2_games": "",
                },
                {
                    "event": 3,
                    "frame": 4770,
                    "timestamp": 79.5,
                    "team_1_score": "40",
                    "team_2_score": "30",
                    "team_1_games": "",
                    "team_2_games": "",
                },
                {
                    "event": 4,
                    "frame": 8970,
                    "timestamp": 149.5,
                    "team_1_score": "15",
                    "team_2_score": "30",
                    "team_1_games": "1",
                    "team_2_games": "0",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scoreboard_path = root / "scoreboard.csv"
            report_path = root / "report.json"
            details_path = root / "details.csv"
            scoreboard.to_csv(scoreboard_path, index=False)

            report = build_scoreboard_transition_expansion_report(
                scoreboard_events_path=scoreboard_path,
                report_path=report_path,
                details_path=details_path,
                initial_points="15-15",
                initial_games="0-0",
            )

            details = pd.read_csv(details_path)
            self.assertEqual(report["minimum_point_events"], 7)
            self.assertEqual(report["hidden_point_events"], 4)
            self.assertEqual(report["impossible_transitions"], 0)
            self.assertEqual(details.iloc[3]["example_winner_sequence"], "team_a team_a team_b team_b")


if __name__ == "__main__":
    unittest.main()
