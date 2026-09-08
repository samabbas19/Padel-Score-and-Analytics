import unittest
import tempfile
from pathlib import Path

from analytics.scoreboard_ground_truth import (
    ScoreboardExtractorConfig,
    is_legal_score_transition,
    load_scoreboard_template_seeds,
)


class PadelScoreTransitionTests(unittest.TestCase):
    def assert_legal(self, before_points, after_points, before_games="0-0", after_games="0-0"):
        self.assertTrue(
            is_legal_score_transition(
                {"points": before_points, "games": before_games},
                {"points": after_points, "games": after_games},
            )
        )

    def assert_illegal(self, before_points, after_points, before_games="0-0", after_games="0-0"):
        self.assertFalse(
            is_legal_score_transition(
                {"points": before_points, "games": before_games},
                {"points": after_points, "games": after_games},
            )
        )

    def test_standard_point_progression(self):
        self.assert_legal("0-0", "15-0")
        self.assert_legal("0-0", "0-15")
        self.assert_legal("15-15", "30-15")
        self.assert_legal("30-30", "40-30")

    def test_game_point_catch_up_progression(self):
        self.assert_legal("40-0", "40-15")
        self.assert_legal("40-15", "40-30")
        self.assert_legal("0-40", "15-40")
        self.assert_legal("15-40", "30-40")
        self.assert_legal("40-30", "40-40")
        self.assert_legal("30-40", "40-40")

    def test_advantage_progression(self):
        self.assert_legal("40-40", "AD-40")
        self.assert_legal("40-40", "40-AD")
        self.assert_legal("AD-40", "40-40")
        self.assert_legal("40-AD", "40-40")

    def test_game_win_progression(self):
        self.assert_legal("40-0", "0-0", "0-0", "1-0")
        self.assert_legal("40-30", "0-0", "0-0", "1-0")
        self.assert_legal("0-40", "0-0", "0-0", "0-1")
        self.assert_legal("40-AD", "0-0", "0-0", "0-1")

    def test_illegal_jump_detection(self):
        self.assert_illegal("0-0", "30-0")
        self.assert_illegal("15-15", "40-15")
        self.assert_illegal("15-40", "40-40")
        self.assert_illegal("30-30", "0-0", "0-0", "1-0")

    def test_load_scoreboard_template_seed_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeds.json"
            path.write_text(
                """
                {
                  "point_template_seeds": [
                    {"frame": 100, "roi_name": "team_1_points", "label": "15"}
                  ],
                  "game_template_seeds": [
                    {"frame": 200, "roi_name": "team_1_games", "label": "1"}
                  ]
                }
                """,
                encoding="utf-8",
            )

            config = load_scoreboard_template_seeds(
                path,
                ScoreboardExtractorConfig(),
            )

            self.assertEqual(len(config.point_template_seeds), 1)
            self.assertEqual(config.point_template_seeds[0].frame, 100)
            self.assertEqual(config.point_template_seeds[0].label, "15")
            self.assertEqual(len(config.game_template_seeds), 1)
            self.assertEqual(config.game_template_seeds[0].label, "1")


if __name__ == "__main__":
    unittest.main()
