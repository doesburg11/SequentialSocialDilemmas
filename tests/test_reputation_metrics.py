"""Unit tests for social_dilemmas/analysis/reputation_metrics.py"""

import unittest

from social_dilemmas.analysis.reputation_metrics import compute_territoriality, compute_turn_taking


class TestTurnTaking(unittest.TestCase):
    def test_empty_sequence(self):
        self.assertEqual(compute_turn_taking([]), 0.0)

    def test_single_agent_monopoly_approaches_zero(self):
        # One agent takes every turn: every gap after the first is 0 -> recency 1.0.
        score = compute_turn_taking(["agent-0"] * 50)
        self.assertLess(score, 0.05)
        self.assertGreaterEqual(score, 0.0)

    def test_perfect_round_robin_is_one(self):
        # 5 agents rotating with no repeats: every gap is 4 (>= 4 -> recency 0),
        # including the first cycle's undefined-gap turns (also treated as 0).
        sequence = ["agent-0", "agent-1", "agent-2", "agent-3", "agent-4"] * 3
        self.assertEqual(compute_turn_taking(sequence), 1.0)

    def test_monopoly_scores_lower_than_round_robin(self):
        monopoly = compute_turn_taking(["agent-0"] * 10)
        round_robin = compute_turn_taking(
            ["agent-0", "agent-1", "agent-2", "agent-3", "agent-4"] * 2
        )
        self.assertLess(monopoly, round_robin)

    def test_partial_rotation_is_between_extremes(self):
        # Matches the qualitative shape of the paper's own "low turn taking"
        # example (mostly one agent, occasional other) without claiming to
        # reproduce its exact 0.24 score -- see module docstring.
        score = compute_turn_taking(["agent-1", "agent-3", "agent-1", "agent-3"] + ["agent-1"] * 6)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class TestTerritoriality(unittest.TestCase):
    def test_no_visits_is_zero(self):
        self.assertEqual(compute_territoriality([]), 0.0)

    def test_fully_disjoint_territories_is_one(self):
        # Each of 2 agents visits only its own exclusive cell.
        river_positions_by_step = [
            {"agent-0": (0, 0)},
            {"agent-1": (0, 1)},
            {"agent-0": (0, 0)},
            {"agent-1": (0, 1)},
        ]
        self.assertAlmostEqual(compute_territoriality(river_positions_by_step), 1.0)

    def test_full_overlap_across_many_locations_scores_lower_than_disjoint(self):
        # With only 1 possible location, min(gamma_d, N_l) saturates at 1
        # regardless of overlap, so the metric can't discriminate -- it
        # needs several distinct locations to show its range. Here both
        # scenarios visit 4 locations total (N_l=4, gamma_d=2 in both), so
        # they're comparable and only composition-per-location differs.
        disjoint_territories = [
            {"agent-0": (0, 0)},
            {"agent-0": (0, 1)},
            {"agent-1": (0, 2)},
            {"agent-1": (0, 3)},
        ]
        fully_overlapping_territories = [
            {"agent-0": (0, 0), "agent-1": (0, 0)},
            {"agent-0": (0, 1), "agent-1": (0, 1)},
            {"agent-0": (0, 2), "agent-1": (0, 2)},
            {"agent-0": (0, 3), "agent-1": (0, 3)},
        ]
        disjoint_score = compute_territoriality(disjoint_territories)
        overlap_score = compute_territoriality(fully_overlapping_territories)
        self.assertAlmostEqual(disjoint_score, 1.0)
        self.assertAlmostEqual(overlap_score, 0.5)
        self.assertLess(overlap_score, disjoint_score)

    def test_partial_overlap_between_extremes(self):
        # agent-0 visits cell A and B; agent-1 visits only cell B: some
        # overlap (at B) but not fully disjoint, not fully shared.
        river_positions_by_step = [
            {"agent-0": (0, 0)},
            {"agent-0": (0, 1), "agent-1": (0, 1)},
        ]
        score = compute_territoriality(river_positions_by_step)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
