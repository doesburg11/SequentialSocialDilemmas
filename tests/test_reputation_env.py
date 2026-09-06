"""Unit tests for CleanupReputationEnv / MapEnvReputation's reputation
intrinsic reward mechanism (see social_dilemmas/envs/map_env_reputation.py).
"""

import unittest

from social_dilemmas.envs.agent import CLEANUP_ACTIONS
from social_dilemmas.envs.cleanup_reputation import CleanupReputationEnv

CLEANUP_ACTION_MAP = {y: x for x, y in CLEANUP_ACTIONS.items()}

# agent-? spawns are assigned to these 'P' points in map order; test cases
# reposition agents explicitly afterwards so which spawn point gets which
# agent id doesn't matter.
TEST_MAP = [
    "@@@@@@",
    "@    @",
    "@HP  @",
    "@    @",
    "@   P@",
    "@@@@@@",
]


class TestReputationReward(unittest.TestCase):
    def _make_env(self, identifiable=True, contribution_scale=1.0):
        # contribution_scale=1.0 (the literal Eq. S3 q=1) keeps the
        # hand-computed expected values below simple; the production default
        # (0.001) is covered by test_default_contribution_scale.
        env = CleanupReputationEnv(
            ascii_map=TEST_MAP,
            num_agents=2,
            reputation_reward=True,
            identifiable=identifiable,
            reputation_alpha_range=(2.5, 2.5),
            reputation_beta_range=(0.18, 0.18),
            contribution_scale=contribution_scale,
        )
        env.reset()
        # Deterministic layout regardless of random spawn assignment:
        # agent-0 next to the waste cell at [2, 1], facing it (LEFT);
        # agent-1 far away, uninvolved.
        env.agents["agent-0"].set_pos([2, 2])
        env.agents["agent-0"].update_agent_rot("LEFT")
        env.agents["agent-1"].set_pos([4, 4])
        env.agents["agent-1"].update_agent_rot("UP")
        # Disable stochastic waste/apple spawning: these tests are about the
        # reputation reward mechanism, not spawn dynamics, and without this
        # the single waste cell at [2, 1] can respawn on the very next
        # custom_map_update() (wasteSpawnProbability=0.5 in this 1-cell test
        # map), making a same-cell "waste is gone" assertion flaky.
        # compute_probabilities() recomputes both spawn probabilities from
        # waste_density on every custom_map_update() call, so a plain
        # attribute assignment alone would get clobbered on the next step --
        # it's overridden to a no-op instead.
        env.compute_probabilities = lambda: None
        env.current_waste_spawn_prob = 0.0
        env.current_apple_spawn_prob = 0.0
        return env

    def test_contribution_flag_only_set_on_successful_clean(self):
        env = self._make_env()
        # First CLEAN: waste is present at [2, 1] -> should succeed.
        env.step({"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["STAY"]})
        self.assertEqual(env.agents["agent-0"].contributed_this_step, 1)
        self.assertEqual(env.agents["agent-1"].contributed_this_step, 0)

        # Second CLEAN: the waste cell is now river ([2, 1] -> b"R"), so
        # there's nothing left to clean -> should NOT count as a contribution
        # even though the agent chose the CLEAN action again.
        env.step({"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["STAY"]})
        self.assertEqual(env.agents["agent-0"].contributed_this_step, 0)

    def test_contribution_ema_updates_with_smoothing(self):
        env = self._make_env()
        env.step({"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["STAY"]})
        # c^1 = 0.97 * 0 + 1 = 1.0
        self.assertAlmostEqual(env.contribution_ema["agent-0"], 1.0)
        self.assertAlmostEqual(env.contribution_ema["agent-1"], 0.0)

        env.step({"agent-0": CLEANUP_ACTION_MAP["STAY"], "agent-1": CLEANUP_ACTION_MAP["STAY"]})
        # No waste left to clean this step -> q=0 -> c^2 = 0.97 * 1.0 + 0 = 0.97
        self.assertAlmostEqual(env.contribution_ema["agent-0"], 0.97)

    def test_reputation_reward_identifiable_matches_formula(self):
        env = self._make_env(identifiable=True)
        _obs, rewards, _dones, _infos = env.step(
            {"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["STAY"]}
        )
        # c_self("agent-0") = 1.0, c_self("agent-1") = 0.0, mean = 0.5.
        # agent-0 is above the mean -> penalized by beta; agent-1 is below
        # the mean -> penalized by alpha. Neither agent collects an apple or
        # fires a penalty beam here, so extrinsic reward is exactly 0 and the
        # full reward is the reputation term.
        alpha, beta = 2.5, 0.18
        expected_reward_0 = -beta * max(1.0 - 0.5, 0.0)  # agent-0 above mean
        expected_reward_1 = -alpha * max(0.5 - 0.0, 0.0)  # agent-1 below mean
        self.assertAlmostEqual(rewards["agent-0"], expected_reward_0)
        self.assertAlmostEqual(rewards["agent-1"], expected_reward_1)

    def test_reputation_reward_anonymous_is_zero(self):
        env = self._make_env(identifiable=False)
        _obs, rewards, _dones, _infos = env.step(
            {"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["STAY"]}
        )
        # Anonymous: no reputation term added, so reward stays at the (here,
        # zero) extrinsic reward, even though contributions are unequal.
        self.assertAlmostEqual(rewards["agent-0"], 0.0)
        self.assertAlmostEqual(rewards["agent-1"], 0.0)
        # But contributions are still tracked for post-hoc metrics.
        self.assertAlmostEqual(env.contribution_ema["agent-0"], 1.0)
        self.assertAlmostEqual(env.contribution_ema["agent-1"], 0.0)

    def test_default_contribution_scale(self):
        env = CleanupReputationEnv(ascii_map=TEST_MAP, num_agents=2, reputation_reward=True)
        self.assertEqual(env.contribution_scale, 0.001)

        env = self._make_env(contribution_scale=0.001)
        _obs, rewards, _dones, _infos = env.step(
            {"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["STAY"]}
        )
        # q = 0.001 -> c_0 = 0.001, c_1 = 0, mean = 0.0005: the penalties
        # are 1000x smaller than with the literal q=1, i.e. the intrinsic
        # term stays small relative to the +1/apple extrinsic reward
        # (paper's Fig. S2).
        self.assertAlmostEqual(env.contribution_ema["agent-0"], 0.001)
        self.assertAlmostEqual(rewards["agent-0"], -0.18 * 0.0005)
        self.assertAlmostEqual(rewards["agent-1"], -2.5 * 0.0005)

    def test_reset_clears_ema_and_flags_but_keeps_alpha_beta(self):
        env = self._make_env()
        alpha_before = dict(env.reputation_alpha)
        beta_before = dict(env.reputation_beta)
        env.step({"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["STAY"]})
        self.assertGreater(env.contribution_ema["agent-0"], 0.0)

        env.reset()

        # alpha/beta are population heterogeneity, fixed for the agent's
        # lifetime -- reset() must not resample them.
        self.assertEqual(env.reputation_alpha, alpha_before)
        self.assertEqual(env.reputation_beta, beta_before)
        # But the episode-local contribution EMA and per-step flags do reset.
        self.assertEqual(env.contribution_ema, {"agent-0": 0.0, "agent-1": 0.0})
        for agent in env.agents.values():
            self.assertEqual(agent.contributed_this_step, 0)

    def test_reputation_seed_makes_alpha_beta_reproducible(self):
        make = lambda: CleanupReputationEnv(  # noqa: E731
            ascii_map=TEST_MAP,
            num_agents=2,
            reputation_reward=True,
            reputation_alpha_range=(2.4, 3.0),
            reputation_beta_range=(0.16, 0.20),
            reputation_seed=123,
        )
        env_a, env_b = make(), make()
        self.assertEqual(env_a.reputation_alpha, env_b.reputation_alpha)
        self.assertEqual(env_a.reputation_beta, env_b.reputation_beta)

    def test_reputation_reward_rejects_collective_and_inequity_combos(self):
        with self.assertRaises(ValueError):
            CleanupReputationEnv(
                ascii_map=TEST_MAP,
                num_agents=2,
                reputation_reward=True,
                use_collective_reward=True,
            )
        with self.assertRaises(ValueError):
            CleanupReputationEnv(
                ascii_map=TEST_MAP,
                num_agents=2,
                reputation_reward=True,
                inequity_averse_reward=True,
                alpha=1.0,
                beta=1.0,
            )

    def test_reputation_reward_disabled_by_default(self):
        env = CleanupReputationEnv(ascii_map=TEST_MAP, num_agents=2)
        env.reset()
        self.assertFalse(env.reputation_reward)
        # No contribution_ema/alpha/beta bookkeeping should exist at all.
        self.assertFalse(hasattr(env, "contribution_ema"))
        _obs, rewards, _dones, _infos = env.step(
            {"agent-0": CLEANUP_ACTION_MAP["STAY"], "agent-1": CLEANUP_ACTION_MAP["STAY"]}
        )
        self.assertAlmostEqual(rewards["agent-0"], 0.0)
        self.assertAlmostEqual(rewards["agent-1"], 0.0)


if __name__ == "__main__":
    unittest.main()
