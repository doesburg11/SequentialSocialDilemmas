"""Env-creator registry for the reputation experiment.

Sibling of `env_creator.py`, scoped to just `CleanupReputationEnv` (the only
env this experiment wires the reputation mechanism into) so the original
`env_creator.py`/`cleanup.py`/`map_env.py` stay untouched.
"""

from social_dilemmas.envs.cleanup_reputation import CleanupReputationEnv


def get_env_creator_reputation(
    env,
    num_agents,
    return_agent_actions=True,
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0,
    reputation_reward=True,
    identifiable=True,
    reputation_alpha_range=(2.4, 3.0),
    reputation_beta_range=(0.16, 0.20),
    contribution_ema_lambda=0.97,
    contribution_scale=0.001,
    reputation_seed=None,
):
    if env == "cleanup_reputation":

        def env_creator(_):
            return CleanupReputationEnv(
                num_agents=num_agents,
                return_agent_actions=return_agent_actions,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                reputation_reward=reputation_reward,
                identifiable=identifiable,
                reputation_alpha_range=reputation_alpha_range,
                reputation_beta_range=reputation_beta_range,
                contribution_ema_lambda=contribution_ema_lambda,
                contribution_scale=contribution_scale,
                reputation_seed=reputation_seed,
            )

    else:
        raise ValueError(
            f"env must be 'cleanup_reputation' for this reputation-experiment registry, "
            f"not {env}. Use social_dilemmas.envs.env_creator.get_env_creator for the "
            f"original harvest/cleanup/gathering/switch envs."
        )

    return env_creator
