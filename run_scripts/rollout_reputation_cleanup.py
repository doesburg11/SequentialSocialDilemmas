# ruff: noqa: E402
"""Roll out CleanupReputationEnv and compute the McKee et al. (2023)
territoriality / turn-taking metrics (see `social_dilemmas/analysis/
reputation_metrics.py`) from the resulting trajectory.

Two policy sources:
- `--random-policy` (default): samples uniform-random actions each step. No
  RLlib/checkpoint dependency -- this is a smoke test that exercises the
  full env -> log -> metrics pipeline end to end, the same role
  `render_rollout.py --random-policy` plays in the sibling Leibo2017 repo.
- `--checkpoint PATH`: loads a trained RLlib `Algorithm` checkpoint (as
  produced by `train_reputation.py` / `run_reputation_cleanup_*.sh`) and
  computes each agent's action from its own policy's RLModule, mirroring
  Leibo2017's `render_rllib_rollout.py` new-API-stack action inference.
  Verified against a real trained checkpoint (2026-09-09): the first attempt
  failed outright (`Algorithm.from_checkpoint()` needs the training env
  registered by name first, which this script didn't do -- see
  `_checkpoint_action_fn`'s docstring for the fix), exactly the risk this
  caveat used to warn about before any checkpoint existed to test against.

A "turn" (for turn-taking) is logged when an agent *enters* the river
region (transition from outside to inside between consecutive steps),
matching the paper's Sec. 7 definition ("the sequence of group members
entering the river"). One continuous stay in the river is one turn however
long it lasts -- counting every cleaning step as a turn instead would read
any sustained cleaning bout as monopolization and bias the score toward 0.
Successful cleans are still counted separately (`num_contributions`).

Usage
-----
    python rollout_reputation_cleanup.py --condition identifiable --num-steps 1000
    python rollout_reputation_cleanup.py --condition anonymous --checkpoint /path/to/checkpoint
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from social_dilemmas.analysis.reputation_metrics import compute_territoriality, compute_turn_taking
from social_dilemmas.envs.cleanup_reputation import CleanupReputationEnv


def _build_env(args):
    return CleanupReputationEnv(
        num_agents=args.num_agents,
        return_agent_actions=False,
        reputation_reward=True,
        identifiable=(args.condition == "identifiable"),
        contribution_ema_lambda=args.contribution_ema_lambda,
        reputation_seed=args.seed,
    )


def _random_action_fn(env):
    action_dim = env.action_space.n

    def choose_actions(_obs):
        return {agent_id: int(np.random.randint(action_dim)) for agent_id in env.agents}

    return choose_actions


def _checkpoint_action_fn(checkpoint_path, args):
    """Load a trained checkpoint and compute greedy actions per agent from
    its own policy's RLModule.

    Regression fix: `Algorithm.from_checkpoint()` reconstructs the full
    algorithm, including its env-runners -- which need the training env
    registered under the same name (`cleanup_reputation_env`) the
    checkpoint was saved with, via `tune.register_env()`. An earlier
    version skipped this, so this path failed on its first real exercise
    against an actual checkpoint (`ray.rllib.utils.error.EnvError: The env
    string you provided ('cleanup_reputation_env') is: a) Not a supported
    or an installed environment...`) -- exactly the risk the module
    docstring's "not yet verified" caveat had flagged. `args` supplies the
    condition/num_agents/etc. needed to reconstruct an env matching what
    the checkpoint was actually trained on; using the wrong condition here
    wouldn't error, just silently evaluate the policy in an env slightly
    different from its own training env.
    """
    import torch
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.core.columns import Columns
    from ray.tune.registry import register_env

    from run_scripts.train_reputation import _RLlibEnvAdapter
    from social_dilemmas.envs.env_creator_reputation import get_env_creator_reputation

    base_env_creator = get_env_creator_reputation(
        env="cleanup_reputation",
        num_agents=args.num_agents,
        return_agent_actions=False,
        reputation_reward=True,
        identifiable=(args.condition == "identifiable"),
        contribution_ema_lambda=args.contribution_ema_lambda,
        reputation_seed=args.seed,
    )

    def env_creator(env_config):
        max_episode_steps = 1000
        if isinstance(env_config, dict):
            max_episode_steps = env_config.get("max_episode_steps", max_episode_steps)
        return _RLlibEnvAdapter(base_env_creator(env_config), max_episode_steps=max_episode_steps)

    register_env("cleanup_reputation_env", env_creator)

    algo = Algorithm.from_checkpoint(checkpoint_path)
    # Regression fix: Algorithm.get_module() takes a *single* module_id
    # (default 'default_policy') and returns *one* RLModule, not a dict of
    # all of them -- calling it with no args (as an earlier version did)
    # silently returns None here, since our modules are named "agent-0" etc,
    # not "default_policy". Cache per-agent lookups instead of refetching
    # every step.
    module_cache = {}

    def get_module(agent_id):
        if agent_id not in module_cache:
            module = algo.get_module(agent_id)
            if module is None:
                raise ValueError(f"No RLModule found for module_id={agent_id!r} in this checkpoint")
            module_cache[agent_id] = module
        return module_cache[agent_id]

    def choose_actions(obs):
        actions = {}
        for agent_id, agent_obs in obs.items():
            # Must match what training fed the module: train_reputation.py's
            # _RLlibEnvAdapter passes `curr_obs` as float32 with its original
            # (2*view+1, 2*view+1, 3) shape and raw 0-255 values (no
            # flattening, no /255) -- the conv_filters in the model config
            # consume it as an image.
            obs_arr = np.asarray(agent_obs["curr_obs"], dtype=np.float32)
            batch = {Columns.OBS: torch.as_tensor(obs_arr).unsqueeze(0)}
            module_out = get_module(agent_id).forward_inference(batch)
            if Columns.ACTIONS in module_out:
                action = module_out[Columns.ACTIONS][0]
            else:
                # PPO-style: sample a deterministic (argmax) action from the
                # action distribution inputs, matching Leibo2017's
                # render_rllib_rollout.py handling of the new API stack.
                dist_inputs = module_out[Columns.ACTION_DIST_INPUTS][0]
                action = int(torch.argmax(dist_inputs).item())
            actions[agent_id] = int(action)
        return actions

    return choose_actions


def rollout(args):
    env = _build_env(args)
    if args.seed is not None:
        env.seed(args.seed)
        np.random.seed(args.seed)

    action_fn = (
        _checkpoint_action_fn(args.checkpoint, args) if args.checkpoint else _random_action_fn(env)
    )

    river_region = {tuple(pos) for pos in env.waste_points}

    obs = env.reset()
    river_positions_by_step = []
    turn_sequence = []
    num_contributions = 0
    in_river_prev = {
        agent_id: tuple(agent.pos.tolist()) in river_region for agent_id, agent in env.agents.items()
    }

    for _ in range(args.num_steps):
        actions = action_fn(obs)
        obs, _rewards, dones, _infos = env.step(actions)

        step_river_positions = {}
        for agent_id, agent in env.agents.items():
            pos = tuple(agent.pos.tolist())
            in_river = pos in river_region
            if in_river:
                step_river_positions[agent_id] = pos
            if in_river and not in_river_prev[agent_id]:
                turn_sequence.append(agent_id)  # entered the river this step
            in_river_prev[agent_id] = in_river
            if getattr(agent, "contributed_this_step", 0):
                num_contributions += 1
        river_positions_by_step.append(step_river_positions)

        if dones.get("__all__", False):
            break

    territoriality = compute_territoriality(river_positions_by_step)
    turn_taking = compute_turn_taking(turn_sequence)

    result = {
        "condition": args.condition,
        "num_agents": args.num_agents,
        "num_steps": len(river_positions_by_step),
        "territoriality": territoriality,
        "turn_taking": turn_taking,
        "num_river_entries": len(turn_sequence),
        "num_contributions": num_contributions,
    }
    print(json.dumps(result, indent=2))

    if args.out is not None:
        log = {
            "result": result,
            "river_positions_by_step": [
                {agent_id: list(pos) for agent_id, pos in step.items()}
                for step in river_positions_by_step
            ],
            "turn_sequence": turn_sequence,
        }
        Path(args.out).write_text(json.dumps(log))
        print(f"Wrote raw rollout log to {args.out}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["identifiable", "anonymous"], default="identifiable")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--contribution-ema-lambda", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained RLlib Algorithm checkpoint. Omit for a random-policy smoke test.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="If given, write the raw per-step rollout log (positions + turn sequence) as JSON.",
    )
    rollout(parser.parse_args())
