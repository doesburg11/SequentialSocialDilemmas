[![Python 3.11.13](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.58.0-blue)](https://docs.ray.io/en/latest/rllib/)

# Multi-Agent RL in Sequential Social Dilemmas — Leibo et al. (2017)

A maintained update of the open-source **Sequential Social Dilemma (SSD)** environments and training code, ported to Ray RLlib's new API stack, replicating the setup from:

> Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). *Multi-agent Reinforcement Learning in Sequential Social Dilemmas.* AAMAS 2017.

The paper's question: matrix-game formulations of the Prisoner's Dilemma reduce a social dilemma to a single simultaneous choice between "cooperate" and "defect." Real social dilemmas — over-fishing a commons, letting pollution accumulate instead of cleaning it up — are neither simultaneous nor single-shot: they unfold over many time steps in a shared space, and whether an individual is "cooperating" at all is something that has to be inferred from a whole trajectory of spatial behaviour, not read off a payoff table. Leibo et al. formalize this as a **Sequential Social Dilemma**: a partially-observable Markov game in which the *same* Prisoner's-Dilemma-defining payoff inequalities hold, but only in aggregate, over policies, not over individual actions.

## What this repository is

This started from Eugene Vinitsky and collaborators' open-source SSD implementation (`sequential_social_dilemma_games`), the codebase most of the field cites for reproducing Leibo et al.'s environments. That codebase is pinned to Ray RLlib 0.8.5, an API generation Ray has since replaced twice over. This repository is a working port to current RLlib (`2.58.0`, the "new API stack": `PPOConfig`/`IMPALAConfig`/`DQNConfig` with `RLModule`, `Learner`, and `EnvRunner`), plus an evaluation layer built specifically to check the paper's own theoretical claims against trained policies rather than only reporting reward curves. It is a port and an extension, not a from-scratch reimplementation — the four environments below are Vinitsky et al.'s design, credited in full in [References](#references).

## The environments

* **Cleanup** — a public-goods dilemma. Apples give reward, but they only regrow in proportion to how clean a shared river is, and cleaning the river takes an agent out of foraging range and yields no reward itself. An agent that never cleans free-rides on whoever does.

  <img src="images/cleanup.png" alt="Image of the cleanup game" width="170" height="246"/>

* **Harvest** — a tragedy-of-the-commons dilemma. Apples regrow faster where more apples already stand nearby, so harvesting too aggressively collapses the local regrowth rate for everyone, including the harvester.

  <img src="images/harvest.png" alt="Image of the Harvest game" width="483" height="187"/>

* **Gathering** — the paper's original two-player game (this repo supports more than two, per `--num_agents`). Agents collect apples for reward and can fire a beam that removes an opponent from the game for a fixed number of steps — a costly, non-productive action whose only function is to reduce a rival's take.

  <img src="images/schelling.png" alt="Schelling diagrams for Harvest and Cleanup" width="953" height="352"/>

  The Schelling diagrams above (from Hughes et al. 2018, cited below) plot the payoff an individual agent gets from a defecting/exploitative strategy (red) against a cooperative one (blue) in Cleanup and Harvest, as a function of how many *other* agents are already cooperating. In both games a lone defector nearly always does better than a lone cooperator — but the more agents defect, the worse the outcome for everyone. That shape is the sequential analogue of the Prisoner's Dilemma payoff structure this repo's evaluation layer checks for directly (see below).

* **Switch** — not one of the paper's SSDs. A single-agent (extendable) coordination/diagnostic map — hold a switch, watch a door open — kept from the upstream codebase as a fast, low-variance environment for exercising the training pipeline without the confound of multi-agent dynamics.

## What changed vs. the upstream codebase

* **RLlib 0.8.5 → 2.58.0.** `run_scripts/train.py` now builds `PPOConfig`, `IMPALAConfig`, or `DQNConfig` objects against `RLModule`/`Learner`/`EnvRunner`, rather than the legacy `Trainer`/`Policy` classes the original codebase targets.
* **Gym → Gymnasium, plus PettingZoo.** Environments expose `Gymnasium`-compatible single-agent interfaces, `RLlib MultiAgentEnv`, and `PettingZoo` AEC/parallel wrappers (`social_dilemmas/envs/pettingzoo_env.py`).
* **DQN support and a working Gathering environment.** DQN was not part of the original codebase's tested algorithm set; Gathering's reward and tagging logic needed real fixes to run cleanly under the new stack.
* **Python 3.8 → 3.11**, dependencies trimmed (Stable-Baselines3 extras and Docker tooling removed as unused).
* **Not ported:** the upstream `MOA` (model-of-other-agents, per Jaques et al. 2018) and `SCM` (social curiosity module) training variants still import from Ray's pre-2.x legacy API and were not migrated. `train.py` enforces this explicitly — passing anything but `--model baseline` raises `Unsupported model '...' for the Ray RLlib 2.40+ stack. Only 'baseline' is supported.` The code remains in `algorithms/` and `models/` for reference, unrun.

## Checking the paper's own claim: the Leibo-style evaluation

Leibo et al.'s formal definition of a sequential social dilemma rests on a small set of payoff inequalities — the same ones that define the classical Prisoner's Dilemma (mutual cooperation beats mutual defection, but each individual is tempted to defect regardless of the other's choice) — holding at the level of *policies* rather than single actions. Reward curves alone don't check this; they show *how much* reward a trained policy earns, not whether the underlying incentive structure it was trained under actually has PD-shaped dilemma properties.

`visualization/visualizer_rllib.py --leibo-eval` (backed by `LeiboMetricsCollector`) does that check directly, on top of a normal rollout:

1. Each agent's use of the tagging/firing action is tracked per episode. An agent whose fire-rate stays under a threshold (default `0.02`) is classified **cooperative (C)** for that episode; otherwise **defective (D)**.
2. For two-agent runs, episodes are bucketed by the resulting `(C, C)`, `(D, D)`, `(C, D)`, `(D, C)` joint classification, and mean returns in each bucket are used to estimate the four payoff values **R** (reward, mutual cooperation), **P** (punishment, mutual defection), **S** (sucker), and **T** (temptation).
3. Those estimates are checked against the paper's own defining inequalities: `R > P`, `R > S`, `2R > T + S`, and the greed/fear conditions `T > R` (greed: defecting against a cooperator pays more than cooperating) and `P > S` (fear: mutual defection beats being exploited). A policy pair for which these hold was trained under a genuine social dilemma, not merely a competitive game that happens to look like one.
4. For runs with more than two agents, the same fire-rate classification instead produces aggregate behavioural metrics — mean social welfare, per-agent fire rates, tagged-out fractions, and episode class-profile counts (`CC`/`DD`/`mixed`) — since a unique R/S/T/P decomposition isn't defined for N > 2.

A rolling two-agent outcome matrix (`--show-outcome-matrix`) can also be printed live during a rollout, showing per-cell episode counts and mean returns as they accumulate, rather than only a final summary.

## Running it

```bash
git clone git@github.com:doesburg11/Leibo2017.git
cd Leibo2017

conda env create --prefix ./.conda --file environment.yml
conda activate "$(pwd)/.conda"
# or: ./run_scripts/create_local_conda_env.sh
```

Train with the repo's current defaults (Gathering, 5 agents, DQN):
```bash
python3 run_scripts/train.py
```

Train PPO on Cleanup with 8 agents:
```bash
python3 run_scripts/train.py --env cleanup --algorithm PPO --num_agents 8
```

Every other option — batch sizes, checkpoint frequency, stopping conditions — is defined in [`config/default_args.py`](config/default_args.py); preconfigured shell scripts for each environment live in [`run_scripts/`](run_scripts/). RLlib startup can take several minutes with larger agent counts, independent of anything in this repo.

Run the Leibo-style evaluation against a checkpoint:
```bash
PYTHONPATH=. ./.conda/bin/python visualization/visualizer_rllib.py \
  <checkpoint_path> --run DQN --env gathering_env --episodes 100 --no-render \
  --leibo-eval --leibo-out output/leibo_eval.json
```

## Tests

`tests/` covers environment mechanics directly — reward accrual, tagging, apple regrowth — and is the most precise reference for rules that are easier to verify in code than to describe in prose. Run with `python -m pytest`.

## Extending with a new environment

Every environment subclasses `MapEnv` (`social_dilemmas/envs/map_env.py`) and overrides four hooks: `setup_agents` (required), plus `custom_reset`, `custom_action`, and `custom_map_update` as needed — respectively, resetting non-agent map state (e.g. respawning apples), handling non-movement actions (e.g. firing or cleaning), and applying map changes that aren't a direct consequence of an agent's action (e.g. apple regrowth).

## Results

Collective reward under un-tuned PPO, Cleanup and Harvest (5 agents, default hyperparameters — no reward shaping or tuning applied):

<img src="images/cleanup_collective_reward.svg" alt="Collective reward plot of cleanup" width="460.8" height="345.6"/>
<img src="images/harvest_collective_reward.svg" alt="Collective reward plot of harvest" width="460.8" height="345.6"/>

These are learning curves, not a `--leibo-eval` report — they show that collective reward improves under training, not that the resulting policies satisfy the paper's dilemma inequalities. Run the evaluation above against a saved checkpoint to check that directly.

## Relation to other repos on this account

This repository sits apart from the [HintonNowlan1987](https://github.com/doesburg11/HintonNowlan1987), [AckleyLittman1991](https://github.com/doesburg11/AckleyLittman1991), and [Prosser2022](https://github.com/doesburg11/Prosser2022) line of replications: those three study how *evolution and lifetime learning interact* (the Baldwin effect and its extensions). This one has no genome and no evolutionary layer at all — it's pure multi-agent reinforcement learning, and the question it answers is narrower and different: whether learning alone, with no evolutionary pressure shaping the agents doing the learning, produces cooperative policies under a genuine social dilemma. It pairs instead with the (one-shot and repeated) Prisoner's Dilemma replications on the learned-cooperation side of the same site, which ask the matrix-game version of the same question this repo asks in a spatial, temporally extended setting.

## References

- Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). AAMAS 2017, 464–473.
- Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Dueñez-Guzman, E., Castañeda, A. G., Dunning, I., Zhu, T., McKee, K., Koster, R., Tina Zhu, Roff, H., & Graepel, T. (2018). [Inequity aversion improves cooperation in intertemporal social dilemmas](https://arxiv.org/abs/1803.08884). NeurIPS 2018.
- Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P. A., Strouse, D. J., Leibo, J. Z., & de Freitas, N. (2019). [Social influence as intrinsic motivation for multi-agent deep reinforcement learning](https://arxiv.org/abs/1810.08647). ICML 2019.
- Vinitsky, E., Jaques, N., Leibo, J., Castañeda, A., Hughes, E., et al. [Sequential Social Dilemma Games](https://github.com/eugenevinitsky/sequential_social_dilemma_games) — the original open-source environments and training code this repository ports and extends.
- Ray Team. [RLlib documentation](https://docs.ray.io/en/latest/rllib/) and [the new API stack migration guide](https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html).
