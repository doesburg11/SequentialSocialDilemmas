[![Python 3.11.13](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.58.0-blue)](https://docs.ray.io/en/latest/rllib/)

# Multi-Agent RL in Sequential Social Dilemmas — Leibo et al. (2017)

> **This is a Vinitsky-codebase port, not a from-scratch paper reproduction.** This repository updates Eugene Vinitsky et al.'s existing open-source [`sequential_social_dilemma_games`](https://github.com/eugenevinitsky/sequential_social_dilemma_games) implementation to Ray RLlib's new API stack (see [What this repository is](#what-this-repository-is)) — it does not reimplement Leibo et al. (2017) from the paper text. For a from-scratch, paper-faithful *direct* reproduction of that paper (its own environments built from the paper's own description, its own independent-DQN training loop, no dependency on Vinitsky's or anyone else's code), see the sibling repo **[Leibo2017](https://github.com/doesburg11/Leibo2017)** instead. The two repos' near-identical names are a common source of confusion — this note is here so the two are never mistaken for each other.

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
git clone git@github.com:doesburg11/SequentialSocialDilemmas.git
cd SequentialSocialDilemmas

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

## The reputation experiment: `cleanup_reputation`

A scaled-down reproduction, layered on top of Cleanup, of:

> McKee, K. R., Hughes, E., Zhu, T. O., Chadwick, M. J., Koster, R., Castañeda, A. G., Beattie, C., Graepel, T., Botvinick, M., & Leibo, J. Z. (2023). [A multi-agent reinforcement learning model of reputation and cooperation in human groups](https://arxiv.org/abs/2103.04982). arXiv:2103.04982.

That paper asks whether an intrinsic motivation for *reputation* — an aversion to having a worse standing than the group average — can, on its own, produce the same group coordination strategies (higher cooperation, less territorial behaviour, more turn-taking) that DeepMind's human participants exhibited when they could see each other's contributions (an *identifiable* condition) versus when they couldn't (*anonymous*). It's a natural fit for this repo's Cleanup environment: same public-goods structure (a river that needs cleaning, an orchard whose regrowth depends on it), just with an added social-cognitive reward term and a manipulation of what a player can see about the group.

Note this is a different DeepMind paper from the one this repository otherwise replicates (Leibo et al. 2017) — it's part of the same lineage (Hughes et al. 2018's Cleanup/Harvest, this repo's base environments) but a separate later contribution, reused here because Cleanup was already the right environment for it.

### The mechanism

Each agent's overall reward is `r = r_e + r_i`: its ordinary Cleanup reward `r_e` (apples), plus a reputation term

```
r_i = -alpha * max(c_bar - c_self, 0) - beta * max(c_self - c_bar, 0)
```

where `c_self` is the agent's own exponentially-smoothed contribution level (`c ← 0.97 * c + q`, `q = 1` on any step its cleaning beam actually removes waste, else `0`) and `c_bar` is the group's mean `c_self` that step. Falling behind the group average costs `alpha`; exceeding it costs the smaller `beta` — so the term mostly punishes free-riding, with a lighter penalty for over-contributing relative to peers. `alpha ~ U(2.4, 3.0)` and `beta ~ U(0.16, 0.20)` are sampled once per agent at construction (population heterogeneity, not re-rolled every episode), matching the paper's own parameterization.

**Identifiable vs. anonymous**, the paper's core manipulation: in the identifiable condition `c_bar` is the full group's mean and the `r_i` term is added to reward as above; in the anonymous condition the term is withheld entirely (contributions are still tracked, just not turned into reward). This is a simplified, binary reading of the paper's actual anonymous condition, which instead restricts *visibility* of other agents' contributions to a limited radius rather than removing the reputation channel outright — documented in [`map_env_reputation.py`](social_dilemmas/envs/map_env_reputation.py)'s docstring as a deliberate simplification, not an oversight.

**A scaling correction worth knowing about**: the paper's Eq. S4 reads literally as `q ∈ {0, 1}` with no further normalization, but taken literally that saturates the EMA near 33 and produces per-step reputation penalties an order of magnitude larger than the +1/apple extrinsic reward — directly contradicting the paper's own Fig. S2 ("agents do not live off intrinsic reward"). Fig. S1 plots the empirical divergence `c_self − c_bar` over ±0.004, which is exactly the unscaled range (±4) divided by 1000 — implying the paper's actual signal is normalized by episode length. `contribution_scale` (default `0.001`, i.e. `1/T` for a 1000-step episode) makes that normalization explicit and tunable; pass `contribution_scale=1.0` (or `--contribution_scale 1.0` on the training CLI) for the literal-but-implausible reading instead.

### New files (nothing existing was touched)

The original `MapEnv`/`CleanupEnv`/`train.py`/`env_creator.py` are untouched; this experiment is entirely new files, mirroring the shape of the originals they sit next to:

* [`social_dilemmas/envs/map_env_reputation.py`](social_dilemmas/envs/map_env_reputation.py) — `MapEnvReputation`, a copy of `MapEnv` (`social_dilemmas/envs/map_env.py`) adding the reputation reward mechanism above. A subclass only needs to set `agent.contributed_this_step = 1` on a successful contribution; the base class resets the flag every step, maintains the EMA, and applies the reward.
* [`social_dilemmas/envs/cleanup_reputation.py`](social_dilemmas/envs/cleanup_reputation.py) — `CleanupReputationEnv`, a copy of `CleanupEnv` (`social_dilemmas/envs/cleanup.py`) subclassing `MapEnvReputation`; the only functional change from `cleanup.py` is flagging a cleaning beam that actually removes waste as a contribution.
* [`social_dilemmas/envs/env_creator_reputation.py`](social_dilemmas/envs/env_creator_reputation.py) — a sibling registry to `env_creator.py`, scoped to `cleanup_reputation`.
* [`social_dilemmas/analysis/reputation_metrics.py`](social_dilemmas/analysis/reputation_metrics.py) — post-hoc **territoriality** and **turn-taking** metrics (below), pure functions over a logged rollout.
* [`run_scripts/train_reputation.py`](run_scripts/train_reputation.py) — a copy of `train.py` wired to the new registry, plus `--condition {identifiable,anonymous}`, `--contribution_scale`, `--reputation_alpha_low/high`, `--reputation_beta_low/high`, `--reputation_seed`.
* [`run_scripts/run_reputation_cleanup_identifiable.sh`](run_scripts/run_reputation_cleanup_identifiable.sh) / [`_anonymous.sh`](run_scripts/run_reputation_cleanup_anonymous.sh) — pilot-scale (10M-step, vs. the paper's ~100M steps × 120 agents) run scripts for the two conditions.
* [`run_scripts/rollout_reputation_cleanup.py`](run_scripts/rollout_reputation_cleanup.py) — rolls out the env (random policy, or a trained checkpoint) and computes the metrics below from the resulting trajectory.
* [`tests/test_reputation_env.py`](tests/test_reputation_env.py), [`tests/test_reputation_metrics.py`](tests/test_reputation_metrics.py) — reward-formula, EMA, seeding, and metrics-correctness tests.

### The metrics

Implementing Eqs. S14-S17 (territoriality) and Table S2 (turn-taking) from the paper's supplementary information — not the full set (contribution *consistency*, Eqs. S18-S19, is a documented gap, not built here):

* **Territoriality** — a normalized beta-diversity score over which agents visit which river cells across an episode. Low = the same group composition tends to show up at any given river location (no territorial division of labor); high = river cells tend to be visited by disjoint subsets of the group (agents have staked out separate "territories"). The paper finds identifiability *lowers* territoriality.
* **Turn-taking** — a recency-weighted score over the sequence of agents entering the river (an outside→inside transition; a continuous stay is one turn regardless of length). Low = one agent dominates the river; high = agents rotate through with long gaps between any one agent's own turns. The paper finds identifiability *raises* turn-taking, and that turn-taking is positively associated with collective return.

### Running it

Smoke test (no training, seconds):
```bash
cd run_scripts
python rollout_reputation_cleanup.py --condition identifiable --num-steps 500 --seed 0
```

Train both conditions (run one at a time if you have a single GPU — each claims it via `--gpus_for_driver 1`):
```bash
bash run_reputation_cleanup_identifiable.sh
bash run_reputation_cleanup_anonymous.sh
```
These default to a 10M-step pilot scale; adjust `--stop_at_timesteps_total` and `--memory` in the scripts for your machine. Progress and checkpoints land in `ray_results/`, same as any other run in this repo.

Compute the territoriality/turn-taking metrics from a trained checkpoint:
```bash
python rollout_reputation_cleanup.py --condition identifiable \
  --checkpoint <path-to-checkpoint>
```
The `--checkpoint` path (RLModule-based greedy action inference) is best-effort — it hasn't been exercised against a real trained checkpoint yet; the `--random-policy` default path (used above) is fully verified.

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
- McKee, K. R., Hughes, E., Zhu, T. O., Chadwick, M. J., Koster, R., Castañeda, A. G., Beattie, C., Graepel, T., Botvinick, M., & Leibo, J. Z. (2023). [A multi-agent reinforcement learning model of reputation and cooperation in human groups](https://arxiv.org/abs/2103.04982). arXiv:2103.04982 — source of the `cleanup_reputation` experiment above.
- Vinitsky, E., Jaques, N., Leibo, J., Castañeda, A., Hughes, E., et al. [Sequential Social Dilemma Games](https://github.com/eugenevinitsky/sequential_social_dilemma_games) — the original open-source environments and training code this repository ports and extends.
- Ray Team. [RLlib documentation](https://docs.ray.io/en/latest/rllib/) and [the new API stack migration guide](https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html).
