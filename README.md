[![Python 3.11.13](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://www.python.org/downloads/release/python-31111/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.58.0-blue)](https://docs.ray.io/en/latest/rllib/)

# Multi-Agent RL in Sequential Social Dilemmas — Leibo et al. (2017)

> **This is a Vinitsky-codebase port, not a from-scratch paper reproduction.** This repository updates Eugene Vinitsky et al.'s existing open-source [`sequential_social_dilemma_games`](https://github.com/eugenevinitsky/sequential_social_dilemma_games) implementation to Ray RLlib's new API stack (see [What this repository is](#what-this-repository-is)) — it does not reimplement Leibo et al. (2017) from the paper text. For a from-scratch, paper-faithful *direct* reproduction of that paper (its own environments built from the paper's own description, its own independent-DQN training loop, no dependency on Vinitsky's or anyone else's code), see the sibling repo **[Leibo2017](https://github.com/doesburg11/Leibo2017)** instead. The two repos' near-identical names are a common source of confusion — this note is here so the two are never mistaken for each other.

> **The `cleanup_reputation` experiment does not currently produce meaningful *cleaning* in either condition, so it cannot test McKee et al. (2023)'s reputation claim.** All 6 training runs (identifiable/anonymous × 3 `reputation_seed` values) end near-zero return and near-zero successful cleaning after 417 iterations; a control run of plain Cleanup (same punishment beam, no reputation shaping) shows the same collapse, so reputation shaping is not *necessary* for it — the punishment beam is a plausible but not yet confirmed explanation. See [Status: training collapses before reputation can matter](#status-training-collapses-before-reputation-can-matter) for the full investigation.

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
The `--checkpoint` path (RLModule-based greedy action inference) is verified against real trained checkpoints (2026-09-09, see below) — the first attempt failed outright (`Algorithm.from_checkpoint()` needs the training env registered by name first; `_checkpoint_action_fn`'s docstring in [`rollout_reputation_cleanup.py`](run_scripts/rollout_reputation_cleanup.py) has the fix), exactly the risk this caveat used to warn about before any checkpoint existed to test against.

### Status: training collapses before reputation can matter

This experiment was smoke-tested (random policy only) but had never actually been trained to completion before 2026-09-09. Doing that surfaced four real, previously-undetected bugs that would have made every documented pilot run (`run_reputation_cleanup_identifiable.sh` / `_anonymous.sh`) crash immediately or fail silently:

1. **`ray.init(memory=..., redis_max_memory=...)`** — both kwargs were removed from Ray's `ray.init()` signature in the installed version; passing either raised `RuntimeError: Unknown keyword argument(s)` before any training happened. Both pilot shell scripts pass `--memory`.
2. **`AlgorithmConfig.training(lr_schedule=...)`** — deprecated in the installed RLlib in favor of passing the schedule directly as `lr=[[step, value], ...]`; the old separate kwarg raised `ValueError: lr_schedule is deprecated and must be None!` Both pilot shell scripts pass `--lr_schedule_steps`/`--lr_schedule_weights`, so this also crashed on the very first config build.
3. **`config/ppo_config.py`'s dashboard override** — even after fixing (2), `apply_ppo_training_config()` unconditionally re-applied a hardcoded constant `lr` (and `entropy_coeff`, `grad_clip`, worker counts) *after* the caller's own settings, silently discarding any `--lr_schedule_steps`/`--lr_schedule_weights` schedule. Fixed by re-applying `lr` once more, guarded on a schedule actually being requested — but the 6-run comparison below (launched before this specific fix landed) ran with the dashboard's constant `lr=0.0001` on **both** conditions instead of the intended decaying schedule. Both conditions saw the identical override, which is symmetric, but that alone doesn't make the 6 runs a reproducibly *paired* comparison in the usual multi-seed sense — see the `reputation_seed` caveat below.
4. **`rollout_reputation_cleanup.py`'s checkpoint loading** — `Algorithm.from_checkpoint()` needs the training env registered under its saved name first (`register_env()`), which the script didn't do; and `Algorithm.get_module()` takes a single `module_id` and returns one `RLModule`, not a dict of all agents' modules, as an earlier version assumed. Both fixed; see `_checkpoint_action_fn` in [`rollout_reputation_cleanup.py`](run_scripts/rollout_reputation_cleanup.py).

With the crash-blocking bugs (1, 2, 4) fixed — bug 3 was still present, see below — a real 6-run comparison (identifiable/anonymous × `reputation_seed` 0/1/2, PPO, 5 agents, 10M/417-iteration each) became possible. Every run's `episode_return_mean` starts hugely negative at iteration 1 — and stays large-magnitude and noisy for a long, seed-dependent stretch of training (identifiable_seed0 and identifiable_seed2 settle under ±50 by iteration ~16-23; identifiable_seed1 and anonymous_seed0/1 don't settle until iteration ~313-374, i.e. nearly the whole run; anonymous_seed2 is intermediate, ~147) — before every run ends near zero:

| Run | Iter-1 return | Final return |
|---|---|---|
| identifiable_seed0 | -17,970 | ~0 |
| identifiable_seed1 | -64,598 | ~-1.6 |
| identifiable_seed2 | -19,223 | ~0 |
| anonymous_seed0 | -19,279 | ~0 |
| anonymous_seed1 | -5,995 | ~2.1 |
| anonymous_seed2 | -20,405 | ~0 |

Rolling out each of the 6 resulting checkpoints for 1000 steps and computing territoriality/turn-taking confirms the near-zero final return corresponds to a real absence of successful cleaning, not a metrics-pipeline artifact — a random policy on the same env produces double-digit river entries and dozens of contributions in 200 steps (`python rollout_reputation_cleanup.py --condition identifiable --num-agents 5 --num-steps 200 --seed 0`; three repeated invocations gave 13/30, 13/30, and 12/37 river-entries/contributions. `--seed` doesn't make this fully reproducible — confirmed empirically, cause not fully traced here. The exact counts vary, but the qualitative gap versus the near-zero trained-checkpoint numbers below is consistent across all three):

| Run | Steps with any agent in river (of 1000) | River-entry transitions | Contributions | Territoriality | Turn-taking |
|---|---|---|---|---|---|
| identifiable_seed0 | 0 | 0 | 1 | 0.0 | 0.0 |
| identifiable_seed1 | 0 | 0 | 1 | 0.0 | 0.0 |
| identifiable_seed2 | 0 | 0 | 0 | 0.0 | 0.0 |
| anonymous_seed0 | 0 | 0 | 0 | 0.0 | 0.0 |
| anonymous_seed1 | 996 | 4 | 3 | 0.89 | 0.81 |
| anonymous_seed2 | 998 | 1 | 0 | 1.0 | 1.0 |

Two distinct patterns, not one uniform collapse. `identifiable_seed{0,1,2}` and `anonymous_seed0` never once have any agent occupy the river across the full 1000-step rollout — genuine avoidance (the 2 single-contribution cases for `identifiable` still register 0 occupied steps; the beam's range is 5 tiles, per `_CLEANUP_ACTIONS` in `cleanup_reputation.py`, so an agent standing just outside the river region can fire `CLEAN` into a waste tile without ever being recorded as "in the river" by this rollout script's position-based check). `anonymous_seed1`/`anonymous_seed2`, by contrast, have an agent occupying the river on 996-998 of 1000 steps — essentially the whole episode — but almost never firing the beam productively (1-3 contributions total) and rarely changing which agent is present (1-4 entry transitions): agents parked at the river without cleaning, not agents avoiding it. With so little underlying activity, neither metric is a stable read for these two: `turn_taking` is computed directly from that 1-4-long transition sequence (near-1.0 here is closer to "only one turn happened" than "agents rotate through fairly"), and `territoriality` (computed from which agents visited which river cells, not from transitions — see [`reputation_metrics.py`](social_dilemmas/analysis/reputation_metrics.py)) is likewise near-degenerate when almost the entire occupancy record belongs to a single dominant agent. Either way, no run in either condition learned meaningful cleaning behavior, so **this training setup cannot test McKee et al.'s identifiable-vs-anonymous claim at all**.

**Diagnostic: is this reputation-specific, or a property of Cleanup itself?** A control run of plain `CleanupEnv` (`train.py --env cleanup`, same 5 agents/PPO/10M-step/entropy_coeff/env_runner setup, no reputation shaping at all) was trained to check this. One caveat on comparability: the control ran under the already-fixed `train.py` (bug 3 above, fixed), so it actually followed the intended decaying `lr` schedule — reaching `~6.4e-4` by the 10M-step stop (partway to the schedule's 20M-step endpoint of `1.2e-5`, since training stopped before then); the 6 reputation runs above ran before that fix landed on `train_reputation.py` and so trained under bug 3's constant `lr=0.0001` throughout — a real difference between the control and the 6-run comparison, not just a labeling detail. With that caveat, the control shows the same qualitative pattern: `episode_return_mean` starts at -42,758 and reaches 0 by iteration 3, staying there for the rest of the 417-iteration/10M-step run — if anything, a *faster* collapse than most of the 6 reputation runs above. Since this happens with reputation shaping entirely absent, the reputation mechanism is not *necessary* for the collapse — though the lr difference means this control isn't a clean apples-to-apples isolation of the punishment beam alone.

A further methodological caveat on the "3 seeds" framing itself: the saved RLlib config for all 6 runs has `seed: null`. `--reputation_seed` does reseed process-global `np.random` at env construction (`map_env_reputation.py`'s `__init__`, used immediately for sampling each agent's `alpha`/`beta`) — but RLlib's own network initialization and action sampling are governed by `config["seed"]`, which none of these 6 runs set, and are therefore not controlled by `reputation_seed` at all. Exactly how far `reputation_seed`'s effect on environment-side randomness (spawn positions, orientations) extends into training — given each of the 24 parallel env-runner workers gets its own process, its own construction call, and its own subsequent stream of `reset()`s across many episodes — isn't fully traced here. What's clear either way: "seed 0/1/2" guarantees only 3 different `alpha`/`beta` draws, not 3 independently-seeded, reproducible training runs in the usual sense — treat the per-seed rows above as informal repeats, not a controlled multi-seed comparison.

A plausible contributor is this repo's Cleanup punishment/sanctioning beam (`social_dilemmas/envs/agent.py`'s `CleanupAgent`: firing costs the firer `-1`, but *being hit* costs the target `-50`). That -50 penalty is ~50× an apple's `+1` and dwarfs the reputation term's ~0.1-per-step magnitude, so it could plausibly dominate early-training gradients — an untrained, near-random policy would fire and get hit often, which is consistent with the huge negative early-training returns observed in all 7 runs (6 reputation + 1 control). This punishment mechanic doesn't exist in the sibling [Hughes2018](https://github.com/doesburg11/Hughes2018) repo's simpler from-scratch Cleanup (which *did* learn to clean under its own investigation) — so if it is the cause, this would be a distinct failure mode from that repo's Cleanup non-replication, not the same one recurring.

**What this does and doesn't establish:** the control run shows reputation shaping is not *necessary* to reproduce the collapse. It does **not** prove the punishment beam is the cause — no fire/hit telemetry was collected during any of these 7 runs, and no punishment-disabled ablation was run, so the beam is a plausible, unconfirmed hypothesis rather than a demonstrated mechanism. It also doesn't establish that removing or down-weighting the beam would fix the collapse, or that a longer training budget wouldn't eventually escape it (the paper's own runs use ~100M steps × 120 agents against this repo's 10M-step × 5-agent pilot scale — untested here). Investigation was stopped at this point (2026-09-09) rather than pursued further; the honest summary is that `cleanup_reputation`'s basics — getting PPO to learn *any* cleaning behavior at all in this repo's Cleanup-with-punishment variant, independent of the reputation mechanism — do not currently work at pilot scale, and the specific cause remains a hypothesis, not a confirmed diagnosis.

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
