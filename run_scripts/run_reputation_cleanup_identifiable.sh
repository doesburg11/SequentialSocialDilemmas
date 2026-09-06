#!/usr/bin/env bash
# Identifiable condition: reputation intrinsic reward active (McKee et al. 2023).
# A scaled-down pilot compared to run_baseline_cleanup.sh's 500M-step budget --
# run the anonymous counterpart (run_reputation_cleanup_anonymous.sh) alongside
# this one and compare collective return / territoriality / turn-taking.

python train_reputation.py \
--env cleanup_reputation \
--model baseline \
--algorithm PPO \
--num_agents 5 \
--condition identifiable \
--num_workers 6 \
--rollout_fragment_length 1000 \
--num_envs_per_worker 16 \
--stop_at_timesteps_total $((10 * 10 ** 6)) \
--memory $((40 * 10 ** 9)) \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--gpus_for_driver 1 \
--cpus_for_driver 0 \
--num_samples 1 \
--entropy_coeff 0.00176 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights .00126 .000012
