#!/usr/bin/env bash
# Anonymous condition: reputation intrinsic reward disabled (McKee et al. 2023).
# Contributions are still tracked internally for post-hoc territoriality/turn-taking
# metrics; only the reward term is withheld. Compare against
# run_reputation_cleanup_identifiable.sh.

python train_reputation.py \
--env cleanup_reputation \
--model baseline \
--algorithm PPO \
--num_agents 5 \
--condition anonymous \
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
