#!/bin/bash

eval_eps=50
eval_interval=100000
num_env_steps=4000000
expert_file="../gail_experts/data/bring_0-expert_data/reset/int_2.pt"
env="bring_0"
num_processes=1
seed=10

python ../main.py \
  --seed "$seed" \
  --num-steps 2048 \
  --lr 3e-4 \
  --entropy-coef 0 \
  --value-loss-coef 0.5 \
  --ppo-epoch 10 \
  --num-mini-batch 32 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --use-linear-lr-decay \
  --use-proper-time-limits \
  --num-processes="$num_processes" \
  --use-gae \
  --algo ppo \
  --gail \
  --eval-interval="$eval_interval" \
  --num-env-steps="$num_env_steps" \
  --gail-experts-file="$expert_file" \
  --env-name="$env" \
  --log-interval 1 \
  --eval-eps "$eval_eps" \
  --no-cuda
#  --train-render
