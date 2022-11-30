#!/bin/bash

SCRIPT=$1
DEVICE=$2
MAIN_TASK=$3

# expert dir should just be the lowest level directory before int_X.gz,
# the rest is handled in the individual script files
EXPERT_DIR=$4
USER_MACHINE=$5
EXPERIMENT_NAME=$6

seeds=(1 2 3 4 5)
for seed in "${seeds[@]}"
do
    bash "${SCRIPT}" "${seed}" "${DEVICE}" "${MAIN_TASK}" "${EXPERT_DIR}" "${USER_MACHINE}" "${EXPERIMENT_NAME}"
done