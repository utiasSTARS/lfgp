#!/bin/bash

# seeds should be in the following format: "1 2 3 4 5"
SEEDS=($1)
SCRIPT=$2
DEVICE=$3
MAIN_TASK=$4

# expert dir should just be the lowest level directory before int_X.gz,
# the rest is handled in the individual script files
EXPERT_DIR=$5
USER_MACHINE=$6
EXPERIMENT_NAME=$7

# optional for DAC/LfGP only
EXPBUF_LAST_SAMPLE_PROP=$8  # default is .95, 0. turns it off
EXPBUF_MODEL_SAMPLE_RATE=$9  # default is .1, 0. turns it off

# optional for LfGP only
SCHEDULER=$10

if [ "${SCRIPT}" = "lfgp.bash" ]; then
    for seed in "${SEEDS[@]}"
    do
        bash "${SCRIPT}" "${seed}" "${DEVICE}" "${MAIN_TASK}" "${EXPERT_DIR}" "${USER_MACHINE}" "${SCHEDULER}" \
            "${EXPBUF_LAST_SAMPLE_PROP}" "${EXPBUF_MODEL_SAMPLE_RATE}" "${EXPERIMENT_NAME}"
    done
elif [ "${SCRIPT}" = "dac.bash" ]; then
    for seed in "${SEEDS[@]}"
    do
        bash "${SCRIPT}" "${seed}" "${DEVICE}" "${MAIN_TASK}" "${EXPERT_DIR}" "${USER_MACHINE}" \
            "${EXPBUF_LAST_SAMPLE_PROP}" "${EXPBUF_MODEL_SAMPLE_RATE}" "${EXPERIMENT_NAME}"
    done
else
    for seed in "${SEEDS[@]}"
    do
        bash "${SCRIPT}" "${seed}" "${DEVICE}" "${MAIN_TASK}" "${EXPERT_DIR}" "${USER_MACHINE}" "${EXPERIMENT_NAME}"
    done
fi