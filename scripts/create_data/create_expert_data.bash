#!/bin/bash

# NUM_STEPS_PER_BUFFER should be either an empty string, or a comma separated list, e.g. "9000, 0, 0, 0, 0, 0"

MAIN_TASK=$1
STEPS_PER_TASK=$2
NUM_STEPS_PER_BUFFER=$3
AUX_OVERRIDE=$4


SAVE_PATH_POST="${STEPS_PER_TASK}_steps_no_extra_final"
DEFAULT_TOP_DIR="../../lfgp_data"
TOP_DIR=${LFGP_TOP_DIR:=${DEFAULT_TOP_DIR}}
echo "Using TOP_DIR OF ${TOP_DIR}"
FULL_PATH="${TOP_DIR}/trained_models/experts/${MAIN_TASK}"
MODEL_PATH="${FULL_PATH}/state_dict.pt"
CONFIG_PATH="${FULL_PATH}/sacx_experiment_setting.pkl"
SAVE_PATH="${TOP_DIR}/custom_expert_data/${MAIN_TASK}/${SAVE_PATH_POST}/"


echo "Generating data for ${MAIN_TASK}, ${STEPS_PER_TASK} steps per task."
if [ -z "${NUM_STEPS_PER_BUFFER}" ]; then
    echo "NUM_STEPS_PER_BUFFER is unset, running for all tasks"
else
    echo "Getting ${NUM_STEPS_PER_BUFFER} for each task, running task ${AUX_OVERRIDE} only."
fi


if [ "${MAIN_TASK}" = "insert" ]; then
    O_PRB="0.08333333333"
    FORCED_SCHEDULE="{0: {0: ([0, 1, 2, 3, 4, 5, 6], [${O_PRB}, ${O_PRB}, .5, ${O_PRB}, ${O_PRB}, ${O_PRB}, ${O_PRB}], ['k', 'd', 'd', 'd', 'd', 'd', 'd']), 70: 0}, 1: {0: 3, 15: 1}}"
    SCHEDULER_PERIOD=90
    if [ -z "${NUM_STEPS_PER_BUFFER}" ]; then
        NUM_STEPS_PER_BUFFER=""
        AUX_OVERRIDE=""
    fi

elif [ "${MAIN_TASK}" = "stack" ]; then
    FORCED_SCHEDULE="{0: {0: ([0, 1, 2, 3, 4, 5], [.1, .1, .5, .1, .1, .1], ['k', 'd', 'd', 'd', 'd', 'd']), 45: 0}, 1: {0: 3, 15: 1}}"
    SCHEDULER_PERIOD=90
    if [ -z "${NUM_STEPS_PER_BUFFER}" ]; then
        NUM_STEPS_PER_BUFFER=""
        AUX_OVERRIDE=""
    fi

elif [ "${MAIN_TASK}" = "bring" ]; then
    FORCED_SCHEDULE="{0: {0: ([0, 1, 3, 4, 5, 6], [.1, .1, .5, .1, .1, .1], ['k', 'd', 'd', 'd', 'd', 'd']), 45: 0}, 1: {0: 3, 15: 1}}"
    SCHEDULER_PERIOD=90
    if [ -z "${NUM_STEPS_PER_BUFFER}" ]; then
        NUM_STEPS_PER_BUFFER=""
        AUX_OVERRIDE="0,1,3,4,5,6"  # to ensure we skip insert from this model
    fi

elif [ "${MAIN_TASK}" = "unstack-stack" ]; then
    FORCED_SCHEDULE="{0: {0: ([0, 1, 2, 4, 5, 6], [.1, .1, .5, .1, .1, .1], ['k', 'd', 'd', 'd', 'd', 'd']), 45: 0}, 1: {0: 3, 15: 1}}"
    SCHEDULER_PERIOD=120
    if [ -z "${NUM_STEPS_PER_BUFFER}" ]; then
        NUM_STEPS_PER_BUFFER=""
        AUX_OVERRIDE="0,1,2,4,5,6"  # to ensure we skip unstack from this model
    fi

fi

echo "Saving to ${SAVE_PATH}"

python ../../rl_sandbox/rl_sandbox/examples/lfgp/experts/create_expert_data.py \
    --model_path="${MODEL_PATH}" \
    --config_path="${CONFIG_PATH}" \
    --save_path="${SAVE_PATH}" \
    --num_episodes=10000000 \
    --num_steps="${STEPS_PER_TASK}" \
    --seed=1 \
    --forced_schedule="${FORCED_SCHEDULE}" \
    --scheduler_period="${SCHEDULER_PERIOD}" \
    --success_only \
    --reset_on_success \
    --reset_between_intentions \
    --aux_override="${AUX_OVERRIDE}"