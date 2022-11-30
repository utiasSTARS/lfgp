#!/bin/bash

SEED=$1
DEVICE=$2
MAIN_TASK=$3
EXPERT_DIR=$4
USER_MACHINE=$5
EXPBUF_LAST_SAMPLE_PROP=$6  # default is .95, 0. turns it off
EXPBUF_MODEL_SAMPLE_RATE=$7  # default is .1, 0. turns it off
EXPERIMENT_NAME="${EXPERT_DIR}_$8"


DEFAULT_TOP_DIR="../../lfgp_data/expert_data/"
TOP_DIR=${LFGP_DATA_TOP_DIR:=${DEFAULT_TOP_DIR}}
echo "Using TOP_DIR OF ${TOP_DIR}"

DEFAULT_STACK_DIR="stack/"
STACK_DIR=${STACK_DIR:=${DEFAULT_STACK_DIR}}

DEFAULT_UNSTACK_DIR="unstack_stack_env_only/"
UNSTACK_DIR=${UNSTACK_DIR:=${DEFAULT_UNSTACK_DIR}}

DEFAULT_BRING_DIR="bring/"
BRING_DIR=${BRING_DIR:=${DEFAULT_BRING_DIR}}

DEFAULT_INSERT_DIR="insert/"
INSERT_DIR=${INSERT_DIR:=${DEFAULT_INSERT_DIR}}


if [ "${MAIN_TASK}" = "stack" ]; then
    EXPERT_PATH_MID="${TOP_DIR}${STACK_DIR}"
    PRE="${EXPERT_PATH_MID}${EXPERT_DIR}/int_"
    EXPERT_PATH="${PRE}2.gz"
    MAX_STEPS=2000000
elif [ "${MAIN_TASK}" = "unstack_stack_env_only" ]; then
    EXPERT_PATH_MID="${TOP_DIR}${UNSTACK_DIR}"
    PRE="${EXPERT_PATH_MID}${EXPERT_DIR}/int_"
    EXPERT_PATH="${PRE}2.gz"
    MAX_STEPS=2000000
elif [ "${MAIN_TASK}" = "bring" ]; then
    EXPERT_PATH_MID="${TOP_DIR}${BRING_DIR}"
    PRE="${EXPERT_PATH_MID}${EXPERT_DIR}/int_"
    EXPERT_PATH="${PRE}2.gz"
    MAX_STEPS=2000000
elif [ "${MAIN_TASK}" = "insert" ]; then
    EXPERT_PATH_MID="${TOP_DIR}${INSERT_DIR}"
    PRE="${EXPERT_PATH_MID}${EXPERT_DIR}/int_"
    EXPERT_PATH="${PRE}2.gz"
    MAX_STEPS=4000000
else
    echo "Invalid MAIN_TASK ${MAIN_TASK}"
    exit 1
fi

echo "Running DAC for seed ${SEED}, on device ${DEVICE}, main task ${MAIN_TASK}, expert dir ${EXPERT_DIR}."
echo "User machine ${USER_MACHINE}, Experiment name ${EXPERIMENT_NAME}."

PYTHON_TO_EXEC=$(cat <<-END
../../rl_sandbox/rl_sandbox/examples/lfgp/run_dac.py
--seed ${SEED}
--user_machine ${USER_MACHINE}
--expert_path ${EXPERT_PATH}
--main_task ${MAIN_TASK}_0
--exp_name ${EXPERIMENT_NAME}
--device ${DEVICE}
--num_evals 50 
--max_steps ${MAX_STEPS}
--expbuf_last_sample_prop=${EXPBUF_LAST_SAMPLE_PROP}
--expbuf_model_sample_rate=${EXPBUF_MODEL_SAMPLE_RATE}
END
)

if [[ "${DEVICE}" == *"cuda"* ]]; then
    PYTHON_TO_EXEC+=" --gpu_buffer"
fi

python ${PYTHON_TO_EXEC}
