#!/bin/bash

SEED=$1
MODEL_PATH_AFTER_TOP="$2"
MODEL_NAME="$3"
CONFIG_NAME="$4"
NUM_EPISODES="$5"
INTENTION="$6"
RENDER="$7"
STOCHASTIC="$8"
FORCED_SCHEDULE="$9"

# some ideas for what you might want to try with forced schedule!
# FORCED_SCHEDULE="{0: 4, 45: 3, 90: 5, 135: 2, 180: 0}"
# FORCED_SCHEDULE="{0: 3, 25: 2, 50: 4, 75: 5, 100: 1, 125: 3, 150: 4, 175: 2, 200: 1}"
# FORCED_SCHEDULE="{0: 3, 90: 2, 180: 0}"
# FORCED_SCHEDULE="{0: 4, 45: 3, 90: 2, 180: 0}"
# FORCED_SCHEDULE="{0: 4, 45: 3, 90: 2, 135: 0, 180: 4, 225: 3, 270: 2, 315: 0}"
# FORCED_SCHEDULE="{0: 3, 45: 2, 90: 0, 135: 3, 180: 2, 225: 0, 270: 3, 315: 2}"
# FORCED_SCHEDULE="{0: 5, 90: 2, 180: 0}"
# FORCED_SCHEDULE="{0: 5, 45: 2, 90: 4, 135: 2, 180: 3, 225: 2, 270: 2, 315: 5}"  # realistic WRS ep


DEFAULT_TOP_DIR="../../lfgp_data/trained_models/"
TOP_DIR=${LFGP_MODEL_TOP_DIR:=${DEFAULT_TOP_DIR}}
echo "Using TOP_DIR OF ${TOP_DIR}"

COMMON_TOP="${TOP_DIR}/${MODEL_PATH_AFTER_TOP}"
MODEL_PATH="${COMMON_TOP}/${MODEL_NAME}"
CONFIG_PATH="${COMMON_TOP}/${CONFIG_NAME}"

PYTHON_TO_EXEC=$(cat <<-END 
../../rl_sandbox/rl_sandbox/examples/eval_tools/evaluate.py 
--seed=${SEED}
--model_path=${MODEL_PATH} 
--config_path=${CONFIG_PATH}
--num_episodes=${NUM_EPISODES}
--intention=${INTENTION}
--model_path=${MODEL_PATH}
--forced_schedule=${FORCED_SCHEDULE}
--force_egl
END
)

if [ "${RENDER}" = "true" ]; then
    PYTHON_TO_EXEC+=" --render"
fi

if [ "${STOCHASTIC}" = "true" ]; then
    PYTHON_TO_EXEC+=" --stochastic"
fi

python ${PYTHON_TO_EXEC}