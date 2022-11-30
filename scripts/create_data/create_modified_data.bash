#!/bin/bash

MAIN_TASK=$1
STEPS_PER_TASK_ORIG=$2
STEPS_PER_TASK_NEW=$3
NUM_EXTRA_LASTS=$4
KEEP_EVERY_NTH=$5


INPUT_PATH_POST="${STEPS_PER_TASK_ORIG}_steps"
DEFAULT_TOP_DIR="../../lfgp_data/expert_data/"
TOP_DIR=${LFGP_DATA_TOP_DIR:=${DEFAULT_TOP_DIR}}
echo "Using TOP_DIR OF ${TOP_DIR}"
OUTPUT_PATH_POST="${STEPS_PER_TASK_NEW}_steps_${KEEP_EVERY_NTH}_ss_${NUM_EXTRA_LASTS}_el/"

echo "Generating smaller dataset, subsampled by ${KEEP_EVERY_NTH}, getting ${NUM_EXTRA_LASTS} extra final transtions, for ${MAIN_TASK}, original: ${INPUT_PATH_POST}, new: ${OUTPUT_PATH_POST}."

IN_PATH="${TOP_DIR}${MAIN_TASK}/${INPUT_PATH_POST}"
OUT_PATH="${TOP_DIR}${MAIN_TASK}/${OUTPUT_PATH_POST}"

echo "Getting data from ${IN_PATH}, Saving new data to ${OUT_PATH}."

python ../../rl_sandbox/rl_sandbox/examples/lfgp/experts/create_subsampled_data.py \
    --seed=0 \
    --keep_last \
    --input_path="${IN_PATH}" \
    --output_path="${OUT_PATH}" \
    --keep_every_nth="${KEEP_EVERY_NTH}" \
    --num_to_subsample_from="${STEPS_PER_TASK_NEW}" \
    --num_extra_lasts="${NUM_EXTRA_LASTS}"