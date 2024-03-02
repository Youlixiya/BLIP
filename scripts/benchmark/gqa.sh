#!/bin/bash

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$GPUS"
CHUNKS=${#GPULIST[@]}

MODEL_LOADER='mobilevlm/eval/model_vqa_loader.py'
MODEL_DIR='ckpts/MobileVLM_V2-3B'
CONV_MODE='v1'
SPLIT_NME='llava_gqa_testdev_balanced'
DATA_ROOT='data/eval/gqa/data'
SAVE_PATH='data/eval/gqa/answers'
# MODEL_LOADER=$1
# MODEL_DIR=$2
# CONV_MODE=$3
# SPLIT_NME=$4
# DATA_ROOT=$5
# SAVE_PATH=$6/${SPLIT_NME}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ${MODEL_LOADER} \
        --model-path ${MODEL_DIR} \
        --question-file ${DATA_ROOT}/${SPLIT_NME}.jsonl \
        --image-folder ${DATA_ROOT}/images \
        --answers-file ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV_MODE} &
done

wait

RESULT_FILE=${SAVE_PATH}/merge.jsonl
> "${RESULT_FILE}"  # Clear out the output file if it exists
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl >> "${RESULT_FILE}"  # Loop through the indices and concatenate each file
    rm ${SAVE_PATH}/${CHUNKS}_${IDX}.jsonl
done

python ${DATA_ROOT}/convert_gqa_for_eval.py --src ${RESULT_FILE} --dst ${SAVE_PATH}/merge.json

python ${DATA_ROOT}/eval.py \
    --questions ${DATA_ROOT}/testdev_balanced_questions.json \
    --predictions ${SAVE_PATH}/merge.json
