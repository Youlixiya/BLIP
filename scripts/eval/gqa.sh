#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# CKPT="llava-clip-vit-l-384-tinyllama-1.1b-3t"
# CKPT="llava-clip-vit-l-224-qwen-v1.5-1.8b"
# CKPT="llava-siglip-vit-l-384-tinyllama-1.1b-chat"
# CKPT="llava-clip-sam-tinyllama-1.1b-chat"
CKPT="llava-clip-vit-l-336-mobilellama-1.7b-chat"
# CKPT="MobileVLM_V2-3B"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m mobilevlm.eval.model_vqa_loader \
        --model-path checkpoints/llava-clip-vit-l-336-mobilellama-1.7b-chat \
        --question-file data/eval/gqa/$SPLIT.jsonl \
        --image-folder data/eval/gqa/data/images \
        --answers-file data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode v1 &
done

# wait

output_file=data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/eval/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
