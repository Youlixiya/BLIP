deepspeed mobilevlm/train/train_mem.py \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path ./ckpts/MobileLLaMA-1.4B-Chat \
    --version plain \
    --data_path data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder data/LLaVA-Pretrain/images \
    --vision_tower ./ckpts/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-siglip-vit-l-384-mobilellama-1.4b-chat-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# deepspeed llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path ./ckpts/TinyLlama-1.1B-Chat-v1.0 \
#     --version llava_llama_2 \
#     --data_path data/llava_v1_5_mix665k.json \
#     --image_folder data \
#     --vision_tower ./ckpts/clip-vit-large-patch14 \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-clip-vit-l-224-tinyllama-1.1b-chat-pretrain/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-clip-vit-l-224-tinyllama-1.1b-chat-lora \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb

deepspeed mobilevlm/train/train_mem.py \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path ./ckpts/MobileLLaMA-1.4B-Chat \
    --version v1 \
    --data_path data/llava_v1_5_mix665k.json \
    --image_folder data \
    --vision_tower ./ckpts/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-siglip-vit-l-384-mobilellama-1.4b-chat-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-siglip-vit-l-384-mobilellama-1.4b-chat \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path ./checkpoints/llava-siglip-vit-l-384-tinyllama-1.1b-chat \
#     --version v1 \
#     --data_path data/coco_train.json+data/llava_instruct_150k.json+data/refcoco3_train.json \
#     --image_folder data/coco/train2017+data/coco/train2017+data/coco/train2014 \
#     --vision_tower ./ckpts/siglip-so400m-patch14-384 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --mm_use_ref_token True \
#     --mm_use_box_token True \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-siglip-vit-l-384-tinyllama-1.1b-chat-rec \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb

# python scripts/merge_lora_weights.py --model-path ./checkpoints/llava-clip-vit-l-224-tinyllama-1.1b-chat-lora \
#                                   --model-base ./ckpts/TinyLlama-1.1B-Chat-v1.0 \
#                                   --save-model-path ./checkpoints/llava-clip-vit-l-224-tinyllama-1.1b-chat
