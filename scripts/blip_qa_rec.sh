export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

python -m torch.distributed.run --nproc_per_node=8 \
         blip_qa_train.py \
        --data_path data/blip/blip_refcoco3_rec_643k.json \
        --model_name_or_path ckpts/blip-vqa-capfilt-large \
        --bf16 \
        --output_dir checkpoints/blip_qa_rec \
        --max_steps 2000    \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000  \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --tf32 True  \
        --model_max_length 256  \
        --gradient_checkpointing True  \
        --lazy_preprocess True \
        --pretraining_length 256