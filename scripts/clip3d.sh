CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 clip3d_train.py --optim adamw --learning_rate 0.001 --weight_decay 0.0005 --batch_size 256 --epochs 50 --ckpt ckpts/clip3d.pt --work_dir work_dir/clip3d
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 blip_train.py --optim adamw --learning_rate 2e-5 --weight_decay 0 --batch_size 64 --epochs 1 --ckpt ckpts/blip --work_dir work_dir/blip
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 blip_qa_train_bak.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 extract_caption_by_blip.py --work_dir work_dir/blip