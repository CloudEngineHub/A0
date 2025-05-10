# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export TEXT_ENCODER_NAME="Qwen/Qwen2.5-7B"  #"google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="/mnt/data/xurongtao/checkpoints/a0-1b"

export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

# export HF_ENDPOINT=https://hf-mirror.com

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export DS_LOG_LEVEL=DEBUG

export WANDB_PROJECT="a0_diffusion_transformer"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi


export CUDA_VISIBLE_DEVICES=1

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

# deepspeed --hostfile=hostfile.txt main.py \
    # --deepspeed="./configs/zero2.json" \



accelerate launch --main_process_port 29501 main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --text_encoder="Qwen2.5-7B" \
    --output_dir=$OUTPUT_DIR \
    --train_datasets="all" \
    --train_batch_size=100 \
    --sample_batch_size=1 \
    --max_train_steps=160000 \
    --checkpointing_period=2000 \
    --sample_period=-1 \
    --checkpoints_total_limit=15 \
    --lr_scheduler="constant_with_warmup" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --crop_rotate_aug \
    --cond_mask_prob=0.0 \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=tensorboard \

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \

    # Use this to load the a0 checkpoint trained on the 1 million pixmo-point datset 
    # --pretrained_model_name_or_path="JianZhangAI/A0-1b-pretrain"

    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    # --precomp_lang_embed \
