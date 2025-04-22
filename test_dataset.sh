export CUDA_VISIBLE_DEVICES=1

export PRETRAINED_MODEL_NAME_OR_PATH='/mnt/data/xurongtao/checkpoints/0402-a0-all-alternate/checkpoint-30000'
export IMAGE_SAVE_PATH=None
DATASET=$1

if [ "$DATASET" = "droid" ]; then
    echo "Running on DROID dataset..."
    python -m scripts.test_droid_dataset --poa  --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH  --image_save_path=$IMAGE_SAVE_PATH
elif [ "$DATASET" = "hoi4d" ]; then
    echo "Running on HOI4D dataset..."
    python -m scripts.test_hoi4d_dataset --poa  --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH --image_save_path=$IMAGE_SAVE_PATH
elif [ "$DATASET" = "hoi4d_frame" ]; then
    echo "Running on HOI4D Frame Selection dataset..."
    python -m scripts.test_hoi4d_frame_dataset --poa --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH --image_save_path=$IMAGE_SAVE_PATH
elif [ "$DATASET" = "maniskill" ]; then
    echo "Running on ManiSkill dataset..."
    python -m scripts.test_maniskill_dataset --poa  --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH --image_save_path=$IMAGE_SAVE_PATH
else
    echo "Unknown dataset: $DATASET"
    echo "Usage: bash test_dataset.sh [droid|hoi4d|maniskill]"
fi