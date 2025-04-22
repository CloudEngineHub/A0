 # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt,
python -m scripts.a0_inference \
    --pretrained_model_name_or_path='/path/to/model' \
    --instruction='put the pear on the plate' \
    --image_path=/path/to/image.jpg 