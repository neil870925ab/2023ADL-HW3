# "${1}" : ./data/train.json

python train.py \
    --base_model_path ./Taiwan-LLM-7B-v2.0-chat \
    --train_file  "${1}" \
    --output_dir ./adapter_checkpoint

