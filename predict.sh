# "${1}" : ./Taiwan-LLM-7B-v2.0-chat
# "${2}" : ./adapter_checkpoint
# "${3}" : ./data/public_test.json
# "${4}" : ./prediction.json

python predict.py \
    --base_model_path "${1}" \
    --peft_path "${2}" \
    --test_file "${3}" \
    --output_file "${4}" \