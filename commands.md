cd /home/yjy/flightgpt/FlightGPT && source /home/yjy/miniconda3/etc/profile.d/conda.sh && conda activate flightgpt && python merge_grpo_lora.py \
    --base_model ./model_weight/Qwen2.5-VL-7B-Instruct \
    --lora_adapter ./experiment/FlightGPT/checkpoint-2379 \
    --output_dir ./model_weight/FlightGPT_GRPO_Merged