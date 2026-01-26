"""
Merge GRPO LoRA adapter weights into the base model.

This script merges the LoRA adapter trained with GRPO into the base Qwen2.5-VL model,
creating a full merged model that can be served directly with vLLM.

Usage:
    python merge_grpo_lora.py \
        --base_model ./model_weight/Qwen2.5-VL-7B-Instruct \
        --lora_adapter ./experiment/FlightGPT/checkpoint-2379 \
        --output_dir ./model_weight/FlightGPT_GRPO_Merged
"""

import argparse
import os
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def merge_lora_model(base_model_path: str, lora_adapter_path: str, output_dir: str):
    """
    Merge LoRA adapter weights into the base model and save the merged model.
    
    Args:
        base_model_path: Path to the base model (e.g., Qwen2.5-VL-7B-Instruct)
        lora_adapter_path: Path to the LoRA adapter checkpoint
        output_dir: Directory to save the merged model
    """
    print("=" * 60)
    print("LoRA Model Merger for FlightGPT GRPO")
    print("=" * 60)
    
    print(f"\n📁 Base model: {base_model_path}")
    print(f"📁 LoRA adapter: {lora_adapter_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load the base model
    print("\n🔄 Step 1/4: Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"   ✓ Base model loaded successfully")
    
    # Step 2: Load and merge the LoRA adapter
    print("\n🔄 Step 2/4: Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.bfloat16,
    )
    print(f"   ✓ LoRA adapter loaded successfully")
    
    # Merge the LoRA weights into the base model
    print("\n🔄 Step 3/4: Merging weights...")
    merged_model = model.merge_and_unload()
    print(f"   ✓ Weights merged successfully")
    
    # Step 3: Save the merged model
    print(f"\n🔄 Step 4/4: Saving merged model to {output_dir}...")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    print(f"   ✓ Model weights saved")
    
    # Step 4: Copy/save tokenizer and processor
    print("\n🔄 Saving tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"   ✓ Tokenizer saved")
    
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_dir)
    print(f"   ✓ Processor saved")
    
    print("\n" + "=" * 60)
    print("✅ MERGE COMPLETE!")
    print("=" * 60)
    print(f"\nMerged model saved to: {output_dir}")
    print("\nYou can now serve the merged model with vLLM:")
    print(f"""
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve {output_dir} \\
  --dtype auto \\
  --trust-remote-code \\
  --served-model-name qwen_2_5_vl_7b \\
  --host 0.0.0.0 \\
  -tp 4 \\
  --uvicorn-log-level debug \\
  --port 8989 \\
  --limit-mm-per-prompt image=2,video=0 \\
  --max-model-len=32000
""")
    print("Then run: python eval.py")
    

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="./model_weight/Qwen2.5-VL-7B-Instruct",
        help="Path to the base model"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default="./experiment/FlightGPT/checkpoint-2379",
        help="Path to the LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_weight/FlightGPT_GRPO_Merged",
        help="Directory to save the merged model"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model not found: {args.base_model}")
    if not os.path.exists(args.lora_adapter):
        raise FileNotFoundError(f"LoRA adapter not found: {args.lora_adapter}")
    if not os.path.exists(os.path.join(args.lora_adapter, "adapter_config.json")):
        raise FileNotFoundError(f"adapter_config.json not found in {args.lora_adapter}")
    if not os.path.exists(os.path.join(args.lora_adapter, "adapter_model.safetensors")):
        raise FileNotFoundError(f"adapter_model.safetensors not found in {args.lora_adapter}")
    
    merge_lora_model(args.base_model, args.lora_adapter, args.output_dir)


if __name__ == "__main__":
    main()
