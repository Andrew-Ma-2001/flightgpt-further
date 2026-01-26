#!/bin/bash
# ==============================================================================
# FlightGPT GRPO LoRA Model Evaluation Script
# ==============================================================================
# 
# This script provides two methods to evaluate your GRPO LoRA model:
# 
# METHOD 1: Merge LoRA and serve (recommended for production)
#   - Merges LoRA adapter into base model
#   - Creates a full model that can be served like a regular model
#   - One-time merge, then serve normally
# 
# METHOD 2: vLLM LoRA support (recommended for quick testing)
#   - No merging needed
#   - Faster to switch between different checkpoints
#   - Uses vLLM's --enable-lora flag
# 
# ==============================================================================

set -e

# Configuration
BASE_MODEL="./model_weight/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER="./experiment/FlightGPT/checkpoint-2379"
MERGED_OUTPUT="./model_weight/FlightGPT_GRPO_Merged"
PORT=8989

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "=============================================="
echo "  FlightGPT GRPO LoRA Evaluation"
echo "=============================================="
echo ""

# Check if argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_grpo_eval.sh <method>"
    echo ""
    echo "Methods:"
    echo "  merge    - Merge LoRA adapter into base model (run once)"
    echo "  serve1   - Serve the merged model with vLLM"
    echo "  serve2   - Serve base model with LoRA adapter (no merge needed)"
    echo "  eval1    - Run evaluation on merged model"
    echo "  eval2    - Run evaluation on LoRA model"
    echo ""
    echo "Typical workflows:"
    echo ""
    echo "  Workflow A (Merge approach):"
    echo "    1. ./run_grpo_eval.sh merge    # Run once to merge"
    echo "    2. ./run_grpo_eval.sh serve1   # Start vLLM server"
    echo "    3. ./run_grpo_eval.sh eval1    # Run evaluation (in another terminal)"
    echo ""
    echo "  Workflow B (Direct LoRA approach - no merge):"
    echo "    1. ./run_grpo_eval.sh serve2   # Start vLLM with LoRA support"
    echo "    2. ./run_grpo_eval.sh eval2    # Run evaluation (in another terminal)"
    echo ""
    exit 1
fi

METHOD=$1

case $METHOD in
    "merge")
        echo -e "${YELLOW}Merging LoRA adapter into base model...${NC}"
        echo "Base model: $BASE_MODEL"
        echo "LoRA adapter: $LORA_ADAPTER"
        echo "Output: $MERGED_OUTPUT"
        echo ""
        python merge_grpo_lora.py \
            --base_model "$BASE_MODEL" \
            --lora_adapter "$LORA_ADAPTER" \
            --output_dir "$MERGED_OUTPUT"
        echo -e "${GREEN}Merge complete!${NC}"
        ;;
    
    "serve1")
        echo -e "${YELLOW}Starting vLLM server with merged model...${NC}"
        echo "Model: $MERGED_OUTPUT"
        echo "Port: $PORT"
        echo ""
        
        if [ ! -d "$MERGED_OUTPUT" ]; then
            echo -e "${RED}Error: Merged model not found at $MERGED_OUTPUT${NC}"
            echo "Please run './run_grpo_eval.sh merge' first"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MERGED_OUTPUT" \
            --dtype auto \
            --trust-remote-code \
            --served-model-name qwen_2_5_vl_7b \
            --host 0.0.0.0 \
            -tp 4 \
            --uvicorn-log-level debug \
            --port $PORT \
            --limit-mm-per-prompt image=2,video=0 \
            --max-model-len=32000
        ;;
    
    "serve2")
        echo -e "${YELLOW}Starting vLLM server with LoRA support...${NC}"
        echo "Base model: $BASE_MODEL"
        echo "LoRA adapter: $LORA_ADAPTER"
        echo "Port: $PORT"
        echo ""
        
        if [ ! -d "$LORA_ADAPTER" ]; then
            echo -e "${RED}Error: LoRA adapter not found at $LORA_ADAPTER${NC}"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$BASE_MODEL" \
            --dtype auto \
            --trust-remote-code \
            --served-model-name qwen_2_5_vl_7b \
            --host 0.0.0.0 \
            -tp 4 \
            --uvicorn-log-level debug \
            --port $PORT \
            --enable-lora \
            --lora-modules "grpo_lora=$LORA_ADAPTER" \
            --limit-mm-per-prompt image=2,video=0 \
            --max-model-len=32000 \
            --max-lora-rank 64
        ;;
    
    "eval1")
        echo -e "${YELLOW}Running evaluation on merged model...${NC}"
        echo ""
        python eval.py
        ;;
    
    "eval2")
        echo -e "${YELLOW}Running evaluation on LoRA model...${NC}"
        echo ""
        python eval_grpo_lora.py
        ;;
    
    *)
        echo -e "${RED}Unknown method: $METHOD${NC}"
        echo "Valid methods: merge, serve1, serve2, eval1, eval2"
        exit 1
        ;;
esac
