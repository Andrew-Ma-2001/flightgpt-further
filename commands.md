#### 合并LoRA adapter into base model
```bash
cd /home/yjy/flightgpt/FlightGPT && source /home/yjy/miniconda3/etc/profile.d/conda.sh && conda activate flightgpt && python merge_grpo_lora.py \
    --base_model ./model_weight/Qwen2.5-VL-7B-Instruct \
    --lora_adapter ./experiment/FlightGPT/checkpoint-2379 \
    --output_dir ./model_weight/FlightGPT_GRPO_Merged
```

#### 启动vLLM server，对应 grpo 训练模型，qwen2.5-vl-7b, flightgpt sft 模型

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ./model_weight/FlightGPT_GRPO_Merged   --dtype auto   --trust-remote-code   --served-model-name qwen_2_5_vl_grpo   --host 0.0.0.0   -tp 4   --port 8989   --limit-mm-per-prompt image=2,video=0   --max-model-len=32000

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ./model_weight/Qwen2.5-VL-7B-Instruct   --dtype auto   --trust-remote-code   --served-model-name qwen_2_5_vl_7b   --host 0.0.0.0   -tp 4   --port 8989   --limit-mm-per-prompt image=2,video=0   --max-model-len=32000

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ./model_weight/Flightgpt_SFT   --dtype auto   --trust-remote-code   --served-model-name qwen_2_5_vl_7b_sft   --host 0.0.0.0   -tp 4   --port 8989   --limit-mm-per-prompt image=2,video=0   --max-model-len=32000
```

##### 可视化部分
streamlit run human_eval/vllm_vision_chat_app.py
streamlit run human_eval/checkpoint_analysis_app.py
streamlit run human_eval/manual_eval_app.py


百度网盘上传备份用法

> https://github.com/qjfoidnh/BaiduPCS-Go?tab=readme-ov-file#%E4%B8%8B%E8%BD%BD%E8%BF%90%E8%A1%8C-%E8%AF%B4%E6%98%8E

Linux 编译 
```bash
sudo apt update && sudo apt install -y git wget curl
git clone https://github.com/qjfoidnh/BaiduPCS-Go.git
cd BaiduPCS-Go
go version
go env -w GOPROXY=https://goproxy.cn,direct
go mod download
CGO_ENABLED=0 GOOS=linux GOARCH=$(go env GOARCH) go build -o BaiduPCS-Go
./BaiduPCS-Go
```

上传文件
```bash
upload /home/yjy/flightgpt/FlightGPT/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl ./ --norapid
```

百度网盘保存文件：
1. refineCityNav 对应 refine_citynav
2. r1-flightgpt 对应 experiment/FlightGPT/checkpoint-2379
3. refineCityNav 保存 flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64 依赖包