# 🚀 FlightGPT： A vision-language model based agent for UAV navigation.
[![arXiv](https://img.shields.io/badge/arXiv-2506.12364-b31b1b.svg)](https://arxiv.org/abs/2506.12364)
[![model](https://img.shields.io/badge/model-Flightgpt-yellow.svg)](https://huggingface.co/ADJHD/Flightgpt)
[![data](https://img.shields.io/badge/data-training-blue.svg)](https://huggingface.co/datasets/ADJHD/flightgpt_training_data)

****
## 📑 Introduction
FlightGPT is a state-of-the-art UAV Vision-and-Language Navigation (VLN) framework designed for applications like disaster response, logistics delivery, and urban inspection. Built on powerful Vision-Language Models (VLMs), FlightGPT employs a two-stage training pipeline: supervised fine-tuning (SFT) with high-quality demonstrations to improve initialization and reasoning, followed by Group Relative Policy Optimization (GRPO) guided by a composite reward considering goal accuracy, reasoning quality, and format compliance to enhance generalization. With a Chain-of-Thought (CoT) reasoning mechanism for interpretable decision-making, FlightGPT achieves state-of-the-art performance on the city-scale CityNav dataset, surpassing the strongest baseline by 9.22% in unseen environments.


## 📢 News
- **2025-12-2**: Our SFT model [Flightgpt_SFT](https://huggingface.co/ADJHD/Flightgpt_SFT) is now publicly available on Hugging Face!
- **2025-9-4**: Our training [data](https://huggingface.co/datasets/ADJHD/flightgpt_training_data) is now publicly available on Hugging Face!
- **2025-9-3**: Our model [Flightgpt](https://huggingface.co/ADJHD/Flightgpt) is now publicly available on Hugging Face!
- **2025-08-21**: Accepted by EMNLP 2025!


## 🛠️ Environment Setup

This project depends on multiple models and tool libraries. It is recommended to use Conda to create an isolated environment.

### Install Conda Environment

```bash
- conda create -n flightgpt python=3.11
- conda activate flightgpt

- pip install -r requirements.txt
```

---

## 🛠️ Model and Data Preparation

* Download model weights to `./model_weight/`  
  Note: Change the value of `max_pixels` in `preprocessor_config.json` to `16032016`.

* Download data to `./data/`

* And for sft, Download the cleaned_final.json to ./LLaMA-Factory/data

### 📦 Project Structure
├── model_weight/ # Directory for model weights (download manually)  
├── experiment/  
├── R1PhotoData/  
├── data/  
│    └── citynav/ # Data annotation directory  
│    └── rgbd-new/ # Raw image files  
│    └── training_data/ # Training data directory  
│    └── ...  
├── data_examples/ # Examples of some training data  
├── eval.py # Model inference and evaluation script  
├── open-r1-multimodal/ # GRPO training directory  
├── LLaMA-Factory/ # SFT training directory  
├── requirements.txt # Combined environment dependency file  
├── README.md # This document  
├── ...  

---

## 🚀 Inference

1. Start the vLLM service
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /home/yjy/flightgpt/FlightGPT/model_weight/Flightgpt_SFT \
  --dtype auto \
  --trust-remote-code \
  --served-model-name qwen_2_5_vl_7b \
  --host 0.0.0.0 \
  -tp 4 \
  --uvicorn-log-level debug \
  --port 8989 \
  --limit-mm-per-prompt image=2,video=0 \
  --max-model-len=32000


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ./model_weight/Qwen2.5-VL-7B-Instruct \
  --dtype auto \
  --trust-remote-code \
  --served-model-name qwen_2_5_vl_7b \
  --host 0.0.0.0 \
  -tp 4 \
  --port 8989 \
  --enable-lora \
  --lora-modules grpo_lora=./experiment/FlightGPT/checkpoint-2379 \
  --limit-mm-per-prompt image=2,video=0 \
  --max-model-len=32000 \
  --max-lora-rank 64
```

2. Start the inference script

```bash
python eval_by_qwen.py
```

3. Result Visualization  
You can use the visualize_prediction function to visualize the predicted target coordinates and the landmark bounding boxes, as well as the actual target coordinates and landmark bounding boxes.

---

## 🚀 Training
1. SFT
```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
llamafactory-cli export ./LLaMA-Factory/examples/merge_lora/qwen2vl_lora_sft.yaml
```


2、GRPO
```bash
sh ./open-r1-multimodal/run_scripts/run_grpo_rec_lora.sh
sh ./open-r1-multimodal/run_scripts/run_grpo_rec_lora_newdataset.sh
sh ./open-r1-multimodal/run_scripts/run_grpo_lora_newdataset_flightgptsft.sh
```

---

## 🧠 VIB-Nav: Learned Visual Information Bottleneck

Empirically, training at 4k map resolution but doing inference at 2k beats 4k/4k.
The hypothesis is that navigation's task-sufficient statistic is low-dimensional, so
4k feeds information *above* the task-optimal rate and the policy overfits to nuisance
detail. **VIB-Nav** replaces the crude "downsample to 2k" bottleneck with a *learned,
instruction-conditioned* Information Bottleneck on the post-merger visual tokens,
decoupling the **information rate** from the **input resolution**.

For each visual token `h_i` (conditioned on the instruction via cross-attention) we learn a
diagonal Gaussian posterior `q(z_i | h_i, c_i) = N(mu_i, diag(exp(logvar_i)))`, sample
`z_i` with the reparameterization trick, and feed **z** (not h) to the LLM during both
training and rollouts. We add a KL rate term to the GRPO loss:

```
L = L_RL + beta * R_hat,   R_hat = mean_i max(KL(q(z_i) || N(0,I)), free_bits)   # nats/token
```

Code: bottleneck in `open-r1-multimodal/src/open_r1/models/vib_adapter.py`, the Qwen2.5-VL
forward hook + `pop_vib_rate()` in `open-r1-multimodal/src/open_r1/models/qwen_vib_patch.py`,
loss/logging in `open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py`. Unit tests:
`open-r1-multimodal/tests/test_vib_adapter.py` (`pytest tests/test_vib_adapter.py`).

### Enable VIB

Add these flags to any GRPO run script (defaults reproduce the original pipeline exactly,
i.e. `--vib_enabled false` changes no numerics):

```bash
    --vib_enabled true \
    --vib_mode fixed \
    --vib_beta 1e-3 \
    --vib_free_bits 0.5 \
    --vib_instruction_conditioned true \
    --vib_eval_mode mean \
    --freeze_vision_modules true       # backbone stays frozen; only VIB (+ LoRA) train
```

A ready-made example is `open-r1-multimodal/run_scripts/run_grpo_vib.sh`.

### Logged diagnostics (every `logging_steps`)

`vib/beta`, `vib/rate` (R_hat), `vib/kl_token_mean`, `vib/suppressed_frac`,
`vib/mu_abs_mean`, `vib/std_mean`, `vib/rl_loss`, `vib/total_loss`. Set
`--vib_artifact_every M` (M>0) to dump per-token KL → image-patch artifacts to
`<output_dir>/vib_artifacts/` for the "which tokens get suppressed" figure.

### Experiments (one-line config changes)

| Goal | Change |
| --- | --- |
| Regression test (identical to original) | `--vib_enabled false` |
| Instruction-agnostic ablation | `--vib_instruction_conditioned false` |
| **Rate–performance inverted-U** (beta sweep) | `--vib_mode fixed` with `--vib_beta` in `{0, 1e-4, 1e-3, 1e-2, 1e-1}` |
| **Matched-rate 4k vs 2k** | `--vib_mode dual --vib_R_star <R>` trained once on 4k data and once on 2k data |
| Free-bits ablation | `--vib_free_bits` in `{0.0, 0.5, 1.0}` |

**Beta sweep** (draws the inverted-U): launch one run per beta with `--vib_mode fixed`,
plotting reported eval metric vs the converged `vib/rate`.

**Matched-rate experiment**: pick a target rate `R*`, run `--vib_mode dual --vib_R_star R*`
on the 4k dataset and again on the 2k dataset. Dual ascent drives `vib/rate -> R*` in both,
so the two runs are compared at *matched information rate* despite different input
resolutions — isolating the effect of rate from resolution.

---

## 🖋️ Citation

If you use FlightGPT in your research, please cite our project:

```bibtex

@article{cai2025flightgpt,
  title={FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models},
  author={Cai, Hengxing and Dong, Jinhan and Tan, Jingjun and Deng, Jingcheng and Li, Sihang and Gao, Zhifeng and Wang, Haidong and Su, Zicheng and Sumalee, Agachai and Zhong, Renxin},
  journal={arXiv preprint arXiv:2505.12835},
  year={2025}
}
```
