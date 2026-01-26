# SwanLab Integration Guide

This document describes how to use SwanLab for experiment tracking and visualization in open-r1-multimodal.

## Overview

SwanLab is an open-source tool for tracking and visualizing AI training processes. It's compatible with various frameworks including PyTorch, Transformers, and TRL. This project now supports both **WandB** and **SwanLab** for experiment tracking.

## Installation

SwanLab is already included in the dependencies. To install the package with SwanLab support:

```bash
pip install -e .
```

Or install SwanLab separately:

```bash
pip install swanlab
```

## Configuration

### Using SwanLab with Training Scripts

To use SwanLab for experiment tracking, you need to:

1. **Set the report_to parameter to include 'swanlab':**

```bash
--report_to swanlab
```

Or if you want to use both wandb and swanlab:

```bash
--report_to wandb swanlab
```

2. **Configure SwanLab project settings (optional):**

You can specify SwanLab-specific configurations:

```bash
--swanlab_project "your_project_name" \
--swanlab_experiment "your_experiment_name"
```

### Example Training Command

Here's an example of running GRPO training with SwanLab:

```bash
accelerate launch --config_file=configs/zero3.yaml src/open_r1/grpo.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name trl-lib/math_reasoning \
    --dataset_train_split train \
    --dataset_test_split test \
    --output_dir checkpoints/qwen2-vl-grpo \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --report_to swanlab \
    --swanlab_project "open-r1-multimodal" \
    --swanlab_experiment "qwen2-vl-grpo-experiment" \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100
```

### Using Environment Variables

You can also set SwanLab configuration via environment variables:

```bash
export SWANLAB_PROJECT="your_project_name"
export SWANLAB_EXPERIMENT="your_experiment_name"
export SWANLAB_API_KEY="your_api_key"  # Optional: for cloud logging
```

## SwanLab vs WandB

Both tools provide similar functionality:

| Feature | WandB | SwanLab |
|---------|-------|---------|
| Experiment Tracking | ✓ | ✓ |
| Metric Visualization | ✓ | ✓ |
| Model Versioning | ✓ | ✓ |
| Cloud Hosting | ✓ | ✓ |
| Open Source | ✗ | ✓ |
| Chinese Documentation | Limited | Extensive |

## Configuration Options

The following configuration options are available in `GRPOConfig` and `SFTConfig`:

- `swanlab_project`: The SwanLab project to store runs under (similar to `wandb_project`)
- `swanlab_experiment`: The SwanLab experiment name (similar to run name in WandB)

## Examples

### Example 1: Simple GRPO Training with SwanLab

```bash
python src/open_r1/grpo.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name trl-lib/math_reasoning \
    --output_dir ./outputs \
    --report_to swanlab \
    --swanlab_project "my-project"
```

### Example 2: SFT Training with SwanLab

```bash
accelerate launch src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name your/dataset \
    --output_dir ./outputs \
    --report_to swanlab \
    --swanlab_project "sft-project" \
    --swanlab_experiment "experiment-1"
```

### Example 3: Using Both WandB and SwanLab

```bash
python src/open_r1/grpo.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name trl-lib/math_reasoning \
    --output_dir ./outputs \
    --report_to wandb swanlab \
    --wandb_project "wandb-project" \
    --swanlab_project "swanlab-project"
```

## Authentication

For cloud logging, you'll need to authenticate SwanLab:

```bash
swanlab login
```

Or set your API key:

```bash
export SWANLAB_API_KEY="your_api_key"
```

## Viewing Results

After training starts, SwanLab will provide a URL to view your experiment:

```
SwanLab: https://swanlab.cn/@your-username/your-project/runs/run-id
```

You can view:
- Training and validation metrics
- Loss curves
- System metrics (GPU, CPU, memory)
- Hyperparameters
- Model artifacts

## Troubleshooting

### SwanLab not logging

Make sure:
1. SwanLab is installed: `pip install swanlab`
2. You've set `--report_to swanlab` in your training command
3. You're authenticated if using cloud features

### Compatibility Issues

If you encounter issues:
1. Update SwanLab to the latest version: `pip install --upgrade swanlab`
2. Check transformers compatibility: SwanLab works with transformers >= 4.30.0
3. Check the SwanLab documentation: https://docs.swanlab.cn/

## Additional Resources

- [SwanLab Documentation](https://docs.swanlab.cn/)
- [SwanLab GitHub](https://github.com/SwanHubX/SwanLab)
- [Transformers Integration](https://docs.swanlab.cn/en/guide_cloud/integration/)

## Notes

- SwanLab integration is fully compatible with the existing WandB setup
- You can use both SwanLab and WandB simultaneously
- All metrics logged to WandB are also available in SwanLab when both are enabled
- SwanLab supports offline mode for air-gapped environments
