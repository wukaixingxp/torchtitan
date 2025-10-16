# Qwen2.5 Model

## Overview
This directory contains the Qwen2.5 model implementation for torchtitan, adapted from the Qwen3 implementation.

## Available features
#### Dense Model
- Qwen2.5 dense model:
    - supports FSDP/HSDP, TP, DDP.
    - Supports AC, torch.compile.

## Model Sizes
The following model sizes are supported:
- debugmodel (256 dim, 8 layers) - for testing
- 0.5B (896 dim, 24 layers)
- 1.5B (1536 dim, 28 layers)
- 3B (2048 dim, 36 layers)
- 7B (3584 dim, 28 layers)
- 14B (5120 dim, 48 layers)
- 32B (5120 dim, 64 layers)
- 72B (8192 dim, 80 layers)

## Download Qwen2.5 tokenizer
```bash
python scripts/download_hf_assets.py --repo_id <hf_repo_name> --assets tokenizer
```

For example:
- For Qwen2.5 0.5B model: `--repo_id Qwen/Qwen2.5-0.5B`
- For Qwen2.5 1.5B model: `--repo_id Qwen/Qwen2.5-1.5B`
- For Qwen2.5 7B model: `--repo_id Qwen/Qwen2.5-7B`

## Key Differences from Qwen3
- No QK normalization (qk_norm=False)
- Attention projections (q_proj, k_proj, v_proj) have biases
- No MoE support in Qwen2.5

## Training
To train Qwen2.5 models, use the training configs in the `train_configs/` directory:
```bash
CONFIG_FILE="./torchtitan/models/qwen2_5/train_configs/qwen2_5_0.5b.toml"
torchrun train.py --job.config_file $CONFIG_FILE
```

## To be added
- Modeling
    - CP is not supported currently because of RoPE embedding implementation details.
    - MoE variants (available in Qwen2.5 MoE models)

- Testing
    - Learning rate verifying: verify learning rate and schedule with real training jobs
    - The model should be tested against established performance benchmarks
    - CI integration
