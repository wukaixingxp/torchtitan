**The Qwen2.5 model is still under development.**


## Available features
#### Dense Model
- Qwen2.5 dense model:
    - supports FSDP/HSDP, TP, DDP.
    - Supports AC, torch.compile.

Other model sizes are added to the args, but toml file configs need to be added and tested.

## Download Qwen2.5 tokenizer
```python scripts/download_hf_assets.py --repo_id <hf_repo_name> --assets tokenizer```

eg, for Qwen2.5 0.5B model, the HF repo name is `Qwen/Qwen2.5-0.5B`. For 7B model, the HF repo name is `Qwen/Qwen2.5-7B`.


## To be added
- Modeling
    - CP is not supported currently because of RoPE embedding implementation details.

- Testing
    - Learning rate verifying: verify learning rate and schedule with real training jobs (eg, 3k stps), or find official references.
    - The model should be tested against established performance benchmarks
    - CI integration
