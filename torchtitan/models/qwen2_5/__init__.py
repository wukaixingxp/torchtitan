# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_qwen2_5
from .model.args import Qwen2_5ModelArgs
from .model.model import Qwen2_5Model
from .model.state_dict_adapter import Qwen2_5StateDictAdapter

__all__ = [
    "parallelize_qwen2_5",
    "Qwen2_5ModelArgs",
    "Qwen2_5Model",
    "qwen2_5_args",
]

# Adding different variants of the model

qwen2_5_args = {
    "debugmodel": Qwen2_5ModelArgs(
        vocab_size=2048,
        max_seq_len=4096,
        head_dim=128,
        dim=256,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "0.5B": Qwen2_5ModelArgs(
        vocab_size=152064,
        max_seq_len=32768,
        head_dim=64,
        dim=896,
        n_layers=24,
        n_heads=14,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=4864,
        rope_theta=1000000,
        enable_weight_tying=True,
    ),
    "1.5B": Qwen2_5ModelArgs(
        vocab_size=152064,
        max_seq_len=32768,
        head_dim=128,
        dim=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=8960,
        rope_theta=1000000,
        enable_weight_tying=False,
    ),
    "3B": Qwen2_5ModelArgs(
        vocab_size=152064,
        max_seq_len=32768,
        head_dim=128,
        dim=2048,
        n_layers=36,
        n_heads=16,
        n_kv_heads=2,
        qk_norm=True,
        hidden_dim=11008,
        rope_theta=1000000,
        enable_weight_tying=False,
    ),
    "7B": Qwen2_5ModelArgs(
        vocab_size=152064,
        max_seq_len=32768,
        head_dim=128,
        dim=3584,
        n_layers=28,
        n_heads=28,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=18944,
        rope_theta=1000000,
        enable_weight_tying=False,
    ),
    "14B": Qwen2_5ModelArgs(
        vocab_size=152064,
        max_seq_len=32768,
        head_dim=128,
        dim=5120,
        n_layers=48,
        n_heads=40,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=13824,
        rope_theta=1000000,
        enable_weight_tying=False,
    ),
    "32B": Qwen2_5ModelArgs(
        vocab_size=152064,
        max_seq_len=32768,
        head_dim=128,
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=27648,
        rope_theta=1000000,
        enable_weight_tying=False,
    ),
    "72B": Qwen2_5ModelArgs(
        vocab_size=152064,
        max_seq_len=32768,
        head_dim=128,
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=29568,
        rope_theta=1000000,
        enable_weight_tying=False,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Qwen2_5Model,
        model_args=qwen2_5_args,  # Change from dict to Mapping
        parallelize_fn=parallelize_qwen2_5,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen2_5StateDictAdapter,
    )
