# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is adapted from torchtitan/models/llama3/model/state_dict_adapter.py.

We can use this script to adapt the checkpoint from HF to the format that we can load into the torchtitan model and vice versa.
This can enable us to do a parity test with the HF implementation and make sure that our results are
aligned with the HF implementation.

"""
import re
from typing import Any

from torchtitan.protocols import StateDictAdapter

from .args import Qwen2_5ModelArgs


class Qwen2_5StateDictAdapter(StateDictAdapter):
    def __init__(self, model_args: Qwen2_5ModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # MLP module
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transformer layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert between the HF shape and the torchtitan shape.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                if key not in to_hf_map:
                    continue
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert between the HF shape and the torchtitan shape.
        """

        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in self.from_hf_map:
                    continue
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                if key not in self.from_hf_map:
                    continue
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
                state_dict[new_key] = value

        return state_dict
