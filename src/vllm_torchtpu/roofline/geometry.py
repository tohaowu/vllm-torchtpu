# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model geometry and 10-family layer registry from HuggingFace config.json."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .constants import LINEAR_ATTENTION_LAYER_TOKENS, VALID_LAYER_TOKENS


def _as_mapping(value: object) -> Mapping[str, object]:
    """Return value as a mapping when compatible with HF config JSON, else empty."""
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object] | None:
    """Return value as a non-string sequence when valid, else None."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return None


def _as_int(value: object, default: int = 0) -> int:
    """Convert JSON scalar number or numeric string to int cleanly."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


@dataclass(frozen=True)
class ModelGeometry:
    """Architecture dimensions shared across attention, recurrence, and FFN families."""

    hidden: int
    num_layers: int
    layer_types: tuple[str, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    attn_output_gate: bool
    linear_k_heads: int
    linear_v_heads: int
    linear_k_dim: int
    linear_v_dim: int
    conv_kernel: int
    num_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    vocab: int
    mtp_layers: int
    dense_inter: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    sliding_window: int = 0
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    ple_dim: int = 0

    @staticmethod
    def from_hf_config(cfg: Mapping[str, object]) -> ModelGeometry:
        """Build ModelGeometry from a raw or nested HF config mapping."""
        return from_hf_config(cfg)


def _is_kda_config(tc: Mapping[str, object], cfg: Mapping[str, object]) -> bool:
    """Return True when config declares KDA-style linear attention capabilities."""
    lin_cfg = _as_mapping(tc.get("linear_attn_config"))
    if lin_cfg.get("short_conv_kernel_size") or lin_cfg.get("kda"):
        return True
    model_type = str(tc.get("model_type") or cfg.get("model_type") or "").lower()
    return model_type in {"kimi_linear", "kda"}


def _resolve_layer_types(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    num_layers: int,
    specs: tuple[int, int, int],
) -> tuple[str, ...]:
    """Resolve sequence of layer type strings for each transformer layer."""
    kv_lora_rank, index_topk, sliding_window = specs
    raw_types = _as_sequence(tc.get("layer_types"))
    if raw_types is not None:
        if len(raw_types) != num_layers:
            msg = (
                f"layer_types length {len(raw_types)} != num_hidden_layers {num_layers}"
            )
            raise ValueError(msg)
        layer_types = tuple(str(x) for x in raw_types)
        for token in layer_types:
            if token not in VALID_LAYER_TOKENS:
                supported = sorted(VALID_LAYER_TOKENS)
                msg = f"Unknown layer_type {token!r}; supported tokens: {supported}"
                raise ValueError(msg)
        return layer_types

    interval = _as_int(tc.get("full_attention_interval"), 0)
    if interval > 0:
        att_tok = (
            "dsa"
            if (kv_lora_rank > 0 and index_topk > 0)
            else ("mla" if kv_lora_rank > 0 else "full_attention")
        )
        lin_tok = "kda" if _is_kda_config(tc, cfg) else "linear_attention"
        return tuple(
            att_tok if (i + 1) % interval == 0 else lin_tok for i in range(num_layers)
        )

    sw_pattern = _as_int(
        tc.get("sliding_window_pattern") or tc.get("_sliding_window_pattern"), 0
    )
    if sliding_window > 0:
        if sw_pattern > 1:
            return tuple(
                "full_attention" if (i + 1) % sw_pattern == 0 else "sliding_attention"
                for i in range(num_layers)
            )
        return ("sliding_attention",) * num_layers

    if kv_lora_rank > 0:
        default_mla = "dsa" if index_topk > 0 else "mla"
        return (default_mla,) * num_layers

    return ("full_attention",) * num_layers


def _resolve_moe_dimensions(
    tc: Mapping[str, object],
) -> tuple[int, int, int, int, int]:
    """Resolve (num_experts, top_k, moe_inter, num_shared_experts, shared_inter)."""
    num_experts = _as_int(
        tc.get("num_experts")
        or tc.get("n_routed_experts")
        or tc.get("num_local_experts")
    )
    dense_inter = _as_int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size")
    )
    moe_inter = _as_int(tc.get("moe_intermediate_size") or tc.get("latent_moe_dim"))
    if num_experts > 0 and moe_inter == 0 and dense_inter > 0:
        moe_inter = dense_inter

    num_shared = _as_int(tc.get("n_shared_experts") or tc.get("num_shared_experts"))
    shared_inter = _as_int(tc.get("shared_expert_intermediate_size"))
    if shared_inter == 0 and num_shared > 0 and moe_inter > 0:
        shared_inter = num_shared * moe_inter

    top_k = _as_int(
        tc.get("num_experts_per_tok")
        or tc.get("num_selected_experts")
        or tc.get("top_k")
    )
    return num_experts, top_k, moe_inter, num_shared, shared_inter


def _resolve_linear_attention_dimensions(
    tc: Mapping[str, object],
    layer_types: tuple[str, ...],
) -> tuple[int, int, int, int, int]:
    """Resolve (lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel)."""
    lin_cfg = _as_mapping(tc.get("linear_attn_config"))
    lin_k_heads = _as_int(
        tc.get("linear_num_key_heads")
        or lin_cfg.get("num_key_heads")
        or lin_cfg.get("num_heads")
    )
    lin_v_heads = _as_int(
        tc.get("linear_num_value_heads")
        or lin_cfg.get("num_value_heads")
        or lin_cfg.get("num_heads")
    )
    lin_k_dim = _as_int(
        tc.get("linear_key_head_dim")
        or lin_cfg.get("key_head_dim")
        or lin_cfg.get("head_dim")
    )
    lin_v_dim = _as_int(
        tc.get("linear_value_head_dim")
        or lin_cfg.get("value_head_dim")
        or lin_cfg.get("head_dim")
    )

    has_linear = any(t in LINEAR_ATTENTION_LAYER_TOKENS for t in layer_types)
    raw_conv = tc.get("linear_conv_kernel_dim") or lin_cfg.get("short_conv_kernel_size")
    if has_linear:
        if raw_conv is None:
            msg = (
                "Model with linear-attention layers missing required field: "
                "'linear_conv_kernel_dim'"
            )
            raise ValueError(msg)
        conv_kernel = _as_int(raw_conv)
    else:
        conv_kernel = 0
    return lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel


def _resolve_mtp_layers(tc: Mapping[str, object], cfg: Mapping[str, object]) -> int:
    """Resolve number of Multi-Token Prediction (MTP) layers."""
    mtp_raw = tc.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = cfg.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = tc.get(
            "num_nextn_predict_layers", cfg.get("num_nextn_predict_layers", 0)
        )
    return _as_int(mtp_raw)


def from_hf_config(cfg: Mapping[str, object]) -> ModelGeometry:
    """Build ModelGeometry from a raw or nested HF config mapping."""
    sub = cfg.get("text_config")
    tc: Mapping[str, object] = _as_mapping(sub) if sub is not None else cfg
    if not tc:
        tc = cfg
    hidden = _as_int(tc["hidden_size"])
    num_layers = _as_int(tc["num_hidden_layers"])
    num_heads = _as_int(tc["num_attention_heads"])
    raw_kv_heads = tc.get("num_key_value_heads")
    num_kv_heads = num_heads if raw_kv_heads is None else _as_int(raw_kv_heads)
    head_dim = _as_int(tc.get("head_dim")) or (hidden // num_heads)

    kv_lora_rank = _as_int(tc.get("kv_lora_rank"))
    q_lora_rank = _as_int(tc.get("q_lora_rank"))
    qk_nope_head_dim = _as_int(tc.get("qk_nope_head_dim"))
    qk_rope_head_dim = _as_int(tc.get("qk_rope_head_dim"))
    v_head_dim = _as_int(tc.get("v_head_dim"))
    index_n_heads = _as_int(tc.get("index_n_heads"))
    index_head_dim = _as_int(tc.get("index_head_dim"))
    index_topk = _as_int(tc.get("index_topk"))
    sliding_window = _as_int(tc.get("sliding_window"))

    layer_types = _resolve_layer_types(
        tc, cfg, num_layers, (kv_lora_rank, index_topk, sliding_window)
    )
    dense_inter = _as_int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size")
    )
    num_experts, top_k, moe_inter, num_shared, shared_inter = _resolve_moe_dimensions(
        tc
    )
    lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel = (
        _resolve_linear_attention_dimensions(tc, layer_types)
    )
    mtp_layers = _resolve_mtp_layers(tc, cfg)
    ple_dim = _as_int(tc.get("per_layer_input_dim") or tc.get("ple_dim"))
    first_k = _as_int(
        tc.get("first_k_dense_replace", cfg.get("first_k_dense_replace", 0))
    )

    return ModelGeometry(
        hidden=hidden,
        num_layers=num_layers,
        layer_types=layer_types,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_output_gate=bool(tc.get("attn_output_gate", False)),
        linear_k_heads=lin_k_heads,
        linear_v_heads=lin_v_heads,
        linear_k_dim=lin_k_dim,
        linear_v_dim=lin_v_dim,
        conv_kernel=conv_kernel,
        num_experts=num_experts,
        top_k=top_k,
        moe_inter=moe_inter,
        shared_inter=shared_inter,
        vocab=_as_int(tc.get("vocab_size", cfg.get("vocab_size", 0))),
        mtp_layers=mtp_layers,
        dense_inter=dense_inter,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        sliding_window=sliding_window,
        num_shared_experts=num_shared,
        first_k_dense_replace=first_k,
        ple_dim=ple_dim,
    )
