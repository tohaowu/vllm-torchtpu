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
"""Model geometry, support guards, and 10-family layer registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .constants import (
    BLOCK_PARALLEL_ENABLED,
    HYBRID_PATTERN_MAP,
    LAYER_FAMILIES,
    LINEAR_ATTENTION_LAYER_TOKENS,
    MULTIMODAL_KEYS,
    UNSUPPORTED_NON_CAUSAL_TYPES,
    UNSUPPORTED_SSM_MODEL_TYPES,
    VALID_LAYER_TOKENS,
)

__all__ = [
    "LAYER_FAMILIES",
    "ModelGeometry",
    "UnsupportedArchitectureError",
    "from_hf_config",
    "supports",
]


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


def _as_float(value: object, default: float = 0.0) -> float:
    """Convert JSON scalar number or numeric string to float cleanly."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


class UnsupportedArchitectureError(ValueError):
    """Raised when a HuggingFace config describes an unmodeled architecture."""


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
    compress_ratios: tuple[float, ...] = ()
    block_parallel: int = 1
    mtp_inter: int = 0
    n_moe_layers: int | None = None
    n_dense_layers: int | None = None
    global_head_dim: int = 0
    ffn_kinds: tuple[str, ...] = ()

    @staticmethod
    def from_hf_config(cfg: Mapping[str, object]) -> ModelGeometry:
        """Build ModelGeometry from a raw or nested HF config mapping."""
        return from_hf_config(cfg)

    def ffn_layer_counts(self) -> tuple[int, int]:
        """Return (n_moe_ffn, n_dense_ffn) counted from layer_ffn_kinds."""
        kinds = layer_ffn_kinds(self)
        return (kinds.count("moe"), kinds.count("dense"))


def layer_ffn_kinds(geom: ModelGeometry) -> tuple[str, ...]:
    """Return per-layer FFN kinds ('dense', 'moe', or 'none') for model geometry."""
    if geom.ffn_kinds and len(geom.ffn_kinds) == geom.num_layers:
        return geom.ffn_kinds
    if geom.num_experts > 0:
        first_k = min(geom.num_layers, geom.first_k_dense_replace)
        return ("dense",) * first_k + ("moe",) * (geom.num_layers - first_k)
    return ("dense",) * geom.num_layers


def _validate_text_config(cfg: Mapping[str, object]) -> Mapping[str, object]:
    """Validate and extract text_config mapping from top-level config."""
    if not cfg:
        msg = "Empty HuggingFace config mapping"
        raise UnsupportedArchitectureError(msg)
    has_text_key = "text_config" in cfg
    sub = _as_mapping(cfg.get("text_config"))
    if has_text_key and not sub:
        msg = "Multimodal config has non-mapping text_config"
        raise UnsupportedArchitectureError(msg)
    tc: Mapping[str, object] = sub if sub else cfg
    is_multimodal = has_text_key or any(k in cfg for k in MULTIMODAL_KEYS)
    if is_multimodal and not tc.get("hidden_size"):
        msg = "Multimodal config missing extractable text_config.hidden_size"
        raise UnsupportedArchitectureError(msg)
    return tc


def _has_gdn_kda_heads(tc: Mapping[str, object]) -> bool:
    """Return whether text config declares linear or recurrent attention heads."""
    lin_map = _as_mapping(tc.get("linear_attn_config"))
    return bool(
        tc.get("linear_num_key_heads")
        or tc.get("linear_num_value_heads")
        or lin_map.get("num_key_heads")
        or lin_map.get("num_value_heads")
        or lin_map.get("num_heads")
        or tc.get("mamba_num_heads")
        or tc.get("attn_type_list")
        or tc.get("hybrid_override_pattern")
    )


def supports(cfg: Mapping[str, object]) -> None:
    """Validate that cfg maps onto modeled attention, recurrence, and FFN families."""
    tc = _validate_text_config(cfg)
    mtype = str(tc.get("model_type") or cfg.get("model_type") or "").strip().lower()
    if mtype in UNSUPPORTED_NON_CAUSAL_TYPES:
        msg = f"Unsupported non-causal-LM model_type {mtype!r}"
        raise UnsupportedArchitectureError(msg)
    has_gdn = _has_gdn_kda_heads(tc)
    if mtype in UNSUPPORTED_SSM_MODEL_TYPES and not has_gdn:
        msg = f"Unsupported state-space model_type {mtype!r} without GDN/KDA head keys"
        raise UnsupportedArchitectureError(msg)
    if ("ssm_cfg" in tc or "state_size" in tc) and not (
        tc.get("num_attention_heads") or has_gdn
    ):
        msg = "Unmodeled state-space config missing attention and GDN/KDA head keys"
        raise UnsupportedArchitectureError(msg)
    raw_types = _as_sequence(tc.get("layer_types"))
    if raw_types is None:
        return
    for token in raw_types:
        tok_s = str(token)
        if tok_s not in VALID_LAYER_TOKENS:
            msg = (
                f"Unknown layer_type {tok_s!r}; supported tokens: "
                f"{sorted(VALID_LAYER_TOKENS)}"
            )
            raise UnsupportedArchitectureError(msg)
        if tok_s == "mamba" and not has_gdn:
            msg = "Layer token 'mamba' requires GDN/KDA head dimensions"
            raise UnsupportedArchitectureError(msg)


def _is_kda_config(tc: Mapping[str, object], cfg: Mapping[str, object]) -> bool:
    """Return True when config declares KDA-style linear attention capabilities."""
    lin_cfg = _as_mapping(tc.get("linear_attn_config"))
    if lin_cfg.get("short_conv_kernel_size") or lin_cfg.get("kda"):
        return True
    model_type = str(tc.get("model_type") or cfg.get("model_type") or "").lower()
    if model_type.startswith("kimi") or model_type == "kda":
        return True
    raw_archs = _as_sequence(cfg.get("architectures") or tc.get("architectures")) or ()
    return any("kimi" in str(arch).lower() for arch in raw_archs)


def _infer_pattern_layer_types(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    num_layers: int,
    specs: tuple[int, int, int],
) -> tuple[str, ...]:
    """Infer layer types from interval, attention list, or override pattern."""
    kv_lora_rank, index_topk, sliding_window = specs
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
    attn_list = _as_sequence(tc.get("attn_type_list"))
    if attn_list is not None and len(attn_list) == num_layers:
        return tuple(
            "full_attention" if _as_int(v, 0) == 1 else "linear_attention"
            for v in attn_list
        )
    hyb_pat = tc.get("hybrid_override_pattern")
    if isinstance(hyb_pat, str) and len(hyb_pat.strip()) == num_layers:
        pat_s = hyb_pat.strip()
        if any(ch not in HYBRID_PATTERN_MAP for ch in pat_s):
            msg = f"Unsupported hybrid_override_pattern tokens in {pat_s!r}"
            raise UnsupportedArchitectureError(msg)
        return tuple(HYBRID_PATTERN_MAP[ch] for ch in pat_s)
    sw_pat = _as_int(
        tc.get("sliding_window_pattern") or tc.get("_sliding_window_pattern"), 0
    )
    if sliding_window > 0:
        default_layers = (
            tuple(
                "full_attention" if (i + 1) % sw_pat == 0 else "sliding_attention"
                for i in range(num_layers)
            )
            if sw_pat > 1
            else ("sliding_attention",) * num_layers
        )
    elif kv_lora_rank > 0:
        default_mla = "dsa" if index_topk > 0 else "mla"
        default_layers = (default_mla,) * num_layers
    else:
        default_layers = ("full_attention",) * num_layers
    return default_layers


def _resolve_layer_types(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    num_layers: int,
    specs: tuple[int, int, int],
) -> tuple[str, ...]:
    """Resolve sequence of layer type strings for each transformer layer."""
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
    return _infer_pattern_layer_types(tc, cfg, num_layers, specs)


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


def _resolve_ffn_kinds(
    tc: Mapping[str, object],
    num_layers: int,
    num_experts: int,
) -> tuple[str, ...]:
    """Resolve (moe, dense, none) kind for each layer's feed-forward network."""
    hyb_s = str(tc.get("hybrid_override_pattern") or "").strip()
    if hyb_s and len(hyb_s) == num_layers:
        return tuple(
            "moe" if ch == "E" else ("dense" if ch == "-" else "none") for ch in hyb_s
        )
    raw_moe_layers = _as_sequence(tc.get("moe_layers"))
    if raw_moe_layers is not None:
        moe_set = {_as_int(x) for x in raw_moe_layers}
        return tuple("moe" if i in moe_set else "dense" for i in range(num_layers))
    interleave_step = _as_int(tc.get("interleave_moe_layer_step"), 0)
    if num_experts > 0 and interleave_step > 1:
        return tuple(
            "moe" if (i + 1) % interleave_step == 0 else "dense"
            for i in range(num_layers)
        )
    first_k_dense = _as_int(tc.get("first_k_dense_replace"), 0)
    if num_experts > 0:
        return tuple("dense" if i < first_k_dense else "moe" for i in range(num_layers))
    return ("dense",) * num_layers


def _resolve_linear_attention_dimensions(
    tc: Mapping[str, object],
    layer_types: tuple[str, ...],
    attn_dims: tuple[int, int, int],
) -> tuple[int, int, int, int, int]:
    """Resolve (lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel)."""
    num_heads, num_kv_heads, head_dim = attn_dims
    lin_cfg = _as_mapping(tc.get("linear_attn_config"))
    has_attn_list = tc.get("attn_type_list") is not None
    lin_k_heads = _as_int(
        tc.get("linear_num_key_heads")
        or lin_cfg.get("num_key_heads")
        or lin_cfg.get("num_heads")
        or tc.get("mamba_num_heads")
        or (num_kv_heads if has_attn_list else 0)
    )
    lin_v_heads = _as_int(
        tc.get("linear_num_value_heads")
        or lin_cfg.get("num_value_heads")
        or lin_cfg.get("num_heads")
        or tc.get("mamba_num_heads")
        or (num_heads if has_attn_list else 0)
    )
    lin_k_dim = _as_int(
        tc.get("linear_key_head_dim")
        or lin_cfg.get("key_head_dim")
        or lin_cfg.get("head_dim")
        or tc.get("mamba_head_dim")
        or (head_dim if has_attn_list else 0)
    )
    lin_v_dim = _as_int(
        tc.get("linear_value_head_dim")
        or lin_cfg.get("value_head_dim")
        or lin_cfg.get("head_dim")
        or tc.get("mamba_head_dim")
        or (head_dim if has_attn_list else 0)
    )

    has_linear = any(t in LINEAR_ATTENTION_LAYER_TOKENS for t in layer_types)
    raw_conv = (
        tc.get("linear_conv_kernel_dim")
        or lin_cfg.get("short_conv_kernel_size")
        or lin_cfg.get("conv_kernel_size")
        or lin_cfg.get("linear_conv_kernel_dim")
        or tc.get("conv_kernel")
        or tc.get("d_conv")
    )
    if has_linear:
        if raw_conv is None:
            if has_attn_list:
                return lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, 0
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


def _resolve_compress_ratios(
    tc: Mapping[str, object],
    cfg: Mapping[str, object],
    num_layers: int,
) -> tuple[float, ...]:
    """Resolve per-layer KV compression or retention ratios."""
    raw_cr = _as_sequence(tc.get("compress_ratios", cfg.get("compress_ratios")))
    if raw_cr is None or not raw_cr:
        return ()
    if len(raw_cr) != num_layers:
        msg = f"compress_ratios length {len(raw_cr)} != num_hidden_layers {num_layers}"
        raise ValueError(msg)
    compress_ratios = tuple(_as_float(x) for x in raw_cr)
    if any(r <= 0.0 for r in compress_ratios):
        msg = f"compress_ratios entries must be > 0, got {compress_ratios}"
        raise ValueError(msg)
    return compress_ratios


def _resolve_block_parallel(tc: Mapping[str, object], cfg: Mapping[str, object]) -> int:
    """Resolve block parallelism factor (1 or 2)."""
    raw_bp = tc.get("block_parallel", cfg.get("block_parallel"))
    if raw_bp is None:
        raw_bp = tc.get("parallel_attn_mlp", cfg.get("parallel_attn_mlp", 1))
    if isinstance(raw_bp, bool):
        return BLOCK_PARALLEL_ENABLED if raw_bp else 1
    return max(1, _as_int(raw_bp, 1))


def from_hf_config(cfg: Mapping[str, object]) -> ModelGeometry:
    """Build ModelGeometry from a raw or nested HF config mapping."""
    supports(cfg)
    sub = _as_mapping(cfg.get("text_config"))
    tc: Mapping[str, object] = sub if sub else cfg
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
    global_head_dim = _as_int(tc.get("global_head_dim"))

    layer_types = _resolve_layer_types(
        tc, cfg, num_layers, (kv_lora_rank, index_topk, sliding_window)
    )
    dense_inter = _as_int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size")
    )
    num_experts, top_k, moe_inter, num_shared, shared_inter = _resolve_moe_dimensions(
        tc
    )
    ffn_kinds = _resolve_ffn_kinds(tc, num_layers, num_experts)
    lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel = (
        _resolve_linear_attention_dimensions(
            tc, layer_types, (num_heads, num_kv_heads, head_dim)
        )
    )
    mtp_layers = _resolve_mtp_layers(tc, cfg)
    ple_dim = _as_int(tc.get("per_layer_input_dim") or tc.get("ple_dim"))
    first_k = _as_int(
        tc.get("first_k_dense_replace", cfg.get("first_k_dense_replace", 0))
    )
    compress_ratios = _resolve_compress_ratios(tc, cfg, num_layers)
    block_parallel = _resolve_block_parallel(tc, cfg)
    mtp_inter = _as_int(
        tc.get("mtp_intermediate_size", cfg.get("mtp_intermediate_size", 0))
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
        compress_ratios=compress_ratios,
        block_parallel=block_parallel,
        mtp_inter=mtp_inter,
        n_moe_layers=ffn_kinds.count("moe"),
        n_dense_layers=ffn_kinds.count("dense"),
        global_head_dim=global_head_dim,
        ffn_kinds=ffn_kinds,
    )
