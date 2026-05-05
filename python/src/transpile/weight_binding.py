from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class WeightBinding:
    path: str
    kind: str  # "weight" | "embedding"
    source_name: str


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_PARAKEET_GLOBAL_FILENAMES: dict[str, str] = {
    "ctc_head.weight": "ctc_head_weight.weights",
    "ctc_head.bias": "ctc_head_bias.bias",
    "encoder.subsampling.layers.0.weight": "subsampling_conv0_weight.weights",
    "encoder.subsampling.layers.0.bias": "subsampling_conv0_bias.bias",
    "encoder.subsampling.layers.2.weight": "subsampling_depthwise1_weight.weights",
    "encoder.subsampling.layers.2.bias": "subsampling_depthwise1_bias.bias",
    "encoder.subsampling.layers.3.weight": "subsampling_pointwise1_weight.weights",
    "encoder.subsampling.layers.3.bias": "subsampling_pointwise1_bias.bias",
    "encoder.subsampling.layers.5.weight": "subsampling_depthwise2_weight.weights",
    "encoder.subsampling.layers.5.bias": "subsampling_depthwise2_bias.bias",
    "encoder.subsampling.layers.6.weight": "subsampling_pointwise2_weight.weights",
    "encoder.subsampling.layers.6.bias": "subsampling_pointwise2_bias.bias",
    "encoder.subsampling.linear.weight": "subsampling_linear_weight.weights",
    "encoder.subsampling.linear.bias": "subsampling_linear_bias.bias",
    "encoder.pre_encode.conv.0.weight": "subsampling_conv0_weight.weights",
    "encoder.pre_encode.conv.0.bias": "subsampling_conv0_bias.bias",
    "encoder.pre_encode.conv.2.weight": "subsampling_depthwise1_weight.weights",
    "encoder.pre_encode.conv.2.bias": "subsampling_depthwise1_bias.bias",
    "encoder.pre_encode.conv.3.weight": "subsampling_pointwise1_weight.weights",
    "encoder.pre_encode.conv.3.bias": "subsampling_pointwise1_bias.bias",
    "encoder.pre_encode.conv.5.weight": "subsampling_depthwise2_weight.weights",
    "encoder.pre_encode.conv.5.bias": "subsampling_depthwise2_bias.bias",
    "encoder.pre_encode.conv.6.weight": "subsampling_pointwise2_weight.weights",
    "encoder.pre_encode.conv.6.bias": "subsampling_pointwise2_bias.bias",
    "encoder.pre_encode.out.weight": "subsampling_linear_weight.weights",
    "encoder.pre_encode.out.bias": "subsampling_linear_bias.bias",
}

_PARAKEET_LAYER_FILENAMES: dict[str, str] = {
    "feed_forward1.linear1.weight": "ff1_linear1.weights",
    "feed_forward1.linear1.bias": "ff1_linear1.bias",
    "feed_forward1.linear2.weight": "ff1_linear2.weights",
    "feed_forward1.linear2.bias": "ff1_linear2.bias",
    "feed_forward2.linear1.weight": "ff2_linear1.weights",
    "feed_forward2.linear1.bias": "ff2_linear1.bias",
    "feed_forward2.linear2.weight": "ff2_linear2.weights",
    "feed_forward2.linear2.bias": "ff2_linear2.bias",
    "self_attn.q_proj.weight": "self_attn_q.weights",
    "self_attn.q_proj.bias": "self_attn_q.bias",
    "self_attn.k_proj.weight": "self_attn_k.weights",
    "self_attn.k_proj.bias": "self_attn_k.bias",
    "self_attn.v_proj.weight": "self_attn_v.weights",
    "self_attn.v_proj.bias": "self_attn_v.bias",
    "self_attn.o_proj.weight": "self_attn_output.weights",
    "self_attn.o_proj.bias": "self_attn_output.bias",
    "self_attn.relative_k_proj.weight": "self_attn_relative_k.weights",
    "self_attn.linear_q.weight": "self_attn_q.weights",
    "self_attn.linear_q.bias": "self_attn_q.bias",
    "self_attn.linear_k.weight": "self_attn_k.weights",
    "self_attn.linear_k.bias": "self_attn_k.bias",
    "self_attn.linear_v.weight": "self_attn_v.weights",
    "self_attn.linear_v.bias": "self_attn_v.bias",
    "self_attn.linear_out.weight": "self_attn_output.weights",
    "self_attn.linear_out.bias": "self_attn_output.bias",
    "self_attn.linear_pos.weight": "self_attn_relative_k.weights",
    "self_attn.bias_u": "self_attn_bias_u.weights",
    "self_attn.bias_v": "self_attn_bias_v.weights",
    "self_attn.pos_bias_u": "self_attn_bias_u.weights",
    "self_attn.pos_bias_v": "self_attn_bias_v.weights",
    "conv.pointwise_conv1.weight": "conv_pointwise1.weights",
    "conv.pointwise_conv1.bias": "conv_pointwise1.bias",
    "conv.depthwise_conv.weight": "conv_depthwise.weights",
    "conv.depthwise_conv.bias": "conv_depthwise.bias",
    "conv.pointwise_conv2.weight": "conv_pointwise2.weights",
    "conv.pointwise_conv2.bias": "conv_pointwise2.bias",
    "conv.norm.weight": "conv_batchnorm_weight.weights",
    "conv.norm.bias": "conv_batchnorm_bias.bias",
    "conv.norm.running_mean": "conv_batchnorm_running_mean.weights",
    "conv.norm.running_var": "conv_batchnorm_running_var.weights",
    "conv.batch_norm.weight": "conv_batchnorm_weight.weights",
    "conv.batch_norm.bias": "conv_batchnorm_bias.bias",
    "conv.batch_norm.running_mean": "conv_batchnorm_running_mean.weights",
    "conv.batch_norm.running_var": "conv_batchnorm_running_var.weights",
    "norm_feed_forward1.weight": "norm_ff1.weights",
    "norm_feed_forward1.bias": "norm_ff1.bias",
    "norm_self_att.weight": "norm_self_attn.weights",
    "norm_self_att.bias": "norm_self_attn.bias",
    "norm_conv.weight": "norm_conv.weights",
    "norm_conv.bias": "norm_conv.bias",
    "norm_feed_forward2.weight": "norm_ff2.weights",
    "norm_feed_forward2.bias": "norm_ff2.bias",
    "norm_out.weight": "norm_out.weights",
    "norm_out.bias": "norm_out.bias",
}


def _candidate_model_dir_names(model_name_or_path: str) -> list[str]:
    candidates: list[str] = []
    raw = model_name_or_path.strip()
    if not raw:
        return candidates

    def _add(name: str) -> None:
        name = name.strip().lower()
        if name and name not in candidates:
            candidates.append(name)

    _add(raw.split("/")[-1])

    path = Path(raw)
    parts = path.parts
    for part in parts:
        if part.startswith("models--"):
            _add(part[len("models--"):].split("--")[-1])
            break

    return candidates


def _default_weights_dir_for_model_name(model_name_or_path: str) -> str | None:
    if not model_name_or_path:
        return None
    for model_dir_name in _candidate_model_dir_names(model_name_or_path):
        candidate = _PROJECT_ROOT / "weights" / model_dir_name
        if candidate.exists():
            return str(candidate)
    return None


def _normalized_source_candidates(source_name: str) -> list[str]:
    candidates: list[str] = []

    def _add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    raw = source_name.strip()
    _add(raw)

    for prefix in ("p_", "b_", "c_"):
        if raw.startswith(prefix):
            _add(raw[len(prefix):])

    for prefix in ("adapter.model.", "adapter.module.", "module."):
        if raw.startswith(prefix):
            _add(raw[len(prefix):])

    stripped = candidates[-1]

    for prefix in (
        "module.backbone.",
        "adapter.backbone.",
        "adapter.model.model.",
        "adapter.model.language_model.",
    ):
        if stripped.startswith(prefix):
            tail = stripped[len(prefix):]
            _add(f"model.{tail}")
            _add(f"model.language_model.{tail}")
            if tail.startswith("layers."):
                _add(f"model.{tail}")
                _add(f"model.language_model.{tail}")
            break

    if stripped == "adapter.model.lm_head.weight":
        _add("lm_head.weight")

    layer_match = re.match(
        r"^(?:(?:module|adapter)_)?backbone_layers_slice_none__\d+__none____modules__(\d+)___(.+)$",
        stripped,
    )
    if layer_match:
        layer_index = int(layer_match.group(1))
        tail = layer_match.group(2)
        tail_map = {
            "self_attn_q_proj_weight": "self_attn.q_proj.weight",
            "self_attn_k_proj_weight": "self_attn.k_proj.weight",
            "self_attn_v_proj_weight": "self_attn.v_proj.weight",
            "self_attn_o_proj_weight": "self_attn.o_proj.weight",
            "self_attn_q_norm_weight": "self_attn.q_norm.weight",
            "self_attn_k_norm_weight": "self_attn.k_norm.weight",
            "mlp_gate_proj_weight": "mlp.gate_proj.weight",
            "mlp_up_proj_weight": "mlp.up_proj.weight",
            "mlp_down_proj_weight": "mlp.down_proj.weight",
            "input_layernorm_weight": "input_layernorm.weight",
            "post_attention_layernorm_weight": "post_attention_layernorm.weight",
            "linear_attn_in_proj_qkv_weight": "linear_attn.in_proj_qkv.weight",
            "linear_attn_conv1d_weight": "linear_attn.conv1d.weight",
            "linear_attn_norm_weight": "linear_attn.norm.weight",
            "linear_attn_dt_bias": "linear_attn.dt_bias",
            "linear_attn_A_log": "linear_attn.A_log",
        }
        dotted_tail = tail_map.get(tail)
        if dotted_tail is not None:
            _add(f"model.layers.{layer_index}.{dotted_tail}")
            _add(f"model.language_model.layers.{layer_index}.{dotted_tail}")

    embed_map = {
        "module_backbone_embed_tokens_weight": [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ],
        "adapter_backbone_embed_tokens_weight": [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ],
        "module_backbone_embed_tokens_per_layer_weight": [
            "model.embed_tokens_per_layer.weight",
            "model.language_model.embed_tokens_per_layer.weight",
        ],
        "adapter_backbone_embed_tokens_per_layer_weight": [
            "model.embed_tokens_per_layer.weight",
            "model.language_model.embed_tokens_per_layer.weight",
        ],
        "module_backbone_per_layer_model_projection_weight": [
            "model.per_layer_model_projection.weight",
            "model.language_model.per_layer_model_projection.weight",
        ],
        "adapter_backbone_per_layer_model_projection_weight": [
            "model.per_layer_model_projection.weight",
            "model.language_model.per_layer_model_projection.weight",
        ],
        "module_backbone_norm_weight": [
            "model.norm.weight",
            "model.language_model.norm.weight",
        ],
        "adapter_backbone_norm_weight": [
            "model.norm.weight",
            "model.language_model.norm.weight",
        ],
        "module_model_lm_head_weight": [
            "lm_head.weight",
        ],
        "adapter_model_lm_head_weight": [
            "lm_head.weight",
        ],
    }
    for key, mapped in embed_map.items():
        if stripped == key:
            for item in mapped:
                _add(item)

    return candidates


def _fallback_filename_candidates(source_name: str) -> list[str]:
    filenames: list[str] = []

    def _add(filename: str | None) -> None:
        if filename and filename not in filenames:
            filenames.append(filename)

    for candidate in _normalized_source_candidates(source_name):
        _add(_PARAKEET_GLOBAL_FILENAMES.get(candidate))

        layer_match = re.match(r"^encoder\.layers\.(\d+)\.(.+)$", candidate)
        if layer_match:
            layer_index = int(layer_match.group(1))
            suffix = layer_match.group(2)
            mapped = _PARAKEET_LAYER_FILENAMES.get(suffix)
            if mapped is not None:
                _add(f"layer_{layer_index}_{mapped}")

    return filenames


def resolve_transpile_weights_dir(graph_meta: dict[str, object]) -> str | None:
    explicit = graph_meta.get("weights_dir")
    if isinstance(explicit, str) and explicit:
        return explicit

    family = str(graph_meta.get("adapter_family", "")).upper()
    family_env = f"CACTUS_TRANSPILER_WEIGHTS_DIR_{family}"
    if family and family_env in os.environ and os.environ[family_env]:
        return os.environ[family_env]

    generic = os.environ.get("CACTUS_TRANSPILER_WEIGHTS_DIR")
    if generic:
        return generic

    model_name_or_path = graph_meta.get("model_name_or_path")
    if isinstance(model_name_or_path, str) and model_name_or_path:
        default_dir = _default_weights_dir_for_model_name(model_name_or_path)
        if default_dir:
            return default_dir
    return None


def resolve_weight_binding(*, weights_dir: str | None, source_name: str) -> WeightBinding | None:
    if not weights_dir:
        return None
    root = Path(weights_dir)
    if not root.exists():
        return None
    manifest_path = root / "weights_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if not isinstance(manifest, dict):
        return None
    for candidate_name in _normalized_source_candidates(source_name):
        entry = manifest.get(candidate_name)
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        kind = entry.get("kind", "weight")
        if not isinstance(filename, str) or not isinstance(kind, str):
            continue
        candidate = root / filename
        if not candidate.exists():
            continue
        return WeightBinding(path=str(candidate), kind=kind, source_name=candidate_name)

    for filename in _fallback_filename_candidates(source_name):
        candidate = root / filename
        if candidate.exists():
            return WeightBinding(path=str(candidate), kind="weight", source_name=source_name)
    return None
