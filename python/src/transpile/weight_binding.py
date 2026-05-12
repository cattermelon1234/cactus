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
    "decoder.prediction.embed.weight": "tdt_predictor_embed.weights",
    "joint.enc.weight": "tdt_joint_enc.weights",
    "joint.enc.bias": "tdt_joint_enc.bias",
    "joint.pred.weight": "tdt_joint_pred.weights",
    "joint.pred.bias": "tdt_joint_pred.bias",
    "joint.joint_net.2.weight": "tdt_joint_out.weights",
    "joint.joint_net.2.bias": "tdt_joint_out.bias",
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

_GEMMA3N_VISION_TOWER_PREFIX = "model.vision_tower.timm_model."
_GEMMA3N_AUDIO_TOWER_PREFIX = "model.audio_tower."
_GEMMA4_VISION_TOWER_PREFIX = "model.vision_tower."
_GEMMA4_AUDIO_TOWER_PREFIX = "model.audio_tower."

_GEMMA_GLOBAL_FILENAMES: dict[str, tuple[str, str]] = {
    "model.embed_tokens.weight": ("token_embeddings.weights", "embedding"),
    "model.language_model.embed_tokens.weight": ("token_embeddings.weights", "embedding"),
    "model.language_model.embed_tokens_per_layer.weight": ("embed_tokens_per_layer.weights", "embedding"),
    "model.language_model.per_layer_model_projection.weight": ("per_layer_model_proj.weights", "weight"),
    "model.language_model.per_layer_projection_norm.weight": ("per_layer_proj_norm.weights", "weight"),
    "model.embed_vision.embedding.weight": ("embed_vision_embedding.weights", "embedding"),
    "model.embed_vision.embedding_projection.weight": ("embed_vision_proj.weights", "weight"),
    "model.embed_vision.soft_embedding_norm.weight": ("embed_vision_soft_norm.weights", "weight"),
    "model.embed_vision.hard_embedding_norm.weight": ("embed_vision_hard_norm.weights", "weight"),
    "model.embed_audio.embedding.weight": ("embed_audio_embedding.weights", "embedding"),
    "model.embed_audio.embedding_projection.weight": ("embed_audio_proj.weights", "weight"),
    "model.embed_audio.soft_embedding_norm.weight": ("embed_audio_soft_norm.weights", "weight"),
    "model.embed_audio.hard_embedding_norm.weight": ("embed_audio_hard_norm.weights", "weight"),
}

_GEMMA_DECODER_LAYER_FILENAMES: dict[str, tuple[str, str]] = {
    "layer_scalar": ("layer_scalar.weights", "weight"),
    "self_attn.q_proj.weight": ("attn_q.weights", "weight"),
    "self_attn.k_proj.weight": ("attn_k.weights", "weight"),
    "self_attn.v_proj.weight": ("attn_v.weights", "weight"),
    "self_attn.o_proj.weight": ("attn_output.weights", "weight"),
    "self_attn.q_norm.weight": ("attn_q_norm.weights", "weight"),
    "self_attn.k_norm.weight": ("attn_k_norm.weights", "weight"),
    "mlp.gate_proj.weight": ("ffn_gate.weights", "weight"),
    "mlp.up_proj.weight": ("ffn_up.weights", "weight"),
    "mlp.down_proj.weight": ("ffn_down.weights", "weight"),
    "input_layernorm.weight": ("input_norm.weights", "weight"),
    "post_attention_layernorm.weight": ("post_attn_norm.weights", "weight"),
    "pre_feedforward_layernorm.weight": ("pre_ffn_norm.weights", "weight"),
    "post_feedforward_layernorm.weight": ("post_ffn_norm.weights", "weight"),
    "post_per_layer_projection_layernorm.weight": ("post_per_layer_norm.weights", "weight"),
    "per_layer_input_gate.weight": ("per_layer_gate.weights", "weight"),
    "per_layer_projection.weight": ("per_layer_proj.weights", "weight"),
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

    def _add_backbone_aliases(name: str) -> None:
        if name.startswith("backbone."):
            tail = name[len("backbone.") :]
            _add(f"model.{tail}")
            _add(f"model.language_model.{tail}")
        if name.startswith("language_model."):
            tail = name[len("language_model.") :]
            _add(f"model.language_model.{tail}")
        if name.startswith("multimodal_backbone."):
            tail = name[len("multimodal_backbone.") :]
            _add(tail)
            _add(f"model.{tail}")
            _add(f"model.multimodal_backbone.{tail}")
        if name.startswith("model.encoder."):
            _add(name[len("model.") :])
        if name.startswith("model.decoder."):
            _add(name[len("model.") :])
        if name.startswith("layers."):
            _add(f"encoder.{name}")
        if name.startswith("pre_encode."):
            _add(f"encoder.{name}")
        if name.startswith("prediction."):
            _add(f"decoder.{name}")

    raw = source_name.strip()
    _add(raw)
    _add_backbone_aliases(raw)

    for prefix in ("p_", "b_", "c_"):
        if raw.startswith(prefix):
            stripped = raw[len(prefix):]
            _add(stripped)
            _add_backbone_aliases(stripped)

    stripped = raw
    while True:
        for prefix in ("adapter.model.", "adapter.module.", "adapter.encoder.", "adapter.decoder.", "module."):
            if stripped.startswith(prefix):
                tail = stripped[len(prefix) :]
                if prefix == "adapter.encoder.":
                    stripped = f"encoder.{tail}"
                elif prefix == "adapter.decoder.":
                    stripped = f"decoder.{tail}"
                else:
                    stripped = tail
                _add(stripped)
                _add_backbone_aliases(stripped)
                break
        else:
            break

    for candidate in tuple(candidates):
        for repeated_prefix, collapsed_prefix in (
            ("model.model.", "model."),
            ("language_model.language_model.", "language_model."),
            ("backbone.backbone.", "backbone."),
        ):
            if candidate.startswith(repeated_prefix):
                tail = candidate[len(repeated_prefix) :]
                _add(f"{collapsed_prefix}{tail}")

        for prefix in (
            "module.backbone.",
            "adapter.backbone.",
            "adapter.model.model.",
            "adapter.model.language_model.",
        ):
            if not candidate.startswith(prefix):
                continue
            tail = candidate[len(prefix):]
            _add(tail)
            _add_backbone_aliases(tail)
            _add(f"model.{tail}")
            _add(f"model.language_model.{tail}")
            if tail.startswith("layers."):
                _add(f"model.{tail}")
                _add(f"model.language_model.{tail}")
            break

    if stripped == "adapter.model.lm_head.weight":
        _add("lm_head.weight")
        _add("model.embed_tokens.weight")
        _add("model.language_model.embed_tokens.weight")
    if stripped == "lm_head.weight":
        _add("model.embed_tokens.weight")
        _add("model.language_model.embed_tokens.weight")

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


def _gemma_tower_output_name(hf_key: str, strip_prefix: str, add_prefix: str) -> str:
    name = hf_key[len(strip_prefix) :]
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
        ext = ".weights"
    elif name.endswith(".bias"):
        name = name[: -len(".bias")]
        ext = ".bias"
    else:
        ext = ".weights"
    if name.endswith(".linear"):
        name = name[: -len(".linear")]
    elif name.endswith("_linear"):
        name = name[: -len("_linear")]
    name = name.replace(".", "_")
    return add_prefix + name + ext


def _fallback_filename_candidates(source_name: str) -> list[tuple[str, str]]:
    filenames: list[tuple[str, str]] = []

    def _add(filename: str | None, *, kind: str = "weight") -> None:
        if not filename:
            return
        candidate = (filename, kind)
        if candidate not in filenames:
            filenames.append(candidate)

    for candidate in _normalized_source_candidates(source_name):
        _add(_PARAKEET_GLOBAL_FILENAMES.get(candidate))
        gemma_global = _GEMMA_GLOBAL_FILENAMES.get(candidate)
        if gemma_global is not None:
            _add(gemma_global[0], kind=gemma_global[1])
        if candidate in {"model.embed_tokens_per_layer.weight", "model.language_model.embed_tokens_per_layer.weight"}:
            _add("embed_tokens_per_layer.weights", kind="embedding")
        elif candidate in {"model.per_layer_model_projection.weight", "model.language_model.per_layer_model_projection.weight"}:
            _add("per_layer_model_proj.weights")
        elif candidate in {"model.per_layer_projection_norm.weight", "model.language_model.per_layer_projection_norm.weight"}:
            _add("per_layer_proj_norm.weights")
        elif candidate in {"lm_head.weight", "model.lm_head.weight"}:
            _add("token_embeddings.weights", kind="embedding")
        elif candidate in {
            "_cactus_vision_post_proj_norm",
            "model._cactus_vision_post_proj_norm",
        }:
            _add("embed_vision_post_proj_norm.weights")

        if candidate.startswith(_GEMMA3N_VISION_TOWER_PREFIX):
            _add(_gemma_tower_output_name(candidate, _GEMMA3N_VISION_TOWER_PREFIX, "vision_"))
        elif candidate.startswith(_GEMMA3N_AUDIO_TOWER_PREFIX):
            _add(_gemma_tower_output_name(candidate, _GEMMA3N_AUDIO_TOWER_PREFIX, "audio_"))
        elif candidate.startswith(_GEMMA4_VISION_TOWER_PREFIX):
            _add(_gemma_tower_output_name(candidate, _GEMMA4_VISION_TOWER_PREFIX, "vision_"))
        elif candidate.startswith(_GEMMA4_AUDIO_TOWER_PREFIX):
            _add(_gemma_tower_output_name(candidate, _GEMMA4_AUDIO_TOWER_PREFIX, "audio_"))

        predictor_match = re.match(r"^decoder\.prediction\.dec_rnn\.lstm\.(\d+)\.(Wx|Wh|bias)$", candidate)
        if predictor_match:
            layer_index = int(predictor_match.group(1))
            suffix = predictor_match.group(2)
            if suffix == "Wx":
                _add(f"tdt_predictor_lstm_{layer_index}_weight_ih.weights")
            elif suffix == "Wh":
                _add(f"tdt_predictor_lstm_{layer_index}_weight_hh.weights")
            elif suffix == "bias":
                _add(f"tdt_predictor_lstm_{layer_index}_bias.weights")

        layer_match = re.match(r"^encoder\.layers\.(\d+)\.(.+)$", candidate)
        if layer_match:
            layer_index = int(layer_match.group(1))
            suffix = layer_match.group(2)
            mapped = _PARAKEET_LAYER_FILENAMES.get(suffix)
            if mapped is not None:
                _add(f"layer_{layer_index}_{mapped}")

        gemma_layer_match = re.match(
            r"^(?:model(?:\.language_model)?\.)?layers\.(\d+)\.(.+)$",
            candidate,
        )
        if gemma_layer_match:
            layer_index = int(gemma_layer_match.group(1))
            suffix = gemma_layer_match.group(2)
            mapped = _GEMMA_DECODER_LAYER_FILENAMES.get(suffix)
            if mapped is not None:
                _add(f"layer_{layer_index}_{mapped[0]}", kind=mapped[1])

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
    manifest: dict[str, object] = {}
    manifest_path = root / "weights_manifest.json"
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text())
        except Exception:
            loaded_manifest = None
        if isinstance(loaded_manifest, dict):
            manifest = loaded_manifest
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

    for filename, kind in _fallback_filename_candidates(source_name):
        candidate = root / filename
        if candidate.exists():
            return WeightBinding(path=str(candidate), kind=kind, source_name=source_name)
    return None
