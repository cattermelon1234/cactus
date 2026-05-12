from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import struct
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.transpile.component_pipeline import ComponentModuleSpec
from src.tensor_io import CACTUS_MAGIC
from src.tensor_io import align_offset


_GEMMA4_SAFE_TEXT_MLP_PRODUCT_SCALE = 1.0 / 64.0


@dataclass
class CanonicalizedModel:
    module: torch.nn.Module
    task: str
    family: str
    input_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Gemma4NativeMergeSegment:
    kind: str
    input_start: int
    length: int
    feature_start: int = 0


@dataclass(frozen=True)
class _Gemma4NativeMergePlan:
    segments: tuple[_Gemma4NativeMergeSegment, ...]
    pli_token_ids: tuple[int, ...]


def _model_name_or_path(model: torch.nn.Module) -> str:
    value = getattr(model, "name_or_path", None)
    if isinstance(value, str) and value:
        return value
    config = getattr(model, "config", None)
    value = getattr(config, "_name_or_path", None)
    if isinstance(value, str) and value:
        return value
    return ""


def _transpile_graph_meta(model: torch.nn.Module, *, adapter_family: str, adapter_type: str, input_names: tuple[str, ...]) -> dict[str, object]:
    return {
        "adapter_family": adapter_family,
        "adapter_type": adapter_type,
        "model_name_or_path": _model_name_or_path(model),
        "input_names": input_names,
    }


def _extract_tensor_output(output: object, *, preferred_field: str | None = None) -> torch.Tensor:
    if preferred_field is not None:
        value = getattr(output, preferred_field, None)
        if isinstance(value, torch.Tensor):
            return value

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]

    for field_name in ("last_hidden_state", "logits"):
        value = getattr(output, field_name, None)
        if isinstance(value, torch.Tensor):
            return value

    raise TypeError(f"could not extract tensor output from {type(output).__name__}")


def _select_last_active_token(hidden_or_logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None or hidden_or_logits.ndim < 3:
        return hidden_or_logits[:, -1:, ...]

    if attention_mask.ndim != 2:
        raise ValueError(f"expected 2D attention mask, got shape {tuple(attention_mask.shape)}")
    if attention_mask.shape[1] != hidden_or_logits.shape[1]:
        raise ValueError(
            "attention mask / hidden sequence length mismatch: "
            f"{tuple(attention_mask.shape)} vs {tuple(hidden_or_logits.shape)}"
        )

    trailing_zero = attention_mask[:, :1] - attention_mask[:, :1]
    shifted_mask = torch.cat((attention_mask[:, 1:], trailing_zero), dim=1)
    last_active_mask = torch.logical_and(attention_mask != 0, shifted_mask == 0)
    expanded_mask = last_active_mask.to(dtype=hidden_or_logits.dtype)
    for _ in range(hidden_or_logits.ndim - 2):
        expanded_mask = expanded_mask.unsqueeze(-1)
    return (hidden_or_logits * expanded_mask).sum(dim=1, keepdim=True)


def _module_floating_dtype(module: torch.nn.Module) -> torch.dtype | None:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return None


def _module_device(module: torch.nn.Module) -> torch.device | None:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return None


def _unique_modules(*candidates: object) -> tuple[torch.nn.Module, ...]:
    modules: list[torch.nn.Module] = []
    seen: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, torch.nn.Module):
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        modules.append(candidate)
    return tuple(modules)


def _gemma4_vision_modules(multimodal_backbone: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    return _unique_modules(
        getattr(multimodal_backbone, "vision_tower", None),
        getattr(multimodal_backbone, "embed_vision", None),
    )


def _gemma4_audio_modules(multimodal_backbone: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    return _unique_modules(
        getattr(multimodal_backbone, "audio_tower", None),
        getattr(multimodal_backbone, "embed_audio", None),
    )


def _gemma4_text_config(multimodal_backbone: torch.nn.Module) -> object:
    config = getattr(multimodal_backbone, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            return get_text_config()
        except Exception:
            pass
    return config


def _gemma4_special_token_ids(multimodal_backbone: torch.nn.Module) -> tuple[int, int, int]:
    config = getattr(multimodal_backbone, "config", None)
    text_config = _gemma4_text_config(multimodal_backbone)
    image_token_id = int(getattr(config, "image_token_id", 0) or 0)
    audio_token_id = int(getattr(config, "audio_token_id", 0) or 0)
    pad_token_id = getattr(text_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(config, "pad_token_id", 0)
    return image_token_id, audio_token_id, int(pad_token_id or 0)


@contextmanager
def _temporary_cpu_float32_modules(modules: tuple[torch.nn.Module, ...]):
    promoted: list[tuple[torch.nn.Module, torch.dtype]] = []
    for module in modules:
        device = _module_device(module)
        dtype = _module_floating_dtype(module)
        if device is None or device.type != "cpu" or dtype is None or dtype == torch.float32:
            continue
        promoted.append((module, dtype))
        module.to(dtype=torch.float32)
    try:
        yield
    finally:
        for module, dtype in reversed(promoted):
            module.to(dtype=dtype)


def _torch_is_compiling() -> bool:
    dynamo = getattr(torch, "_dynamo", None)
    is_compiling = getattr(dynamo, "is_compiling", None)
    if callable(is_compiling):
        try:
            return bool(is_compiling())
        except Exception:
            return False
    return False


def _gemma4_cpu_safe_text_mlp_enabled(layer: torch.nn.Module, hidden_states: torch.Tensor) -> bool:
    if hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float16:
        return False
    if bool(getattr(layer, "enable_moe_block", False)):
        return False
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return False
    for attr_name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, attr_name):
            return False
    return True


def _gemma4_cpu_safe_text_mlp_forward(mlp: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    gate = F.linear(
        hidden_states,
        mlp.gate_proj.weight,
        mlp.gate_proj.bias,
    )
    up = F.linear(
        hidden_states,
        mlp.up_proj.weight,
        mlp.up_proj.bias,
    )
    # Gemma4 text MLPs can overflow in FP16 at the gated product.
    # Scaling one branch here keeps the product and down-projection finite.
    # The following post-feedforward RMSNorm is scale-invariant, so this
    # constant factor cancels back out in the normalized output.
    activated = mlp.act_fn(gate) * _GEMMA4_SAFE_TEXT_MLP_PRODUCT_SCALE
    return F.linear(
        activated * up,
        mlp.down_proj.weight,
        mlp.down_proj.bias,
    )


def _gemma4_text_attention_forward(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    past_key_values: object | None,
    use_cache: bool,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
) -> torch.Tensor:
    if shared_kv_states is None:
        attn_out, _ = attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return attn_out

    from transformers.models.gemma4.modeling_gemma4 import ALL_ATTENTION_FUNCTIONS  # type: ignore
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb  # type: ignore
    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward  # type: ignore

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    if position_embeddings is None:
        raise ValueError("Gemma4 text attention requires position embeddings")
    cos, sin = position_embeddings

    query_states = attn.q_proj(hidden_states).view(hidden_shape)
    query_states = attn.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    kv_shared_layer_index = getattr(attn, "kv_shared_layer_index", None)
    if bool(getattr(attn, "is_kv_shared_layer", False)) and kv_shared_layer_index in shared_kv_states:
        key_states, value_states = shared_kv_states[int(kv_shared_layer_index)]
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = attn.k_proj(hidden_states).view(hidden_shape)
        value_states = attn.v_proj(hidden_states).view(hidden_shape) if attn.v_proj is not None else key_states

        key_states = attn.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = attn.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if bool(getattr(attn, "store_full_length_kv", False)):
            shared_kv_states[int(attn.layer_idx)] = (key_states, value_states)

    attention_interface = eager_attention_forward
    if getattr(attn.config, "_attn_implementation", "eager") != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

    attn_output, _ = attention_interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=attn.attention_dropout if attn.training else 0.0,
        scaling=attn.scaling,
        sliding_window=attn.sliding_window,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return attn.o_proj(attn_output)


def _gemma4_text_decoder_layer_forward(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    per_layer_input: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: object | None = None,
    use_cache: bool = False,
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    if not _gemma4_cpu_safe_text_mlp_enabled(layer, hidden_states):
        return layer(
            hidden_states,
            position_embeddings=position_embeddings,
            per_layer_input=per_layer_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    residual = hidden_states

    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = _gemma4_text_attention_forward(
        layer.self_attn,
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        shared_kv_states=shared_kv_states,
    )
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    mlp_out = _gemma4_cpu_safe_text_mlp_forward(layer.mlp, hidden_states)
    hidden_states = layer.post_feedforward_layernorm(mlp_out)
    hidden_states = residual.float() + hidden_states
    hidden_states = hidden_states.to(dtype=residual.dtype)

    if getattr(layer, "hidden_size_per_layer_input", 0):
        residual = hidden_states
        hidden_states = layer.per_layer_input_gate(hidden_states)
        hidden_states = layer.act_fn(hidden_states)
        if per_layer_input is None:
            raise ValueError("Gemma4 layer expected per_layer_input but none was provided")
        hidden_states = torch.multiply(hidden_states, per_layer_input)
        hidden_states = layer.per_layer_projection(hidden_states)
        hidden_states = layer.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = hidden_states * layer.layer_scalar
    return hidden_states


def _gemma4_text_backbone_forward(
    backbone: torch.nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    per_layer_inputs: torch.Tensor | None,
    causal_mask_mapping: dict[str, torch.Tensor],
    position_ids: torch.LongTensor,
) -> torch.Tensor:
    hidden_states = inputs_embeds
    layer_types = tuple(dict.fromkeys(getattr(backbone.config, "layer_types", ())))
    position_embeddings = {
        layer_type: backbone.rotary_emb(hidden_states, position_ids, layer_type)
        for layer_type in layer_types
    }
    shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for decoder_layer in backbone.layers[: backbone.config.num_hidden_layers]:
        layer_per_input = None
        if per_layer_inputs is not None:
            layer_per_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]
        hidden_states = _gemma4_text_decoder_layer_forward(
            decoder_layer,
            hidden_states,
            per_layer_input=layer_per_input,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_embeddings=position_embeddings[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            shared_kv_states=shared_kv_states,
        )

    return backbone.norm(hidden_states)


def _gemma4_strip_audio_padding(audio_output: object) -> torch.Tensor:
    audio_features = getattr(audio_output, "pooler_output", None)
    audio_mask_from_encoder = getattr(audio_output, "audio_mel_mask", None)
    if not isinstance(audio_features, torch.Tensor):
        raise TypeError("Gemma4 audio output did not expose tensor pooler_output")
    if not isinstance(audio_mask_from_encoder, torch.Tensor):
        return audio_features
    all_real_tokens: list[torch.Tensor] = []
    for encodings, padding_mask in zip(audio_features, audio_mask_from_encoder, strict=True):
        all_real_tokens.append(encodings[~padding_mask])
    return torch.cat(all_real_tokens, dim=0).unsqueeze(0)


def _gemma4_rms_norm_no_scale(hidden_states: torch.Tensor, *, eps: float) -> torch.Tensor:
    hidden_states_f32 = hidden_states.float()
    mean_squared = hidden_states_f32.pow(2).mean(-1, keepdim=True) + eps
    return hidden_states_f32 * torch.pow(mean_squared, -0.5)


def _gemma4_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    normed = _gemma4_rms_norm_no_scale(hidden_states, eps=eps)
    return normed * weight.to(device=normed.device, dtype=normed.dtype)


def _gemma4_layer_norm_eps(multimodal_backbone: torch.nn.Module) -> float:
    config = getattr(multimodal_backbone, "config", None)
    text_config = None
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config()
        except Exception:
            text_config = None
    for candidate in (text_config, config):
        value = getattr(candidate, "rms_norm_eps", None)
        if value is not None:
            return float(value)
        value = getattr(candidate, "layer_norm_eps", None)
        if value is not None:
            return float(value)
    return 1e-6


def _gemma4_load_cactus_fp_tensor(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        header = handle.read(84)
    if len(header) < 84 or header[:4] != CACTUS_MAGIC:
        raise RuntimeError(f"Gemma4 Cactus tensor is missing a valid header: {path}")

    alignment = max(1, int(struct.unpack_from("<I", header, 8)[0]))
    ndim = int(struct.unpack_from("<I", header, 12)[0])
    dims = struct.unpack_from("<QQQQ", header, 16)
    shape = tuple(int(dim) for dim in dims[:ndim] if int(dim) > 0)
    precision = int(struct.unpack_from("<I", header, 48)[0])
    byte_size = int(struct.unpack_from("<Q", header, 52)[0])
    scales_bytes = int(struct.unpack_from("<Q", header, 60)[0])
    dtype = {1: np.float16, 2: np.float32}.get(precision)
    if dtype is None:
        raise RuntimeError(f"Gemma4 Cactus tensor must be FP16/FP32, got precision={precision}: {path}")

    aligned_header = align_offset(84, alignment)
    scales_offset = aligned_header if scales_bytes > 0 else 0
    data_offset = (
        align_offset(scales_offset + scales_bytes, alignment)
        if scales_bytes > 0
        else aligned_header
    )
    data_count = byte_size // np.dtype(dtype).itemsize
    array = np.memmap(path, mode="r", dtype=dtype, offset=data_offset, shape=(data_count,))
    tensor = torch.from_numpy(np.array(array, copy=True))
    if shape:
        tensor = tensor.reshape(shape)
    return tensor


def _gemma4_load_vision_post_proj_norm(weights_dir: str | Path | None) -> torch.Tensor | None:
    if weights_dir is None:
        return None
    path = Path(weights_dir).expanduser() / "embed_vision_post_proj_norm.weights"
    if not path.exists():
        return None
    tensor = _gemma4_load_cactus_fp_tensor(path)
    if tensor.ndim != 1:
        raise RuntimeError(f"Gemma4 vision post-projection norm must be 1D, got shape={tuple(tensor.shape)}")
    return tensor.float()


def _gemma4_can_use_native_like_vision_features(multimodal_backbone: torch.nn.Module) -> bool:
    vision_tower = getattr(multimodal_backbone, "vision_tower", None)
    embed_vision = getattr(multimodal_backbone, "embed_vision", None)
    return (
        isinstance(vision_tower, torch.nn.Module)
        and isinstance(embed_vision, torch.nn.Module)
        and hasattr(vision_tower, "patch_embedder")
        and hasattr(vision_tower, "encoder")
        and hasattr(embed_vision, "embedding_projection")
    )


def _gemma4_can_use_native_like_audio_features(multimodal_backbone: torch.nn.Module) -> bool:
    audio_tower = getattr(multimodal_backbone, "audio_tower", None)
    embed_audio = getattr(multimodal_backbone, "embed_audio", None)
    return (
        isinstance(audio_tower, torch.nn.Module)
        and isinstance(embed_audio, torch.nn.Module)
        and hasattr(audio_tower, "subsample_conv_projection")
        and hasattr(audio_tower, "conformer")
        and hasattr(embed_audio, "embedding_projection")
    )


def _gemma4_pool_vision_hidden_native_like(
    vision_tower: torch.nn.Module,
    hidden_states: torch.Tensor,
    pixel_position_ids: torch.Tensor,
) -> torch.Tensor:
    output_length = int(getattr(vision_tower, "default_output_length"))
    pooling_kernel_size = int(getattr(vision_tower, "pooling_kernel_size"))
    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    pooled_batches: list[torch.Tensor] = []
    for hidden_row, position_row, padding_row in zip(hidden_states, pixel_position_ids, padding_positions, strict=True):
        valid_hidden = hidden_row[~padding_row].float()
        valid_positions = position_row[~padding_row].clamp(min=0)
        max_x = valid_positions[:, 0].max() + 1
        kernel_positions = torch.div(valid_positions, pooling_kernel_size, rounding_mode="floor")
        kernel_indices = kernel_positions[:, 0] + torch.div(
            max_x,
            pooling_kernel_size,
            rounding_mode="floor",
        ) * kernel_positions[:, 1]
        weights = F.one_hot(kernel_indices.long(), output_length).float() / float(pooling_kernel_size**2)
        pooled_full = weights.transpose(0, 1) @ valid_hidden
        valid_bins = torch.logical_not((weights == 0).all(dim=0))
        pooled_batches.append(pooled_full[valid_bins])
    return torch.cat(pooled_batches, dim=0)


def _gemma4_compute_native_like_image_features(
    multimodal_backbone: torch.nn.Module,
    pixel_values: torch.Tensor,
    pixel_position_ids: torch.Tensor | None,
    *,
    post_proj_norm_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if pixel_position_ids is None:
        raise TypeError("Gemma4 native-like vision feature path requires pixel_position_ids")
    vision_tower = getattr(multimodal_backbone, "vision_tower", None)
    embed_vision = getattr(multimodal_backbone, "embed_vision", None)
    if not isinstance(vision_tower, torch.nn.Module) or not isinstance(embed_vision, torch.nn.Module):
        raise TypeError("Gemma4 multimodal backbone is missing native-like vision modules")

    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    vision_inputs = vision_tower.patch_embedder(
        pixel_values,
        pixel_position_ids,
        padding_positions,
    )
    vision_outputs = vision_tower.encoder(
        inputs_embeds=vision_inputs,
        attention_mask=~padding_positions,
        pixel_position_ids=pixel_position_ids,
    )
    vision_hidden = _extract_tensor_output(vision_outputs, preferred_field="last_hidden_state")
    pooled_hidden = _gemma4_pool_vision_hidden_native_like(
        vision_tower,
        vision_hidden,
        pixel_position_ids,
    )
    projection = getattr(embed_vision, "embedding_projection", None)
    if not isinstance(projection, torch.nn.Linear):
        raise TypeError("Gemma4 vision embedder is missing embedding_projection")
    projected = F.linear(
        pooled_hidden,
        projection.weight.float(),
        None if projection.bias is None else projection.bias.float(),
    )
    if post_proj_norm_weight is not None and post_proj_norm_weight.numel() > 0:
        return _gemma4_rms_norm(
            projected,
            post_proj_norm_weight,
            eps=_gemma4_layer_norm_eps(multimodal_backbone),
        )
    return _gemma4_rms_norm_no_scale(projected, eps=float(getattr(embed_vision, "eps", 1e-6)))


def _gemma4_compute_native_like_audio_features(
    multimodal_backbone: torch.nn.Module,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
) -> torch.Tensor:
    audio_tower = getattr(multimodal_backbone, "audio_tower", None)
    embed_audio = getattr(multimodal_backbone, "embed_audio", None)
    if not isinstance(audio_tower, torch.nn.Module) or not isinstance(embed_audio, torch.nn.Module):
        raise TypeError("Gemma4 multimodal backbone is missing native-like audio modules")

    audio_output = audio_tower(input_features, ~input_features_mask)
    if not isinstance(audio_output, tuple) or len(audio_output) != 2:
        raise TypeError("Gemma4 audio tower did not return (audio_encodings, audio_mel_mask)")
    audio_encodings, audio_mel_mask = audio_output
    if not isinstance(audio_encodings, torch.Tensor):
        raise TypeError("Gemma4 audio tower did not return tensor audio encodings")
    if not isinstance(audio_mel_mask, torch.Tensor):
        raise TypeError("Gemma4 audio tower did not return tensor audio mask")

    projection = getattr(embed_audio, "embedding_projection", None)
    if not isinstance(projection, torch.nn.Linear):
        raise TypeError("Gemma4 audio embedder is missing embedding_projection")
    normed = _gemma4_rms_norm_no_scale(audio_encodings, eps=float(getattr(embed_audio, "eps", 1e-6)))
    projected = F.linear(
        normed,
        projection.weight.float(),
        None if projection.bias is None else projection.bias.float(),
    )
    projected = projected * (1.0 / 16.0)
    all_real_tokens: list[torch.Tensor] = []
    for encodings, padding_mask in zip(projected, audio_mel_mask, strict=True):
        all_real_tokens.append(encodings[~padding_mask])
    return torch.cat(all_real_tokens, dim=0).unsqueeze(0)


def _gemma4_feature_token_count(features: torch.Tensor | None) -> int:
    if features is None or features.numel() == 0:
        return 0
    if features.ndim == 3:
        return int(features.shape[1])
    if features.ndim == 2:
        return int(features.shape[0])
    if features.ndim >= 1:
        return int(features.reshape(-1, features.shape[-1]).shape[0])
    return 0


def _gemma4_build_native_merge_plan(
    multimodal_backbone: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    image_feature_count: int,
    audio_feature_count: int,
) -> _Gemma4NativeMergePlan:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError(
            "Gemma4 native multimodal merge currently expects a static batch-1 input_ids tensor, "
            f"got shape {tuple(input_ids.shape)}"
        )

    image_token_id, audio_token_id, pad_token_id = _gemma4_special_token_ids(multimodal_backbone)
    token_ids = [int(token) for token in input_ids.detach().cpu().reshape(-1).tolist()]
    segments: list[_Gemma4NativeMergeSegment] = []
    pli_token_ids: list[int] = []
    image_offset = 0
    audio_offset = 0
    index = 0

    while index < len(token_ids):
        token_id = token_ids[index]
        is_image = image_token_id != 0 and token_id == image_token_id
        is_audio = audio_token_id != 0 and token_id == audio_token_id

        if is_image or is_audio:
            region_start = index
            while index < len(token_ids) and token_ids[index] == token_id:
                index += 1
            placeholder_count = index - region_start
            if is_image:
                insert_count = min(placeholder_count, max(0, image_feature_count - image_offset))
                if insert_count > 0:
                    segments.append(
                        _Gemma4NativeMergeSegment(
                            kind="image",
                            input_start=region_start,
                            length=insert_count,
                            feature_start=image_offset,
                        )
                    )
                    pli_token_ids.extend([pad_token_id] * insert_count)
                    image_offset += insert_count
            else:
                insert_count = min(placeholder_count, max(0, audio_feature_count - audio_offset))
                if insert_count > 0:
                    segments.append(
                        _Gemma4NativeMergeSegment(
                            kind="audio",
                            input_start=region_start,
                            length=insert_count,
                            feature_start=audio_offset,
                        )
                    )
                    pli_token_ids.extend([pad_token_id] * insert_count)
                    audio_offset += insert_count
            continue

        text_start = index
        while index < len(token_ids):
            next_token = token_ids[index]
            if image_token_id != 0 and next_token == image_token_id:
                break
            if audio_token_id != 0 and next_token == audio_token_id:
                break
            index += 1
        text_tokens = token_ids[text_start:index]
        if text_tokens:
            segments.append(
                _Gemma4NativeMergeSegment(
                    kind="text",
                    input_start=text_start,
                    length=len(text_tokens),
                )
            )
            pli_token_ids.extend(text_tokens)

    if not segments:
        raise RuntimeError("Gemma4 native multimodal merge built no input segments")
    return _Gemma4NativeMergePlan(
        segments=tuple(segments),
        pli_token_ids=tuple(pli_token_ids),
    )


def _gemma4_feature_sequence(
    features: torch.Tensor,
    *,
    batch_size: int,
    feature_dim: int,
) -> torch.Tensor:
    if features.ndim == 3:
        if int(features.shape[0]) == batch_size:
            return features
        if int(features.shape[0]) == 1:
            return features.expand(batch_size, -1, -1)
    if features.ndim == 2:
        if int(features.shape[-1]) != feature_dim:
            raise ValueError(f"Gemma4 feature dim mismatch: expected {feature_dim}, got {tuple(features.shape)}")
        return features.unsqueeze(0).expand(batch_size, -1, -1)
    if features.ndim >= 1 and int(features.shape[-1]) == feature_dim:
        flattened = features.reshape(-1, feature_dim)
        return flattened.unsqueeze(0).expand(batch_size, -1, -1)
    raise ValueError(f"unsupported Gemma4 feature tensor shape: {tuple(features.shape)}")


def _gemma4_apply_native_merge_plan(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    image_features: torch.Tensor,
    audio_features: torch.Tensor,
    merge_plan: _Gemma4NativeMergePlan,
    pli_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = model.get_input_embeddings()
    if not isinstance(embedding, torch.nn.Module):
        raise TypeError("Gemma4 model is missing input embeddings")
    batch_size = int(input_ids.shape[0])
    feature_dim = int(getattr(embedding, "embedding_dim", 0) or image_features.shape[-1] or audio_features.shape[-1])
    target_dtype = _module_floating_dtype(embedding) or image_features.dtype
    image_sequence = _gemma4_feature_sequence(
        image_features,
        batch_size=batch_size,
        feature_dim=feature_dim,
    ).to(device=input_ids.device, dtype=target_dtype)
    audio_sequence = _gemma4_feature_sequence(
        audio_features,
        batch_size=batch_size,
        feature_dim=feature_dim,
    ).to(device=input_ids.device, dtype=target_dtype)

    embedded_segments: list[torch.Tensor] = []
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None)
    hidden_scale = float(getattr(model_config, "hidden_size", 0) or 0)
    if hidden_scale <= 0.0:
        hidden_scale = float(getattr(text_config, "hidden_size", 0) or 0)
    if hidden_scale <= 0.0:
        hidden_scale = float(feature_dim)
    hidden_scale = float(hidden_scale) ** 0.5
    for segment in merge_plan.segments:
        if segment.kind == "text":
            text_tokens = input_ids[:, segment.input_start : segment.input_start + segment.length]
            embedded_segments.append(embedding(text_tokens) * hidden_scale)
        elif segment.kind == "image":
            embedded_segments.append(
                image_sequence[:, segment.feature_start : segment.feature_start + segment.length, :]
            )
        elif segment.kind == "audio":
            embedded_segments.append(
                audio_sequence[:, segment.feature_start : segment.feature_start + segment.length, :]
            )
        else:
            raise RuntimeError(f"unknown Gemma4 merge segment kind: {segment.kind!r}")

    inputs_embeds = torch.cat(embedded_segments, dim=1)
    if pli_token_ids.numel() != inputs_embeds.shape[1]:
        raise RuntimeError(
            "Gemma4 native merge PLI token count mismatch: "
            f"{int(pli_token_ids.numel())} vs {int(inputs_embeds.shape[1])}"
        )
    pli_tokens = pli_token_ids.to(device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
    if batch_size != 1:
        pli_tokens = pli_tokens.expand(batch_size, -1)
    return inputs_embeds, pli_tokens


def _gemma4_remap_sequence_tensor(
    tensor: torch.Tensor | None,
    *,
    merge_plan: _Gemma4NativeMergePlan,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim != 2:
        raise ValueError(f"Gemma4 merge remap expects a rank-2 tensor, got shape {tuple(tensor.shape)}")
    remapped_segments: list[torch.Tensor] = []
    for segment in merge_plan.segments:
        remapped_segments.append(
            tensor[:, segment.input_start : segment.input_start + segment.length]
        )
    if not remapped_segments:
        return tensor[:, :0]
    return torch.cat(remapped_segments, dim=1)


def _gemma4_build_standard_causal_mask_mapping(
    *,
    create_causal_mask: Callable[..., torch.Tensor] | None,
    create_sliding_window_causal_mask: Callable[..., torch.Tensor] | None,
    config: object,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not callable(create_causal_mask) or not callable(create_sliding_window_causal_mask):
        raise RuntimeError("Gemma4 standard causal mask helpers are unavailable")
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    return {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }


def _infer_input_names(module: torch.nn.Module, *, preferred: tuple[str, ...]) -> tuple[str, ...]:
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return preferred[:1]

    control_names = {
        "self",
        "return_dict",
        "use_cache",
        "past_key_values",
        "cache_position",
        "position_ids",
        "labels",
        "decoder_input_ids",
        "decoder_attention_mask",
        "output_attentions",
        "output_hidden_states",
    }
    available = [
        name
        for name, parameter in signature.parameters.items()
        if name not in control_names
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    matched = [name for name in preferred if name in available]
    if matched:
        return tuple(matched)
    if available:
        return tuple(available[: min(2, len(available))])
    return preferred[:1]


class BoundInputAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...], family: str, metadata_task: str):
        super().__init__()
        self.model = model
        self.input_names = tuple(input_names)
        self.family = family
        self.metadata_task = metadata_task

    def _kwargs_from_bound_inputs(self, *bound_inputs: torch.Tensor | None) -> dict[str, torch.Tensor]:
        provided = tuple(bound_inputs)
        if len(self.input_names) > len(provided):
            raise ValueError(
                f"adapter expected at most {len(provided)} bound inputs, got {len(self.input_names)} names"
            )
        kwargs: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.input_names):
            value = provided[index]
            if value is None:
                raise ValueError(f"missing required bound input {index} for {name}")
            kwargs[name] = value
        return kwargs

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family=self.family,
                    adapter_type=type(self).__name__,
                    input_names=self.input_names,
                ),
                "task": self.metadata_task,
            }
        }


def _gemma4_import_hints(backbone: torch.nn.Module, *, module_path_suffix_prefix: str) -> list[dict[str, object]]:
    sliding_window = getattr(backbone.config, "sliding_window", None)
    layer_types = list(getattr(backbone.config, "layer_types", []))
    import_hints: list[dict[str, object]] = []
    for layer_index, layer_type in enumerate(layer_types):
        attrs: dict[str, object] = {}
        if layer_type == "sliding_attention" and sliding_window is not None:
            attrs["window_size"] = int(sliding_window)
        import_hints.append(
            {
                "module_path_suffix": f"{module_path_suffix_prefix}.layers.{layer_index}.self_attn",
                "op": "scaled_dot_product_attention",
                "attrs": attrs,
                "meta": {
                    "attention_layer_type": layer_type,
                    "attention_layer_index": layer_index,
                },
            }
        )
    return import_hints


class CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        return self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )[0]

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="generic",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
            }
        }


class GemmaCausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.backbone = model.model
        from transformers.models.gemma.modeling_gemma import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask = self._create_causal_mask(
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.backbone.rotary_emb(hidden_states, position_ids=position_ids)
        checkpoints: list[torch.Tensor] = []

        for decoder_layer in self.backbone.layers[: self.backbone.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            checkpoints.append(hidden_states)

        hidden_states = self.backbone.norm(hidden_states)
        checkpoints.append(hidden_states)
        return _gemma4_apply_final_logit_softcapping(self.model, self.model.lm_head(hidden_states)), checkpoints

    def get_transpile_metadata(self):
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="gemma",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
            }
        }


class Gemma3CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.backbone = model.model
        from transformers.models.gemma3.modeling_gemma3 import create_causal_mask  # type: ignore
        from transformers.models.gemma3.modeling_gemma3 import create_sliding_window_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask
        self._create_sliding_window_causal_mask = create_sliding_window_causal_mask

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        mask_kwargs = {
            "config": self.backbone.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": self._create_causal_mask(**mask_kwargs),
            "sliding_attention": self._create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        checkpoints: list[torch.Tensor] = []
        position_embeddings = {
            layer_type: self.backbone.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.backbone.config.layer_types
        }

        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.backbone.config.layer_types[i]],
                position_embeddings=position_embeddings[self.backbone.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=None,
            )
            checkpoints.append(hidden_states)

        hidden_states = self.backbone.norm(hidden_states)
        checkpoints.append(hidden_states)
        return self.model.lm_head(hidden_states), checkpoints

    def get_transpile_metadata(self):
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="gemma3",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
            },
            "import_hints": _gemma4_import_hints(self.backbone, module_path_suffix_prefix="backbone"),
        }


class Gemma4CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        model_backbone = model.model
        self.backbone = getattr(model_backbone, "language_model", model_backbone)
        from transformers.models.gemma4.modeling_gemma4 import create_causal_mask  # type: ignore
        from transformers.models.gemma4.modeling_gemma4 import create_sliding_window_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask
        self._create_sliding_window_causal_mask = create_sliding_window_causal_mask

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        per_layer_inputs = None
        if self.backbone.hidden_size_per_layer_input:
            per_layer_inputs = self.backbone.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        mask_kwargs = {
            "config": self.backbone.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": self._create_causal_mask(**mask_kwargs),
            "sliding_attention": self._create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        checkpoints: list[torch.Tensor] = []
        position_embeddings = {
            layer_type: self.backbone.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.backbone.unique_layer_types
        }
        shared_kv_states: dict[str, torch.Tensor] = {}

        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.backbone.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.backbone.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=None,
            )
            checkpoints.append(hidden_states)

        hidden_states = self.backbone.norm(hidden_states)
        checkpoints.append(hidden_states)
        return self.model.lm_head(hidden_states), checkpoints

    def debug_first_block(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        per_layer_inputs = None
        if self.backbone.hidden_size_per_layer_input:
            per_layer_inputs = self.backbone.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        mask_kwargs = {
            "config": self.backbone.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": self._create_causal_mask(**mask_kwargs),
            "sliding_attention": self._create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.backbone.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.backbone.unique_layer_types
        }
        shared_kv_states: dict[str, torch.Tensor] = {}
        layer = self.backbone.layers[0]
        layer_type = self.backbone.config.layer_types[0]
        per_layer_input = per_layer_inputs[:, :, 0, :] if per_layer_inputs is not None else None

        checkpoints: dict[str, torch.Tensor] = {}

        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        checkpoints["pre_attn_norm"] = normed

        attn_out = layer.self_attn(
            normed,
            position_embeddings=position_embeddings[layer_type],
            attention_mask=causal_mask_mapping[layer_type],
            position_ids=position_ids,
            past_key_values=None,
            shared_kv_states=shared_kv_states,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        checkpoints["attn_o_proj"] = attn_out

        post_attn_norm = layer.post_attention_layernorm(attn_out)
        checkpoints["post_attn_norm"] = post_attn_norm

        after_attention = residual + post_attn_norm
        checkpoints["after_attention_residual"] = after_attention

        pre_ffn_norm = layer.pre_feedforward_layernorm(after_attention)
        checkpoints["pre_ffn_norm"] = pre_ffn_norm

        mlp_out = layer.mlp(pre_ffn_norm)
        checkpoints["mlp_down"] = mlp_out

        post_ffn_norm = layer.post_feedforward_layernorm(mlp_out)
        checkpoints["post_ffn_norm"] = post_ffn_norm

        after_ffn = after_attention + post_ffn_norm
        checkpoints["after_ffn_residual"] = after_ffn

        if per_layer_input is not None:
            gated = layer.per_layer_input_gate(after_ffn)
            gated = layer.act_fn(gated)
            projected = gated * per_layer_input
            per_layer_proj = layer.per_layer_projection(projected)
            checkpoints["per_layer_input_proj"] = per_layer_proj
            post_per_layer_input_norm = layer.post_per_layer_input_norm(per_layer_proj)
            checkpoints["post_per_layer_input_norm"] = post_per_layer_input_norm
            after_ffn = after_ffn + post_per_layer_input_norm

        layer_scalar = getattr(layer, "layer_scalar", None)
        if layer_scalar is not None:
            after_ffn = after_ffn * layer_scalar
        checkpoints["layer_scalar_out"] = after_ffn
        return checkpoints

    def get_transpile_metadata(self):
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="gemma4",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
            },
            "import_hints": _gemma4_import_hints(self.backbone, module_path_suffix_prefix="backbone"),
        }


def _gemma4_apply_final_logit_softcapping(model: torch.nn.Module, logits: torch.Tensor) -> torch.Tensor:
    config = getattr(model, "config", None)
    text_config = None
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            text_config = get_text_config()
        except Exception:
            text_config = None
    cap = getattr(text_config, "final_logit_softcapping", None)
    if cap is None:
        cap = getattr(config, "final_logit_softcapping", None)
    if cap is None:
        return logits
    cap_value = float(cap)
    if cap_value <= 0.0:
        return logits
    return torch.tanh(logits / cap_value) * cap_value


class Gemma4MultimodalCausalLMLogitsAdapter(BoundInputAdapter):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...], weights_dir: str | None = None):
        super().__init__(
            model,
            input_names=input_names,
            family="gemma4",
            metadata_task="multimodal_causal_lm_logits",
        )
        model_backbone = model.model
        self.multimodal_backbone = model_backbone
        self.backbone = getattr(model_backbone, "language_model", model_backbone)
        self.last_token_logits_only = False
        self._use_cached_multimodal_features = False
        self._native_merge_plan: _Gemma4NativeMergePlan | None = None
        self.register_buffer("_cached_image_features", torch.empty(0), persistent=False)
        self.register_buffer("_cached_audio_features", torch.empty(0), persistent=False)
        self.register_buffer("_native_merge_pli_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        vision_post_proj_norm = _gemma4_load_vision_post_proj_norm(weights_dir)
        if vision_post_proj_norm is None:
            vision_post_proj_norm = torch.empty(0)
        self.register_buffer("_cactus_vision_post_proj_norm", vision_post_proj_norm, persistent=False)
        self._capture_cpu_float32_text_modules: list[tuple[torch.nn.Module, torch.dtype]] = []
        self._create_causal_mask_mapping = None
        self._create_masks_for_generate = None
        self._create_causal_mask = None
        self._create_sliding_window_causal_mask = None
        try:
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_sliding_window_causal_mask  # type: ignore

            self._create_causal_mask = create_causal_mask
            self._create_causal_mask_mapping = create_causal_mask_mapping
            self._create_masks_for_generate = create_masks_for_generate
            self._create_sliding_window_causal_mask = create_sliding_window_causal_mask
        except Exception:
            pass

    def _apply_final_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        return _gemma4_apply_final_logit_softcapping(self.model, logits)

    def _capture_text_modules(self) -> tuple[torch.nn.Module, ...]:
        return _unique_modules(
            self.backbone,
            getattr(self.model, "get_input_embeddings", lambda: None)(),
            getattr(self.model, "lm_head", None),
        )

    def _compute_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor | None,
        get_image_features: Callable[..., object],
    ) -> torch.Tensor:
        multimodal_backbone = self.multimodal_backbone
        if _gemma4_can_use_native_like_vision_features(multimodal_backbone):
            return _gemma4_compute_native_like_image_features(
                multimodal_backbone,
                pixel_values,
                pixel_position_ids,
                post_proj_norm_weight=self._cactus_vision_post_proj_norm,
            )
        vision_tower = getattr(multimodal_backbone, "vision_tower", None)
        embed_vision = getattr(multimodal_backbone, "embed_vision", None)
        vision_modules = _gemma4_vision_modules(multimodal_backbone)
        if (
            pixel_values.device.type != "cpu"
            or pixel_values.dtype != torch.float16
            or len(vision_modules) != 2
            or _torch_is_compiling()
        ):
            return get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

        vision_dtype = _module_floating_dtype(vision_tower)
        embed_dtype = _module_floating_dtype(embed_vision)
        if vision_dtype != torch.float16 and embed_dtype != torch.float16:
            return get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

        # Gemma4's CPU float16 vision path can emit non-finite soft tokens; upcast only
        # the static image feature extraction path and restore the original module dtypes.
        with _temporary_cpu_float32_modules(vision_modules):
            return get_image_features(
                pixel_values.float(),
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

    def _compute_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        get_audio_features: Callable[..., object],
    ) -> torch.Tensor:
        multimodal_backbone = self.multimodal_backbone
        if _gemma4_can_use_native_like_audio_features(multimodal_backbone):
            return _gemma4_compute_native_like_audio_features(
                multimodal_backbone,
                input_features,
                input_features_mask,
            )
        audio_modules = _gemma4_audio_modules(multimodal_backbone)
        if (
            input_features.device.type != "cpu"
            or input_features.dtype != torch.float16
            or len(audio_modules) != 2
            or _torch_is_compiling()
        ):
            audio_output = get_audio_features(input_features, ~input_features_mask, return_dict=True)
            return _gemma4_strip_audio_padding(audio_output)

        if all(_module_floating_dtype(module) != torch.float16 for module in audio_modules):
            audio_output = get_audio_features(input_features, ~input_features_mask, return_dict=True)
            return _gemma4_strip_audio_padding(audio_output)

        with _temporary_cpu_float32_modules(audio_modules):
            audio_output = get_audio_features(input_features.float(), ~input_features_mask, return_dict=True)
            return _gemma4_strip_audio_padding(audio_output)

    def prepare_cpu_float32_capture(self) -> None:
        self._capture_cpu_float32_text_modules.clear()
        if os.environ.get("CACTUS_GEMMA4_CAPTURE_FP32") != "1":
            return
        for module in self._capture_text_modules():
            device = _module_device(module)
            dtype = _module_floating_dtype(module)
            if device is None or device.type != "cpu" or dtype is None or dtype == torch.float32:
                continue
            self._capture_cpu_float32_text_modules.append((module, dtype))
            module.to(dtype=torch.float32)

    def restore_cpu_float32_capture(self) -> None:
        for module, dtype in reversed(self._capture_cpu_float32_text_modules):
            module.to(dtype=dtype)
        self._capture_cpu_float32_text_modules.clear()

    def _resolve_image_features(
        self,
        *,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor | None,
        pixel_position_ids: torch.Tensor | None,
        get_image_features: Callable[..., object],
    ) -> torch.Tensor | None:
        if pixel_values is None:
            return None
        if self._use_cached_multimodal_features and self._cached_image_features.numel() > 0:
            image_features = self._cached_image_features
        else:
            image_features = self._compute_image_features(
                pixel_values,
                pixel_position_ids,
                get_image_features,
            )
        return image_features.to(inputs_embeds.device, inputs_embeds.dtype)

    def _resolve_audio_features(
        self,
        *,
        inputs_embeds: torch.Tensor,
        input_features: torch.Tensor | None,
        input_features_mask: torch.Tensor | None,
        get_audio_features: Callable[..., object],
    ) -> torch.Tensor | None:
        if input_features is None or input_features_mask is None:
            return None
        if self._use_cached_multimodal_features and self._cached_audio_features.numel() > 0:
            audio_features = self._cached_audio_features
        else:
            audio_features = self._compute_audio_features(
                input_features,
                input_features_mask,
                get_audio_features,
            )
        return audio_features.to(inputs_embeds.device, inputs_embeds.dtype)

    def _prepare_text_backbone_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        pixel_position_ids: torch.Tensor | None,
        input_features: torch.Tensor | None,
        input_features_mask: torch.Tensor | None,
        get_placeholder_mask: Callable[..., object],
        get_image_features: Callable[..., object],
        get_audio_features: Callable[..., object],
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor], torch.LongTensor]:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        image_features = self._resolve_image_features(
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            get_image_features=get_image_features,
        )
        audio_features = self._resolve_audio_features(
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            input_features_mask=input_features_mask,
            get_audio_features=get_audio_features,
        )

        if image_features is not None and audio_features is not None and self._native_merge_plan is not None:
            inputs_embeds, per_layer_inputs_tokens = _gemma4_apply_native_merge_plan(
                self.model,
                input_ids=input_ids,
                image_features=image_features,
                audio_features=audio_features,
                merge_plan=self._native_merge_plan,
                pli_token_ids=self._native_merge_pli_token_ids,
            )
            attention_mask = _gemma4_remap_sequence_tensor(
                attention_mask,
                merge_plan=self._native_merge_plan,
            )
            token_type_ids = _gemma4_remap_sequence_tensor(
                token_type_ids,
                merge_plan=self._native_merge_plan,
            )
            if token_type_ids is None:
                raise RuntimeError("Gemma4 native merge requires token_type_ids for multimodal attention masking")
        else:
            text_mask, image_mask, audio_mask = get_placeholder_mask(
                token_type_ids=token_type_ids,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )
            if image_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    image_features,
                )
            if audio_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    audio_features,
                )
            per_layer_inputs_tokens = input_ids * text_mask.to(dtype=input_ids.dtype)

        per_layer_inputs = None
        text_config = _gemma4_text_config(self.multimodal_backbone)
        if getattr(text_config, "hidden_size_per_layer_input", None):
            per_layer_inputs = self.backbone.get_per_layer_inputs(per_layer_inputs_tokens)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        if self._native_merge_plan is not None:
            causal_mask_mapping = _gemma4_build_standard_causal_mask_mapping(
                create_causal_mask=self._create_causal_mask,
                create_sliding_window_causal_mask=self._create_sliding_window_causal_mask,
                config=self.backbone.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        elif getattr(text_config, "use_bidirectional_attention", None) == "vision":
            causal_mask_mapping = self._create_causal_mask_mapping(
                self.multimodal_backbone.config,
                inputs_embeds,
                attention_mask,
                None,
                position_ids,
                token_type_ids,
                pixel_values,
                is_training=self.training,
            )
        else:
            causal_mask_mapping = self._create_masks_for_generate(
                self.multimodal_backbone.config,
                inputs_embeds,
                attention_mask,
                None,
                position_ids,
            )
        return inputs_embeds, per_layer_inputs, causal_mask_mapping, position_ids

    def _forward_hidden_states(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        pixel_position_ids: torch.Tensor | None,
        input_features: torch.Tensor | None,
        input_features_mask: torch.Tensor | None,
        get_placeholder_mask: Callable[..., object],
        get_image_features: Callable[..., object],
        get_audio_features: Callable[..., object],
    ) -> torch.Tensor:
        inputs_embeds, per_layer_inputs, causal_mask_mapping, position_ids = self._prepare_text_backbone_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            get_placeholder_mask=get_placeholder_mask,
            get_image_features=get_image_features,
            get_audio_features=get_audio_features,
        )
        return _gemma4_text_backbone_forward(
            self.backbone,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            causal_mask_mapping=causal_mask_mapping,
            position_ids=position_ids,
        )

    def prime_static_multimodal_features(self, *bound_inputs: torch.Tensor | None) -> None:
        kwargs = self._kwargs_from_bound_inputs(*bound_inputs)
        input_ids = kwargs["input_ids"]
        token_type_ids = kwargs.get("token_type_ids")
        pixel_values = kwargs.get("pixel_values")
        pixel_position_ids = kwargs.get("pixel_position_ids")
        input_features = kwargs.get("input_features")
        input_features_mask = kwargs.get("input_features_mask")

        multimodal_backbone = self.multimodal_backbone
        get_placeholder_mask = getattr(multimodal_backbone, "get_placeholder_mask", None)
        get_image_features = getattr(multimodal_backbone, "get_image_features", None)
        get_audio_features = getattr(multimodal_backbone, "get_audio_features", None)
        if not callable(get_placeholder_mask):
            return

        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            _, image_mask, audio_mask = get_placeholder_mask(
                token_type_ids=token_type_ids,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )

            if pixel_values is not None and callable(get_image_features) and image_mask.any():
                image_features = self._compute_image_features(
                    pixel_values,
                    pixel_position_ids,
                    get_image_features,
                )
                self._cached_image_features = image_features.detach()
            else:
                self._cached_image_features = self._cached_image_features.new_empty(0)

            if (
                input_features is not None
                and input_features_mask is not None
                and callable(get_audio_features)
                and audio_mask.any()
            ):
                self._cached_audio_features = self._compute_audio_features(
                    input_features,
                    input_features_mask,
                    get_audio_features,
                ).detach()
            else:
                self._cached_audio_features = self._cached_audio_features.new_empty(0)

            if self._cached_image_features.numel() > 0 and self._cached_audio_features.numel() > 0:
                plan = _gemma4_build_native_merge_plan(
                    self.multimodal_backbone,
                    input_ids,
                    image_feature_count=_gemma4_feature_token_count(self._cached_image_features),
                    audio_feature_count=_gemma4_feature_token_count(self._cached_audio_features),
                )
                self._native_merge_plan = plan
                self._native_merge_pli_token_ids = torch.tensor(
                    plan.pli_token_ids,
                    dtype=torch.long,
                )
            else:
                self._native_merge_plan = None
                self._native_merge_pli_token_ids = self._native_merge_pli_token_ids.new_empty(0)

        self._use_cached_multimodal_features = True

    def forward(self, *bound_inputs: torch.Tensor | None) -> torch.Tensor:
        kwargs = self._kwargs_from_bound_inputs(*bound_inputs)
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask")
        token_type_ids = kwargs.get("token_type_ids")
        pixel_values = kwargs.get("pixel_values")
        pixel_position_ids = kwargs.get("pixel_position_ids")
        input_features = kwargs.get("input_features")
        input_features_mask = kwargs.get("input_features_mask")

        multimodal_backbone = self.multimodal_backbone
        get_placeholder_mask = getattr(multimodal_backbone, "get_placeholder_mask", None)
        get_image_features = getattr(multimodal_backbone, "get_image_features", None)
        get_audio_features = getattr(multimodal_backbone, "get_audio_features", None)
        lm_head = getattr(self.model, "lm_head", None)
        if (
            not callable(get_placeholder_mask)
            or not callable(get_image_features)
            or not callable(get_audio_features)
            or not callable(self._create_causal_mask_mapping)
            or not callable(self._create_masks_for_generate)
            or not isinstance(lm_head, torch.nn.Module)
        ):
            outputs = self.model(
                return_dict=True,
                use_cache=False,
                **kwargs,
            )
            logits = _extract_tensor_output(outputs, preferred_field="logits")
            if self.last_token_logits_only and logits.ndim >= 3:
                return _select_last_active_token(logits, attention_mask)
            return logits

        hidden_states = self._forward_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            get_placeholder_mask=get_placeholder_mask,
            get_image_features=get_image_features,
            get_audio_features=get_audio_features,
        )
        if self.last_token_logits_only:
            hidden_states = _select_last_active_token(hidden_states, attention_mask)
        return self._apply_final_logit_softcapping(lm_head(hidden_states))

    def get_transpile_metadata(self):
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="gemma4",
                    adapter_type=type(self).__name__,
                    input_names=self.input_names,
                ),
                "task": self.metadata_task,
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
                "last_token_logits_only": bool(self.last_token_logits_only),
            },
            "import_hints": _gemma4_import_hints(
                self.backbone,
                module_path_suffix_prefix="backbone",
            ),
        }


_GEMMA4_DECODER_PIPELINE_OUTPUT_KEYS = (
    "inputs_embeds",
    "per_layer_inputs",
    "full_attention_mask",
    "sliding_attention_mask",
    "position_ids",
)


class _Gemma4MultimodalComponentBase(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        input_names: tuple[str, ...],
        weights_dir: str | None = None,
        native_merge_plan: _Gemma4NativeMergePlan | None = None,
    ):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.multimodal_backbone = model.model
        model_backbone = model.model
        self.backbone = getattr(model_backbone, "language_model", model_backbone)
        self._native_merge_plan = native_merge_plan
        self.register_buffer(
            "_native_merge_pli_token_ids",
            torch.tensor(native_merge_plan.pli_token_ids, dtype=torch.long)
            if native_merge_plan is not None
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        vision_post_proj_norm = _gemma4_load_vision_post_proj_norm(weights_dir)
        if vision_post_proj_norm is None:
            vision_post_proj_norm = torch.empty(0)
        self.register_buffer("_cactus_vision_post_proj_norm", vision_post_proj_norm, persistent=False)
        self._create_causal_mask_mapping = None
        self._create_masks_for_generate = None
        self._create_causal_mask = None
        self._create_sliding_window_causal_mask = None
        try:
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_sliding_window_causal_mask  # type: ignore

            self._create_causal_mask = create_causal_mask
            self._create_causal_mask_mapping = create_causal_mask_mapping
            self._create_masks_for_generate = create_masks_for_generate
            self._create_sliding_window_causal_mask = create_sliding_window_causal_mask
        except Exception:
            pass
        self._capture_modules: list[tuple[torch.nn.Module, torch.dtype]] = []

    def _base_graph_meta(self, *, adapter_type: str, input_names: tuple[str, ...]) -> dict[str, object]:
        sliding_window = getattr(self.backbone.config, "sliding_window", None)
        layer_types = list(getattr(self.backbone.config, "layer_types", []))
        return {
            **_transpile_graph_meta(
                self.model,
                adapter_family="gemma4",
                adapter_type=adapter_type,
                input_names=input_names,
            ),
            "task": "multimodal_causal_lm_logits",
            "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
            "layer_types": tuple(layer_types),
            "sliding_window": None if sliding_window is None else int(sliding_window),
        }

    def _compute_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if _gemma4_can_use_native_like_vision_features(self.multimodal_backbone):
            return _gemma4_compute_native_like_image_features(
                self.multimodal_backbone,
                pixel_values,
                pixel_position_ids,
                post_proj_norm_weight=self._cactus_vision_post_proj_norm,
            )
        get_image_features = getattr(self.multimodal_backbone, "get_image_features", None)
        vision_tower = getattr(self.multimodal_backbone, "vision_tower", None)
        embed_vision = getattr(self.multimodal_backbone, "embed_vision", None)
        vision_modules = _gemma4_vision_modules(self.multimodal_backbone)
        if not callable(get_image_features):
            raise TypeError("Gemma4 multimodal backbone is missing get_image_features")
        if (
            pixel_values.device.type != "cpu"
            or pixel_values.dtype != torch.float16
            or len(vision_modules) != 2
            or _torch_is_compiling()
        ):
            return get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

        vision_dtype = _module_floating_dtype(vision_tower)
        embed_dtype = _module_floating_dtype(embed_vision)
        if vision_dtype != torch.float16 and embed_dtype != torch.float16:
            return get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

        with _temporary_cpu_float32_modules(vision_modules):
            return get_image_features(
                pixel_values.float(),
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output

    def _compute_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
    ) -> torch.Tensor:
        if _gemma4_can_use_native_like_audio_features(self.multimodal_backbone):
            return _gemma4_compute_native_like_audio_features(
                self.multimodal_backbone,
                input_features,
                input_features_mask,
            )
        get_audio_features = getattr(self.multimodal_backbone, "get_audio_features", None)
        if not callable(get_audio_features):
            raise TypeError("Gemma4 multimodal backbone is missing get_audio_features")
        audio_modules = _gemma4_audio_modules(self.multimodal_backbone)
        if (
            input_features.device.type == "cpu"
            and input_features.dtype == torch.float16
            and len(audio_modules) == 2
            and not _torch_is_compiling()
            and any(_module_floating_dtype(module) == torch.float16 for module in audio_modules)
        ):
            with _temporary_cpu_float32_modules(audio_modules):
                audio_output = get_audio_features(input_features.float(), ~input_features_mask, return_dict=True)
                return _gemma4_strip_audio_padding(audio_output)

        audio_output = get_audio_features(input_features, ~input_features_mask, return_dict=True)
        return _gemma4_strip_audio_padding(audio_output)

    def _modules_to_prepare_for_capture(self) -> tuple[torch.nn.Module, ...]:
        return ()

    def prepare_for_capture(self, **_: object) -> None:
        self.restore_after_capture()
        modules = self._modules_to_prepare_for_capture()
        if not modules:
            return
        promoted: list[tuple[torch.nn.Module, torch.dtype]] = []
        for module in modules:
            device = _module_device(module)
            dtype = _module_floating_dtype(module)
            if device is None or device.type != "cpu" or dtype is None or dtype == torch.float32:
                continue
            promoted.append((module, dtype))
            module.to(dtype=torch.float32)
        self._capture_modules = promoted

    def restore_after_capture(self, **_: object) -> None:
        for module, dtype in reversed(self._capture_modules):
            module.to(dtype=dtype)
        self._capture_modules.clear()

    def _prepare_decoder_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self._native_merge_plan is not None:
            inputs_embeds, per_layer_inputs_tokens = _gemma4_apply_native_merge_plan(
                self.model,
                input_ids=input_ids,
                image_features=image_features,
                audio_features=audio_features,
                merge_plan=self._native_merge_plan,
                pli_token_ids=self._native_merge_pli_token_ids,
            )
            attention_mask = _gemma4_remap_sequence_tensor(
                attention_mask,
                merge_plan=self._native_merge_plan,
            )
            token_type_ids = _gemma4_remap_sequence_tensor(
                token_type_ids,
                merge_plan=self._native_merge_plan,
            )
            if token_type_ids is None:
                raise RuntimeError("Gemma4 native merge requires token_type_ids for multimodal attention masking")
        else:
            get_placeholder_mask = getattr(self.multimodal_backbone, "get_placeholder_mask", None)
            if not callable(get_placeholder_mask):
                raise TypeError("Gemma4 multimodal backbone is missing get_placeholder_mask")

            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            text_mask, image_mask, audio_mask = get_placeholder_mask(
                token_type_ids=token_type_ids,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_features.to(inputs_embeds.device, inputs_embeds.dtype),
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask.unsqueeze(-1).expand_as(inputs_embeds),
                audio_features.to(inputs_embeds.device, inputs_embeds.dtype),
            )
            per_layer_inputs_tokens = input_ids * text_mask.to(dtype=input_ids.dtype)

        per_layer_inputs: torch.Tensor
        text_config = _gemma4_text_config(self.multimodal_backbone)
        if getattr(text_config, "hidden_size_per_layer_input", None):
            per_layer_inputs = self.backbone.get_per_layer_inputs(per_layer_inputs_tokens)
            per_layer_inputs = self.backbone.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        else:
            per_layer_inputs = inputs_embeds.new_empty((inputs_embeds.shape[0], inputs_embeds.shape[1], 0, 0))

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        if self._native_merge_plan is not None:
            causal_mask_mapping = _gemma4_build_standard_causal_mask_mapping(
                create_causal_mask=self._create_causal_mask,
                create_sliding_window_causal_mask=self._create_sliding_window_causal_mask,
                config=self.backbone.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        elif getattr(text_config, "use_bidirectional_attention", None) == "vision":
            if not callable(self._create_causal_mask_mapping):
                raise RuntimeError("Gemma4 component LM encoder requires create_causal_mask_mapping")
            causal_mask_mapping = self._create_causal_mask_mapping(
                self.multimodal_backbone.config,
                inputs_embeds,
                attention_mask,
                None,
                position_ids,
                token_type_ids,
                None,
                is_training=self.training,
            )
        else:
            if not callable(self._create_masks_for_generate):
                raise RuntimeError("Gemma4 component LM encoder requires create_masks_for_generate")
            causal_mask_mapping = self._create_masks_for_generate(
                self.multimodal_backbone.config,
                inputs_embeds,
                attention_mask,
                None,
                position_ids,
            )
        full_attention_mask = causal_mask_mapping["full_attention"]
        sliding_attention_mask = causal_mask_mapping.get("sliding_attention", full_attention_mask)
        return (
            inputs_embeds,
            per_layer_inputs,
            full_attention_mask,
            sliding_attention_mask,
            position_ids,
        )


class Gemma4VisionEncoderAdapter(_Gemma4MultimodalComponentBase):
    def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__(model, input_names=("pixel_values", "pixel_position_ids"), weights_dir=weights_dir)

    def _modules_to_prepare_for_capture(self) -> tuple[torch.nn.Module, ...]:
        return ()

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor | None) -> torch.Tensor:
        return self._compute_image_features(pixel_values, pixel_position_ids)

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=("pixel_values", "pixel_position_ids"),
            ),
        }


class Gemma4AudioEncoderAdapter(_Gemma4MultimodalComponentBase):
    def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__(model, input_names=("input_features", "input_features_mask"), weights_dir=weights_dir)

    def _modules_to_prepare_for_capture(self) -> tuple[torch.nn.Module, ...]:
        return ()

    def forward(self, input_features: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
        return self._compute_audio_features(input_features, input_features_mask)

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=("input_features", "input_features_mask"),
            ),
        }


class Gemma4LMEncoderAdapter(_Gemma4MultimodalComponentBase):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        weights_dir: str | None = None,
        native_merge_plan: _Gemma4NativeMergePlan | None = None,
    ):
        super().__init__(
            model,
            input_names=("input_ids", "attention_mask", "token_type_ids", "image_features", "audio_features"),
            weights_dir=weights_dir,
            native_merge_plan=native_merge_plan,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        return self._prepare_decoder_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_features=image_features,
            audio_features=audio_features,
        )

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=("input_ids", "attention_mask", "token_type_ids", "image_features", "audio_features"),
            ),
        }


class Gemma4DecoderAdapter(_Gemma4MultimodalComponentBase):
    def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__(model, input_names=_GEMMA4_DECODER_PIPELINE_OUTPUT_KEYS, weights_dir=weights_dir)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        full_attention_mask: torch.Tensor,
        sliding_attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        normalized_per_layer_inputs = per_layer_inputs
        if normalized_per_layer_inputs.numel() == 0:
            normalized_per_layer_inputs = None
        hidden_states = _gemma4_text_backbone_forward(
            self.backbone,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=normalized_per_layer_inputs,
            causal_mask_mapping={
                "full_attention": full_attention_mask,
                "sliding_attention": sliding_attention_mask,
            },
            position_ids=position_ids,
        )
        logits = self.model.lm_head(hidden_states[:, -1:, :])
        return _gemma4_apply_final_logit_softcapping(self.model, logits)

    def get_transpile_metadata(self):
        return {
            "graph": self._base_graph_meta(
                adapter_type=type(self).__name__,
                input_names=_GEMMA4_DECODER_PIPELINE_OUTPUT_KEYS,
            ),
            "import_hints": _gemma4_import_hints(
                self.backbone,
                module_path_suffix_prefix="backbone",
            ),
        }


def _build_gemma4_multimodal_component_specs(
    model: torch.nn.Module,
    *,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None,
    components: tuple[str, ...] | None = None,
) -> list[ComponentModuleSpec]:
    pixel_values = named_tensors["pixel_values"]
    pixel_position_ids = named_tensors.get("pixel_position_ids")
    input_features = named_tensors["input_features"]
    input_features_mask = named_tensors["input_features_mask"]
    input_ids = named_tensors["input_ids"]
    attention_mask = named_tensors.get("attention_mask")
    token_type_ids = named_tensors["token_type_ids"]

    requested_components = tuple(components or ("vision_encoder", "audio_encoder", "lm_encoder", "decoder"))
    requested_set = set(requested_components)
    if not requested_set:
        return []

    expanded_components: list[str] = []

    def _require(component: str) -> None:
        if component not in expanded_components:
            expanded_components.append(component)

    if "decoder" in requested_set:
        _require("vision_encoder")
        _require("audio_encoder")
        _require("lm_encoder")
        _require("decoder")
    elif "lm_encoder" in requested_set:
        _require("vision_encoder")
        _require("audio_encoder")
        _require("lm_encoder")
    else:
        if "vision_encoder" in requested_set:
            _require("vision_encoder")
        if "audio_encoder" in requested_set:
            _require("audio_encoder")

    vision_encoder = Gemma4VisionEncoderAdapter(model, weights_dir=weights_dir).eval()
    audio_encoder = Gemma4AudioEncoderAdapter(model, weights_dir=weights_dir).eval()

    image_features: torch.Tensor | None = None
    audio_features: torch.Tensor | None = None
    decoder_inputs: tuple[torch.Tensor, ...] | None = None
    native_merge_plan: _Gemma4NativeMergePlan | None = None

    with torch.no_grad():
        if "vision_encoder" in expanded_components and ("lm_encoder" in expanded_components or "decoder" in expanded_components):
            image_features = vision_encoder(pixel_values, pixel_position_ids)
        if "audio_encoder" in expanded_components and ("lm_encoder" in expanded_components or "decoder" in expanded_components):
            audio_features = audio_encoder(input_features, input_features_mask)
        if "lm_encoder" in expanded_components or "decoder" in expanded_components:
            if image_features is None:
                image_features = vision_encoder(pixel_values, pixel_position_ids)
            if audio_features is None:
                audio_features = audio_encoder(input_features, input_features_mask)
            native_merge_plan = _gemma4_build_native_merge_plan(
                getattr(model, "model", model),
                input_ids,
                image_feature_count=_gemma4_feature_token_count(image_features),
                audio_feature_count=_gemma4_feature_token_count(audio_features),
            )
    lm_encoder = Gemma4LMEncoderAdapter(
        model,
        weights_dir=weights_dir,
        native_merge_plan=native_merge_plan,
    ).eval()
    decoder = Gemma4DecoderAdapter(model, weights_dir=weights_dir).eval()

    with torch.no_grad():
        if "decoder" in expanded_components:
            decoder_inputs = lm_encoder(
                input_ids,
                attention_mask,
                token_type_ids,
                image_features,
                audio_features,
            )

    common_graph_meta = {
        "weights_dir": weights_dir,
        "task": "multimodal_causal_lm_logits",
        "adapter_family": "gemma4",
    }
    specs: list[ComponentModuleSpec] = []
    if "vision_encoder" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="vision_encoder",
            module=vision_encoder,
            example_inputs=(pixel_values, pixel_position_ids),
            input_keys=("pixel_values", "pixel_position_ids"),
            output_keys=("image_features",),
            graph_meta={**common_graph_meta, "component": "vision_encoder"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "audio_encoder" in expanded_components:
        specs.append(ComponentModuleSpec(
            component="audio_encoder",
            module=audio_encoder,
            example_inputs=(input_features, input_features_mask),
            input_keys=("input_features", "input_features_mask"),
            output_keys=("audio_features",),
            graph_meta={**common_graph_meta, "component": "audio_encoder"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "lm_encoder" in expanded_components:
        if image_features is None or audio_features is None:
            raise RuntimeError("Gemma4 lm_encoder spec requires precomputed image/audio features")
        specs.append(ComponentModuleSpec(
            component="lm_encoder",
            module=lm_encoder,
            example_inputs=(input_ids, attention_mask, token_type_ids, image_features, audio_features),
            input_keys=("input_ids", "attention_mask", "token_type_ids", "image_features", "audio_features"),
            output_keys=_GEMMA4_DECODER_PIPELINE_OUTPUT_KEYS,
            graph_meta={**common_graph_meta, "component": "lm_encoder"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    if "decoder" in expanded_components:
        if decoder_inputs is None:
            raise RuntimeError("Gemma4 decoder spec requires precomputed decoder inputs")
        specs.append(ComponentModuleSpec(
            component="decoder",
            module=decoder,
            example_inputs=tuple(decoder_inputs),
            input_keys=_GEMMA4_DECODER_PIPELINE_OUTPUT_KEYS,
            output_keys=("logits",),
            graph_meta={**common_graph_meta, "component": "decoder"},
            metadata={"family": "gemma4", "task": "multimodal_causal_lm_logits"},
        ))
    return specs


class Qwen35CausalLMLogitsAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.backbone = model.model
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask  # type: ignore

        self._create_causal_mask = create_causal_mask

    def forward(self, input_ids: torch.Tensor):
        return self.debug_forward(input_ids)[0]

    def debug_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        text_position_ids = position_ids[0]
        multimodal_position_ids = position_ids[1:]

        causal_mask = self._create_causal_mask(
            config=self.backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self.backbone._update_linear_attn_mask(None, None)

        hidden_states = inputs_embeds
        checkpoints: list[torch.Tensor] = []
        position_embeddings = self.backbone.rotary_emb(hidden_states, multimodal_position_ids)

        for i, decoder_layer in enumerate(self.backbone.layers[: self.backbone.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if self.backbone.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
            checkpoints.append(hidden_states)

        hidden_states = self.backbone.norm(hidden_states)
        checkpoints.append(hidden_states)
        return self.model.lm_head(hidden_states), checkpoints

    def get_transpile_metadata(self):
        layer_types = tuple(getattr(self.backbone.config, "layer_types", ()))
        import_hints: list[dict[str, object]] = []
        for layer_index, layer_type in enumerate(layer_types):
            import_hints.append(
                {
                    "module_path_suffix": f"backbone.layers.{layer_index}.self_attn",
                    "meta": {
                        "attention_layer_type": layer_type,
                        "attention_layer_index": layer_index,
                    },
                }
            )
        return {
            "graph": {
                **_transpile_graph_meta(
                    self.model,
                    adapter_family="qwen3_5",
                    adapter_type=type(self).__name__,
                    input_names=("input_ids",),
                ),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": layer_types,
            },
            "import_hints": import_hints,
        }


class CTCLogitsAdapter(BoundInputAdapter):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...], family: str):
        super().__init__(model, input_names=input_names, family=family, metadata_task="ctc_logits")

    def forward(self, *bound_inputs: torch.Tensor | None) -> torch.Tensor:
        outputs = self.model(return_dict=True, **self._kwargs_from_bound_inputs(*bound_inputs))
        return _extract_tensor_output(outputs, preferred_field="logits")


class EncoderHiddenStatesAdapter(BoundInputAdapter):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...], family: str):
        encoder = None
        get_encoder = getattr(model, "get_encoder", None)
        if callable(get_encoder):
            encoder = get_encoder()
        if encoder is None:
            encoder = getattr(model, "encoder", None)
        if encoder is None:
            model_attr = getattr(model, "model", None)
            if model_attr is not None:
                encoder = getattr(model_attr, "encoder", None)
        if not isinstance(encoder, torch.nn.Module):
            raise NotImplementedError(f"{type(model).__name__} does not expose an encoder module")
        super().__init__(model, input_names=input_names, family=family, metadata_task="encoder_hidden_states")
        self.encoder = encoder

    def forward(self, *bound_inputs: torch.Tensor | None) -> torch.Tensor:
        outputs = self.encoder(return_dict=True, **self._kwargs_from_bound_inputs(*bound_inputs))
        return _extract_tensor_output(outputs, preferred_field="last_hidden_state")


def _family_key(model: torch.nn.Module) -> str:
    explicit_family = getattr(model, "family", None)
    if isinstance(explicit_family, str) and explicit_family:
        return explicit_family.lower()
    module_name = type(model).__module__
    if module_name.startswith("transformers.models.whisper."):
        return "whisper"
    if module_name.startswith("transformers.models.gemma4."):
        return "gemma4"
    if module_name.startswith("transformers.models.gemma3."):
        return "gemma3"
    if module_name.startswith("transformers.models.gemma."):
        return "gemma"
    if module_name.startswith("transformers.models.qwen3_5."):
        return "qwen3_5"
    if module_name.startswith("transformers.models.lfm2_vl."):
        return "lfm2_vl"
    if module_name.startswith("transformers.models.lfm2_moe."):
        return "lfm2_moe"
    if module_name.startswith("transformers.models.lfm2."):
        return "lfm2"
    return "generic"


def build_component_module_specs(
    model: torch.nn.Module,
    *,
    task: str,
    named_tensors: dict[str, torch.Tensor],
    weights_dir: str | None = None,
    components: tuple[str, ...] | None = None,
) -> list[ComponentModuleSpec] | None:
    family = _family_key(model)
    if family == "gemma4" and task == "multimodal_causal_lm_logits":
        return _build_gemma4_multimodal_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
            components=components,
        )
    if family == "parakeet_tdt" and task == "tdt_transcription":
        from src.transpile.parakeet_tdt_local import build_parakeet_tdt_component_specs

        return build_parakeet_tdt_component_specs(
            model,
            named_tensors=named_tensors,
            weights_dir=weights_dir,
        )
    return None


def canonicalize_model_interface(
    model: torch.nn.Module,
    task: str = "causal_lm_logits",
    *,
    input_names: tuple[str, ...] | None = None,
    weights_dir: str | None = None,
) -> CanonicalizedModel:
    family = _family_key(model)
    adapter_factory: Callable[[torch.nn.Module], torch.nn.Module]
    resolved_input_names = tuple(input_names or ())

    if task == "causal_lm_logits":
        if family == "gemma":
            adapter_factory = GemmaCausalLMLogitsAdapter
        elif family == "gemma4":
            adapter_factory = Gemma4CausalLMLogitsAdapter
        elif family == "gemma3":
            adapter_factory = Gemma3CausalLMLogitsAdapter
        elif family == "qwen3_5":
            adapter_factory = Qwen35CausalLMLogitsAdapter
        else:
            adapter_factory = CausalLMLogitsAdapter
        resolved_input_names = ("input_ids",)
    elif task == "multimodal_causal_lm_logits":
        if family != "gemma4":
            raise NotImplementedError(f"{type(model).__name__} does not support task={task}")
        if not resolved_input_names:
            resolved_input_names = (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "pixel_values",
                "pixel_position_ids",
                "input_features",
                "input_features_mask",
            )
        adapter_factory = lambda inner_model: Gemma4MultimodalCausalLMLogitsAdapter(  # type: ignore[assignment]
            inner_model,
            input_names=resolved_input_names,
            weights_dir=weights_dir,
        )
    elif task == "ctc_logits":
        if not resolved_input_names:
            resolved_input_names = _infer_input_names(
                model,
                preferred=("input_values", "input_features", "attention_mask"),
            )
        adapter_factory = lambda inner_model: CTCLogitsAdapter(  # type: ignore[assignment]
            inner_model,
            input_names=resolved_input_names,
            family=family,
        )
    elif task == "encoder_hidden_states":
        encoder_module = None
        get_encoder = getattr(model, "get_encoder", None)
        if callable(get_encoder):
            encoder_module = get_encoder()
        if encoder_module is None:
            encoder_module = getattr(model, "encoder", None)
        if encoder_module is None and getattr(model, "model", None) is not None:
            encoder_module = getattr(model.model, "encoder", None)
        if not isinstance(encoder_module, torch.nn.Module):
            raise NotImplementedError(f"{type(model).__name__} does not support task={task}")
        if not resolved_input_names:
            resolved_input_names = _infer_input_names(
                encoder_module,
                preferred=("input_features", "input_values", "attention_mask"),
            )
        adapter_factory = lambda inner_model: EncoderHiddenStatesAdapter(  # type: ignore[assignment]
            inner_model,
            input_names=resolved_input_names,
            family=family,
        )
    else:
        raise NotImplementedError(f"unsupported canonicalization task: {task}")

    return CanonicalizedModel(
        module=adapter_factory(model).eval(),
        task=task,
        family=family,
        input_names=resolved_input_names,
    )
