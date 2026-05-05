from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable

import torch


@dataclass
class CanonicalizedModel:
    module: torch.nn.Module
    task: str
    family: str
    input_names: tuple[str, ...] = ()


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
        return self.model.lm_head(hidden_states), checkpoints

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


class Gemma4MultimodalCausalLMLogitsAdapter(BoundInputAdapter):
    def __init__(self, model: torch.nn.Module, *, input_names: tuple[str, ...]):
        super().__init__(
            model,
            input_names=input_names,
            family="gemma4",
            metadata_task="multimodal_causal_lm_logits",
        )
        model_backbone = model.model
        self.multimodal_backbone = model_backbone
        self.backbone = getattr(model_backbone, "language_model", model_backbone)
        self._create_causal_mask_mapping = None
        self._create_masks_for_generate = None
        try:
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping  # type: ignore
            from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate  # type: ignore

            self._create_causal_mask_mapping = create_causal_mask_mapping
            self._create_masks_for_generate = create_masks_for_generate
        except Exception:
            pass

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
            return _extract_tensor_output(outputs, preferred_field="logits")

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        text_mask, image_mask, audio_mask = get_placeholder_mask(
            token_type_ids=token_type_ids,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        per_layer_inputs = None
        text_config = self.multimodal_backbone.config.get_text_config()
        if getattr(text_config, "hidden_size_per_layer_input", None):
            per_layer_inputs_tokens = input_ids * text_mask.to(dtype=input_ids.dtype)
            per_layer_inputs = self.backbone.get_per_layer_inputs(per_layer_inputs_tokens)

        if pixel_values is not None:
            image_features = get_image_features(
                pixel_values,
                pixel_position_ids,
                None,
                return_dict=True,
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_features)

        if input_features is not None and input_features_mask is not None:
            audio_output = get_audio_features(input_features, ~input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask_expanded = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask_expanded, audio_features)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        if getattr(text_config, "use_bidirectional_attention", None) == "vision":
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

        hidden_states = self.backbone(
            input_ids=None,
            per_layer_inputs=per_layer_inputs,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return lm_head(hidden_states)

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
            },
            "import_hints": _gemma4_import_hints(
                self.backbone,
                module_path_suffix_prefix="model.model.language_model",
            ),
        }


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
    return "generic"


def canonicalize_model_interface(
    model: torch.nn.Module,
    task: str = "causal_lm_logits",
    *,
    input_names: tuple[str, ...] | None = None,
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
