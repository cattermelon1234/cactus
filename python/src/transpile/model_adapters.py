from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class CanonicalizedModel:
    module: torch.nn.Module
    task: str
    family: str


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
                "adapter_family": "generic",
                "adapter_type": type(self).__name__,
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
                "adapter_family": "gemma",
                "adapter_type": type(self).__name__,
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
        import_hints: list[dict[str, object]] = []
        for layer_index, layer_type in enumerate(layer_types):
            attrs: dict[str, object] = {}
            if layer_type == "sliding_attention" and sliding_window is not None:
                attrs["window_size"] = int(sliding_window)
            import_hints.append(
                {
                    "module_path_suffix": f"backbone.layers.{layer_index}.self_attn",
                    "op": "scaled_dot_product_attention",
                    "attrs": attrs,
                    "meta": {
                        "attention_layer_type": layer_type,
                        "attention_layer_index": layer_index,
                    },
                }
            )
        return {
            "graph": {
                "adapter_family": "gemma3",
                "adapter_type": type(self).__name__,
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
            },
            "import_hints": import_hints,
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
        import_hints: list[dict[str, object]] = []
        for layer_index, layer_type in enumerate(layer_types):
            attrs: dict[str, object] = {}
            if layer_type == "sliding_attention" and sliding_window is not None:
                attrs["window_size"] = int(sliding_window)
            import_hints.append(
                {
                    "module_path_suffix": f"backbone.layers.{layer_index}.self_attn",
                    "op": "scaled_dot_product_attention",
                    "attrs": attrs,
                    "meta": {
                        "attention_layer_type": layer_type,
                        "attention_layer_index": layer_index,
                    },
                }
            )
        return {
            "graph": {
                "adapter_family": "gemma4",
                "adapter_type": type(self).__name__,
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(layer_types),
                "sliding_window": None if sliding_window is None else int(sliding_window),
            },
            "import_hints": import_hints,
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
        return {
            "graph": {
                "adapter_family": "qwen3_5",
                "adapter_type": type(self).__name__,
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "layer_types": tuple(getattr(self.backbone.config, "layer_types", [])),
            }
        }


def _family_key(model: torch.nn.Module) -> str:
    module_name = type(model).__module__
    if module_name.startswith("transformers.models.gemma4."):
        return "gemma4"
    if module_name.startswith("transformers.models.gemma3."):
        return "gemma3"
    if module_name.startswith("transformers.models.gemma."):
        return "gemma"
    if module_name.startswith("transformers.models.qwen3_5."):
        return "qwen3_5"
    return "generic"


def canonicalize_model_interface(model: torch.nn.Module, task: str = "causal_lm_logits") -> CanonicalizedModel:
    if task != "causal_lm_logits":
        raise NotImplementedError(f"unsupported canonicalization task: {task}")

    family = _family_key(model)
    adapter_factory: Callable[[torch.nn.Module], torch.nn.Module]
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

    return CanonicalizedModel(
        module=adapter_factory(model).eval(),
        task=task,
        family=family,
    )
