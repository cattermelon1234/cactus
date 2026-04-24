from __future__ import annotations

import copy
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transpile.capture_pytorch import capture_model
from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
from src.transpile.canonicalize.cleanup import summarize_unsupported_ops
from src.transpile.lower import transpile_captured
from src.transpile.model_adapters import canonicalize_model_interface
from src.transpile.optimize_graph import optimize_graph
from src.transpile.optimize_graph import summarize_detected_gold_patterns


SUPPORTED_MODEL_IDS = (
    "Qwen/Qwen3.5-2B",
    "google/gemma-4-E2B",
    "google/gemma-2b-it",
    "google/gemma-3-270m-it",
)

DEFAULT_PROMPTS = {
    "gemma": "The capital of France is",
    "gemma3": "The capital of France is",
    "gemma4": "The capital of France is",
    "qwen3_5": "The capital of France is",
}

STAGES_BY_FAMILY = {
    "gemma": ("pre_attn_norm", "attention_out", "post_attn_norm", "layer_out"),
    "gemma3": ("pre_attn_norm", "attention_out", "post_attn_norm", "layer_out"),
    "gemma4": ("pre_attn_norm", "attention_out", "post_attn_norm", "layer_out"),
    "qwen3_5": ("pre_attn_norm", "attention_out", "post_attn_norm", "layer_out"),
}


def _require_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_MODEL_DEBUG_MATRIX_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_MODEL_DEBUG_MATRIX_TEST=1 to run the model debug matrix test."
        )


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise unittest.SkipTest(f"transformers is not available: {exc}") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _resolve_local_snapshot(model_id: str) -> str | None:
    explicit = Path(model_id)
    if explicit.exists():
        return str(explicit)

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + model_id.replace("/", "--"))
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return str(snapshots[-1])


def _discover_local_models() -> list[str]:
    available: list[str] = []
    for model_id in SUPPORTED_MODEL_IDS:
        if _resolve_local_snapshot(model_id) is not None:
            available.append(model_id)
    return available


def _max_abs_diff(ref: np.ndarray, got: np.ndarray) -> float:
    return float(np.max(np.abs(ref.astype(np.float32) - got.astype(np.float32))))


def _mean_abs_diff(ref: np.ndarray, got: np.ndarray) -> float:
    return float(np.mean(np.abs(ref.astype(np.float32) - got.astype(np.float32))))


def _topk_ids(logits: np.ndarray, k: int = 5) -> list[int]:
    return [int(index) for index in np.argsort(logits)[-k:][::-1]]


def _print_graph_summary(graph, label: str, *, max_nodes: int = 12) -> None:
    op_counts = Counter(graph.nodes[node_id].op for node_id in graph.order)
    semantic_counts = Counter(
        graph.nodes[node_id].op for node_id in graph.order if graph.nodes[node_id].kind == "semantic"
    )
    print(f"\n=== {label} ===")
    print(f"node_count={len(graph.order)}")
    print("top_ops=" + ", ".join(f"{op}={count}" for op, count in op_counts.most_common(12)))
    if semantic_counts:
        print("semantic_ops=" + ", ".join(f"{op}={count}" for op, count in semantic_counts.most_common()))
    unsupported = summarize_unsupported_ops(graph)
    if unsupported:
        print("distinct_unknown_ops=" + ", ".join(f"{op}={count}" for op, count in unsupported.items()))
    else:
        print("distinct_unknown_ops=<none>")
    patterns = graph.meta.get("detected_gold_patterns", ())
    if patterns:
        pattern_counts = Counter(pattern.name for pattern in patterns)
        print("patterns=" + ", ".join(f"{name}={count}" for name, count in pattern_counts.most_common()))
    for node_id in graph.order[:max_nodes]:
        node = graph.nodes[node_id]
        print(f"  {node_id}: op={node.op} kind={node.kind} attrs={node.attrs}")
    if len(graph.order) > max_nodes:
        print(f"  ... ({len(graph.order) - max_nodes} more nodes omitted)")


class FullModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.adapter = canonicalize_model_interface(model, task="causal_lm_logits").module

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.adapter(input_ids)


def _gemma_first_layer_stages(adapter_module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb  # type: ignore

    backbone = adapter_module.backbone
    inputs_embeds = backbone.embed_tokens(input_ids)
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    causal_mask = adapter_module._create_causal_mask(
        config=backbone.config,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = backbone.rotary_emb(inputs_embeds, position_ids=position_ids)
    cos, sin = position_embeddings
    layer = backbone.layers[0]

    residual = inputs_embeds
    pre_attn_norm = layer.input_layernorm(inputs_embeds)

    input_shape = pre_attn_norm.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    q_proj = layer.self_attn.q_proj(pre_attn_norm).view(hidden_shape).transpose(1, 2)
    k_proj = layer.self_attn.k_proj(pre_attn_norm).view(hidden_shape).transpose(1, 2)
    _q_rope, _k_rope = apply_rotary_pos_emb(q_proj, k_proj, cos, sin)

    attention_out, _ = layer.self_attn(
        hidden_states=pre_attn_norm,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    post_attn_residual = residual + attention_out
    post_attn_norm = layer.post_attention_layernorm(post_attn_residual)
    mlp_out = layer.mlp(post_attn_norm)
    layer_out = post_attn_residual + mlp_out
    return {
        "pre_attn_norm": pre_attn_norm,
        "attention_out": attention_out,
        "post_attn_norm": post_attn_norm,
        "layer_out": layer_out,
    }


def _gemma3_first_layer_stages(adapter_module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    backbone = adapter_module.backbone
    inputs_embeds = backbone.embed_tokens(input_ids)
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    layer_type = backbone.config.layer_types[0]
    mask_kwargs = {
        "config": backbone.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask = (
        adapter_module._create_sliding_window_causal_mask(**mask_kwargs)
        if layer_type == "sliding_attention"
        else adapter_module._create_causal_mask(**mask_kwargs)
    )
    position_embeddings = backbone.rotary_emb(inputs_embeds, position_ids, layer_type)
    layer = backbone.layers[0]

    residual = inputs_embeds
    pre_attn_norm = layer.input_layernorm(inputs_embeds)
    attention_out = layer.self_attn(
        pre_attn_norm,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=None,
    )
    if isinstance(attention_out, tuple):
        attention_out = attention_out[0]
    post_attn_norm = layer.post_attention_layernorm(attention_out)
    after_attention = residual + post_attn_norm
    pre_ffn_norm = layer.pre_feedforward_layernorm(after_attention)
    mlp_out = layer.mlp(pre_ffn_norm)
    post_ffn_norm = layer.post_feedforward_layernorm(mlp_out)
    layer_out = after_attention + post_ffn_norm
    return {
        "pre_attn_norm": pre_attn_norm,
        "attention_out": attention_out,
        "post_attn_norm": pre_ffn_norm,
        "layer_out": layer_out,
    }


def _gemma4_first_layer_stages(adapter_module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    checkpoints = adapter_module.debug_first_block(input_ids)
    return {
        "pre_attn_norm": checkpoints["pre_attn_norm"],
        "attention_out": checkpoints["attn_o_proj"],
        "post_attn_norm": checkpoints["pre_ffn_norm"],
        "layer_out": checkpoints.get("layer_scalar_out", checkpoints["after_ffn_residual"]),
    }


def _qwen35_first_layer_stages(adapter_module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    backbone = adapter_module.backbone
    inputs_embeds = backbone.embed_tokens(input_ids)
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    text_position_ids = position_ids[0]
    linear_attn_mask = backbone._update_linear_attn_mask(None, None)
    layer = backbone.layers[0]

    residual = inputs_embeds
    pre_attn_norm = layer.input_layernorm(inputs_embeds)
    if hasattr(layer, "linear_attn"):
        attention_out = layer.linear_attn(pre_attn_norm, cache_params=None, attention_mask=linear_attn_mask)
    else:
        position_embeddings = backbone.rotary_emb(inputs_embeds, position_ids[1:])
        causal_mask = adapter_module._create_causal_mask(
            config=backbone.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        attention_out = layer.self_attn(
            pre_attn_norm,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
        )
        if isinstance(attention_out, tuple):
            attention_out = attention_out[0]
    post_attn_residual = residual + attention_out
    post_attn_norm = layer.post_attention_layernorm(post_attn_residual)
    mlp_out = layer.mlp(post_attn_norm)
    layer_out = post_attn_residual + mlp_out
    return {
        "pre_attn_norm": pre_attn_norm,
        "attention_out": attention_out,
        "post_attn_norm": post_attn_norm,
        "layer_out": layer_out,
    }


def _first_layer_stages(adapter_module, family: str, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    if family == "gemma":
        return _gemma_first_layer_stages(adapter_module, input_ids)
    if family == "gemma3":
        return _gemma3_first_layer_stages(adapter_module, input_ids)
    if family == "gemma4":
        return _gemma4_first_layer_stages(adapter_module, input_ids)
    if family == "qwen3_5":
        return _qwen35_first_layer_stages(adapter_module, input_ids)
    raise NotImplementedError(f"unsupported family for first-layer stages: {family}")


class FirstLayerStageWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, stage_name: str):
        super().__init__()
        self.adapter = canonicalize_model_interface(model, task="causal_lm_logits").module
        self.family = canonicalize_model_interface(model, task="causal_lm_logits").family
        self.stage_name = stage_name

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        stages = _first_layer_stages(self.adapter, self.family, input_ids)
        if self.stage_name not in stages:
            available = ", ".join(sorted(stages))
            raise KeyError(f"unknown stage {self.stage_name!r}; available: {available}")
        return stages[self.stage_name]


class TestTranspileDebugMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_opt_in()
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        cls.AutoModelForCausalLM = AutoModelForCausalLM
        cls.AutoTokenizer = AutoTokenizer
        cls.available_model_ids = _discover_local_models()
        if not cls.available_model_ids:
            raise unittest.SkipTest("no supported local Hugging Face snapshots found for the debug matrix test")

    def test_model_debug_matrix(self) -> None:
        successful_models = 0

        for model_id in self.available_model_ids:
            with self.subTest(model_id=model_id):
                snapshot = _resolve_local_snapshot(model_id)
                self.assertIsNotNone(snapshot)

                tokenizer = self.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
                model = self.AutoModelForCausalLM.from_pretrained(
                    snapshot,
                    torch_dtype=torch.float16,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                ).eval()
                canonical = canonicalize_model_interface(model, task="causal_lm_logits")
                family = canonical.family
                prompt = os.environ.get(
                    f"CACTUS_DEBUG_MATRIX_PROMPT_{family.upper()}",
                    os.environ.get("CACTUS_DEBUG_MATRIX_PROMPT", DEFAULT_PROMPTS.get(family, "The capital of France is")),
                )
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

                print(f"\n\n### Model: {model_id}")
                print(f"family={family} prompt_len={int(input_ids.shape[1])}")

                full_capture = capture_model(FullModelWrapper(model).eval(), (input_ids,))
                raw_graph = copy.deepcopy(full_capture.ir_graph)
                canonicalize_exported_graph(full_capture.ir_graph)
                optimize_graph(full_capture.ir_graph)

                _print_graph_summary(raw_graph, f"{model_id} Raw IR")
                _print_graph_summary(full_capture.ir_graph, f"{model_id} After Fusion")
                print("pattern_summary=" + str(summarize_detected_gold_patterns(full_capture.ir_graph)))

                print("\nFirst-layer stage diffs:")
                ref_stages = _first_layer_stages(canonical.module, family, input_ids)
                for stage_name in STAGES_BY_FAMILY[family]:
                    stage_capture = capture_model(FirstLayerStageWrapper(model, stage_name).eval(), (input_ids,))
                    canonicalize_exported_graph(stage_capture.ir_graph)
                    optimize_graph(stage_capture.ir_graph)
                    stage_tg = transpile_captured(stage_capture)
                    stage_tg.set_inputs([input_ids.cpu().numpy()])
                    got = stage_tg.execute()[0].numpy().astype(np.float32)
                    ref = ref_stages[stage_name].detach().float().cpu().numpy()
                    print(
                        f"  {stage_name}: shape={tuple(got.shape)} "
                        f"max_abs_diff={_max_abs_diff(ref, got):.6f} "
                        f"mean_abs_diff={_mean_abs_diff(ref, got):.6f}"
                    )

                print("\nFinal inference:")
                try:
                    tg = transpile_captured(full_capture)
                    tg.set_inputs([input_ids.cpu().numpy()])
                    transpiled_logits = tg.execute()[0].numpy().astype(np.float32)
                    with torch.no_grad():
                        hf_logits = model(input_ids=input_ids, use_cache=False, return_dict=False)[0].detach().float().cpu().numpy()
                    hf_next = int(np.argmax(hf_logits[0, -1]))
                    transpiled_next = int(np.argmax(transpiled_logits[0, -1]))
                    print(f"  input_ids={input_ids.tolist()}")
                    print(f"  hf_next={hf_next} top5={_topk_ids(hf_logits[0, -1])}")
                    print(f"  transpiled_next={transpiled_next} top5={_topk_ids(transpiled_logits[0, -1])}")
                    print(f"  logits_max_abs_diff={_max_abs_diff(hf_logits, transpiled_logits):.6f}")
                    print(f"  logits_mean_abs_diff={_mean_abs_diff(hf_logits, transpiled_logits):.6f}")
                    successful_models += 1
                except Exception as exc:
                    print(f"  transpile_or_inference_error={type(exc).__name__}: {exc}")

        self.assertGreater(successful_models, 0, "expected at least one model to complete full transpiled inference")


if __name__ == "__main__":
    unittest.main()
