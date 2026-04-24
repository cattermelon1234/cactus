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
from src.transpile.fusion.attention import _extract_attention_input
from src.transpile.fusion.common import producer
from src.transpile.fusion.mlp import match_gated_mlp
from src.transpile.lower import BroadcastAlias
from src.transpile.lower import transpile_captured
from src.transpile.model_adapters import canonicalize_model_interface
from src.transpile.optimize_graph import optimize_graph
from src.transpile.optimize_graph import summarize_detected_gold_patterns


def _require_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_GEMMA2B_DEBUG_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_GEMMA2B_DEBUG_TEST=1 to run the Gemma 2B debug transpiler test."
        )


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise unittest.SkipTest(f"transformers is not available: {exc}") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _resolve_local_snapshot(model_id: str) -> str:
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
        raise unittest.SkipTest(f"no local Hugging Face snapshot found for {model_id!r}")
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        raise unittest.SkipTest(f"no local Hugging Face snapshots found for {model_id!r}")
    return str(snapshots[-1])


class GemmaFullModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.module = canonicalize_model_interface(model, task="causal_lm_logits").module

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.module(input_ids)


def _graph_summary(graph, label: str, *, max_nodes: int = 30) -> None:
    op_counts = Counter(graph.nodes[node_id].op for node_id in graph.order)
    semantic_counts = Counter(
        graph.nodes[node_id].op
        for node_id in graph.order
        if graph.nodes[node_id].kind == "semantic"
    )
    print(f"\n=== {label} ===")
    print(f"node_count={len(graph.order)}")
    print("top_ops=" + ", ".join(f"{op}={count}" for op, count in op_counts.most_common(12)))
    if semantic_counts:
        print("semantic_ops=" + ", ".join(f"{op}={count}" for op, count in semantic_counts.most_common()))
    patterns = graph.meta.get("detected_gold_patterns", ())
    if patterns:
        pattern_counts = Counter(pattern.name for pattern in patterns)
        print("patterns=" + ", ".join(f"{name}={count}" for name, count in pattern_counts.most_common()))
    for node_id in graph.order[:max_nodes]:
        node = graph.nodes[node_id]
        print(f"  {node_id}: op={node.op} kind={node.kind} outputs={node.outputs} attrs={node.attrs}")
    if len(graph.order) > max_nodes:
        print(f"  ... ({len(graph.order) - max_nodes} more nodes omitted)")


def _extract_checkpoint_value_ids(ir_graph):
    checkpoints = []

    for node_id in ir_graph.order:
        node = ir_graph.nodes[node_id]
        if node.kind == "semantic" and node.op in {"attention_block", "decoder_block", "transformer_block"}:
            if node.outputs:
                checkpoints.append((f"{node.op}_{len(checkpoints)}", node.outputs[0]))

    last_linear_id = next(
        (node_id for node_id in reversed(ir_graph.order) if ir_graph.nodes[node_id].op == "linear"),
        None,
    )
    if last_linear_id is not None:
        last_linear = ir_graph.nodes[last_linear_id]
        if last_linear.inputs:
            checkpoints.append(("pre_lm_head", last_linear.inputs[0]))

    return checkpoints


def _transpile_single_output(ir_graph, output_value_id: str):
    subgraph = copy.deepcopy(ir_graph)
    subgraph.outputs = [output_value_id]
    return transpile_captured(type("Captured", (), {"ir_graph": subgraph})())


def _max_abs_diff(ref: np.ndarray, got: np.ndarray) -> float:
    return float(np.max(np.abs(ref - got)))


def _mean_abs_diff(ref: np.ndarray, got: np.ndarray) -> float:
    return float(np.mean(np.abs(ref - got)))


def _debug_output_to_numpy(output) -> np.ndarray:
    if isinstance(output, BroadcastAlias):
        base = output.tensor.numpy().astype(np.float32)
        logical_shape = tuple(int(v) for v in output.logical_shape)
        if base.shape == logical_shape:
            return base
        if len(base.shape) == 4 and len(logical_shape) == 4:
            if (
                base.shape[0] == logical_shape[0]
                and base.shape[2] == logical_shape[2]
                and base.shape[3] == logical_shape[3]
                and logical_shape[1] % max(base.shape[1], 1) == 0
            ):
                return np.repeat(base, logical_shape[1] // base.shape[1], axis=1)
        raise ValueError(f"cannot materialize BroadcastAlias from {base.shape} to {logical_shape}")
    return output.numpy().astype(np.float32)


def _repeat_heads_to_match(ref: np.ndarray, got_shape: tuple[int, ...]) -> np.ndarray:
    if ref.shape == got_shape:
        return ref
    if len(ref.shape) != 4 or len(got_shape) != 4:
        return ref
    if ref.shape[0] != got_shape[0] or ref.shape[2] != got_shape[2] or ref.shape[3] != got_shape[3]:
        return ref
    if ref.shape[1] <= 0 or got_shape[1] % ref.shape[1] != 0:
        return ref
    return np.repeat(ref, got_shape[1] // ref.shape[1], axis=1)


def _first_user_with_op(ir_graph, value_id: str, allowed_ops: set[str]) -> str | None:
    value = ir_graph.values.get(value_id)
    if value is None:
        return None
    for user_id in value.users:
        user = ir_graph.nodes.get(user_id)
        if user is not None and user.op in allowed_ops:
            return user_id
    return None


def _other_input(node, known_input: str) -> str:
    if len(node.inputs) != 2:
        raise ValueError(f"expected binary node for {node.id}, got {len(node.inputs)} inputs")
    if node.inputs[0] == known_input:
        return node.inputs[1]
    if node.inputs[1] == known_input:
        return node.inputs[0]
    raise ValueError(f"node {node.id} does not consume {known_input}")


def _collect_block_debug_value_ids(ir_graph) -> list[dict[str, str]]:
    attention_block_ids = [node_id for node_id in ir_graph.order if ir_graph.nodes[node_id].op == "attention_block"]
    mlp_matches: list[tuple[str, object]] = []
    for node_id in ir_graph.order:
        node = ir_graph.nodes[node_id]
        match = match_gated_mlp(ir_graph, node)
        if match is not None:
            mlp_matches.append((node_id, match))

    stages: list[dict[str, str]] = []
    count = min(len(attention_block_ids), len(mlp_matches))
    for index in range(count):
        attn_node = ir_graph.nodes[attention_block_ids[index]]
        mlp_node_id, mlp_match = mlp_matches[index]
        mlp_node = ir_graph.nodes[mlp_node_id]
        q_info = _extract_attention_input(ir_graph, attn_node.inputs[0], role="q")
        if q_info is None:
            continue
        q_rope = producer(ir_graph, attn_node.inputs[0])
        k_rope = producer(ir_graph, attn_node.inputs[1])
        q_pre_rope = q_rope.inputs[0] if q_rope is not None and q_rope.op == "rope" and q_rope.inputs else attn_node.inputs[0]
        k_pre_rope = k_rope.inputs[0] if k_rope is not None and k_rope.op == "rope" and k_rope.inputs else attn_node.inputs[1]

        post_attn_add_id = _first_user_with_op(ir_graph, attn_node.outputs[0], {"add", "add_clipped"})
        if post_attn_add_id is None:
            continue
        post_attn_add = ir_graph.nodes[post_attn_add_id]
        layer_add_id = _first_user_with_op(ir_graph, mlp_node.outputs[0], {"add", "add_clipped"})
        if layer_add_id is None:
            continue
        layer_add = ir_graph.nodes[layer_add_id]

        stages.append(
            {
                "pre_attn_norm": str(q_info["source_input"]),
                "q_pre_rope": q_pre_rope,
                "k_pre_rope": k_pre_rope,
                "v_pre_attention": attn_node.inputs[2],
                "q_for_attention": attn_node.inputs[0],
                "k_for_attention": attn_node.inputs[1],
                "v_for_attention": attn_node.inputs[2],
                "attention_out": attn_node.outputs[0],
                "post_attn_residual": post_attn_add.outputs[0],
                "post_attn_norm": str(mlp_match.input_value_id),
                "mlp_out": mlp_node.outputs[0],
                "layer_out": layer_add.outputs[0],
            }
        )

    return stages


def _gemma2b_reference_layers(adapter_module, input_ids: torch.Tensor) -> list[dict[str, np.ndarray]]:
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

    hidden_states = inputs_embeds
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids=position_ids)
    cos, sin = position_embeddings
    refs: list[dict[str, np.ndarray]] = []

    with torch.no_grad():
        for decoder_layer in backbone.layers[: backbone.config.num_hidden_layers]:
            residual = hidden_states
            pre_attn_norm = decoder_layer.input_layernorm(hidden_states)

            input_shape = pre_attn_norm.shape[:-1]
            hidden_shape = (*input_shape, -1, decoder_layer.self_attn.head_dim)
            q_proj = decoder_layer.self_attn.q_proj(pre_attn_norm).view(hidden_shape).transpose(1, 2)
            k_proj = decoder_layer.self_attn.k_proj(pre_attn_norm).view(hidden_shape).transpose(1, 2)
            v_proj = decoder_layer.self_attn.v_proj(pre_attn_norm).view(hidden_shape).transpose(1, 2)
            q_rope, k_rope = apply_rotary_pos_emb(q_proj, k_proj, cos, sin)
            raw_scores = torch.matmul(q_rope.float(), k_rope.float().transpose(-1, -2)) * float(decoder_layer.self_attn.scaling)
            if causal_mask is None:
                masked_scores = raw_scores
            else:
                masked_scores = raw_scores + causal_mask.float()
            attention_probs = torch.softmax(masked_scores, dim=-1)
            attention_context = torch.matmul(attention_probs, v_proj.float())
            attention_context_bshd = attention_context.transpose(1, 2).contiguous()
            attention_flat = attention_context_bshd.reshape(*input_shape, -1)
            attention_o_proj = decoder_layer.self_attn.o_proj(attention_flat.to(pre_attn_norm.dtype)).float()

            attn_out, _ = decoder_layer.self_attn(
                hidden_states=pre_attn_norm,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            post_attn_residual = residual + attn_out

            residual = post_attn_residual
            post_attn_norm = decoder_layer.post_attention_layernorm(post_attn_residual)
            mlp_out = decoder_layer.mlp(post_attn_norm)
            hidden_states = residual + mlp_out

            refs.append(
                {
                    "pre_attn_norm": pre_attn_norm.detach().float().cpu().numpy(),
                    "q_pre_rope": q_proj.detach().float().cpu().numpy(),
                    "k_pre_rope": k_proj.detach().float().cpu().numpy(),
                    "v_pre_attention": v_proj.detach().float().cpu().numpy(),
                    "q_for_attention": q_rope.detach().float().cpu().numpy(),
                    "k_for_attention": k_rope.detach().float().cpu().numpy(),
                    "v_for_attention": v_proj.detach().float().cpu().numpy(),
                    "raw_attention_scores": raw_scores.detach().float().cpu().numpy(),
                    "masked_attention_scores": masked_scores.detach().float().cpu().numpy(),
                    "attention_probs": attention_probs.detach().float().cpu().numpy(),
                    "attention_context": attention_context.detach().float().cpu().numpy(),
                    "attention_context_bshd": attention_context_bshd.detach().float().cpu().numpy(),
                    "attention_flat": attention_flat.detach().float().cpu().numpy(),
                    "attention_o_proj": attention_o_proj.detach().float().cpu().numpy(),
                    "attention_scale": float(decoder_layer.self_attn.scaling),
                    "attention_out": attn_out.detach().float().cpu().numpy(),
                    "post_attn_residual": post_attn_residual.detach().float().cpu().numpy(),
                    "post_attn_norm": post_attn_norm.detach().float().cpu().numpy(),
                    "mlp_out": mlp_out.detach().float().cpu().numpy(),
                    "layer_out": hidden_states.detach().float().cpu().numpy(),
                }
            )

    return refs


def _gemma2b_rope_reference(adapter_module, input_ids: torch.Tensor) -> dict[str, object]:
    backbone = adapter_module.backbone
    inputs_embeds = backbone.embed_tokens(input_ids)
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    cos, sin = backbone.rotary_emb(inputs_embeds, position_ids=position_ids)
    return {
        "position_ids_shape": tuple(int(v) for v in position_ids.shape),
        "cos_shape": tuple(int(v) for v in cos.shape),
        "sin_shape": tuple(int(v) for v in sin.shape),
        "seq_len": int(inputs_embeds.shape[1]),
    }


def _value_shape(ir_graph, value_id: str):
    value = ir_graph.values.get(value_id)
    if value is None or value.shape is None:
        return None
    return tuple(int(v) for v in value.shape)


def _rope_diagnostics(ir_graph, rope_value_id: str) -> dict[str, object] | None:
    rope_node = producer(ir_graph, rope_value_id)
    if rope_node is None or rope_node.op != "rope":
        return None
    input_value_id = rope_node.inputs[0]
    return {
        "node_id": rope_node.id,
        "theta": rope_node.attrs.get("theta"),
        "position_offset": rope_node.attrs.get("position_offset", 0),
        "input_shape": _value_shape(ir_graph, input_value_id),
        "output_shape": _value_shape(ir_graph, rope_value_id),
    }


def _compute_attention_debug_numpy(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    q_t = torch.from_numpy(q).float()
    k_t = torch.from_numpy(k).float()
    v_t = torch.from_numpy(v).float()
    mask_t = torch.from_numpy(mask).float()
    raw_scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * float(scale)
    masked_scores = raw_scores + mask_t
    attention_probs = torch.softmax(masked_scores, dim=-1)
    attention_context = torch.matmul(attention_probs, v_t)
    return {
        "raw_attention_scores": raw_scores.cpu().numpy(),
        "masked_attention_scores": masked_scores.cpu().numpy(),
        "attention_probs": attention_probs.cpu().numpy(),
        "attention_context": attention_context.cpu().numpy(),
    }


def _project_attention_debug_numpy(
    context_bhsd: np.ndarray,
    weight: torch.Tensor,
) -> dict[str, np.ndarray]:
    context_t = torch.from_numpy(context_bhsd).float()
    context_bshd = context_t.transpose(1, 2).contiguous()
    flat = context_bshd.reshape(context_bshd.shape[0], context_bshd.shape[1], -1)
    projected = torch.nn.functional.linear(flat.to(weight.dtype), weight).float()
    return {
        "attention_context_bshd": context_bshd.cpu().numpy(),
        "attention_flat": flat.cpu().numpy(),
        "attention_o_proj": projected.cpu().numpy(),
    }


class TestGemma2BDebug(unittest.TestCase):
    model_id = os.environ.get("CACTUS_GEMMA2B_HF_MODEL_ID", "google/gemma-2b-it")

    @classmethod
    def setUpClass(cls) -> None:
        _require_opt_in()
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        snapshot = _resolve_local_snapshot(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        cls.model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()

    def test_gemma2b_layer_debug(self) -> None:
        prompt = os.environ.get("CACTUS_GEMMA2B_DEBUG_PROMPT", "The capital of France is")
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

        adapter = canonicalize_model_interface(self.model, task="causal_lm_logits")
        if adapter.family != "gemma":
            raise unittest.SkipTest(f"expected gemma adapter, got family={adapter.family}")

        captured = capture_model(GemmaFullModelWrapper(self.model).eval(), (input_ids,))
        raw_graph = copy.deepcopy(captured.ir_graph)
        canonicalize_exported_graph(captured.ir_graph)
        cleanup_graph = copy.deepcopy(captured.ir_graph)
        optimize_graph(captured.ir_graph)

        _graph_summary(raw_graph, f"{self.model_id} Raw IR")
        _graph_summary(cleanup_graph, f"{self.model_id} After Cleanup")
        _graph_summary(captured.ir_graph, f"{self.model_id} After Fusion")
        print("pattern_summary=" + str(summarize_detected_gold_patterns(captured.ir_graph)))

        tg = transpile_captured(captured)
        tg.set_inputs([input_ids.cpu().numpy()])
        transpiled_logits = tg.execute()[0].numpy().astype(np.float32)

        with torch.no_grad():
            ref_logits_t, ref_checkpoints_t = adapter.module.debug_forward(input_ids)
        ref_logits = ref_logits_t.detach().float().cpu().numpy()
        ref_checkpoints = [tensor.detach().float().cpu().numpy() for tensor in ref_checkpoints_t]
        ref_layer_stages = _gemma2b_reference_layers(adapter.module, input_ids)
        ref_rope = _gemma2b_rope_reference(adapter.module, input_ids)
        attn_o_proj_weights = [
            layer.self_attn.o_proj.weight.detach().cpu()
            for layer in adapter.module.backbone.layers[: adapter.module.backbone.config.num_hidden_layers]
        ]

        checkpoint_value_ids = _extract_checkpoint_value_ids(captured.ir_graph)
        print("\nLayer checkpoint diffs:")
        print(f"adapter_checkpoints={len(ref_checkpoints)} ir_checkpoints={len(checkpoint_value_ids)}")

        compared = min(len(ref_checkpoints), len(checkpoint_value_ids))
        worst_label = None
        worst_diff = -1.0
        for index in range(compared):
            label, value_id = checkpoint_value_ids[index]
            debug_graph = _transpile_single_output(captured.ir_graph, value_id)
            debug_graph.set_inputs([input_ids.cpu().numpy()])
            got = debug_graph.execute()[0].numpy().astype(np.float32)
            ref = ref_checkpoints[index]
            max_diff = _max_abs_diff(ref, got)
            mean_diff = _mean_abs_diff(ref, got)
            print(
                f"  {label}: shape={tuple(got.shape)} "
                f"max_abs_diff={max_diff:.6f} mean_abs_diff={mean_diff:.6f}"
            )
            if max_diff > worst_diff:
                worst_diff = max_diff
                worst_label = label

        block_debug_value_ids = _collect_block_debug_value_ids(captured.ir_graph)
        print("\nPer-layer stage diffs:")
        print(f"ref_layers={len(ref_layer_stages)} fused_layers={len(block_debug_value_ids)}")
        print(
            "rope_reference="
            + str(
                {
                    "position_ids_shape": ref_rope["position_ids_shape"],
                    "cos_shape": ref_rope["cos_shape"],
                    "sin_shape": ref_rope["sin_shape"],
                    "hf_qk_layout": "(batch, heads, seq, dim)",
                }
            )
        )
        detailed = min(len(ref_layer_stages), len(block_debug_value_ids))
        worst_stage_layer = None
        worst_stage_name = None
        worst_stage_diff = -1.0
        first_bad_layer = None
        for layer_index in range(detailed):
            stage_ids = block_debug_value_ids[layer_index]
            layer_worst_diff = -1.0
            got_stage_outputs: dict[str, np.ndarray] = {}
            print(f"  layer_{layer_index}:")
            q_rope_diag = _rope_diagnostics(captured.ir_graph, stage_ids["q_for_attention"])
            k_rope_diag = _rope_diagnostics(captured.ir_graph, stage_ids["k_for_attention"])
            if q_rope_diag is not None or k_rope_diag is not None:
                print(f"    q_rope_diag={q_rope_diag}")
                print(f"    k_rope_diag={k_rope_diag}")
            for stage_name in (
                "pre_attn_norm",
                "q_pre_rope",
                "k_pre_rope",
                "v_pre_attention",
                "q_for_attention",
                "k_for_attention",
                "v_for_attention",
                "attention_out",
                "post_attn_residual",
                "post_attn_norm",
                "mlp_out",
                "layer_out",
            ):
                debug_graph = _transpile_single_output(captured.ir_graph, stage_ids[stage_name])
                debug_graph.set_inputs([input_ids.cpu().numpy()])
                got = _debug_output_to_numpy(debug_graph.execute()[0])
                got_stage_outputs[stage_name] = got
                ref = _repeat_heads_to_match(ref_layer_stages[layer_index][stage_name], tuple(got.shape))
                max_diff = _max_abs_diff(ref, got)
                mean_diff = _mean_abs_diff(ref, got)
                print(
                    f"    {stage_name}: shape={tuple(got.shape)} "
                    f"max_abs_diff={max_diff:.6f} mean_abs_diff={mean_diff:.6f}"
                )
                if max_diff > layer_worst_diff:
                    layer_worst_diff = max_diff
                if max_diff > worst_stage_diff:
                    worst_stage_diff = max_diff
                    worst_stage_layer = layer_index
                    worst_stage_name = stage_name

            attention_debug = _compute_attention_debug_numpy(
                got_stage_outputs["q_for_attention"],
                got_stage_outputs["k_for_attention"],
                got_stage_outputs["v_for_attention"],
                ref_layer_stages[layer_index]["masked_attention_scores"] - ref_layer_stages[layer_index]["raw_attention_scores"],
                ref_layer_stages[layer_index]["attention_scale"],
            )
            projection_debug = _project_attention_debug_numpy(
                attention_debug["attention_context"],
                attn_o_proj_weights[layer_index],
            )
            print("    attention_math:")
            for attn_name in (
                "raw_attention_scores",
                "masked_attention_scores",
                "attention_probs",
                "attention_context",
                "attention_context_bshd",
                "attention_flat",
                "attention_o_proj",
            ):
                got_source = projection_debug if attn_name in projection_debug else attention_debug
                ref = _repeat_heads_to_match(
                    ref_layer_stages[layer_index][attn_name],
                    tuple(got_source[attn_name].shape),
                )
                got = got_source[attn_name].astype(np.float32)
                max_diff = _max_abs_diff(ref, got)
                mean_diff = _mean_abs_diff(ref, got)
                print(
                    f"      {attn_name}: shape={tuple(got.shape)} "
                    f"max_abs_diff={max_diff:.6f} mean_abs_diff={mean_diff:.6f}"
                )
            if first_bad_layer is None and layer_worst_diff > 1e-1:
                first_bad_layer = layer_index

        hf_next = int(np.argmax(ref_logits[0, -1]))
        transpiled_next = int(np.argmax(transpiled_logits[0, -1]))
        print("\nFinal token check:")
        print(f"  input_ids={input_ids.tolist()}")
        print(f"  hf_next={hf_next}")
        print(f"  transpiled_next={transpiled_next}")
        print(f"  logits_max_abs_diff={_max_abs_diff(ref_logits, transpiled_logits):.6f}")
        print(f"  logits_mean_abs_diff={_mean_abs_diff(ref_logits, transpiled_logits):.6f}")
        if worst_label is not None:
            print(f"  worst_layer={worst_label} worst_max_abs_diff={worst_diff:.6f}")
        if worst_stage_name is not None and worst_stage_layer is not None:
            print(
                f"  worst_stage=layer_{worst_stage_layer}:{worst_stage_name} "
                f"worst_max_abs_diff={worst_stage_diff:.6f}"
            )
        if first_bad_layer is not None:
            print(f"  first_bad_layer=layer_{first_bad_layer}")

        self.assertGreater(compared, 0)
        self.assertGreater(detailed, 0)


if __name__ == "__main__":
    unittest.main()
