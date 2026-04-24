from __future__ import annotations

from dataclasses import dataclass

import torch

from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
from src.transpile.canonicalize.utils import rebuild_graph
from src.transpile.fusion import match_attention
from src.transpile.fusion import match_attention_block
from src.transpile.fusion import match_gated_deltanet
from src.transpile.fusion import match_gated_mlp
from src.transpile.fusion import match_rms_norm
from src.transpile.fusion import match_rope
from src.transpile.fusion.common import producer
from src.transpile.fusion.common import strip_passthrough
from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.graph_ir import IRValue
from src.transpile.graph_ir import verify_ir
from src.transpile.model_patterns import GOLD_PATTERNS
from src.transpile.normalize import dtype_to_ir


@dataclass(frozen=True)
class DetectedPattern:
    name: str
    anchor_node_id: str
    node_ids: tuple[str, ...]
    value_ids: tuple[str, ...]
    details: dict[str, object]


@dataclass(frozen=True)
class FusionConfig:
    enable_gated_deltanet: bool = True
    enable_rms_norm: bool = True
    enable_rope: bool = True
    enable_attention: bool = True
    enable_attention_block: bool = True
    enable_add_clipped: bool = True


def optimize_graph(graph: IRGraph, *, max_passes: int = 8, config: FusionConfig | None = None) -> IRGraph:
    config = config or FusionConfig()
    verify_ir(graph)
    canonicalize_exported_graph(graph)

    for _ in range(max_passes):
        changed = False
        if config.enable_gated_deltanet and fuse_gated_deltanet(graph):
            changed = True
        if config.enable_rms_norm and fuse_rms_norm(graph):
            changed = True
        if config.enable_rope and fuse_rope(graph):
            changed = True
        if config.enable_attention and fuse_attention(graph):
            changed = True
        if config.enable_attention_block and fuse_attention_blocks(graph):
            changed = True
        if config.enable_add_clipped and fuse_add_clipped(graph):
            changed = True
        if not changed:
            break
        canonicalize_exported_graph(graph)

    annotate_gold_patterns(graph)
    verify_ir(graph)
    return graph


def fuse_gated_deltanet(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_gated_deltanet(graph, node)
        if match is None:
            continue

        inputs = [
            match.normalized_input_value_id,
            match.qkv_weight_value_id,
            match.a_weight_value_id,
            match.b_weight_value_id,
            match.norm_weight_value_id,
        ]
        if match.z_weight_value_id is not None:
            inputs.append(match.z_weight_value_id)
        if match.dt_bias_value_id is not None:
            inputs.append(match.dt_bias_value_id)
        if match.a_log_value_id is not None:
            inputs.append(match.a_log_value_id)
        if match.conv_weight_value_id is not None:
            inputs.append(match.conv_weight_value_id)

        node.op = f"gated_deltanet_{match.mode}"
        node.inputs = inputs
        node.attrs = {
            "num_k_heads": int(match.num_k_heads),
            "num_v_heads": int(match.num_v_heads),
            "key_dim": int(match.key_dim),
            "value_dim": int(match.value_dim),
            "eps": float(match.eps),
            "chunk_size": int(match.chunk_size),
            "has_z": bool(match.z_weight_value_id is not None),
            "has_dt_bias": bool(match.dt_bias_value_id is not None),
            "has_a_log": bool(match.a_log_value_id is not None),
            "has_conv": bool(match.conv_weight_value_id is not None),
        }
        node.kind = "semantic"
        node.meta["gated_deltanet_mode"] = match.mode
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def annotate_gold_patterns(graph: IRGraph) -> list[DetectedPattern]:
    _clear_gold_pattern_annotations(graph)
    patterns = [
        *_detect_gated_mlps(graph),
        *_detect_decoder_attentions(graph),
        *_detect_transformer_blocks(graph),
    ]

    for pattern in patterns:
        for node_id in pattern.node_ids:
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            node.meta.setdefault("gold_patterns", [])
            node.meta["gold_patterns"].append(pattern.name)

        anchor = graph.nodes.get(pattern.anchor_node_id)
        if anchor is not None:
            anchor.meta["gold_pattern_anchor"] = True
            anchor.meta["gold_pattern_details"] = pattern.details

    graph.meta["gold_patterns_catalog"] = tuple(pattern.name for pattern in GOLD_PATTERNS)
    graph.meta["detected_gold_patterns"] = patterns
    return patterns


def summarize_detected_gold_patterns(graph: IRGraph) -> dict[str, int]:
    summary: dict[str, int] = {}
    for pattern in annotate_gold_patterns(graph):
        summary[pattern.name] = summary.get(pattern.name, 0) + 1
    return summary


def fuse_rms_norm(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"multiply", "type_as", "precision_cast"}:
            continue

        match = match_rms_norm(graph, node)
        if match is None:
            continue

        weight_value_id = match.weight_value_id
        if weight_value_id is None:
            input_value = graph.values.get(match.input_value_id)
            if input_value is None or input_value.shape is None or not input_value.shape:
                continue
            hidden_dim = input_value.shape[-1]
            if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                continue
            weight_value_id = _materialize_ones_constant(
                graph,
                hidden_dim,
                dtype=input_value.dtype,
                suffix="rms_norm_ones",
            )
        if float(match.weight_offset) != 0.0:
            weight_value_id = _materialize_shifted_constant(
                graph,
                weight_value_id,
                float(match.weight_offset),
                suffix="rms_norm_scale",
            )

        node.op = "rms_norm"
        node.inputs = [match.input_value_id, weight_value_id]
        node.attrs = {"eps": float(match.eps)}
        node.kind = "semantic"
        node.meta["rms_weight_offset"] = float(match.weight_offset)
        node.meta["rms_input_value_id"] = match.input_value_id
        node.meta["rms_weight_value_id"] = weight_value_id
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_rope(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or not node.outputs:
            continue

        match = match_rope(graph, node.outputs[0])
        if match is None or match.partial:
            continue

        node.op = "rope"
        node.inputs = [match.input_value_id]
        node.attrs = {
            "theta": float(match.theta),
            "position_offset": int(match.position_offset),
        }
        node.kind = "semantic"
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_attention(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"scaled_dot_product_attention", "attention"}:
            continue

        match = match_attention(graph, node)
        if match is None:
            continue

        window_size = int(node.attrs.get("window_size", 0))
        if len(node.inputs) > 3:
            mask_info = _extract_sliding_window_mask(graph, node.inputs[3])
            if mask_info is not None:
                node.meta["mask_window_size_hint"] = int(mask_info["window_size"])
                if window_size == 0:
                    window_size = int(mask_info["window_size"])
                    node.meta["window_size_source"] = "mask_pattern"

        if window_size != 0 and "window_size_source" not in node.meta:
            node.meta["window_size_source"] = "import_attr"

        semantic_attrs = {
            key: value
            for key, value in node.attrs.items()
            if key not in {"mask", "dropout_p", "scale", "is_causal"}
        }
        semantic_attrs.update(
            {
                "scale": float(node.attrs.get("scale", 1.0)),
                "is_causal": bool(node.attrs.get("is_causal", True)),
                "window_size": window_size,
            }
        )

        if (
            node.op == "attention"
            and node.inputs[:3] == [match.query_value_id, match.key_value_id, match.value_value_id]
            and node.attrs == semantic_attrs
        ):
            continue

        node.op = "attention"
        node.inputs = [match.query_value_id, match.key_value_id, match.value_value_id]
        node.attrs = semantic_attrs
        node.kind = "semantic"
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_attention_blocks(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_attention_block(graph, node)
        if match is None:
            continue

        inputs = [match.query_value_id, match.key_value_id, match.value_value_id]
        if match.gate_value_id is not None:
            inputs.append(match.gate_value_id)
        inputs.append(match.output_projection_weight_value_id)
        if match.output_projection_bias_value_id is not None:
            inputs.append(match.output_projection_bias_value_id)

        node.op = "attention_block"
        node.inputs = inputs
        node.attrs = {
            "scale": float(match.scale),
            "is_causal": bool(match.is_causal),
            "window_size": int(match.window_size),
            "has_gate": bool(match.gate_value_id is not None),
            "has_bias": bool(match.output_projection_bias_value_id is not None),
            "attention_output_shape": tuple(int(dim) for dim in match.attention_output_shape),
        }
        node.kind = "semantic"
        node.meta["attention_block_source"] = match.attention_node_id
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_add_clipped(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op != "add" or len(node.inputs) != 2:
            continue

        lhs = strip_passthrough(graph, node.inputs[0])
        rhs = strip_passthrough(graph, node.inputs[1])
        if not (_looks_like_gemma_residual_add(graph, lhs, rhs) or _looks_like_gemma_residual_add(graph, rhs, lhs)):
            continue

        node.op = "add_clipped"
        node.kind = "semantic"
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def _materialize_shifted_constant(graph: IRGraph, value_id: str, offset: float, *, suffix: str) -> str:
    base = graph.constants[value_id]
    if not isinstance(base, torch.Tensor):
        raise NotImplementedError(f"expected tensor constant for {value_id}, got {type(base).__name__}")

    new_value_id = f"{value_id}_{suffix}"
    if new_value_id in graph.constants:
        return new_value_id

    shifted = base.detach().cpu() + offset
    graph.constants[new_value_id] = shifted
    graph.values[new_value_id] = IRValue(
        id=new_value_id,
        shape=tuple(shifted.shape),
        dtype=dtype_to_ir(shifted.dtype),
        producer=None,
        users=[],
    )
    return new_value_id


def _materialize_ones_constant(graph: IRGraph, size: int, *, dtype: str | None, suffix: str) -> str:
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    new_value_id = f"c_{suffix}_{size}_{dtype or 'fp32'}"
    if new_value_id in graph.constants:
        return new_value_id

    value = torch.ones((size,), dtype=torch_dtype)
    graph.constants[new_value_id] = value
    graph.values[new_value_id] = IRValue(
        id=new_value_id,
        shape=(size,),
        dtype=dtype_to_ir(value.dtype),
        producer=None,
        users=[],
    )
    return new_value_id


def _clear_gold_pattern_annotations(graph: IRGraph) -> None:
    for node in graph.nodes.values():
        node.meta.pop("gold_patterns", None)
        node.meta.pop("gold_pattern_anchor", None)
        node.meta.pop("gold_pattern_details", None)


def _detect_gated_mlps(graph: IRGraph) -> list[DetectedPattern]:
    patterns: list[DetectedPattern] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        match = match_gated_mlp(graph, node)
        if match is None:
            continue

        pattern_name = "gated_mlp_gelu" if match.activation == "gelu" else "gated_mlp_silu"
        patterns.append(
            DetectedPattern(
                name=pattern_name,
                anchor_node_id=node.id,
                node_ids=match.node_ids,
                value_ids=(match.input_value_id,),
                details={
                    "activation": match.activation,
                    "input_value_id": match.input_value_id,
                    "gate_weight_value_id": match.gate_weight_value_id,
                    "up_weight_value_id": match.up_weight_value_id,
                    "down_weight_value_id": match.down_weight_value_id,
                },
            )
        )
    return patterns


def _detect_decoder_attentions(graph: IRGraph) -> list[DetectedPattern]:
    patterns: list[DetectedPattern] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"scaled_dot_product_attention", "attention"}:
            continue

        match = match_attention(graph, node)
        if match is None:
            continue

        q_input, k_input, v_input = match.source_input_value_ids
        pattern_name = "decoder_attention_gqa" if match.has_gqa_repeat else "decoder_attention"

        node_ids = set(match.node_ids)
        details = {
            "input_value_ids": (q_input, k_input, v_input),
            "q_linear_weight_value_id": match.weight_value_ids[0],
            "k_linear_weight_value_id": match.weight_value_ids[1],
            "v_linear_weight_value_id": match.weight_value_ids[2],
            "has_rope": bool(match.has_rope),
            "has_qk_rms_norm": bool(match.has_qk_norm),
            "has_gqa_repeat": bool(match.has_gqa_repeat),
            "is_causal": bool(match.is_causal),
            "scale": float(match.scale),
            "window_size_hint": int(match.window_size),
        }

        if len(node.inputs) > 3:
            details["mask_value_id"] = node.inputs[3]
            mask_info = _extract_sliding_window_mask(graph, node.inputs[3])
            if mask_info is not None:
                details["mask_pattern"] = "sliding_window_attention_mask"
                details["window_size_hint"] = int(mask_info["window_size"])
                node_ids.update(mask_info["node_ids"])

        patterns.append(
            DetectedPattern(
                name=pattern_name,
                anchor_node_id=node.id,
                node_ids=tuple(sorted(node_ids)),
                value_ids=(q_input, k_input, v_input),
                details=details,
            )
        )
    return patterns


def _detect_transformer_blocks(graph: IRGraph) -> list[DetectedPattern]:
    patterns: list[DetectedPattern] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"add", "add_clipped"} or len(node.inputs) != 2:
            continue

        rhs_pattern = _find_anchor_pattern(graph, node.inputs[1], {"gated_mlp_gelu", "gated_mlp_silu"})
        lhs_pattern = _find_anchor_pattern(graph, node.inputs[0], {"gated_mlp_gelu", "gated_mlp_silu"})
        mlp_pattern = rhs_pattern or lhs_pattern
        if mlp_pattern is None:
            continue

        residual_value_id = node.inputs[0] if rhs_pattern is not None else node.inputs[1]
        attn_pattern = _find_anchor_pattern(
            graph,
            residual_value_id,
            {"decoder_attention_gqa", "decoder_attention", "gemma4_partial_rope_attention"},
        )
        if attn_pattern is None:
            continue

        node_ids = {node.id}
        for candidate in graph.nodes.values():
            gold_patterns = candidate.meta.get("gold_patterns", [])
            if attn_pattern in gold_patterns or mlp_pattern in gold_patterns:
                node_ids.add(candidate.id)

        patterns.append(
            DetectedPattern(
                name="decoder_block_post_attn_norm" if node.op == "add_clipped" else "decoder_block_simple_residual",
                anchor_node_id=node.id,
                node_ids=tuple(sorted(node_ids)),
                value_ids=(residual_value_id,),
                details={
                    "residual_value_id": residual_value_id,
                    "attention_pattern": attn_pattern,
                    "mlp_pattern": mlp_pattern,
                    "residual_op": node.op,
                },
            )
        )
    return patterns


def _looks_like_gemma_residual_add(graph: IRGraph, residual_value_id: str, branch_value_id: str) -> bool:
    branch_node = producer(graph, strip_passthrough(graph, branch_value_id))
    if branch_node is None or branch_node.op != "rms_norm":
        return False
    return strip_passthrough(graph, residual_value_id) != strip_passthrough(graph, branch_node.inputs[0])


def _find_anchor_pattern(graph: IRGraph, value_id: str, names: set[str]) -> str | None:
    current = value_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        current = strip_passthrough(graph, current)
        node = producer(graph, current)
        if node is None:
            return None

        for name in node.meta.get("gold_patterns", []):
            if name in names and node.meta.get("gold_pattern_anchor"):
                return name

        if len(node.inputs) != 1:
            return None
        current = node.inputs[0]
    return None


def _extract_sliding_window_mask(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    node_ids: set[str] = set()
    top_and = producer(graph, strip_passthrough(graph, value_id))
    if top_and is None or top_and.op not in {"aten.__and__.Tensor", "logical_and"}:
        return None
    node_ids.add(top_and.id)

    stack = list(top_and.inputs)
    saw_diff = False
    saw_cumsum = False
    window_candidates: list[int] = []

    while stack:
        current = strip_passthrough(graph, stack.pop())
        node = producer(graph, current)
        if node is None or node.id in node_ids:
            continue
        node_ids.add(node.id)

        if node.op in {"aten.diff.default", "diff"}:
            saw_diff = True
        elif node.op in {"aten.cumsum.default", "cumsum"}:
            saw_cumsum = True
        elif node.op == "scalar_subtract":
            maybe_window = int(node.attrs.get("value", 0))
            if maybe_window > 0:
                window_candidates.append(maybe_window)

        stack.extend(node.inputs)

    if not saw_diff or not saw_cumsum:
        return None

    return {
        "window_size": max((candidate for candidate in window_candidates if candidate > 1), default=0),
        "node_ids": tuple(sorted(node_ids)),
    }
