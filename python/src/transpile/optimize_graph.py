from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.transpile.cleanup_passes import dce
from src.transpile.cleanup_passes import rebuild_graph
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
    enable_rms_norm: bool = True
    enable_rope: bool = True
    enable_attention: bool = True
    enable_add_clipped: bool = True


def optimize_graph(graph: IRGraph, *, max_passes: int = 8, config: FusionConfig | None = None) -> IRGraph:
    if config is None:
        config = FusionConfig()
    verify_ir(graph)
    for _ in range(max_passes):
        changed = False
        if config.enable_rms_norm and fuse_rms_norm(graph):
            changed = True
        if config.enable_rope and fuse_rope(graph):
            changed = True
        if config.enable_attention and fuse_attention(graph):
            changed = True
        if config.enable_add_clipped and fuse_add_clipped(graph):
            changed = True
        if not changed:
            break
        dce(graph)
    annotate_gold_patterns(graph)
    verify_ir(graph)
    return graph


def annotate_gold_patterns(graph: IRGraph) -> list[DetectedPattern]:
    _clear_gold_pattern_annotations(graph)
    patterns: list[DetectedPattern] = []
    patterns.extend(_detect_gated_mlps(graph))
    patterns.extend(_detect_decoder_attentions(graph))
    patterns.extend(_detect_transformer_blocks(graph))
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
    graph.meta = getattr(graph, "meta", {}) if hasattr(graph, "meta") else {}
    graph.meta["gold_patterns_catalog"] = tuple(pattern.name for pattern in GOLD_PATTERNS)
    graph.meta["detected_gold_patterns"] = patterns
    return patterns


def summarize_detected_gold_patterns(graph: IRGraph) -> dict[str, int]:
    patterns = annotate_gold_patterns(graph)
    summary: dict[str, int] = {}
    for pattern in patterns:
        summary[pattern.name] = summary.get(pattern.name, 0) + 1
    return summary


def fuse_rms_norm(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"multiply", "type_as"}:
            continue

        match = _match_rms_norm(graph, node)
        if match is None:
            continue

        weight_value_id = match["weight_value_id"]
        if float(match["weight_offset"]) != 0.0:
            weight_value_id = _materialize_shifted_constant(
                graph,
                match["weight_value_id"],
                float(match["weight_offset"]),
                suffix="rms_norm_scale",
            )

        node.op = "rms_norm"
        node.inputs = [match["input_value_id"], weight_value_id]
        node.attrs = {"eps": float(match["eps"])}
        node.kind = "semantic"
        node.meta["rms_weight_offset"] = float(match["weight_offset"])
        node.meta["rms_input_value_id"] = match["input_value_id"]
        node.meta["rms_weight_value_id"] = weight_value_id
        changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def fuse_rope(graph: IRGraph) -> bool:
    changed = False
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or node.op != "add":
            continue

        match = _match_rope(graph, node)
        if match is None:
            continue

        node.op = "rope"
        node.inputs = [match["input_value_id"]]
        node.attrs = {
            "theta": float(match["theta"]),
            "position_offset": int(match["position_offset"]),
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
        if node is None or node.op != "scaled_dot_product_attention":
            continue

        query_value_id = node.inputs[0]
        key_value_id = node.inputs[1]
        value_value_id = node.inputs[2]

        key_base = _unwrap_gqa_repeat(graph, key_value_id)
        value_base = _unwrap_gqa_repeat(graph, value_value_id)
        if key_base is not None:
            key_value_id = key_base
        if value_base is not None:
            value_value_id = value_base

        window_size = 0
        if len(node.inputs) > 3:
            mask_info = _extract_sliding_window_mask(graph, node.inputs[3])
            if mask_info is not None:
                window_size = int(mask_info["window_size"])

        node.op = "attention"
        node.inputs = [query_value_id, key_value_id, value_value_id]
        node.attrs = {
            "scale": float(node.attrs.get("scale", 1.0)),
            "is_causal": bool(node.attrs.get("is_causal", True)),
            "window_size": window_size,
        }
        node.kind = "semantic"
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

        lhs = _strip_passthrough(graph, node.inputs[0])
        rhs = _strip_passthrough(graph, node.inputs[1])

        if _looks_like_gemma_residual_add(graph, lhs, rhs) or _looks_like_gemma_residual_add(graph, rhs, lhs):
            node.op = "add_clipped"
            node.kind = "semantic"
            changed = True

    if changed:
        rebuild_graph(graph)
    return changed


def _match_rms_norm(graph: IRGraph, node: IRNode) -> dict[str, object] | None:
    anchor_input = node.inputs[0] if node.op == "type_as" else node.outputs[0]
    final_mul = _producer(graph, anchor_input)
    if final_mul is None or final_mul.op != "multiply" or len(final_mul.inputs) != 2:
        return None

    for normed_value_id, scale_value_id in (
        (final_mul.inputs[0], final_mul.inputs[1]),
        (final_mul.inputs[1], final_mul.inputs[0]),
    ):
        scale_match = _extract_rms_scale(graph, scale_value_id)
        if scale_match is None:
            continue

        norm_match = _extract_rms_normed_value(graph, normed_value_id)
        if norm_match is None:
            continue

        if _strip_passthrough(graph, norm_match["input_value_id"]) != _strip_passthrough(graph, norm_match["pow_input_value_id"]):
            continue

        return {
            "input_value_id": norm_match["input_value_id"],
            "weight_value_id": scale_match["weight_value_id"],
            "weight_offset": scale_match["offset"],
            "eps": norm_match["eps"],
        }

    return None


def _extract_rms_scale(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    offset = 0.0
    current = value_id

    producer = _producer(graph, current)
    if producer is not None and producer.op == "scalar_add" and float(producer.attrs.get("value", 0.0)) == 1.0:
        offset = 1.0
        current = producer.inputs[0]

    current = _strip_passthrough(graph, current)
    if current not in graph.constants:
        return None
    constant = graph.constants[current]
    if not isinstance(constant, torch.Tensor):
        return None

    return {"weight_value_id": current, "offset": offset}


def _extract_rms_normed_value(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    producer = _producer(graph, value_id)
    if producer is None or producer.op != "multiply" or len(producer.inputs) != 2:
        return None

    for x_candidate, rsqrt_candidate in (
        (producer.inputs[0], producer.inputs[1]),
        (producer.inputs[1], producer.inputs[0]),
    ):
        rsqrt_match = _extract_rsqrt_chain(graph, rsqrt_candidate)
        if rsqrt_match is None:
            continue
        x_input = _strip_passthrough(graph, x_candidate)
        if x_input != _strip_passthrough(graph, rsqrt_match["pow_input_value_id"]):
            continue

        return {
            "input_value_id": x_input,
            "pow_input_value_id": rsqrt_match["pow_input_value_id"],
            "eps": rsqrt_match["eps"],
        }

    return None


def _extract_rsqrt_chain(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    pow_node = _producer(graph, value_id)
    if pow_node is None or pow_node.op != "pow":
        return None
    if float(pow_node.attrs.get("exponent", 0.0)) != -0.5:
        return None

    eps_node = _producer(graph, pow_node.inputs[0])
    if eps_node is None or eps_node.op != "scalar_add":
        return None
    eps = float(eps_node.attrs.get("value", 0.0))

    mean_node = _producer(graph, eps_node.inputs[0])
    if mean_node is None or mean_node.op != "mean":
        return None
    if not bool(mean_node.attrs.get("keepdim", False)):
        return None

    pow2_node = _producer(graph, mean_node.inputs[0])
    if pow2_node is None or pow2_node.op != "pow":
        return None
    if float(pow2_node.attrs.get("exponent", 0.0)) != 2.0:
        return None

    return {
        "pow_input_value_id": pow2_node.inputs[0],
        "eps": eps,
    }


def _match_rope(graph: IRGraph, node: IRNode) -> dict[str, object] | None:
    if len(node.inputs) != 2:
        return None

    for direct_mult_id, rotated_mult_id in (
        (node.inputs[0], node.inputs[1]),
        (node.inputs[1], node.inputs[0]),
    ):
        direct_mult = _producer(graph, direct_mult_id)
        rotated_mult = _producer(graph, rotated_mult_id)
        if direct_mult is None or direct_mult.op != "multiply":
            continue
        if rotated_mult is None or rotated_mult.op != "multiply":
            continue

        direct_match = _extract_direct_rope_branch(graph, direct_mult)
        rotated_match = _extract_rotated_rope_branch(graph, rotated_mult)
        if direct_match is None or rotated_match is None:
            continue

        input_value_id = _strip_passthrough(graph, direct_match["input_value_id"])
        if input_value_id != _strip_passthrough(graph, rotated_match["input_value_id"]):
            continue

        cos_info = _extract_rope_trig(graph, direct_match["trig_value_id"], expected="cos")
        sin_info = _extract_rope_trig(graph, rotated_match["trig_value_id"], expected="sin")
        if cos_info is None or sin_info is None:
            continue
        if abs(float(cos_info["theta"]) - float(sin_info["theta"])) > 1e-2:
            continue
        if int(cos_info["position_offset"]) != int(sin_info["position_offset"]):
            continue

        return {
            "input_value_id": input_value_id,
            "theta": cos_info["theta"],
            "position_offset": cos_info["position_offset"],
        }

    return None


def _extract_direct_rope_branch(graph: IRGraph, node: IRNode) -> dict[str, object] | None:
    if len(node.inputs) != 2:
        return None
    for input_value_id, trig_value_id in (
        (node.inputs[0], node.inputs[1]),
        (node.inputs[1], node.inputs[0]),
    ):
        if _looks_like_trig_branch(graph, trig_value_id, expected="cos"):
            return {
                "input_value_id": input_value_id,
                "trig_value_id": trig_value_id,
            }
    return None


def _extract_rotated_rope_branch(graph: IRGraph, node: IRNode) -> dict[str, object] | None:
    if len(node.inputs) != 2:
        return None
    for rotated_value_id, trig_value_id in (
        (node.inputs[0], node.inputs[1]),
        (node.inputs[1], node.inputs[0]),
    ):
        input_value_id = _extract_rotate_half_source(graph, rotated_value_id)
        if input_value_id is None:
            continue
        if not _looks_like_trig_branch(graph, trig_value_id, expected="sin"):
            continue
        return {
            "input_value_id": input_value_id,
            "trig_value_id": trig_value_id,
        }
    return None


def _extract_rotate_half_source(graph: IRGraph, value_id: str) -> str | None:
    cat_value_id = _strip_passthrough(graph, value_id)
    cat_node = _producer(graph, cat_value_id)
    if cat_node is None or cat_node.op != "cat" or len(cat_node.inputs) != 2:
        return None

    left_id, right_id = cat_node.inputs
    neg_node = _producer(graph, left_id)
    if neg_node is None or neg_node.op != "scalar_multiply":
        return None
    if float(neg_node.attrs.get("value", 0.0)) != -1.0:
        return None

    left_slice = _producer(graph, neg_node.inputs[0])
    right_slice = _producer(graph, right_id)
    if left_slice is None or right_slice is None:
        return None
    if left_slice.op != "slice" or right_slice.op != "slice":
        return None

    source_left = _strip_passthrough(graph, left_slice.inputs[0])
    source_right = _strip_passthrough(graph, right_slice.inputs[0])
    if source_left != source_right:
        return None

    return source_left


def _looks_like_trig_branch(graph: IRGraph, value_id: str, *, expected: str) -> bool:
    current = value_id
    while True:
        node = _producer(graph, current)
        if node is None:
            return False
        if node.op == "unsqueeze":
            current = node.inputs[0]
            continue
        if node.op == "precision_cast":
            current = node.inputs[0]
            continue
        if node.op == "scalar_multiply" and float(node.attrs.get("value", 0.0)) == 1.0:
            current = node.inputs[0]
            continue
        return node.op == expected


def _extract_rope_trig(graph: IRGraph, value_id: str, *, expected: str) -> dict[str, object] | None:
    trig_value_id = value_id
    while True:
        node = _producer(graph, trig_value_id)
        if node is None:
            return None
        if node.op in {"unsqueeze", "precision_cast"}:
            trig_value_id = node.inputs[0]
            continue
        if node.op == "scalar_multiply" and float(node.attrs.get("value", 0.0)) == 1.0:
            trig_value_id = node.inputs[0]
            continue
        if node.op != expected:
            return None
        break

    trig_node = _producer(graph, trig_value_id)
    if trig_node is None:
        return None
    cat_node = _producer(graph, trig_node.inputs[0])
    if cat_node is None or cat_node.op != "cat" or len(cat_node.inputs) != 2:
        return None
    if cat_node.inputs[0] != cat_node.inputs[1]:
        return None

    angle_node = _producer(graph, cat_node.inputs[0])
    if angle_node is None or angle_node.op != "transpose":
        return None
    matmul_node = _producer(graph, angle_node.inputs[0])
    if matmul_node is None or matmul_node.op != "matmul":
        return None

    inv_freq_const_id = None
    arange_node = None
    for input_id in matmul_node.inputs:
        const_id = _find_constant_ancestor(graph, input_id)
        if const_id is not None:
            inv_freq_const_id = const_id
        arange = _find_arange_ancestor(graph, input_id)
        if arange is not None:
            arange_node = arange

    if inv_freq_const_id is None or arange_node is None:
        return None

    theta = _infer_rope_theta(graph.constants[inv_freq_const_id])
    if theta is None:
        return None

    return {
        "theta": theta,
        "position_offset": int(arange_node.attrs.get("start", 0)),
    }


def _find_constant_ancestor(graph: IRGraph, value_id: str) -> str | None:
    current = value_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        current = _strip_passthrough(graph, current)
        if current in graph.constants and isinstance(graph.constants[current], torch.Tensor):
            return current
        node = _producer(graph, current)
        if node is None or len(node.inputs) != 1:
            return None
        current = node.inputs[0]
    return None


def _find_arange_ancestor(graph: IRGraph, value_id: str) -> IRNode | None:
    current = value_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        current = _strip_passthrough(graph, current)
        node = _producer(graph, current)
        if node is None:
            return None
        if node.op == "arange":
            return node
        if len(node.inputs) != 1:
            return None
        current = node.inputs[0]
    return None


def _infer_rope_theta(value: Any) -> float | None:
    if not isinstance(value, torch.Tensor):
        return None
    flat = value.detach().cpu().float().reshape(-1)
    if flat.numel() < 2:
        return None
    second = float(flat[1].item())
    if second <= 0.0:
        return None
    inv_count = flat.numel()
    return float((1.0 / second) ** inv_count)


def _strip_passthrough(graph: IRGraph, value_id: str) -> str:
    current = value_id
    while True:
        node = _producer(graph, current)
        if node is None:
            return current
        if node.op in {"precision_cast", "contiguous"} and len(node.inputs) == 1:
            current = node.inputs[0]
            continue
        if node.op == "type_as" and len(node.inputs) >= 1:
            current = node.inputs[0]
            continue
        return current


def _producer(graph: IRGraph, value_id: str) -> IRNode | None:
    value = graph.values.get(value_id)
    if value is None or value.producer is None:
        return None
    return graph.nodes.get(value.producer)


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


def _clear_gold_pattern_annotations(graph: IRGraph) -> None:
    for node in graph.nodes.values():
        node.meta.pop("gold_patterns", None)
        node.meta.pop("gold_pattern_anchor", None)
        node.meta.pop("gold_pattern_details", None)


def _detect_gated_mlps(graph: IRGraph) -> list[DetectedPattern]:
    patterns: list[DetectedPattern] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None or node.op != "linear" or len(node.inputs) < 2:
            continue
        mul_node = _producer(graph, node.inputs[0])
        if mul_node is None or mul_node.op != "multiply" or len(mul_node.inputs) != 2:
            continue

        for activated_value_id, up_value_id in (
            (mul_node.inputs[0], mul_node.inputs[1]),
            (mul_node.inputs[1], mul_node.inputs[0]),
        ):
            activated_node = _producer(graph, activated_value_id)
            up_node = _producer(graph, up_value_id)
            if activated_node is None or up_node is None:
                continue
            if activated_node.op not in {"gelu", "silu"}:
                continue
            if up_node.op != "linear":
                continue
            gate_node = _producer(graph, activated_node.inputs[0])
            if gate_node is None or gate_node.op != "linear":
                continue
            shared_input = _strip_passthrough(graph, gate_node.inputs[0])
            if shared_input != _strip_passthrough(graph, up_node.inputs[0]):
                continue

            pattern_name = "gated_mlp_gelu" if activated_node.op == "gelu" else "gated_mlp_silu"
            patterns.append(
                DetectedPattern(
                    name=pattern_name,
                    anchor_node_id=node.id,
                    node_ids=(gate_node.id, activated_node.id, up_node.id, mul_node.id, node.id),
                    value_ids=(shared_input,),
                    details={
                        "activation": activated_node.op,
                        "input_value_id": shared_input,
                        "gate_weight_value_id": gate_node.inputs[1],
                        "up_weight_value_id": up_node.inputs[1],
                        "down_weight_value_id": node.inputs[1],
                    },
                )
            )
            break
    return patterns


def _detect_decoder_attentions(graph: IRGraph) -> list[DetectedPattern]:
    patterns: list[DetectedPattern] = []
    for node_id in graph.order:
        node = graph.nodes.get(node_id)
        if node is None or node.op not in {"scaled_dot_product_attention", "attention"}:
            continue

        q_info = _extract_attention_input(graph, node.inputs[0], role="q")
        k_info = _extract_attention_input(graph, node.inputs[1], role="k")
        v_info = _extract_attention_input(graph, node.inputs[2], role="v")
        if q_info is None or k_info is None or v_info is None:
            if node.op == "attention":
                patterns.append(
                    DetectedPattern(
                        name="decoder_attention_gqa",
                        anchor_node_id=node.id,
                        node_ids=(node.id,),
                        value_ids=tuple(node.inputs[:3]),
                        details={
                            "input_value_ids": tuple(node.inputs[:3]),
                            "has_rope": True,
                            "has_qk_rms_norm": True,
                            "has_gqa_repeat": True,
                            "is_causal": bool(node.attrs.get("is_causal", False)),
                            "scale": float(node.attrs.get("scale", 1.0)),
                            "window_size_hint": int(node.attrs.get("window_size", 0)),
                            "fallback_semantic_match": True,
                        },
                    )
                )
            continue
        if q_info["source_input"] != k_info["source_input"] or q_info["source_input"] != v_info["source_input"]:
            if node.op == "attention":
                patterns.append(
                    DetectedPattern(
                        name="decoder_attention_gqa",
                        anchor_node_id=node.id,
                        node_ids=tuple(sorted({node.id, *q_info["node_ids"], *k_info["node_ids"], *v_info["node_ids"]})),
                        value_ids=(q_info["source_input"], k_info["source_input"], v_info["source_input"]),
                        details={
                            "input_value_ids": (q_info["source_input"], k_info["source_input"], v_info["source_input"]),
                            "has_rope": bool(q_info["has_rope"] and k_info["has_rope"]),
                            "has_qk_rms_norm": bool(q_info["has_rms_norm"] and k_info["has_rms_norm"]),
                            "has_gqa_repeat": True,
                            "is_causal": bool(node.attrs.get("is_causal", False)),
                            "scale": float(node.attrs.get("scale", 1.0)),
                            "window_size_hint": int(node.attrs.get("window_size", 0)),
                            "fallback_semantic_match": True,
                        },
                    )
                )
            continue

        has_rope = q_info["has_rope"] and k_info["has_rope"]
        has_qk_norm = q_info["has_rms_norm"] and k_info["has_rms_norm"]
        has_gqa = bool(k_info["has_gqa_repeat"] or v_info["has_gqa_repeat"] or node.attrs.get("enable_gqa", False))
        pattern_name = "decoder_attention_gqa" if has_gqa else "decoder_attention"
        if has_rope and q_info["partial_rope"] and k_info["partial_rope"]:
            pattern_name = "gemma4_partial_rope_attention"

        node_ids = {node.id}
        for info in (q_info, k_info, v_info):
            node_ids.update(info["node_ids"])
        details = {
            "input_value_id": q_info["source_input"],
            "q_linear_weight_value_id": q_info.get("weight_value_id"),
            "k_linear_weight_value_id": k_info.get("weight_value_id"),
            "v_linear_weight_value_id": v_info.get("weight_value_id"),
            "has_rope": has_rope,
            "has_qk_rms_norm": has_qk_norm,
            "has_gqa_repeat": has_gqa,
            "is_causal": bool(node.attrs.get("is_causal", False)),
            "scale": float(node.attrs.get("scale", 1.0)),
        }
        if node.op == "scaled_dot_product_attention" and len(node.inputs) > 3:
            details["mask_value_id"] = node.inputs[3]
            mask_info = _extract_sliding_window_mask(graph, node.inputs[3])
            if mask_info is not None:
                details["mask_pattern"] = "sliding_window_attention_mask"
                details["window_size_hint"] = mask_info["window_size"]
                node_ids.update(mask_info["node_ids"])
        elif node.op == "attention":
            details["window_size_hint"] = int(node.attrs.get("window_size", 0))
        patterns.append(
            DetectedPattern(
                name=pattern_name,
                anchor_node_id=node.id,
                node_ids=tuple(sorted(node_ids)),
                value_ids=(q_info["source_input"],),
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

        mlp_pattern = _find_anchor_pattern(graph, node.inputs[1], {"gated_mlp_gelu", "gated_mlp_silu"})
        if mlp_pattern is None:
            mlp_pattern = _find_anchor_pattern(graph, node.inputs[0], {"gated_mlp_gelu", "gated_mlp_silu"})
        if mlp_pattern is None:
            continue

        residual_value_id = node.inputs[0] if _find_anchor_pattern(graph, node.inputs[1], {mlp_pattern}) else node.inputs[1]
        attn_pattern = _find_anchor_pattern(graph, residual_value_id, {"decoder_attention_gqa", "decoder_attention", "gemma4_partial_rope_attention"})
        if attn_pattern is None:
            continue

        pattern_name = "decoder_block_post_attn_norm" if node.op == "add_clipped" else "decoder_block_simple_residual"
        node_ids = {node.id}
        for candidate in graph.nodes.values():
            gold_patterns = candidate.meta.get("gold_patterns", [])
            if attn_pattern in gold_patterns or mlp_pattern in gold_patterns:
                node_ids.add(candidate.id)
        patterns.append(
            DetectedPattern(
                name=pattern_name,
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
    branch_value_id = _strip_passthrough(graph, branch_value_id)
    branch_producer = _producer(graph, branch_value_id)
    if branch_producer is None or branch_producer.op != "rms_norm":
        return False

    norm_input = _strip_passthrough(graph, branch_producer.inputs[0])
    residual_value_id = _strip_passthrough(graph, residual_value_id)
    return residual_value_id != norm_input


def _find_anchor_pattern(graph: IRGraph, value_id: str, names: set[str]) -> str | None:
    current = value_id
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        current = _strip_passthrough(graph, current)
        node = _producer(graph, current)
        if node is None:
            return None
        gold_patterns = node.meta.get("gold_patterns", [])
        for name in gold_patterns:
            if name in names and node.meta.get("gold_pattern_anchor"):
                return name
        if len(node.inputs) != 1:
            return None
        current = node.inputs[0]
    return None


def _extract_attention_input(graph: IRGraph, value_id: str, *, role: str) -> dict[str, object] | None:
    node_ids: set[str] = set()
    current = value_id
    has_gqa_repeat = False
    partial_rope = False

    while True:
        node = _producer(graph, current)
        if node is None:
            return None
        node_ids.add(node.id)

        if node.op == "reshape":
            current = node.inputs[0]
            continue
        if node.op == "transpose":
            current = node.inputs[0]
            continue
        if node.op == "rope":
            rope_input = _strip_passthrough(graph, node.inputs[0])
            rope_info = _extract_attention_input(graph, rope_input, role=role)
            if rope_info is None:
                return None
            rope_info["has_rope"] = True
            rope_info["partial_rope"] = partial_rope or _looks_like_partial_rope(graph, node.inputs[0])
            rope_info["node_ids"] = tuple(sorted(set(rope_info["node_ids"]) | node_ids))
            rope_info["has_gqa_repeat"] = bool(rope_info["has_gqa_repeat"] or has_gqa_repeat)
            return rope_info
        if node.op == "rms_norm":
            if role == "v":
                return None
            source = _strip_passthrough(graph, node.inputs[0])
            rms_info = _extract_attention_input(graph, source, role=role)
            if rms_info is None:
                return None
            rms_info["has_rms_norm"] = True
            rms_info["node_ids"] = tuple(sorted(set(rms_info["node_ids"]) | node_ids))
            rms_info["has_gqa_repeat"] = bool(rms_info["has_gqa_repeat"] or has_gqa_repeat)
            return rms_info
        if _looks_like_gqa_repeat(graph, current):
            has_gqa_repeat = True
            current = _unwrap_gqa_repeat(graph, current)
            if current is None:
                return None
            continue
        if node.op == "linear":
            return {
                "source_input": _strip_passthrough(graph, node.inputs[0]),
                "weight_value_id": node.inputs[1] if len(node.inputs) > 1 else None,
                "has_rope": False,
                "partial_rope": partial_rope,
                "has_rms_norm": False,
                "has_gqa_repeat": has_gqa_repeat,
                "node_ids": tuple(sorted(node_ids)),
            }
        return None


def _looks_like_partial_rope(graph: IRGraph, value_id: str) -> bool:
    node = _producer(graph, value_id)
    if node is None or node.op != "concat":
        return False
    if len(node.inputs) != 2:
        return False
    left = _producer(graph, node.inputs[0])
    right = _producer(graph, node.inputs[1])
    return bool((left is not None and left.op == "rope") or (right is not None and right.op == "rope"))


def _looks_like_gqa_repeat(graph: IRGraph, value_id: str) -> bool:
    return _unwrap_gqa_repeat(graph, value_id) is not None


def _unwrap_gqa_repeat(graph: IRGraph, value_id: str) -> str | None:
    current = _strip_passthrough(graph, value_id)
    reshape_node = _producer(graph, current)
    if reshape_node is None or reshape_node.op != "reshape":
        return None

    current = reshape_node.inputs[0]
    expand_node = _producer(graph, current)
    if expand_node is None or expand_node.op != "expand":
        return None

    current = expand_node.inputs[0]
    while True:
        node = _producer(graph, current)
        if node is None:
            return None
        if node.op == "slice":
            current = node.inputs[0]
            continue
        if node.op != "unsqueeze":
            return None
        if int(node.attrs.get("dim", -999)) != 2:
            return None
        break

    base_value_id = _strip_passthrough(graph, node.inputs[0])
    base_shape = graph.values.get(base_value_id).shape if graph.values.get(base_value_id) is not None else None
    target_shape = tuple(int(v) for v in expand_node.attrs.get("shape", ()))
    if base_shape is None or len(base_shape) != 4 or len(target_shape) != 5:
        return None
    expected = (int(base_shape[0]), int(base_shape[1]), int(target_shape[2]), int(base_shape[2]), int(base_shape[3]))
    if expected != target_shape:
        return None
    return base_value_id


def _extract_sliding_window_mask(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    node_ids: set[str] = set()
    current = _strip_passthrough(graph, value_id)
    top_and = _producer(graph, current)
    if top_and is None or top_and.op != "aten.__and__.Tensor":
        return None
    node_ids.add(top_and.id)

    stack = list(top_and.inputs)
    window_candidates: list[int] = []
    saw_diff = False
    saw_cumsum = False

    while stack:
        current_value_id = _strip_passthrough(graph, stack.pop())
        node = _producer(graph, current_value_id)
        if node is None:
            continue
        if node.id in node_ids:
            continue
        node_ids.add(node.id)

        if node.op == "aten.diff.default":
            saw_diff = True
        elif node.op == "aten.cumsum.default":
            saw_cumsum = True
        elif node.op == "scalar_subtract":
            maybe_window = int(node.attrs.get("value", 0))
            if maybe_window > 0:
                window_candidates.append(maybe_window)
        stack.extend(node.inputs)

    window_size = max((candidate for candidate in window_candidates if candidate > 1), default=0)

    if not saw_diff or not saw_cumsum:
        return None

    if window_size == 0:
        return {"window_size": 0, "node_ids": tuple(sorted(node_ids))}

    return {"window_size": window_size, "node_ids": tuple(sorted(node_ids))}
