from __future__ import annotations

from dataclasses import dataclass

from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.fusion.common import producer
from src.transpile.fusion.common import strip_passthrough
from src.transpile.fusion.linear import match_linear
from src.transpile.fusion.rms_norm import match_rms_norm
from src.transpile.fusion.rope import match_rope


@dataclass(frozen=True)
class AttentionMatch:
    query_value_id: str
    key_value_id: str
    value_value_id: str
    source_input_value_ids: tuple[str, str, str]
    weight_value_ids: tuple[str | None, str | None, str | None]
    has_rope: bool
    has_qk_norm: bool
    has_gqa_repeat: bool
    is_causal: bool
    scale: float
    window_size: int
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class AttentionBlockMatch:
    attention_node_id: str
    query_value_id: str
    key_value_id: str
    value_value_id: str
    gate_value_id: str | None
    output_projection_weight_value_id: str
    output_projection_bias_value_id: str | None
    attention_output_shape: tuple[int, ...]
    is_causal: bool
    scale: float
    window_size: int
    node_ids: tuple[str, ...]


def match_attention(graph: IRGraph, node: IRNode) -> AttentionMatch | None:
    if node.op not in {"attention", "scaled_dot_product_attention"} or len(node.inputs) < 3:
        return None

    q_info = _extract_attention_input(graph, node.inputs[0], role="q")
    k_info = _extract_attention_input(graph, node.inputs[1], role="k")
    v_info = _extract_attention_input(graph, node.inputs[2], role="v")
    if q_info is None or k_info is None or v_info is None:
        return None

    has_rope = bool(q_info["has_rope"] and k_info["has_rope"])
    has_qk_norm = bool(q_info["has_rms_norm"] and k_info["has_rms_norm"])
    has_gqa = bool(k_info["has_gqa_repeat"] or v_info["has_gqa_repeat"] or node.attrs.get("enable_gqa", False))
    node_ids = {node.id, *q_info["node_ids"], *k_info["node_ids"], *v_info["node_ids"]}

    return AttentionMatch(
        query_value_id=node.inputs[0],
        key_value_id=node.inputs[1],
        value_value_id=node.inputs[2],
        source_input_value_ids=(q_info["source_input"], k_info["source_input"], v_info["source_input"]),
        weight_value_ids=(q_info.get("weight_value_id"), k_info.get("weight_value_id"), v_info.get("weight_value_id")),
        has_rope=has_rope,
        has_qk_norm=has_qk_norm,
        has_gqa_repeat=has_gqa,
        is_causal=bool(node.attrs.get("is_causal", False)),
        scale=float(node.attrs.get("scale", 1.0)),
        window_size=int(node.attrs.get("window_size", 0)),
        node_ids=tuple(sorted(node_ids)),
    )


def match_attention_block(graph: IRGraph, node: IRNode) -> AttentionBlockMatch | None:
    projection = match_linear(graph, node)
    if projection is None:
        return None

    output_path = _extract_attention_output_path(graph, projection.input_value_id)
    if output_path is None:
        return None
    attention_node = graph.nodes.get(output_path["attention_node_id"])
    if attention_node is None or attention_node.op not in {"attention", "scaled_dot_product_attention"} or len(attention_node.inputs) < 3:
        return None

    return AttentionBlockMatch(
        attention_node_id=attention_node.id,
        query_value_id=attention_node.inputs[0],
        key_value_id=attention_node.inputs[1],
        value_value_id=attention_node.inputs[2],
        gate_value_id=output_path["gate_value_id"],
        output_projection_weight_value_id=projection.weight_value_id,
        output_projection_bias_value_id=projection.bias_value_id,
        attention_output_shape=output_path["attention_output_shape"],
        is_causal=bool(attention_node.attrs.get("is_causal", True)),
        scale=float(attention_node.attrs.get("scale", 1.0)),
        window_size=int(attention_node.attrs.get("window_size", 0)),
        node_ids=tuple(sorted({attention_node.id, *projection.node_ids, *output_path["node_ids"]})),
    )


def _extract_attention_output_path(graph: IRGraph, value_id: str) -> dict[str, object] | None:
    current = value_id
    gate_value_id: str | None = None

    while True:
        node = producer(graph, current)
        if node is None:
            return None
        if node.op in {"precision_cast", "contiguous"} and len(node.inputs) == 1:
            current = node.inputs[0]
            continue
        if node.op == "type_as" and len(node.inputs) >= 1:
            current = node.inputs[0]
            continue
        break

    node = producer(graph, current)
    if node is not None and node.op == "multiply" and len(node.inputs) == 2:
        for attn_candidate, gate_candidate in ((node.inputs[0], node.inputs[1]), (node.inputs[1], node.inputs[0])):
            attn_match = _extract_attention_output_path(graph, attn_candidate)
            if attn_match is None:
                continue
            attn_match["gate_value_id"] = gate_candidate
            attn_match["node_ids"] = tuple(sorted({node.id, *attn_match["node_ids"]}))
            return attn_match

    reshape_node = producer(graph, current)
    if reshape_node is None or reshape_node.op not in {"reshape", "view"} or len(reshape_node.inputs) != 1:
        return None

    attention_output_shape = tuple(int(v) for v in reshape_node.attrs.get("shape", ()))
    current = reshape_node.inputs[0]
    transpose_node = producer(graph, current)
    if transpose_node is None:
        return None

    if transpose_node.op == "transpose":
        if int(transpose_node.attrs.get("dim0", -1)) != 1 or int(transpose_node.attrs.get("dim1", -1)) != 2:
            return None
        current = transpose_node.inputs[0]
    elif transpose_node.op == "permute":
        permutation = tuple(int(v) for v in transpose_node.attrs.get("permutation", ()))
        if permutation != (0, 2, 1, 3):
            return None
        current = transpose_node.inputs[0]
    else:
        return None

    attention_node = producer(graph, current)
    if attention_node is None or attention_node.op not in {"attention", "scaled_dot_product_attention"}:
        return None

    return {
        "attention_node_id": attention_node.id,
        "gate_value_id": gate_value_id,
        "attention_output_shape": attention_output_shape,
        "node_ids": tuple(sorted({reshape_node.id, transpose_node.id})),
    }


def _extract_attention_input(graph: IRGraph, value_id: str, *, role: str) -> dict[str, object] | None:
    node_ids: set[str] = set()
    current = strip_passthrough(graph, value_id)
    has_gqa_repeat = False

    while True:
        node = producer(graph, current)
        if node is None:
            return {
                "source_input": current,
                "weight_value_id": None,
                "has_rope": False,
                "has_rms_norm": False,
                "has_gqa_repeat": has_gqa_repeat,
                "node_ids": tuple(sorted(node_ids)),
            }
        node_ids.add(node.id)

        if node.op in {"reshape", "view", "transpose", "permute"}:
            current = node.inputs[0]
            continue

        rope = match_rope(graph, current)
        if rope is not None:
            rope_info = _extract_attention_input(graph, rope.input_value_id, role=role)
            if rope_info is None:
                return None
            rope_info["has_rope"] = True
            rope_info["node_ids"] = tuple(sorted(set(rope_info["node_ids"]) | node_ids | set(rope.node_ids)))
            rope_info["has_gqa_repeat"] = bool(rope_info["has_gqa_repeat"] or has_gqa_repeat)
            return rope_info

        rms = match_rms_norm(graph, node)
        if rms is not None:
            if role == "v":
                return None
            rms_info = _extract_attention_input(graph, rms.input_value_id, role=role)
            if rms_info is None:
                return None
            rms_info["has_rms_norm"] = True
            rms_info["node_ids"] = tuple(sorted(set(rms_info["node_ids"]) | node_ids | set(rms.node_ids)))
            rms_info["has_gqa_repeat"] = bool(rms_info["has_gqa_repeat"] or has_gqa_repeat)
            return rms_info

        if _looks_like_gqa_repeat(graph, current):
            has_gqa_repeat = True
            current = _unwrap_gqa_repeat(graph, current)
            if current is None:
                return None
            continue

        linear_node = producer(graph, current)
        if linear_node is None:
            return {
                "source_input": strip_passthrough(graph, current),
                "weight_value_id": None,
                "has_rope": False,
                "has_rms_norm": False,
                "has_gqa_repeat": has_gqa_repeat,
                "node_ids": tuple(sorted(node_ids)),
            }
        linear = match_linear(graph, linear_node)
        if linear is None:
            return {
                "source_input": strip_passthrough(graph, current),
                "weight_value_id": None,
                "has_rope": False,
                "has_rms_norm": False,
                "has_gqa_repeat": has_gqa_repeat,
                "node_ids": tuple(sorted(node_ids)),
            }
        return {
            "source_input": strip_passthrough(graph, linear.input_value_id),
            "weight_value_id": linear.weight_value_id,
            "has_rope": False,
            "has_rms_norm": False,
            "has_gqa_repeat": has_gqa_repeat,
            "node_ids": tuple(sorted(set(node_ids) | set(linear.node_ids))),
        }


def _looks_like_gqa_repeat(graph: IRGraph, value_id: str) -> bool:
    return _unwrap_gqa_repeat(graph, value_id) is not None


def _unwrap_gqa_repeat(graph: IRGraph, value_id: str) -> str | None:
    current = strip_passthrough(graph, value_id)
    reshape_node = producer(graph, current)
    if reshape_node is None or reshape_node.op not in {"reshape", "view"}:
        return None
    current = reshape_node.inputs[0]
    expand_node = producer(graph, current)
    if expand_node is None or expand_node.op != "expand":
        return None
    current = expand_node.inputs[0]
    while True:
        node = producer(graph, current)
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
    base_value_id = strip_passthrough(graph, node.inputs[0])
    base_shape = graph.values.get(base_value_id).shape if graph.values.get(base_value_id) is not None else None
    target_shape = tuple(int(v) for v in expand_node.attrs.get("shape", ()))
    if base_shape is None or len(base_shape) != 4 or len(target_shape) != 5:
        return None
    expected = (int(base_shape[0]), int(base_shape[1]), int(target_shape[2]), int(base_shape[2]), int(base_shape[3]))
    if expected != target_shape:
        return None
    return base_value_id
