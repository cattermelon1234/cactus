from __future__ import annotations

from typing import Any

import torch

from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.graph_ir import IRValue
from src.transpile.graph_ir import verify_ir


def run_cleanup_passes(graph: IRGraph) -> IRGraph:
    verify_ir(graph)
    normalize_axes(graph)
    fold_scalar_constant_ops(graph)
    canonicalize_binary_scalar_forms(graph)
    legalize_precisions(graph)
    fold_identity_layout_ops(graph)
    dce(graph)
    verify_ir(graph)
    return graph


def normalize_axes(graph: IRGraph) -> IRGraph:
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or not node.inputs:
            continue

        input_value = graph.values.get(node.inputs[0])
        input_shape = input_value.shape if input_value is not None else None
        input_rank = len(input_shape) if input_shape is not None else None

        if node.op in {"softmax", "sum", "mean", "variance", "min", "max", "slice", "index", "gather"}:
            if input_rank is not None and "axis" in node.attrs:
                axis = node.attrs["axis"]
                if isinstance(axis, int):
                    node.attrs["axis"] = _normalize_dim(axis, input_rank)
                elif isinstance(axis, (list, tuple)) and len(axis) == 1 and isinstance(axis[0], int):
                    node.attrs["axis"] = [_normalize_dim(axis[0], input_rank)] if isinstance(axis, list) else (_normalize_dim(axis[0], input_rank),)

        if node.op == "cat":
            if "axis" in node.attrs and isinstance(node.attrs["axis"], int) and node.inputs:
                first_input = graph.values.get(node.inputs[0])
                if first_input is not None and first_input.shape is not None:
                    node.attrs["axis"] = _normalize_dim(node.attrs["axis"], len(first_input.shape))

        if node.op == "transpose" and input_rank is not None:
            if "dim0" in node.attrs and isinstance(node.attrs["dim0"], int):
                node.attrs["dim0"] = _normalize_dim(node.attrs["dim0"], input_rank)
            if "dim1" in node.attrs and isinstance(node.attrs["dim1"], int):
                node.attrs["dim1"] = _normalize_dim(node.attrs["dim1"], input_rank)

        if node.op == "permute" and input_rank is not None:
            permutation = node.attrs.get("permutation")
            if isinstance(permutation, (tuple, list)):
                node.attrs["permutation"] = tuple(_normalize_dim(int(dim), input_rank) for dim in permutation)

        if node.op == "flatten" and input_rank is not None:
            if "start_dim" in node.attrs and isinstance(node.attrs["start_dim"], int):
                node.attrs["start_dim"] = _normalize_dim(node.attrs["start_dim"], input_rank)
            if "end_dim" in node.attrs and isinstance(node.attrs["end_dim"], int):
                node.attrs["end_dim"] = _normalize_dim(node.attrs["end_dim"], input_rank)

    rebuild_graph(graph)
    return graph


def fold_identity_layout_ops(graph: IRGraph) -> IRGraph:
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or len(node.inputs) != 1 or len(node.outputs) != 1:
            continue

        input_id = node.inputs[0]
        output_id = node.outputs[0]
        input_value = graph.values.get(input_id)
        output_value = graph.values.get(output_id)

        if input_value is None or output_value is None:
            continue

        if node.op == "reshape":
            target_shape = node.attrs.get("shape")
            if input_value.shape is not None and target_shape is not None:
                if tuple(target_shape) == tuple(input_value.shape):
                    _bypass_value(graph, output_id, input_id)
                    _remove_node(graph, node_id)
                    continue

        if node.op == "permute":
            permutation = node.attrs.get("permutation")
            if isinstance(permutation, (tuple, list)) and tuple(int(v) for v in permutation) == tuple(range(len(permutation))):
                _bypass_value(graph, output_id, input_id)
                _remove_node(graph, node_id)
                continue

        if node.op == "transpose":
            dim0 = node.attrs.get("dim0")
            dim1 = node.attrs.get("dim1")
            if isinstance(dim0, int) and isinstance(dim1, int) and dim0 == dim1:
                _bypass_value(graph, output_id, input_id)
                _remove_node(graph, node_id)
                continue

        if node.op == "flatten":
            if input_value.shape is None:
                continue
            start_dim = node.attrs.get("start_dim")
            end_dim = node.attrs.get("end_dim")
            if isinstance(start_dim, int) and isinstance(end_dim, int) and start_dim == end_dim:
                _bypass_value(graph, output_id, input_id)
                _remove_node(graph, node_id)
                continue

            if output_value.shape is not None and tuple(output_value.shape) == tuple(input_value.shape):
                _bypass_value(graph, output_id, input_id)
                _remove_node(graph, node_id)
                continue

        if node.op == "expand":
            target_shape = node.attrs.get("shape")
            if input_value.shape is not None and target_shape is not None:
                resolved_shape = []
                padded_input = (1,) * (len(target_shape) - len(input_value.shape)) + tuple(input_value.shape)
                for in_dim, tgt_dim in zip(padded_input, target_shape):
                    resolved_shape.append(in_dim if int(tgt_dim) == -1 else int(tgt_dim))
                if tuple(resolved_shape) == tuple(padded_input):
                    _bypass_value(graph, output_id, input_id)
                    _remove_node(graph, node_id)
                    continue

        if node.op in {"identity", "contiguous"}:
            _bypass_value(graph, output_id, input_id)
            _remove_node(graph, node_id)
            continue

    rebuild_graph(graph)
    return graph


def dce(graph: IRGraph) -> IRGraph:
    live_values = set(graph.outputs)
    live_nodes: set[str] = set()

    changed = True
    while changed:
        changed = False
        for node_id in reversed(graph.order):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if any(output in live_values for output in node.outputs):
                if node_id not in live_nodes:
                    live_nodes.add(node_id)
                    changed = True
                for input_id in node.inputs:
                    if input_id not in live_values:
                        live_values.add(input_id)
                        changed = True

    graph.order = [node_id for node_id in graph.order if node_id in live_nodes]
    graph.nodes = {node_id: node for node_id, node in graph.nodes.items() if node_id in live_nodes}

    live_value_ids = set(graph.inputs) | set(graph.outputs)
    for node in graph.nodes.values():
        live_value_ids.update(node.inputs)
        live_value_ids.update(node.outputs)

    graph.values = {value_id: value for value_id, value in graph.values.items() if value_id in live_value_ids}
    graph.constants = {value_id: value for value_id, value in graph.constants.items() if value_id in live_value_ids}

    rebuild_graph(graph)
    return graph


def fold_scalar_constant_ops(graph: IRGraph) -> IRGraph:
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or len(node.outputs) != 1:
            continue

        input_scalar = None
        if len(node.inputs) == 1:
            input_scalar = _extract_scalar_constant(graph, node.inputs[0])

        if node.op == "precision_cast" and input_scalar is not None:
            graph.constants[node.outputs[0]] = input_scalar
            _remove_node(graph, node_id)

    rebuild_graph(graph)
    return graph


def canonicalize_binary_scalar_forms(graph: IRGraph) -> IRGraph:
    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None or len(node.inputs) != 2 or len(node.outputs) != 1:
            continue
        if node.op not in {"add", "subtract", "multiply", "multiply_inplace", "divide"}:
            continue

        lhs_id, rhs_id = node.inputs
        lhs_scalar = _extract_scalar_constant(graph, lhs_id)
        rhs_scalar = _extract_scalar_constant(graph, rhs_id)

        if lhs_scalar is None and rhs_scalar is None:
            continue

        if node.op == "add":
            if rhs_scalar is not None:
                node.op = "scalar_add"
                node.inputs = [lhs_id]
                node.attrs = {"value": rhs_scalar}
            elif lhs_scalar is not None:
                node.op = "scalar_add"
                node.inputs = [rhs_id]
                node.attrs = {"value": lhs_scalar}
        elif node.op in {"multiply", "multiply_inplace"}:
            if rhs_scalar is not None:
                node.op = "scalar_multiply"
                node.inputs = [lhs_id]
                node.attrs = {"value": rhs_scalar}
            elif lhs_scalar is not None:
                node.op = "scalar_multiply"
                node.inputs = [rhs_id]
                node.attrs = {"value": lhs_scalar}
        elif node.op == "subtract":
            if rhs_scalar is not None:
                node.op = "scalar_subtract"
                node.inputs = [lhs_id]
                node.attrs = {"value": rhs_scalar}
            elif lhs_scalar is not None:
                node.op = "scalar_subtract_reverse"
                node.inputs = [rhs_id]
                node.attrs = {"value": lhs_scalar}
        elif node.op == "divide":
            if rhs_scalar is not None:
                node.op = "scalar_divide"
                node.inputs = [lhs_id]
                node.attrs = {"value": rhs_scalar}
            elif lhs_scalar is not None:
                node.op = "scalar_divide_reverse"
                node.inputs = [rhs_id]
                node.attrs = {"value": lhs_scalar}

    rebuild_graph(graph)
    return graph


FP32_SUPPORTED_ALL_INPUT_OPS = {
    "precision_cast",
    "type_as",
    "reshape",
    "flatten",
    "expand",
    "slice",
    "index",
    "gather",
    "layer_norm",
    "batch_norm",
    "glu",
}

FP32_SUPPORTED_INPUT_INDICES = {
    "embedding": {1},
}

FP16_ONLY_OUTPUT_OPS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "scalar_add",
    "scalar_subtract",
    "scalar_subtract_reverse",
    "scalar_multiply",
    "scalar_divide",
    "scalar_divide_reverse",
    "not_equal",
    "scalar_not_equal",
    "abs",
    "negate",
    "pow",
    "scalar_exp",
    "scalar_sqrt",
    "scalar_log",
    "cos",
    "sin",
    "transpose",
    "permute",
    "matmul",
    "cat",
    "softmax",
    "sum",
    "mean",
    "variance",
    "min",
    "max",
    "rms_norm",
    "embedding",
    "group_norm",
    "relu",
    "silu",
    "gelu",
    "gelu_erf",
    "sigmoid",
    "tanh",
    "linear",
    "addmm",
}


def legalize_precisions(graph: IRGraph) -> IRGraph:
    counter = 0

    for node_id in list(graph.order):
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        if node.op in FP32_SUPPORTED_ALL_INPUT_OPS:
            continue

        allowed_indices = FP32_SUPPORTED_INPUT_INDICES.get(node.op, set())
        inserted: list[IRNode] = []

        for input_index, value_id in enumerate(list(node.inputs)):
            if input_index in allowed_indices:
                continue
            value = graph.values.get(value_id)
            if value is None or value.dtype != "fp32":
                continue

            counter += 1
            cast_node_id = f"{node.id}_legalize_fp16_{counter}"
            cast_value_id = f"{node.outputs[0]}_legalize_fp16_{counter}"
            cast_node = IRNode(
                id=cast_node_id,
                op="precision_cast",
                inputs=[value_id],
                outputs=[cast_value_id],
                attrs={"dtype": "fp16"},
                meta={
                    "shape": value.shape,
                    "dtype": "fp16",
                    "inserted_by": "legalize_precisions",
                },
            )
            inserted.append(cast_node)
            node.inputs[input_index] = cast_value_id
            graph.values[cast_value_id] = IRValue(
                id=cast_value_id,
                shape=value.shape,
                dtype="fp16",
                producer=cast_node_id,
                users=[],
            )

        if inserted:
            insert_at = graph.order.index(node_id)
            for offset, cast_node in enumerate(inserted):
                graph.nodes[cast_node.id] = cast_node
                graph.order.insert(insert_at + offset, cast_node.id)

        if node.op in FP16_ONLY_OUTPUT_OPS:
            for output_id in node.outputs:
                if output_id in graph.values:
                    graph.values[output_id].dtype = "fp16"
            node.meta["dtype"] = "fp16"

    rebuild_graph(graph)
    return graph


def rebuild_graph(graph: IRGraph) -> None:
    new_values: dict[str, IRValue] = {}

    for value_id in graph.inputs:
        old = graph.values.get(value_id)
        if old is None:
            old = IRValue(id=value_id)
        new_values[value_id] = IRValue(
            id=value_id,
            shape=old.shape,
            dtype=old.dtype,
            producer=None,
            users=[],
        )

    for value_id in graph.constants.keys():
        old = graph.values.get(value_id)
        if old is None:
            old = IRValue(id=value_id)
        new_values[value_id] = IRValue(
            id=value_id,
            shape=old.shape,
            dtype=old.dtype,
            producer=None,
            users=[],
        )

    for node_id in graph.order:
        node = graph.nodes[node_id]
        for output_id in node.outputs:
            old = graph.values.get(output_id)
            if old is None:
                old = IRValue(id=output_id)
            new_values[output_id] = IRValue(
                id=output_id,
                shape=old.shape,
                dtype=old.dtype,
                producer=node_id,
                users=[],
            )

    for node_id in graph.order:
        node = graph.nodes[node_id]
        for input_id in node.inputs:
            if input_id not in new_values:
                old = graph.values.get(input_id)
                if old is None:
                    new_values[input_id] = IRValue(id=input_id, users=[])
                else:
                    new_values[input_id] = IRValue(
                        id=input_id,
                        shape=old.shape,
                        dtype=old.dtype,
                        producer=old.producer,
                        users=[],
                    )
            new_values[input_id].users.append(node_id)

    for output_id in graph.outputs:
        if output_id not in new_values:
            old = graph.values.get(output_id)
            if old is None:
                new_values[output_id] = IRValue(id=output_id, users=[])
            else:
                new_values[output_id] = IRValue(
                    id=output_id,
                    shape=old.shape,
                    dtype=old.dtype,
                    producer=old.producer,
                    users=[],
                )

    graph.values = new_values
    verify_ir(graph)


def _bypass_value(graph: IRGraph, old_value_id: str, new_value_id: str) -> None:
    for node in graph.nodes.values():
        node.inputs = [new_value_id if value_id == old_value_id else value_id for value_id in node.inputs]
    graph.outputs = [new_value_id if value_id == old_value_id else value_id for value_id in graph.outputs]


def _remove_node(graph: IRGraph, node_id: str) -> None:
    if node_id in graph.nodes:
        del graph.nodes[node_id]
    graph.order = [existing_id for existing_id in graph.order if existing_id != node_id]


def _extract_scalar_constant(graph: IRGraph, value_id: str) -> float | int | None:
    if value_id not in graph.constants:
        return None

    value = graph.constants[value_id]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        scalar = value.detach().cpu().item()
        if isinstance(scalar, bool):
            return int(scalar)
        if isinstance(scalar, (int, float)):
            return scalar
    return None


def _normalize_dim(dim: int, rank: int) -> int:
    if dim < 0:
        dim += rank
    return dim
