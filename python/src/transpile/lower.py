from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.graph import Graph
from src.graph import Tensor
from src.transpile.capture_pytorch import CapturedModel
from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.graph_ir import IRValue
from src.transpile.graph_ir import verify_ir


@dataclass
class TranspiledGraph:
    graph: Graph
    runtime_inputs: list[Tensor]
    bound_constants: list[Tensor]
    outputs: list[Tensor]

    def set_input(self, index: int, data: Any, *, dtype: int | None = None) -> None:
        if index < 0 or index >= len(self.runtime_inputs):
            raise IndexError(
                f"runtime input index out of range: {index} (have {len(self.runtime_inputs)})"
            )
        self.graph.set_input(self.runtime_inputs[index], data, dtype=dtype)

    def set_inputs(self, inputs: list[Any] | tuple[Any, ...]) -> None:
        if len(inputs) != len(self.runtime_inputs):
            raise ValueError(
                f"expected {len(self.runtime_inputs)} runtime inputs, got {len(inputs)}"
            )
        for index, value in enumerate(inputs):
            self.set_input(index, value)

    def execute(self) -> list[Tensor]:
        self.graph.execute()
        return self.outputs


@dataclass
class BroadcastAlias:
    tensor: Tensor
    logical_shape: tuple[int, ...]
    kind: str


def transpile_captured(captured: CapturedModel) -> TranspiledGraph:
    return transpile_ir(captured.ir_graph)


def transpile_ir(ir: IRGraph) -> TranspiledGraph:
    verify_ir(ir)
    g = Graph()
    env: dict[str, Any] = {}
    runtime_inputs: list[Tensor] = []
    bound_constants: list[Tensor] = []

    for value_id in ir.inputs:
        value = ir.values[value_id]
        tensor = _lower_input_value(g, value)
        env[value_id] = tensor
        runtime_inputs.append(tensor)

    for value_id, const in ir.constants.items():
        value = ir.values[value_id]
        lowered_const = _lower_constant_value(g, value, const)
        env[value_id] = lowered_const
        if isinstance(lowered_const, Tensor):
            bound_constants.append(lowered_const)

    for node_id in ir.order:
        node = ir.nodes[node_id]
        outputs = _lower_ir_node(g, node, env, ir)
        if len(outputs) != len(node.outputs):
            raise ValueError(
                f"node {node.id} produced {len(outputs)} outputs, expected {len(node.outputs)}"
            )
        for output_id, tensor in zip(node.outputs, outputs):
            env[output_id] = tensor

    outputs = [env[value_id] for value_id in ir.outputs]
    return TranspiledGraph(
        graph=g,
        runtime_inputs=runtime_inputs,
        bound_constants=bound_constants,
        outputs=outputs,
    )


def _lower_input_value(g: Graph, value: IRValue) -> Tensor:
    if value.shape is None or value.dtype is None:
        raise ValueError(f"IR input missing shape or dtype: {value.id}")
    return g.input(shape=value.shape, dtype=_map_ir_dtype(value.dtype))


def _lower_constant_value(g: Graph, value: IRValue, const: Any) -> Any:
    if isinstance(const, torch.nn.Parameter):
        const = const.detach()
    if isinstance(const, torch.Tensor):
        tensor_value = const.detach().cpu()
    else:
        raise NotImplementedError(
            f"unsupported IR constant type for {value.id}: {type(const).__name__}"
        )

    if tensor_value.numel() == 1:
        return tensor_value.item()

    return _materialize_constant_tensor(g, tensor_value)


def _lower_ir_node(g: Graph, node: IRNode, env: dict[str, Any], ir: IRGraph) -> list[Any]:
    op = node.op

    if op == "arange":
        start = int(node.attrs.get("start", 0))
        end = int(node.attrs["end"])
        step = int(node.attrs.get("step", 1))
        dtype = _materialize_constant_torch_dtype(node.attrs.get("dtype"))
        tensor_value = torch.arange(start, end, step=step, dtype=dtype)
        return [_materialize_constant_tensor(g, tensor_value)]

    if op == "add":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "add")]

    if op == "add_clipped":
        lhs, rhs = _legalize_elementwise_binary_inputs(g, _tensor(env, node.inputs[0]), _tensor(env, node.inputs[1]))
        return [g.add_clipped(lhs, rhs)]

    if op == "subtract":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "subtract")]

    if op == "multiply":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "multiply")]

    if op == "multiply_inplace":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "multiply")]

    if op == "divide":
        return [_lower_binary_op(g, env[node.inputs[0]], env[node.inputs[1]], "divide")]

    if op == "not_equal":
        return [_lower_compare_op(g, env[node.inputs[0]], env[node.inputs[1]], "not_equal")]

    if op == "scalar_add":
        return [g.scalar_add(_tensor(env, node.inputs[0]), float(node.attrs["value"]))]

    if op == "scalar_subtract":
        return [g.scalar_subtract(_tensor(env, node.inputs[0]), float(node.attrs["value"]))]

    if op == "scalar_subtract_reverse":
        x = _tensor(env, node.inputs[0])
        return [g.scalar_add(g.scalar_multiply(x, -1.0), float(node.attrs["value"]))]

    if op == "scalar_multiply":
        return [g.scalar_multiply(_tensor(env, node.inputs[0]), float(node.attrs["value"]))]

    if op == "scalar_divide":
        return [g.scalar_divide(_tensor(env, node.inputs[0]), float(node.attrs["value"]))]

    if op == "scalar_not_equal":
        return [g.scalar_not_equal(_tensor(env, node.inputs[0]), float(node.attrs["value"]))]

    if op == "scalar_divide_reverse":
        raise NotImplementedError("scalar_divide_reverse is not directly supported by Cactus graph ops")

    if op == "precision_cast":
        target_dtype = node.attrs.get("dtype")
        if target_dtype is None:
            return [_tensor(env, node.inputs[0])]
        return [g.precision_cast(_tensor(env, node.inputs[0]), _map_ir_or_torch_dtype(target_dtype))]

    if op == "type_as":
        source = _tensor(env, node.inputs[0])
        target = _tensor(env, node.inputs[1])
        return [g.precision_cast(source, target.dtype)]

    if op == "abs":
        return [g.abs(_tensor(env, node.inputs[0]))]

    if op == "negate":
        return [g.scalar_multiply(_tensor(env, node.inputs[0]), -1.0)]

    if op == "pow":
        exponent = node.attrs.get("exponent")
        if exponent is None:
            raise NotImplementedError("tensor-tensor pow is not supported by Cactus graph ops")
        return [g.pow(_tensor(env, node.inputs[0]), float(exponent))]

    if op == "scalar_exp":
        return [g.scalar_exp(_tensor(env, node.inputs[0]))]

    if op == "scalar_sqrt":
        return [g.scalar_sqrt(_tensor(env, node.inputs[0]))]

    if op == "scalar_log":
        return [g.scalar_log(_tensor(env, node.inputs[0]))]

    if op == "cos":
        return [g.scalar_cos(_tensor(env, node.inputs[0]))]

    if op == "sin":
        return [g.scalar_sin(_tensor(env, node.inputs[0]))]

    if op == "reshape":
        source = env[node.inputs[0]]
        if isinstance(source, BroadcastAlias):
            input_shape = tuple(source.logical_shape)
        else:
            input_shape = tuple(_tensor(env, node.inputs[0]).shape)
        target_shape = _resolve_reshape_shape(input_shape, tuple(node.attrs["shape"]))
        if isinstance(source, BroadcastAlias):
            if source.kind != "gqa_repeat_kv":
                raise NotImplementedError(f"unsupported broadcast alias kind in reshape: {source.kind}")
            base_shape = tuple(source.tensor.shape)
            if len(base_shape) != 4:
                raise NotImplementedError(f"gqa_repeat_kv alias requires 4D base tensor, got {base_shape}")
            if len(target_shape) != 4:
                raise NotImplementedError(f"gqa_repeat_kv reshape requires 4D target shape, got {target_shape}")
            batch, kv_heads, seq_len, head_dim = base_shape
            if target_shape[0] != batch or target_shape[2] != seq_len or target_shape[3] != head_dim:
                raise NotImplementedError(
                    f"gqa_repeat_kv reshape mismatch: base {base_shape}, target {target_shape}"
                )
            if target_shape[1] % max(kv_heads, 1) != 0:
                raise NotImplementedError(
                    f"gqa_repeat_kv target head count must be a multiple of kv heads: {base_shape} -> {target_shape}"
                )
            return [BroadcastAlias(source.tensor, target_shape, source.kind)]
        return [g.reshape(_tensor(env, node.inputs[0]), target_shape)]

    if op == "flatten":
        return [
            g.flatten(
                _tensor(env, node.inputs[0]),
                start_dim=int(node.attrs["start_dim"]),
                end_dim=int(node.attrs["end_dim"]),
            )
        ]

    if op == "unsqueeze":
        x = _tensor(env, node.inputs[0])
        target_shape = node.attrs.get("shape")
        if target_shape is None:
            dim = _normalize_dim(int(node.attrs["dim"]), len(x.shape) + 1)
            shape_list = list(x.shape)
            shape_list.insert(dim, 1)
            target_shape = tuple(shape_list)
        return [g.reshape(x, tuple(target_shape))]

    if op == "expand":
        matched_base = _match_gqa_expand_alias(ir, node)
        if matched_base is not None:
            return [BroadcastAlias(_tensor(env, matched_base), tuple(node.attrs["shape"]), "gqa_repeat_kv")]
        x = _tensor(env, node.inputs[0])
        target_shape = _resolve_expand_shape(tuple(x.shape), tuple(node.attrs["shape"]))
        if target_shape == tuple(x.shape):
            return [x]
        raise NotImplementedError(
            f"expand requires broadcast materialization unsupported by Cactus graph ops: "
            f"{tuple(x.shape)} -> {target_shape}"
        )

    if op == "transpose":
        x = _tensor(env, node.inputs[0])
        dim0 = _normalize_dim(int(node.attrs["dim0"]), len(x.shape))
        dim1 = _normalize_dim(int(node.attrs["dim1"]), len(x.shape))
        rank = len(x.shape)
        permutation = list(range(rank))
        permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
        if rank == 2 and permutation == [1, 0]:
            return [g.transpose(x)]
        return [g.permute(x, permutation)]

    if op == "permute":
        x = _tensor(env, node.inputs[0])
        permutation = tuple(_normalize_dim(int(dim), len(x.shape)) for dim in node.attrs["permutation"])
        if len(permutation) == 2 and permutation == (1, 0):
            return [g.transpose(x)]
        return [g.permute(x, permutation)]

    if op == "matmul":
        lhs = _tensor(env, node.inputs[0])
        rhs = _tensor(env, node.inputs[1])
        if len(lhs.shape) == 2 and len(rhs.shape) == 2:
            return [g.matmul(lhs, rhs)]
        legalized = _legalize_matmul_inputs(g, lhs, rhs, node)
        if legalized is None:
            return [g.matmul(lhs, rhs)]
        lhs_2d, rhs_2d, output_shape = legalized
        out = g.matmul(lhs_2d, rhs_2d)
        return [g.reshape(out, output_shape)]

    if op == "linear":
        x = _tensor(env, node.inputs[0])
        weight = _tensor(env, node.inputs[1])
        reshape_back: tuple[int, ...] | None = None
        if len(x.shape) > 2:
            x = _flatten_to_2d_for_linear(g, x)
            output_value = ir.values.get(node.outputs[0])
            output_shape = output_value.shape if output_value is not None else None
            if output_shape is None:
                output_shape = node.meta.get("shape")
            if not isinstance(output_shape, tuple):
                raise NotImplementedError(f"linear missing output shape metadata for node {node.id}")
            reshape_back = tuple(int(v) for v in output_shape)

        out = g.matmul(x, weight, pretransposed_rhs=True)
        if node.attrs.get("has_bias"):
            out = g.add(out, _tensor(env, node.inputs[2]))
        if reshape_back is not None:
            out = g.reshape(out, reshape_back)
        return [out]

    if op == "addmm":
        bias = _tensor(env, node.inputs[0])
        lhs = _tensor(env, node.inputs[1])
        rhs = _tensor(env, node.inputs[2])
        return [g.add(bias, g.matmul(lhs, rhs))]

    if op == "relu":
        return [g.relu(_tensor(env, node.inputs[0]))]

    if op == "silu":
        return [g.silu(_tensor(env, node.inputs[0]))]

    if op == "gelu":
        return [g.gelu(_tensor(env, node.inputs[0]))]

    if op == "gelu_erf":
        return [g.gelu_erf(_tensor(env, node.inputs[0]))]

    if op == "sigmoid":
        return [g.sigmoid(_tensor(env, node.inputs[0]))]

    if op == "tanh":
        return [g.tanh(_tensor(env, node.inputs[0]))]

    if op == "softmax":
        return [g.softmax(_tensor(env, node.inputs[0]), axis=int(node.attrs.get("axis", -1)))]

    if op in {"scaled_dot_product_attention", "attention"}:
        mask = node.attrs.get("mask")
        mask_tensor: Tensor | None = None
        if op == "scaled_dot_product_attention" and len(node.inputs) > 3:
            mask_tensor = _tensor(env, node.inputs[3])
        elif mask is not None:
            raise NotImplementedError(f"{op} with literal mask is not supported yet")
        if op == "scaled_dot_product_attention":
            dropout_p = float(node.attrs.get("dropout_p", 0.0))
            if dropout_p != 0.0:
                raise NotImplementedError("scaled_dot_product_attention with dropout is not supported yet")

        # PyTorch SDPA exports tensors as [batch, heads, seq, dim], while Cactus attention
        # expects [batch, seq, heads, dim]. Convert into Cactus layout, call attention, then
        # convert back so downstream exported ops still see PyTorch layout.
        query = g.permute(_attention_tensor(env, node.inputs[0]), (0, 2, 1, 3))
        key = g.permute(_attention_tensor(env, node.inputs[1]), (0, 2, 1, 3))
        value = g.permute(_attention_tensor(env, node.inputs[2]), (0, 2, 1, 3))
        out = g.attention(
            query,
            key,
            value,
            scale=float(node.attrs.get("scale", 1.0)),
            is_causal=bool(node.attrs.get("is_causal", False)),
            window_size=int(node.attrs.get("window_size", 0)),
            mask=mask_tensor,
        )
        return [g.permute(out, (0, 2, 1, 3))]

    if op in ("sum", "mean", "variance", "min", "max"):
        axis = node.attrs.get("axis")
        if isinstance(axis, (list, tuple)) and len(axis) == 1 and isinstance(axis[0], int):
            axis = int(axis[0])
        if not isinstance(axis, int):
            raise NotImplementedError(f"{op} currently requires a single integer axis")
        fn = getattr(g, op)
        x = _tensor(env, node.inputs[0])
        out = fn(x, axis)
        if bool(node.attrs.get("keepdim", False)):
            reduced_axis = _normalize_dim(axis, len(x.shape))
            new_shape = list(x.shape)
            new_shape[reduced_axis] = 1
            out = g.reshape(out, tuple(new_shape))
        return [out]

    if op == "cat":
        tensors = [_tensor(env, value_id) for value_id in node.inputs]
        return [g.cat(tensors, axis=int(node.attrs.get("axis", 0)))]

    if op == "slice":
        x = _tensor(env, node.inputs[0])
        axis = _normalize_dim(int(node.attrs["axis"]), len(x.shape))
        start = int(node.attrs["start"])
        end = int(node.attrs["end"])
        step = int(node.attrs.get("step", 1))
        if step != 1:
            raise NotImplementedError("slice with step != 1 is not supported by Cactus graph ops")
        dim_size = x.shape[axis]
        start = _normalize_index(start, dim_size)
        end = _normalize_slice_end(end, dim_size)
        length = max(0, end - start)
        return [g.slice(x, axis=axis, start=start, length=length)]

    if op == "index":
        x = _tensor(env, node.inputs[0])
        axis = _normalize_dim(int(node.attrs.get("axis", 0)), len(x.shape))
        index_value = _normalize_index(int(node.attrs["index_value"]), x.shape[axis])
        return [g.index(x, index_value=index_value, axis=axis)]

    if op == "gather":
        return [g.gather(_tensor(env, node.inputs[0]), _tensor(env, node.inputs[1]))]

    if op == "embedding":
        return [g.embedding_from_tensor(_tensor(env, node.inputs[0]), _tensor(env, node.inputs[1]))]

    if op == "layer_norm":
        x = _tensor(env, node.inputs[0])
        weight = _tensor(env, node.inputs[1])
        bias = _tensor(env, node.inputs[2]) if len(node.inputs) > 2 else None
        return [g.layer_norm(x, weight, bias=bias, eps=float(node.attrs["eps"]))]

    if op == "rms_norm":
        x = _tensor(env, node.inputs[0])
        weight = _tensor(env, node.inputs[1])
        reshape_back: tuple[int, ...] | None = None
        if len(x.shape) > 2:
            reshape_back = tuple(int(dim) for dim in x.shape)
            x = _flatten_to_2d_for_linear(g, x)
        out = g.rms_norm(x, weight, eps=float(node.attrs["eps"]))
        if reshape_back is not None:
            out = g.reshape(out, reshape_back)
        return [out]

    if op == "rope":
        return [
            g.rope(
                _tensor(env, node.inputs[0]),
                float(node.attrs["theta"]),
                position_offset=int(node.attrs.get("position_offset", 0)),
            )
        ]

    if op == "group_norm":
        return [
            g.group_norm(
                _tensor(env, node.inputs[0]),
                _tensor(env, node.inputs[1]),
                _tensor(env, node.inputs[2]),
                num_groups=int(node.attrs["num_groups"]),
                eps=float(node.attrs["eps"]),
            )
        ]

    if op == "batch_norm":
        axis = int(node.attrs.get("axis", 1))
        return [
            g.batch_norm(
                _tensor(env, node.inputs[0]),
                _tensor(env, node.inputs[1]),
                _tensor(env, node.inputs[2]),
                _tensor(env, node.inputs[3]),
                _tensor(env, node.inputs[4]),
                axis=axis,
                eps=float(node.attrs["eps"]),
            )
        ]

    if op == "identity":
        return [env[node.inputs[0]]]

    if op == "contiguous":
        return [_tensor(env, node.inputs[0])]

    if op == "getitem":
        source = env[node.inputs[0]]
        index = int(node.attrs["index"])
        if not isinstance(source, (tuple, list)):
            raise NotImplementedError(f"getitem source is not tuple/list for node {node.id}")
        return [source[index]]

    raise NotImplementedError(f"unsupported IR op in lowering: {op}")


def _tensor(env: dict[str, Any], value_id: str) -> Tensor:
    try:
        value = env[value_id]
    except KeyError as exc:
        raise NotImplementedError(f"missing IR value during lowering: {value_id}") from exc
    if not isinstance(value, Tensor):
        raise TypeError(f"expected lowered tensor for {value_id}, got {type(value).__name__}")
    return value


def _attention_tensor(env: dict[str, Any], value_id: str) -> Tensor:
    try:
        value = env[value_id]
    except KeyError as exc:
        raise NotImplementedError(f"missing IR value during lowering: {value_id}") from exc
    if isinstance(value, BroadcastAlias):
        if value.kind != "gqa_repeat_kv":
            raise TypeError(f"unsupported broadcast alias for attention input {value_id}: {value.kind}")
        return value.tensor
    if not isinstance(value, Tensor):
        raise TypeError(f"expected lowered tensor for {value_id}, got {type(value).__name__}")
    return value


def _normalize_dim(dim: int, rank: int) -> int:
    if dim < 0:
        dim += rank
    return dim


def _normalize_index(index: int, dim_size: int) -> int:
    if index < 0:
        index += dim_size
    return index


def _normalize_slice_end(end: int, dim_size: int) -> int:
    if end < 0:
        end += dim_size
    return max(0, min(end, dim_size))


def _legalize_elementwise_binary_inputs(g: Graph, lhs: Tensor, rhs: Tensor) -> tuple[Tensor, Tensor]:
    if lhs.shape == rhs.shape:
        return lhs, rhs

    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)

    if lhs_rank > rhs_rank:
        rhs = _reshape_for_trailing_broadcast(g, rhs, lhs.shape)
    elif rhs_rank > lhs_rank:
        lhs = _reshape_for_trailing_broadcast(g, lhs, rhs.shape)

    if lhs.shape == rhs.shape:
        return lhs, rhs

    return lhs, rhs


def _is_scalar_like(value: Any) -> bool:
    return isinstance(value, (int, float, bool))


def _lower_binary_op(g: Graph, lhs_value: Any, rhs_value: Any, op: str) -> Tensor:
    if isinstance(lhs_value, Tensor) and isinstance(rhs_value, Tensor):
        lhs, rhs = _legalize_elementwise_binary_inputs(g, lhs_value, rhs_value)
        if op == "add":
            return g.add(lhs, rhs)
        if op == "subtract":
            return g.subtract(lhs, rhs)
        if op == "multiply":
            return g.multiply(lhs, rhs)
        if op == "divide":
            return g.divide(lhs, rhs)
        raise NotImplementedError(f"unsupported binary op: {op}")

    if isinstance(lhs_value, Tensor) and _is_scalar_like(rhs_value):
        scalar = float(rhs_value)
        if op == "add":
            return g.scalar_add(lhs_value, scalar)
        if op == "subtract":
            return g.scalar_subtract(lhs_value, scalar)
        if op == "multiply":
            return g.scalar_multiply(lhs_value, scalar)
        if op == "divide":
            return g.scalar_divide(lhs_value, scalar)
        raise NotImplementedError(f"unsupported binary op: {op}")

    if _is_scalar_like(lhs_value) and isinstance(rhs_value, Tensor):
        scalar = float(lhs_value)
        if op == "add":
            return g.scalar_add(rhs_value, scalar)
        if op == "subtract":
            return g.scalar_add(g.scalar_multiply(rhs_value, -1.0), scalar)
        if op == "multiply":
            return g.scalar_multiply(rhs_value, scalar)
        if op == "divide":
            raise NotImplementedError("scalar/tensor divide is not directly supported by Cactus graph ops")
        raise NotImplementedError(f"unsupported binary op: {op}")

    raise TypeError(
        f"unsupported lowered operand types for {op}: "
        f"{type(lhs_value).__name__}, {type(rhs_value).__name__}"
    )


def _lower_compare_op(g: Graph, lhs_value: Any, rhs_value: Any, op: str) -> Tensor:
    if op != "not_equal":
        raise NotImplementedError(f"unsupported compare op: {op}")

    if isinstance(lhs_value, Tensor) and isinstance(rhs_value, Tensor):
        lhs, rhs = _legalize_elementwise_binary_inputs(g, lhs_value, rhs_value)
        return g.not_equal(lhs, rhs)

    if isinstance(lhs_value, Tensor) and _is_scalar_like(rhs_value):
        return g.scalar_not_equal(lhs_value, float(rhs_value))

    if _is_scalar_like(lhs_value) and isinstance(rhs_value, Tensor):
        return g.scalar_not_equal(rhs_value, float(lhs_value))

    raise TypeError(
        f"unsupported lowered operand types for {op}: "
        f"{type(lhs_value).__name__}, {type(rhs_value).__name__}"
    )


def _reshape_for_trailing_broadcast(g: Graph, tensor: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    tensor_shape = tuple(tensor.shape)
    target_rank = len(target_shape)
    tensor_rank = len(tensor_shape)

    if tensor_rank > target_rank:
        return tensor

    padded_shape = (1,) * (target_rank - tensor_rank) + tensor_shape

    # Only legalize cases that are valid trailing broadcasts, e.g.:
    # (H,) -> (1, H), (1, 1, H), etc.
    for src_dim, tgt_dim in zip(padded_shape, target_shape):
        if src_dim != 1 and src_dim != tgt_dim:
            return tensor

    if padded_shape == tensor_shape:
        return tensor

    return g.reshape(tensor, padded_shape)


def _flatten_to_2d_for_linear(g: Graph, tensor: Tensor) -> Tensor:
    shape = tuple(tensor.shape)
    if len(shape) <= 2:
        return tensor
    leading = 1
    for dim in shape[:-1]:
        leading *= int(dim)
    return g.reshape(tensor, (leading, int(shape[-1])))


def _legalize_matmul_inputs(
    g: Graph,
    lhs: Tensor,
    rhs: Tensor,
    node: IRNode,
) -> tuple[Tensor, Tensor, tuple[int, ...]] | None:
    lhs_shape = tuple(lhs.shape)
    rhs_shape = tuple(rhs.shape)
    if len(lhs_shape) <= 2 and len(rhs_shape) <= 2:
        return None

    # Cactus matmul is 2D-only. Legalize the narrow case where both operands have
    # only singleton leading dims, e.g. rotary helper matmuls:
    # (1, M, K) @ (1, K, N) -> reshape to (M, K) @ (K, N) -> reshape back.
    if any(dim != 1 for dim in lhs_shape[:-2]):
        return None
    if any(dim != 1 for dim in rhs_shape[:-2]):
        return None

    lhs_2d_shape = lhs_shape[-2:]
    rhs_2d_shape = rhs_shape[-2:]
    if lhs_2d_shape[-1] != rhs_2d_shape[0]:
        return None

    output_shape = node.meta.get("shape")
    if not isinstance(output_shape, tuple):
        output_shape = lhs_shape[:-2] + (lhs_2d_shape[0], rhs_2d_shape[1])
    output_shape = tuple(int(v) for v in output_shape)

    lhs_2d = g.reshape(lhs, lhs_2d_shape)
    rhs_2d = g.reshape(rhs, rhs_2d_shape)
    return lhs_2d, rhs_2d, output_shape


def _resolve_expand_shape(input_shape: tuple[int, ...], requested_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(requested_shape) < len(input_shape):
        raise NotImplementedError(f"expand cannot reduce rank: {input_shape} -> {requested_shape}")

    padded_input = (1,) * (len(requested_shape) - len(input_shape)) + input_shape
    resolved: list[int] = []
    for in_dim, req_dim in zip(padded_input, requested_shape):
        if req_dim == -1:
            resolved.append(in_dim)
            continue
        if req_dim < -1:
            raise NotImplementedError(f"invalid expand dimension: {req_dim}")
        resolved.append(int(req_dim))
    return tuple(resolved)


def _resolve_reshape_shape(input_shape: tuple[int, ...], requested_shape: tuple[int, ...]) -> tuple[int, ...]:
    resolved = [int(v) for v in requested_shape]
    unknown_indices = [idx for idx, dim in enumerate(resolved) if dim == -1]
    if len(unknown_indices) > 1:
        raise NotImplementedError(f"reshape with multiple inferred dimensions is unsupported: {requested_shape}")
    if not unknown_indices:
        return tuple(resolved)

    input_elements = 1
    for dim in input_shape:
        input_elements *= int(dim)

    known_elements = 1
    for dim in resolved:
        if dim != -1:
            known_elements *= int(dim)

    if known_elements == 0 or input_elements % known_elements != 0:
        raise NotImplementedError(f"cannot infer reshape target {requested_shape} from input shape {input_shape}")

    resolved[unknown_indices[0]] = input_elements // known_elements
    return tuple(resolved)


def _match_gqa_expand_alias(ir: IRGraph, expand_node: IRNode) -> str | None:
    target_shape = tuple(int(v) for v in expand_node.attrs["shape"])
    if len(target_shape) != 5:
        return None

    current_value_id = expand_node.inputs[0]
    unsqueeze_node: IRNode | None = None

    while True:
        producer_id = ir.values[current_value_id].producer
        if producer_id is None:
            return None
        producer = ir.nodes[producer_id]
        if producer.op == "slice":
            axis = int(producer.attrs.get("axis", -1))
            input_shape = ir.values[producer.inputs[0]].shape
            if input_shape is None:
                return None
            normalized_axis = _normalize_dim(axis, len(input_shape))
            start = int(producer.attrs.get("start", 0))
            end = int(producer.attrs.get("end", input_shape[normalized_axis]))
            if start != 0 or end < input_shape[normalized_axis]:
                return None
            current_value_id = producer.inputs[0]
            continue
        if producer.op == "unsqueeze":
            unsqueeze_node = producer
            break
        return None

    base_value_id = unsqueeze_node.inputs[0]
    base_shape = ir.values[base_value_id].shape
    if base_shape is None or len(base_shape) != 4:
        return None

    unsqueezed_dim = _normalize_dim(int(unsqueeze_node.attrs.get("dim", 0)), len(base_shape) + 1)
    if unsqueezed_dim != 2:
        return None

    expected_target = (
        int(base_shape[0]),
        int(base_shape[1]),
        int(target_shape[2]),
        int(base_shape[2]),
        int(base_shape[3]),
    )
    if target_shape != expected_target:
        return None
    if int(target_shape[2]) <= 1:
        return None

    return base_value_id


def _map_ir_dtype(dtype: str) -> int:
    if dtype == "bf16":
        return Graph.FP32
    if dtype == "fp16":
        return Graph.FP16
    if dtype in ("fp32", "fp64"):
        return Graph.FP32
    if dtype == "int8":
        return Graph.INT8
    if dtype in ("int32", "int64", "bool"):
        return Graph.FP32

    raise NotImplementedError(f"unsupported IR dtype: {dtype}")


def _map_ir_or_torch_dtype(dtype: Any) -> int:
    if dtype is None:
        raise NotImplementedError("missing dtype for precision_cast")
    if isinstance(dtype, str):
        if dtype.startswith("torch."):
            return _map_torch_dtype(getattr(torch, dtype.split(".", 1)[1]))
        return _map_ir_dtype(dtype)
    return _map_torch_dtype(dtype)


def _map_torch_dtype(dtype: Any) -> int:
    if dtype == torch.bfloat16:
        return Graph.FP32
    if dtype == torch.float16:
        return Graph.FP16
    if dtype == torch.float32:
        return Graph.FP32
    if dtype == torch.float64:
        return Graph.FP32
    if dtype == torch.int8:
        return Graph.INT8
    if dtype in (torch.int16, torch.int32, torch.int64, torch.bool):
        return Graph.FP32
    raise NotImplementedError(f"unsupported torch dtype: {dtype}")


def _materialize_constant_tensor(g: Graph, tensor_value: torch.Tensor) -> Tensor:
    graph_dtype = _map_torch_dtype(tensor_value.dtype)
    materialized = tensor_value.detach().cpu()
    if graph_dtype == Graph.FP32 and materialized.dtype not in (torch.float32, torch.float64):
        materialized = materialized.to(torch.float32)
    elif graph_dtype == Graph.FP16 and materialized.dtype != torch.float16:
        materialized = materialized.to(torch.float16)
    elif graph_dtype == Graph.INT8 and materialized.dtype != torch.int8:
        materialized = materialized.to(torch.int8)

    tensor = g.input(shape=tuple(materialized.shape), dtype=graph_dtype)
    g.set_input(tensor, materialized, dtype=graph_dtype)
    return tensor


def _materialize_constant_torch_dtype(dtype: Any) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, str):
        if dtype.startswith("torch."):
            return getattr(torch, dtype.split(".", 1)[1])
        if dtype in ("fp16", "bf16"):
            return torch.float16
        if dtype in ("fp32", "fp64", "int16", "int32", "int64", "bool"):
            return torch.float32
        if dtype == "int8":
            return torch.int8
    if isinstance(dtype, torch.dtype):
        return dtype
    raise NotImplementedError(f"unsupported dtype for constant materialization: {dtype}")
