from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.graph import Graph, Tensor
from src.transpile.capture_pytorch import (
    CapturedModel,
    format_target,
    get_dtype,
    get_shape,
    resolve_attr,
)

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


def transpile_captured(captured: CapturedModel) -> TranspiledGraph:
    g = Graph()
    env: dict[Any, Tensor] = {}
    runtime_inputs: list[Tensor] = []
    bound_constants: list[Tensor] = []
    outputs: list[Tensor] = []

    for node in captured.graph.nodes:
        if node.op == "placeholder":
            tensor = _lower_placeholder(g, node)
            env[node] = tensor
            runtime_inputs.append(tensor)
        elif node.op == "call_function":
            env[node] = _lower_call_function(g, node, env)
        elif node.op == "output":
            outputs = _lower_output(node, env)
        elif node.op == "get_attr":
            tensor = _lower_get_attr(captured, g, node)
            env[node] = tensor
            bound_constants.append(tensor)
        else:
            raise NotImplementedError(f"unsupported FX node op: {node.op}")

    return TranspiledGraph(
        graph=g,
        runtime_inputs=runtime_inputs,
        bound_constants=bound_constants,
        outputs=outputs,
    )


def _lower_placeholder(g: Graph, node: Any) -> Tensor:
    shape = get_shape(node)
    dtype = get_dtype(node)

    if shape is None or dtype is None:
        raise ValueError(f"placeholder node missing shape or dtype: {node}")

    return g.input(shape=shape, dtype=_map_torch_dtype(dtype))


def _lower_call_function(g: Graph, node: Any, env: dict[Any, Tensor]) -> Tensor:
    target = format_target(node)

    if target == "aten.add.Tensor":
        return g.add(_resolve_tensor_arg(node.args[0], env), _resolve_tensor_arg(node.args[1], env))

    if target == "aten.sub.Tensor":
        return g.subtract(_resolve_tensor_arg(node.args[0], env), _resolve_tensor_arg(node.args[1], env))

    if target == "aten.mul.Tensor":
        return g.multiply(_resolve_tensor_arg(node.args[0], env), _resolve_tensor_arg(node.args[1], env))

    if target == "aten.div.Tensor":
        return g.divide(_resolve_tensor_arg(node.args[0], env), _resolve_tensor_arg(node.args[1], env))

    if target in ("aten.view.default", "aten.reshape.default"):
        x = _resolve_tensor_arg(node.args[0], env)
        shape = _resolve_shape_arg(node.args[1])
        return g.reshape(x, shape)

    if target == "aten.flatten.using_ints":
        x = _resolve_tensor_arg(node.args[0], env)
        start_dim = _resolve_int_arg(node.args[1])
        end_dim = _resolve_int_arg(node.args[2])
        return g.flatten(x, start_dim=start_dim, end_dim=end_dim)

    if target in ("aten.t.default", "aten.transpose.int"):
        x = _resolve_tensor_arg(node.args[0], env)
        if target == "aten.t.default":
            return g.transpose(x)
        dim0 = _normalize_dim(_resolve_int_arg(node.args[1]), len(x.shape))
        dim1 = _normalize_dim(_resolve_int_arg(node.args[2]), len(x.shape))
        rank = len(x.shape)
        permutation = list(range(rank))
        permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
        if rank == 2 and permutation == [1, 0]:
            return g.transpose(x)
        return g.permute(x, permutation)

    if target == "aten.permute.default":
        x = _resolve_tensor_arg(node.args[0], env)
        permutation = tuple(_normalize_dim(dim, len(x.shape)) for dim in _resolve_shape_arg(node.args[1]))
        if len(permutation) == 2 and permutation == (1, 0):
            return g.transpose(x)
        return g.permute(x, permutation)

    if target in ("aten.mm.default", "aten.matmul.default"):
        return g.matmul(_resolve_tensor_arg(node.args[0], env), _resolve_tensor_arg(node.args[1], env))

    if target == "aten.linear.default":
        x = _resolve_tensor_arg(node.args[0], env)
        weight = _resolve_tensor_arg(node.args[1], env)
        bias = None if len(node.args) < 3 or node.args[2] is None else _resolve_tensor_arg(node.args[2], env)
        out = g.matmul(x, weight, pretransposed_rhs=True)
        if bias is not None:
            out = g.add(out, bias)
        return out

    if target == "aten.addmm.default":
        bias = _resolve_tensor_arg(node.args[0], env)
        lhs = _resolve_tensor_arg(node.args[1], env)
        rhs = _resolve_tensor_arg(node.args[2], env)
        out = g.matmul(lhs, rhs)
        return g.add(bias, out)

    if target == "aten.relu.default":
        return g.relu(_resolve_tensor_arg(node.args[0], env))

    if target in ("aten.gelu.default", "aten.gelu.erf"):
        return g.gelu(_resolve_tensor_arg(node.args[0], env))

    if target == "aten.sigmoid.default":
        return g.sigmoid(_resolve_tensor_arg(node.args[0], env))

    if target == "aten.tanh.default":
        return g.tanh(_resolve_tensor_arg(node.args[0], env))

    if target == "aten.softmax.int":
        x = _resolve_tensor_arg(node.args[0], env)
        axis = int(node.args[1])
        return g.softmax(x, axis=axis)

    raise NotImplementedError(f"unsupported call_function target: {target}")


def _lower_get_attr(captured: CapturedModel, g: Graph, node: Any) -> Tensor:
    try:
        value = resolve_attr(captured.graph_module, node.target)
    except AttributeError:
        if node.target in captured.state_dict:
            value = captured.state_dict[node.target]
        else:
            raise NotImplementedError(f"could not resolve get_attr target: {node.target}") from None

    if isinstance(value, torch.nn.Parameter):
        value = value.detach()

    if not isinstance(value, torch.Tensor):
        raise NotImplementedError(
            f"unsupported get_attr type for {node.target}: {type(value).__name__}"
        )

    tensor_value = value.detach().cpu()
    cactus_dtype = _map_torch_dtype(tensor_value.dtype)
    cactus_tensor = g.input(shape=tuple(tensor_value.shape), dtype=cactus_dtype)
    g.set_input(cactus_tensor, tensor_value, dtype=cactus_dtype)
    return cactus_tensor


def _lower_output(node: Any, env: dict[Any, Tensor]) -> list[Tensor]:
    returned = node.args[0]

    if isinstance(returned, (tuple, list)):
        return [_resolve_tensor_arg(item, env) for item in returned]

    return [_resolve_tensor_arg(returned, env)]


def _resolve_tensor_arg(arg: Any, env: dict[Any, Tensor]) -> Tensor:
    if arg in env:
        return env[arg]
    raise NotImplementedError(f"unsupported tensor argument: {arg!r}")


def _resolve_shape_arg(arg: Any) -> tuple[int, ...]:
    if isinstance(arg, (tuple, list)):
        return tuple(int(v) for v in arg)
    raise NotImplementedError(f"unsupported shape argument: {arg!r}")


def _resolve_int_arg(arg: Any) -> int:
    if isinstance(arg, bool):
        return int(arg)
    if isinstance(arg, int):
        return arg
    raise NotImplementedError(f"unsupported int argument: {arg!r}")


def _normalize_dim(dim: int, rank: int) -> int:
    if dim < 0:
        dim += rank
    return dim


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
    raise NotImplementedError(f"unsupported torch dtype: {dtype}")
