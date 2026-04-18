import ctypes
from os import wait
import numpy as np

from .cactus import _lib, cactus_node_t, cactus_tensor_info_t

class Graph:
    INT8 = 0
    FP16 = 1
    FP32 = 2
    INT4 = 3
    CPU = 0
    NPU = 1

    def __init__(self):
        self.h = _lib.cactus_graph_create()
        if not self.h:
            raise RuntimeError("cactus_graph_create failed")
    
    def save(self, filename):
        rc = _lib.cactus_graph_save(self.h, str(filename).encode())
        if rc != 0:
            raise RuntimeError("graph_save failed")

    @classmethod
    def load(cls, filename):
        h = _lib.cactus_graph_load(str(filename).encode())
        if not h:
            raise RuntimeError("cactus_graph_load failed")
        obj = cls.__new__(cls)
        obj.h = h
        return obj

    def __del__(self):
        h = getattr(self, "h", None)
        if h:
            _lib.cactus_graph_destroy(h)
            self.h = None

    def input(self, shape, dtype=FP16):
        shape = tuple(int(x) for x in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_input(self.h, arr, len(shape), int(dtype), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_input failed")
        return self._tensor_from_node(out.value)

    def set_input(self, tensor, data, dtype=None):
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a Tensor")
        if tensor.g is not self:
            raise ValueError("tensor belongs to a different graph")
        target_dtype = int(tensor.dtype if dtype is None else dtype)
        arr = self._coerce_input_array(data, target_dtype)
        rc = _lib.cactus_graph_set_input(
            self.h,
            cactus_node_t(tensor.id),
            arr.ctypes.data_as(ctypes.c_void_p),
            target_dtype,
        )
        if rc != 0:
            raise RuntimeError("graph_set_input failed")

    def set_external_input(self, tensor, data_ptr, dtype=None):
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a Tensor")
        if tensor.g is not self:
            raise ValueError("tensor belongs to a different graph")
        target_dtype = int(tensor.dtype if dtype is None else dtype)
        ptr = ctypes.c_void_p(data_ptr if isinstance(data_ptr, int) else int(data_ptr))
        rc = _lib.cactus_graph_set_external_input(
            self.h,
            cactus_node_t(tensor.id),
            ptr,
            target_dtype,
        )
        if rc != 0:
            raise RuntimeError("graph_set_external_input failed")

    def hard_reset(self):
        rc = _lib.cactus_graph_hard_reset(self.h)
        if rc != 0:
            raise RuntimeError("graph_hard_reset failed")

    def execute(self):
        rc = _lib.cactus_graph_execute(self.h)
        if rc != 0:
            raise RuntimeError("graph_execute failed")

    def add(self, a, b):
        return self._binary("cactus_graph_add", a, b)

    def add_clipped(self, a, b):
        return self._binary("cactus_graph_add_clipped", a, b)

    def subtract(self, a, b):
        return self._binary("cactus_graph_subtract", a, b)

    def multiply(self, a, b):
        return self._binary("cactus_graph_multiply", a, b)

    def divide(self, a, b):
        return self._binary("cactus_graph_divide", a, b)

    def abs(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_abs(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_abs failed")
        return self._tensor_from_node(out.value)

    def pow(self, x, exponent):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_pow(self.h, cactus_node_t(x.id), ctypes.c_float(float(exponent)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_pow failed")
        return self._tensor_from_node(out.value)

    def precision_cast(self, x, dtype):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_precision_cast(self.h, cactus_node_t(x.id), int(dtype), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_precision_cast failed")
        return self._tensor_from_node(out.value)

    def quantize_activations(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_quantize_activations(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_quantize_activations failed")
        return self._tensor_from_node(out.value)

    def _scalar(self, fn_name, x, value=None):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        fn = getattr(_lib, fn_name)
        if value is None:
            rc = fn(self.h, cactus_node_t(x.id), ctypes.byref(out))
        else:
            rc = fn(self.h, cactus_node_t(x.id), ctypes.c_float(float(value)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"{fn_name} failed")
        return self._tensor_from_node(out.value)

    def scalar_add(self, x, value):
        return self._scalar("cactus_graph_scalar_add", x, value)

    def scalar_subtract(self, x, value):
        return self._scalar("cactus_graph_scalar_subtract", x, value)

    def scalar_multiply(self, x, value):
        return self._scalar("cactus_graph_scalar_multiply", x, value)

    def scalar_divide(self, x, value):
        return self._scalar("cactus_graph_scalar_divide", x, value)

    def scalar_exp(self, x):
        return self._scalar("cactus_graph_scalar_exp", x)

    def scalar_sqrt(self, x):
        return self._scalar("cactus_graph_scalar_sqrt", x)

    def scalar_cos(self, x):
        return self._scalar("cactus_graph_scalar_cos", x)

    def scalar_sin(self, x):
        return self._scalar("cactus_graph_scalar_sin", x)

    def scalar_log(self, x):
        return self._scalar("cactus_graph_scalar_log", x)

    def view(self, x, shape):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_view(self.h, cactus_node_t(x.id), arr, len(shape), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_view failed")
        return self._tensor_from_node(out.value)

    def reshape(self, x, shape):
        x = self._ensure_tensor(x)
        shape = tuple(int(v) for v in shape)
        arr = (ctypes.c_size_t * len(shape))(*shape)
        out = cactus_node_t()
        rc = _lib.cactus_graph_reshape(self.h, cactus_node_t(x.id), arr, len(shape), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_reshape failed")
        return self._tensor_from_node(out.value)

    def flatten(self, x, start_dim=0, end_dim=-1):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_flatten(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_int32(int(start_dim)),
            ctypes.c_int32(int(end_dim)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_flatten failed")
        return self._tensor_from_node(out.value)

    def slice(self, x, axis, start, length=0):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_slice(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_int32(int(axis)),
            ctypes.c_size_t(int(start)),
            ctypes.c_size_t(int(length)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_slice failed")
        return self._tensor_from_node(out.value)

    def index(self, x, index_value, axis=0):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_index(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_size_t(int(index_value)),
            ctypes.c_int32(int(axis)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_index failed")
        return self._tensor_from_node(out.value)

    def transpose(self, x, backend=CPU):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_transpose(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_int32(int(backend)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_transpose failed")
        return self._tensor_from_node(out.value)

    def permute(self, x, permutation, backend=CPU):
        x = self._ensure_tensor(x)
        permutation = tuple(int(v) for v in permutation)
        arr = (ctypes.c_size_t * len(permutation))(*permutation)
        out = cactus_node_t()
        rc = _lib.cactus_graph_transpose_n(
            self.h,
            cactus_node_t(x.id),
            arr,
            len(permutation),
            ctypes.c_int32(int(backend)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_transpose_n failed")
        return self._tensor_from_node(out.value)

    def matmul(self, a, b, pretransposed_rhs=False, backend=CPU):
        a = self._ensure_tensor(a)
        b = self._ensure_tensor(b)
        out = cactus_node_t()
        rc = _lib.cactus_graph_matmul(
            self.h,
            cactus_node_t(a.id),
            cactus_node_t(b.id),
            ctypes.c_bool(bool(pretransposed_rhs)),
            ctypes.c_int32(int(backend)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_matmul failed")
        return self._tensor_from_node(out.value)

    def concat(self, a, b, axis=0):
        a = self._ensure_tensor(a)
        b = self._ensure_tensor(b)
        out = cactus_node_t()
        rc = _lib.cactus_graph_concat(
            self.h,
            cactus_node_t(a.id),
            cactus_node_t(b.id),
            ctypes.c_int32(int(axis)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_concat failed")
        return self._tensor_from_node(out.value)

    def cat(self, tensors, axis=0):
        tensors = [self._ensure_tensor(t) for t in tensors]
        if not tensors:
            raise ValueError("cat requires at least one tensor")
        ids = (cactus_node_t * len(tensors))(*(cactus_node_t(t.id) for t in tensors))
        out = cactus_node_t()
        rc = _lib.cactus_graph_cat(
            self.h,
            ids,
            ctypes.c_size_t(len(tensors)),
            ctypes.c_int32(int(axis)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_cat failed")
        return self._tensor_from_node(out.value)

    def group_norm(self, x, normalized_shape, eps=1e-5):
        raise RuntimeError("graph_group_norm signature changed; use group_norm(x, weight, bias, num_groups, eps)")

    def group_norm(self, x, weight, bias, num_groups, eps=1e-5):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        bias = self._ensure_tensor(bias)
        out = cactus_node_t()
        rc = _lib.cactus_graph_group_norm(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(weight.id),
            cactus_node_t(bias.id),
            ctypes.c_size_t(int(num_groups)),
            ctypes.c_float(float(eps)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_group_norm failed")
        return self._tensor_from_node(out.value)

    def layer_norm(self, x, weight, bias=None, eps=1e-5):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        has_bias = bias is not None
        bias_node = cactus_node_t(0)
        if has_bias:
            bias = self._ensure_tensor(bias)
            bias_node = cactus_node_t(bias.id)
        out = cactus_node_t()
        rc = _lib.cactus_graph_layer_norm(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(weight.id),
            bias_node,
            ctypes.c_float(float(eps)),
            ctypes.c_bool(has_bias),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_layer_norm failed")
        return self._tensor_from_node(out.value)

    def batch_norm(self, x, weight, bias, running_mean, running_var, axis=1, eps=1e-5):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        bias = self._ensure_tensor(bias)
        running_mean = self._ensure_tensor(running_mean)
        running_var = self._ensure_tensor(running_var)
        out = cactus_node_t()
        rc = _lib.cactus_graph_batchnorm(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(weight.id),
            cactus_node_t(bias.id),
            cactus_node_t(running_mean.id),
            cactus_node_t(running_var.id),
            ctypes.c_int32(int(axis)),
            ctypes.c_float(float(eps)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_batchnorm failed")
        return self._tensor_from_node(out.value)

    def rms_norm(self, x, weight, eps=1e-5):
        x = self._ensure_tensor(x)
        weight = self._ensure_tensor(weight)
        out = cactus_node_t()
        rc = _lib.cactus_graph_rms_norm(
            self.h,
            cactus_node_t(x.id),
            cactus_node_t(weight.id),
            ctypes.c_float(float(eps)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_rms_norm failed")
        return self._tensor_from_node(out.value)

    def _reduce(self, fn_name, x, axis):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = getattr(_lib, fn_name)(self.h, cactus_node_t(x.id), ctypes.c_int32(int(axis)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"{fn_name} failed")
        return self._tensor_from_node(out.value)

    def sum(self, x, axis):
        return self._reduce("cactus_graph_sum", x, axis)

    def mean(self, x, axis):
        return self._reduce("cactus_graph_mean", x, axis)

    def variance(self, x, axis):
        return self._reduce("cactus_graph_variance", x, axis)

    def min(self, x, axis):
        return self._reduce("cactus_graph_min", x, axis)

    def max(self, x, axis):
        return self._reduce("cactus_graph_max", x, axis)
    
    def softmax(self, x, axis=-1):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_softmax(
            self.h,
            cactus_node_t(x.id),
            ctypes.c_int32(int(axis)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError("graph_softmax failed")
        return self._tensor_from_node(out.value)
    
    def relu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_relu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_relu failed")
        return self._tensor_from_node(out.value)

    def silu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_silu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_silu failed")
        return self._tensor_from_node(out.value)

    def gelu(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gelu(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_gelu failed")
        return self._tensor_from_node(out.value)

    def gelu_erf(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_gelu_erf(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_gelu_erf failed")
        return self._tensor_from_node(out.value)

    def sigmoid(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_sigmoid(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_sigmoid failed")
        return self._tensor_from_node(out.value)

    def tanh(self, x):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_tanh(self.h, cactus_node_t(x.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_tanh failed")
        return self._tensor_from_node(out.value)

    def glu(self, x, axis=-1):
        x = self._ensure_tensor(x)
        out = cactus_node_t()
        rc = _lib.cactus_graph_glu(self.h, cactus_node_t(x.id), ctypes.c_int32(int(axis)), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError("graph_glu failed")
        return self._tensor_from_node(out.value)

    def output_info(self, x):
        x = self._ensure_tensor(x)
        return self._get_output_info(x.id)

    def _binary(self, fn_name, a, b):
        a = self._ensure_tensor(a)
        b = self._ensure_tensor(b)
        out = cactus_node_t()
        rc = getattr(_lib, fn_name)(self.h, cactus_node_t(a.id), cactus_node_t(b.id), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f"{fn_name} failed")
        return self._tensor_from_node(out.value)

    def _ensure_tensor(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("expected Tensor")
        if x.g is not self:
            raise ValueError("tensor belongs to a different graph")
        return x

    def _get_output_info(self, node_id):
        info = cactus_tensor_info_t()
        rc = _lib.cactus_graph_get_output_info(self.h, cactus_node_t(node_id), ctypes.byref(info))
        if rc != 0:
            raise RuntimeError("graph_get_output_info failed")
        shape = tuple(int(info.shape[i]) for i in range(int(info.rank)))
        return {
            "precision": int(info.precision),
            "rank": int(info.rank),
            "shape": shape,
            "num_elements": int(info.num_elements),
            "byte_size": int(info.byte_size),
        }

    def _tensor_from_node(self, node_id):
        meta = self._get_output_info(node_id)
        return Tensor(self, int(node_id), meta["shape"], meta["precision"])

    def _coerce_input_array(self, data, precision):
        if isinstance(data, Tensor):
            arr = data.numpy()
        else:
            arr = np.asarray(data)
        if precision == self.INT8:
            arr = np.ascontiguousarray(arr, dtype=np.int8)
        elif precision == self.FP16:
            arr = np.ascontiguousarray(arr, dtype=np.float16)
        elif precision == self.FP32:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        elif precision == self.INT4:
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
        else:
            raise RuntimeError("unsupported precision")
        return arr


class Tensor:
    def __init__(self, g, node_id, shape, dtype):
        self.g = g
        self.id = int(node_id)
        self.shape = tuple(shape)
        self.dtype = int(dtype)

    def __add__(self, other):
        return self.g.add(self, other)

    def __sub__(self, other):
        return self.g.subtract(self, other)

    def __mul__(self, other):
        return self.g.multiply(self, other)

    def __truediv__(self, other):
        return self.g.divide(self, other)

    def abs(self):
        return self.g.abs(self)

    def pow(self, exponent):
        return self.g.pow(self, exponent)

    def precision_cast(self, dtype):
        return self.g.precision_cast(self, dtype)

    def quantize_activations(self):
        return self.g.quantize_activations(self)

    def scalar_add(self, value):
        return self.g.scalar_add(self, value)

    def scalar_subtract(self, value):
        return self.g.scalar_subtract(self, value)

    def scalar_multiply(self, value):
        return self.g.scalar_multiply(self, value)

    def scalar_divide(self, value):
        return self.g.scalar_divide(self, value)

    def scalar_exp(self):
        return self.g.scalar_exp(self)

    def scalar_sqrt(self):
        return self.g.scalar_sqrt(self)

    def scalar_cos(self):
        return self.g.scalar_cos(self)

    def scalar_sin(self):
        return self.g.scalar_sin(self)

    def scalar_log(self):
        return self.g.scalar_log(self)

    def relu(self):
        return self.g.relu(self)

    def sigmoid(self):
        return self.g.sigmoid(self)

    def tanh(self):
        return self.g.tanh(self)

    def gelu(self):
        return self.g.gelu(self)

    def gelu_erf(self):
        return self.g.gelu_erf(self)

    def silu(self):
        return self.g.silu(self)

    def view(self, shape):
        return self.g.view(self, shape)

    def reshape(self, shape):
        return self.g.reshape(self, shape)

    def flatten(self, start_dim=0, end_dim=-1):
        return self.g.flatten(self, start_dim=start_dim, end_dim=end_dim)

    def slice(self, axis, start, length=0):
        return self.g.slice(self, axis, start, length=length)

    def index(self, index_value, axis=0):
        return self.g.index(self, index_value, axis=axis)

    def transpose(self, backend=Graph.CPU):
        return self.g.transpose(self, backend=backend)

    def permute(self, permutation, backend=Graph.CPU):
        return self.g.permute(self, permutation, backend=backend)

    def concat(self, other, axis=0):
        return self.g.concat(self, other, axis=axis)

    def cat(self, tensors, axis=0):
        return self.g.cat([self] + tensors, axis=axis)

    def group_norm(self, normalized_shape, eps=1e-5):
        raise RuntimeError("Tensor.group_norm signature changed; use graph.group_norm(x, weight, bias, num_groups, eps)")

    def layer_norm(self, weight, bias=None, eps=1e-5):
        return self.g.layer_norm(self, weight, bias=bias, eps=eps)

    def batch_norm(self, weight, bias, running_mean, running_var, axis=1, eps=1e-5):
        return self.g.batch_norm(self, weight, bias, running_mean, running_var, axis=axis, eps=eps)

    def rms_norm(self, weight, eps=1e-5):
        return self.g.rms_norm(self, weight, eps=eps)
    
    def softmax(self, axis=-1):
        return self.g.softmax(self, axis)

    def glu(self, axis=-1):
        return self.g.glu(self, axis=axis)

    def matmul(self, other, pretransposed_rhs=False, backend=Graph.CPU):
        return self.g.matmul(self, other, pretransposed_rhs=pretransposed_rhs, backend=backend)

    def sum(self, axis):
        return self.g.sum(self, axis)

    def mean(self, axis):
        return self.g.mean(self, axis)

    def variance(self, axis):
        return self.g.variance(self, axis)

    def min(self, axis):
        return self.g.min(self, axis)

    def max(self, axis):
        return self.g.max(self, axis)

    def numpy(self):
        info = cactus_tensor_info_t()
        rc = _lib.cactus_graph_get_output_info(self.g.h, cactus_node_t(self.id), ctypes.byref(info))
        if rc != 0:
            raise RuntimeError("graph_get_output_info failed")

        out_ptr = ctypes.c_void_p()
        rc = _lib.cactus_graph_get_output_ptr(self.g.h, cactus_node_t(self.id), ctypes.byref(out_ptr))
        if rc != 0 or not out_ptr.value:
            raise RuntimeError("graph_get_output_ptr failed")

        rank = int(info.rank)
        shape = tuple(int(info.shape[i]) for i in range(rank))
        num_elements = int(info.num_elements)
        precision = int(info.precision)

        if precision == Graph.FP16:
            arr = np.ctypeslib.as_array((ctypes.c_uint16 * num_elements).from_address(out_ptr.value)).view(np.float16)
        elif precision == Graph.FP32:
            arr = np.ctypeslib.as_array((ctypes.c_float * num_elements).from_address(out_ptr.value))
        elif precision == Graph.INT8:
            arr = np.ctypeslib.as_array((ctypes.c_int8 * num_elements).from_address(out_ptr.value))
        elif precision == Graph.INT4:
            arr = np.ctypeslib.as_array((ctypes.c_uint8 * int(info.byte_size)).from_address(out_ptr.value))
            return arr.copy()
        else:
            raise RuntimeError("unsupported precision")

        return arr.reshape(shape).copy()

    def __repr__(self):
        return f"Tensor(id={self.id}, shape={self.shape}, dtype={self.dtype})"
