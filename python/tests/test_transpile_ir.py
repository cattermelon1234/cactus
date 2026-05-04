import unittest
import tempfile
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from src.graph import Graph
from src.transpile.capture_pytorch import capture_model
from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.graph_ir import IRValue
from src.transpile.graph_ir import verify_ir
from src.transpile.import_semantics import apply_import_semantics
from src.transpile.lower import transpile_captured
from src.transpile.lower import _match_gqa_expand_alias
from src.transpile.optimize_graph import optimize_graph


class Toy(nn.Module):
    def forward(self, x, y):
        z = x + y
        z = z * 0.5
        z = torch.nn.functional.gelu(z)
        z = torch.softmax(z, dim=-1)
        return z


class UnsupportedToy(nn.Module):
    def forward(self, x):
        return torch.square(x)


class NotEqualToy(nn.Module):
    def forward(self, x, y):
        return x != y


class ScalarNotEqualToy(nn.Module):
    def forward(self, x):
        return x != 0


class ScaleLessRmsNormToy(nn.Module):
    def forward(self, x):
        y = x.float()
        rms = torch.pow(torch.mean(torch.pow(y, 2), dim=-1, keepdim=True) + 1e-6, -0.5)
        return (y * rms).type_as(x)


class AttentionInner(nn.Module):
    def forward(self, x):
        return torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True, scale=1.0)


class AttentionHintToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = AttentionInner()

    def get_transpile_metadata(self):
        return {
            "graph": {
                "adapter_family": "test_attention_hint",
            },
            "import_hints": [
                {
                    "module_path_suffix": "inner",
                    "op": "scaled_dot_product_attention",
                    "attrs": {"window_size": 7},
                    "meta": {"attention_layer_type": "sliding_attention"},
                }
            ],
        }

    def forward(self, x):
        return self.inner(x)


class AttentionBlockToy(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float16)

    def forward(self, q, k, v, gate):
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        attn = attn.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        attn = attn * torch.sigmoid(gate)
        attn = attn.to(self.out_proj.weight.dtype)
        return self.out_proj(attn)


class ConstantExpandToy(nn.Module):
    def forward(self, x):
        pos = torch.arange(5, dtype=torch.float32).view(1, 1, -1).expand(4, 1, -1)
        return pos.cos() + x.reshape(1, 1, 1) * 0


class RuntimeExpandToy(nn.Module):
    def forward(self, x):
        return x.view(3, 1).expand(3, 4)


class DepthwiseCausalConv1dToy(nn.Module):
    def __init__(self, channels: int = 4, kernel_size: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, dtype=torch.float16))

    def forward(self, x):
        y = torch.nn.functional.conv1d(x, self.weight, bias=None, stride=1, padding=self.weight.shape[-1] - 1, dilation=1, groups=x.shape[1])
        return y[:, :, : x.shape[-1]]


class SplitWithSizesToy(nn.Module):
    def forward(self, x):
        a, b, c = torch.split(x, [2, 2, 4], dim=-1)
        return b


class ChunkToy(nn.Module):
    def forward(self, x):
        a, b, c = torch.chunk(x, 3, dim=-1)
        return c


class SoftplusToy(nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)


class ConstantPadToy(nn.Module):
    def forward(self, x):
        return torch.nn.functional.pad(x, (1, 2), mode="constant", value=0.5)


class OnesToy(nn.Module):
    def forward(self, x):
        return torch.ones((2, 3), dtype=torch.float16)


class WrapSetGradEnabledToy(nn.Module):
    def forward(self, x):
        with torch.set_grad_enabled(False):
            y = torch.cos(x)
            z = y * 2
        return z + 1


class _FakeGemmaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False, dtype=torch.float16)

    def forward(self, x):
        return self.q_proj(x)


class _FakeGemmaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeGemmaAttention()

    def forward(self, x):
        return self.self_attn(x)


class FakeGemmaWeightBindingToy(nn.Module):
    def __init__(self, weights_dir: str):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([_FakeGemmaLayer()])
        self._weights_dir = weights_dir

    def get_transpile_metadata(self):
        return {"graph": {"adapter_family": "gemma", "weights_dir": self._weights_dir}}

    def forward(self, x):
        return self.backbone.layers[0](x)




class TestTranspileIR(unittest.TestCase):
    def test_toy_graph_transpiles_and_matches(self):
        model = Toy().eval()
        x = torch.tensor(
            [[-1.0, -0.5, 0.0, 0.5], [1.0, -0.25, 0.25, 0.75]],
            dtype=torch.float16,
        )
        y = torch.randn(2, 4, dtype=torch.float16)

        captured = capture_model(model, (x, y))
        canonicalize_exported_graph(captured.ir_graph)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy(), y.numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = model(x, y).float().numpy()
        got = outputs[0].numpy().astype(np.float32)
        np.testing.assert_allclose(ref, got, atol=1e-2)

    def test_unknown_import_op_is_preserved_until_lowering(self):
        model = UnsupportedToy().eval()
        x = torch.randn(2, 4, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        self.assertIn("aten.square.default", {captured.ir_graph.nodes[node_id].op for node_id in captured.ir_graph.order})

        with self.assertRaises(NotImplementedError):
            transpile_captured(captured)

    def test_not_equal_transpiles_and_matches(self):
        model = NotEqualToy().eval()
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        y = torch.tensor([[1, 0, 3], [0, 5, 7]], dtype=torch.float16)

        captured = capture_model(model, (x, y))
        canonicalize_exported_graph(captured.ir_graph)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy(), y.numpy()])
        outputs = tg.execute()

        ref = model(x, y).to(torch.float16).numpy()
        got = outputs[0].numpy()
        np.testing.assert_allclose(ref, got, atol=1e-2)

    def test_scalar_not_equal_transpiles_and_matches(self):
        model = ScalarNotEqualToy().eval()
        x = torch.tensor([[0, 2, 0], [4, 0, 6]], dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).to(torch.float16).numpy()
        got = outputs[0].numpy()
        np.testing.assert_allclose(ref, got, atol=1e-2)

    def test_scale_less_rms_norm_is_fused_and_matches(self):
        model = ScaleLessRmsNormToy().eval()
        x = torch.randn(2, 3, 8, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        optimize_graph(captured.ir_graph)

        rms_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "rms_norm"]
        self.assertEqual(len(rms_nodes), 1)
        self.assertIn(rms_nodes[0].inputs[1], captured.ir_graph.constants)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).numpy()
        got = outputs[0].numpy()
        np.testing.assert_allclose(ref, got, atol=1e-2)

    def test_verify_ir_catches_missing_inputs(self):
        graph = IRGraph(
            values={"v_out": IRValue(id="v_out", producer="n0")},
            nodes={"n0": IRNode(id="n0", op="add", inputs=["v_missing"], outputs=["v_out"])},
            order=["n0"],
            inputs=[],
            outputs=["v_out"],
            constants={},
        )
        with self.assertRaises(ValueError):
            verify_ir(graph)

    def test_capture_metadata_preserves_attention_window_size(self):
        model = AttentionHintToy().eval()
        x = torch.randn(1, 2, 4, 8, dtype=torch.float16)

        captured = capture_model(model, (x,))
        self.assertEqual(captured.ir_graph.meta.get("adapter_family"), "test_attention_hint")

        attention_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "attention"]
        self.assertEqual(len(attention_nodes), 1)
        self.assertEqual(attention_nodes[0].attrs.get("window_size"), 7)
        self.assertEqual(attention_nodes[0].meta.get("attention_layer_type"), "sliding_attention")
        self.assertEqual(attention_nodes[0].meta.get("semantic_source"), "import")

        optimize_graph(captured.ir_graph)
        attention_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "attention"]
        self.assertEqual(len(attention_nodes), 1)
        self.assertEqual(attention_nodes[0].attrs.get("window_size"), 7)
        self.assertEqual(attention_nodes[0].meta.get("window_size_source"), "import_attr")

    def test_attention_block_is_fused_and_matches(self):
        model = AttentionBlockToy().eval()
        q = torch.randn(1, 2, 3, 4, dtype=torch.float16)
        k = torch.randn(1, 2, 3, 4, dtype=torch.float16)
        v = torch.randn(1, 2, 3, 4, dtype=torch.float16)
        gate = torch.randn(1, 3, 8, dtype=torch.float16)

        captured = capture_model(model, (q, k, v, gate))
        optimize_graph(captured.ir_graph)

        attention_block_nodes = [
            captured.ir_graph.nodes[node_id]
            for node_id in captured.ir_graph.order
            if captured.ir_graph.nodes[node_id].op == "attention_block"
        ]
        self.assertEqual(len(attention_block_nodes), 1)

        tg = transpile_captured(captured)
        tg.set_inputs([q.numpy(), k.numpy(), v.numpy(), gate.numpy()])
        outputs = tg.execute()

        ref = model(q, k, v, gate).detach().float().numpy()
        got = outputs[0].numpy().astype(np.float32)
        np.testing.assert_allclose(ref, got, atol=5e-2)

    def test_constant_expand_subgraph_is_folded(self):
        model = ConstantExpandToy().eval()
        x = torch.randn(1, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        self.assertNotIn("expand", {captured.ir_graph.nodes[node_id].op for node_id in captured.ir_graph.order})

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).numpy()
        got = outputs[0].numpy()
        np.testing.assert_allclose(ref, got, atol=5e-4)

    def test_runtime_expand_transpiles_and_matches(self):
        model = RuntimeExpandToy().eval()
        x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).numpy()
        got = outputs[0].numpy()
        np.testing.assert_allclose(ref, got, atol=1e-2)

    def test_depthwise_causal_conv1d_pattern_transpiles_and_matches(self):
        model = DepthwiseCausalConv1dToy().eval()
        x = torch.randn(1, 4, 5, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        conv_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "conv1d"]
        self.assertEqual(len(conv_nodes), 1)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).detach().float().numpy()
        got = outputs[0].numpy().astype(np.float32)
        np.testing.assert_allclose(ref, got, atol=5e-2)

    def test_split_with_sizes_lowering_matches(self):
        model = SplitWithSizesToy().eval()
        x = torch.randn(1, 3, 8, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        split_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "split_with_sizes"]
        self.assertEqual(len(split_nodes), 1)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).detach().float().numpy()
        got = outputs[0].numpy().astype(np.float32)
        np.testing.assert_allclose(ref, got, atol=1e-2)

    def test_chunk_lowering_matches(self):
        model = ChunkToy().eval()
        x = torch.randn(1, 3, 8, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        chunk_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "chunk"]
        self.assertEqual(len(chunk_nodes), 1)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).detach().float().numpy()
        got = outputs[0].numpy().astype(np.float32)
        np.testing.assert_allclose(ref, got, atol=1e-2)

    def test_parameter_constant_gets_weight_binding_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "layer_0_attn_q.weights").touch()
            Path(tmpdir, "weights_manifest.json").write_text(
                json.dumps(
                    {
                        "backbone.layers.0.self_attn.q_proj.weight": {
                            "filename": "layer_0_attn_q.weights",
                            "kind": "weight",
                        }
                    }
                )
            )
            model = FakeGemmaWeightBindingToy(tmpdir).eval()
            x = torch.randn(1, 4, dtype=torch.float16)

            captured = capture_model(model, (x,))

            bindings = captured.ir_graph.meta.get("weight_bindings", {})
            self.assertTrue(bindings)
            self.assertTrue(
                any(
                    isinstance(binding, dict)
                    and binding.get("path", "").endswith("layer_0_attn_q.weights")
                    and binding.get("kind") == "weight"
                    for binding in bindings.values()
                )
            )

    def test_softplus_lowering_matches(self):
        model = SoftplusToy().eval()
        x = torch.randn(2, 4, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        softplus_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "softplus"]
        self.assertEqual(len(softplus_nodes), 1)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).detach().float().numpy()
        got = outputs[0].numpy().astype(np.float32)
        np.testing.assert_allclose(ref, got, atol=1.2e-1)

    def test_constant_pad_lowering_matches(self):
        model = ConstantPadToy().eval()
        x = torch.randn(2, 4, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        pad_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "pad"]
        self.assertEqual(len(pad_nodes), 1)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).detach().float().numpy()
        got = outputs[0].numpy().astype(np.float32)
        np.testing.assert_allclose(ref, got, atol=1e-3)

    def test_ones_lowering_matches(self):
        model = OnesToy().eval()
        x = torch.randn(2, 4, dtype=torch.float16)

        captured = capture_model(model, (x,))
        canonicalize_exported_graph(captured.ir_graph)

        ones_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "ones"]
        self.assertEqual(len(ones_nodes), 1)

        tg = transpile_captured(captured)
        tg.set_inputs([x.numpy()])
        outputs = tg.execute()

        ref = model(x).detach().float().numpy()
        got = outputs[0].numpy().astype(np.float32)
        np.testing.assert_allclose(ref, got, atol=1e-3)

    def test_wrap_with_set_grad_enabled_is_inlined_without_leaking_wrapper_name(self):
        model = WrapSetGradEnabledToy().eval()
        x = torch.randn(2, 3, dtype=torch.float32)

        captured = capture_model(model, (x,))

        self.assertFalse(any("wrap_with_set_grad_enabled" in node_id for node_id in captured.ir_graph.order))
        self.assertEqual([captured.ir_graph.nodes[node_id].op for node_id in captured.ir_graph.order], ["cos", "scalar_multiply", "scalar_add"])

    def test_final_canonicalization_rewrites_to_registry_ops(self):
        graph = IRGraph(
            values={
                "x": IRValue(id="x", shape=(2, 3), dtype="fp16"),
                "y": IRValue(id="y", shape=(3, 2), dtype="fp32"),
                "v0": IRValue(id="v0", shape=(3, 2), dtype="fp16", producer="n0"),
                "v1": IRValue(id="v1", shape=(3, 2), dtype="fp16", producer="n1"),
                "v2": IRValue(id="v2", shape=(3, 2), dtype="fp32", producer="n2"),
            },
            nodes={
                "n0": IRNode(id="n0", op="transpose", inputs=["x"], outputs=["v0"], attrs={"dim0": 0, "dim1": 1}),
                "n1": IRNode(id="n1", op="reshape", inputs=["v0"], outputs=["v1"], attrs={"shape": (3, 2)}),
                "n2": IRNode(id="n2", op="type_as", inputs=["v1", "y"], outputs=["v2"]),
            },
            order=["n0", "n1", "n2"],
            inputs=["x", "y"],
            outputs=["v2"],
            constants={},
        )
        graph.values["x"].users = ["n0"]
        graph.values["y"].users = ["n2"]
        graph.values["v0"].users = ["n1"]
        graph.values["v1"].users = ["n2"]
        graph.values["v2"].users = []

        canonicalize_exported_graph(graph)

        self.assertEqual(graph.nodes["n0"].op, "permute")
        self.assertEqual(graph.nodes["n0"].attrs["permutation"], (1, 0))
        self.assertEqual(graph.nodes["n2"].op, "precision_cast")
        self.assertEqual(graph.nodes["n2"].inputs, ["v0"])
        self.assertEqual(graph.nodes["n2"].attrs["dtype"], "fp32")

    def test_canonical_cleanup_materializes_new_ones_and_keeps_arange_supported(self):
        graph = IRGraph(
            values={
                "v0": IRValue(id="v0", shape=(4,), dtype="int64", producer="n0", users=["n1"]),
                "v1": IRValue(id="v1", shape=(), dtype="bool", producer="n1", users=[]),
            },
            nodes={
                "n0": IRNode(
                    id="n0",
                    op="arange",
                    inputs=[],
                    outputs=["v0"],
                    attrs={"start": 0, "end": 4, "dtype": "int64"},
                ),
                "n1": IRNode(
                    id="n1",
                    op="aten.new_ones.default",
                    inputs=["v0"],
                    outputs=["v1"],
                    attrs={"opaque": True},
                ),
            },
            order=["n0", "n1"],
            inputs=[],
            outputs=["v1"],
            constants={},
        )

        canonicalize_exported_graph(graph)

        self.assertNotIn("n0", graph.nodes)
        self.assertNotIn("n1", graph.nodes)
        self.assertIn("v1", graph.constants)
        self.assertEqual(bool(graph.constants["v1"].item()), True)
        self.assertEqual(graph.meta.get("canonical_unsupported_op_counts"), {})

    def test_canonical_cleanup_removes_noop_casts_and_views(self):
        graph = IRGraph(
            values={
                "x": IRValue(id="x", shape=(2, 3), dtype="fp16", users=["n0"]),
                "v0": IRValue(id="v0", shape=(2, 3), dtype="fp16", producer="n0", users=["n1"]),
                "v1": IRValue(id="v1", shape=(2, 3), dtype="fp16", producer="n1", users=[]),
            },
            nodes={
                "n0": IRNode(id="n0", op="precision_cast", inputs=["x"], outputs=["v0"], attrs={"dtype": "torch.float16"}),
                "n1": IRNode(id="n1", op="reshape", inputs=["v0"], outputs=["v1"], attrs={"shape": (2, 3)}),
            },
            order=["n0", "n1"],
            inputs=["x"],
            outputs=["v1"],
            constants={},
        )

        canonicalize_exported_graph(graph)

        self.assertEqual(graph.order, [])
        self.assertEqual(graph.outputs, ["x"])
        self.assertEqual(graph.meta.get("canonical_unsupported_op_counts"), {})

    def test_canonical_cleanup_renames_compare_and_logical_ops(self):
        graph = IRGraph(
            values={
                "x": IRValue(id="x", shape=(2, 2), dtype="bool", users=["n0"]),
                "y": IRValue(id="y", shape=(2, 2), dtype="bool", users=["n0"]),
                "v0": IRValue(id="v0", shape=(2, 2), dtype="bool", producer="n0", users=[]),
            },
            nodes={
                "n0": IRNode(id="n0", op="aten.__and__.Tensor", inputs=["x", "y"], outputs=["v0"]),
            },
            order=["n0"],
            inputs=["x", "y"],
            outputs=["v0"],
            constants={},
        )

        canonicalize_exported_graph(graph)

        self.assertEqual(graph.nodes["n0"].op, "logical_and")
        self.assertEqual(graph.meta.get("canonical_unsupported_op_counts"), {"logical_and": 1})

    def test_import_semantics_detects_raw_rope_early(self):
        graph = IRGraph(
            values={
                "x": IRValue(id="x", shape=(1, 2, 4), dtype="fp16", users=["n_x_cos", "n_slice_left", "n_slice_right"]),
                "v_arange": IRValue(id="v_arange", shape=(2,), dtype="fp32", producer="n_arange", users=["n_matmul"]),
                "v_inv": IRValue(id="v_inv", shape=(2,), dtype="fp32", producer=None, users=["n_matmul"]),
                "v_angles": IRValue(id="v_angles", shape=(2, 2), dtype="fp32", producer="n_matmul", users=["n_transpose"]),
                "v_angles_t": IRValue(id="v_angles_t", shape=(2, 2), dtype="fp32", producer="n_transpose", users=["n_cat_angles", "n_cat_angles"]),
                "v_cat_angles": IRValue(id="v_cat_angles", shape=(2, 4), dtype="fp32", producer="n_cat_angles", users=["n_cos", "n_sin"]),
                "v_cos": IRValue(id="v_cos", shape=(2, 4), dtype="fp32", producer="n_cos", users=["n_x_cos"]),
                "v_sin": IRValue(id="v_sin", shape=(2, 4), dtype="fp32", producer="n_sin", users=["n_rot_sin"]),
                "v_x_cos": IRValue(id="v_x_cos", shape=(1, 2, 4), dtype="fp16", producer="n_x_cos", users=["n_add"]),
                "v_left": IRValue(id="v_left", shape=(1, 2, 2), dtype="fp16", producer="n_slice_left", users=["n_neg_left"]),
                "v_right": IRValue(id="v_right", shape=(1, 2, 2), dtype="fp16", producer="n_slice_right", users=["n_cat_rot"]),
                "v_neg_left": IRValue(id="v_neg_left", shape=(1, 2, 2), dtype="fp16", producer="n_neg_left", users=["n_cat_rot"]),
                "v_rot": IRValue(id="v_rot", shape=(1, 2, 4), dtype="fp16", producer="n_cat_rot", users=["n_rot_sin"]),
                "v_rot_sin": IRValue(id="v_rot_sin", shape=(1, 2, 4), dtype="fp16", producer="n_rot_sin", users=["n_add"]),
                "v_out": IRValue(id="v_out", shape=(1, 2, 4), dtype="fp16", producer="n_add", users=[]),
            },
            nodes={
                "n_arange": IRNode(id="n_arange", op="arange", inputs=[], outputs=["v_arange"], attrs={"start": 0, "end": 2, "step": 1, "dtype": "fp32"}),
                "n_matmul": IRNode(id="n_matmul", op="matmul", inputs=["v_arange", "v_inv"], outputs=["v_angles"]),
                "n_transpose": IRNode(id="n_transpose", op="transpose", inputs=["v_angles"], outputs=["v_angles_t"], attrs={"dim0": 0, "dim1": 1}),
                "n_cat_angles": IRNode(id="n_cat_angles", op="cat", inputs=["v_angles_t", "v_angles_t"], outputs=["v_cat_angles"], attrs={"axis": -1}),
                "n_cos": IRNode(id="n_cos", op="cos", inputs=["v_cat_angles"], outputs=["v_cos"]),
                "n_sin": IRNode(id="n_sin", op="sin", inputs=["v_cat_angles"], outputs=["v_sin"]),
                "n_x_cos": IRNode(id="n_x_cos", op="multiply", inputs=["x", "v_cos"], outputs=["v_x_cos"]),
                "n_slice_left": IRNode(id="n_slice_left", op="slice", inputs=["x"], outputs=["v_left"], attrs={"axis": -1, "start": 2, "end": 4, "step": 1}),
                "n_slice_right": IRNode(id="n_slice_right", op="slice", inputs=["x"], outputs=["v_right"], attrs={"axis": -1, "start": 0, "end": 2, "step": 1}),
                "n_neg_left": IRNode(id="n_neg_left", op="scalar_multiply", inputs=["v_left"], outputs=["v_neg_left"], attrs={"value": -1.0}),
                "n_cat_rot": IRNode(id="n_cat_rot", op="cat", inputs=["v_neg_left", "v_right"], outputs=["v_rot"], attrs={"axis": -1}),
                "n_rot_sin": IRNode(id="n_rot_sin", op="multiply", inputs=["v_rot", "v_sin"], outputs=["v_rot_sin"]),
                "n_add": IRNode(id="n_add", op="add", inputs=["v_x_cos", "v_rot_sin"], outputs=["v_out"]),
            },
            order=[
                "n_arange",
                "n_matmul",
                "n_transpose",
                "n_cat_angles",
                "n_cos",
                "n_sin",
                "n_x_cos",
                "n_slice_left",
                "n_slice_right",
                "n_neg_left",
                "n_cat_rot",
                "n_rot_sin",
                "n_add",
            ],
            inputs=["x"],
            outputs=["v_out"],
            constants={"v_inv": torch.tensor([1.0, 0.01], dtype=torch.float32)},
        )

        apply_import_semantics(graph)

        rope_node = graph.nodes["n_add"]
        self.assertEqual(rope_node.op, "rope")
        self.assertEqual(rope_node.inputs, ["x"])
        self.assertEqual(rope_node.kind, "semantic")
        self.assertEqual(rope_node.meta.get("semantic_source"), "import")

    def test_match_gqa_expand_alias_accepts_view_expand_view_pattern(self):
        graph = IRGraph(
            values={
                "base": IRValue(id="base", shape=(1, 1, 6, 256), dtype="fp16", users=["n_view"]),
                "v_view": IRValue(id="v_view", shape=(1, 1, 1, 6, 256), dtype="fp16", producer="n_view", users=["n_expand"]),
                "v_expand": IRValue(id="v_expand", shape=(1, 1, 8, 6, 256), dtype="fp16", producer="n_expand", users=["n_view_2"]),
                "v_out": IRValue(id="v_out", shape=(1, 8, 6, 256), dtype="fp16", producer="n_view_2", users=[]),
            },
            nodes={
                "n_view": IRNode(id="n_view", op="view", inputs=["base"], outputs=["v_view"], attrs={"shape": (1, 1, 1, 6, 256)}),
                "n_expand": IRNode(id="n_expand", op="expand", inputs=["v_view"], outputs=["v_expand"], attrs={"shape": (1, 1, 8, 6, 256)}),
                "n_view_2": IRNode(id="n_view_2", op="view", inputs=["v_expand"], outputs=["v_out"], attrs={"shape": (1, 8, 6, 256)}),
            },
            order=["n_view", "n_expand", "n_view_2"],
            inputs=["base"],
            outputs=["v_out"],
            constants={},
        )

        self.assertEqual(_match_gqa_expand_alias(graph, graph.nodes["n_expand"]), "base")

    def test_rope_lowering_translates_bhsd_logical_layout_to_backend_layout(self):
        graph = IRGraph(
            values={
                "x": IRValue(id="x", shape=(1, 6, 8, 256), dtype="fp16", users=["n_perm"]),
                "v_perm": IRValue(id="v_perm", shape=(1, 8, 6, 256), dtype="fp16", producer="n_perm", users=["n_rope"]),
                "v_out": IRValue(id="v_out", shape=(1, 8, 6, 256), dtype="fp16", producer="n_rope", users=[]),
            },
            nodes={
                "n_perm": IRNode(
                    id="n_perm",
                    op="permute",
                    inputs=["x"],
                    outputs=["v_perm"],
                    attrs={"permutation": (0, 2, 1, 3)},
                ),
                "n_rope": IRNode(
                    id="n_rope",
                    op="rope",
                    inputs=["v_perm"],
                    outputs=["v_out"],
                    attrs={"theta": 10000.0, "position_offset": 0},
                    kind="semantic",
                ),
            },
            order=["n_perm", "n_rope"],
            inputs=["x"],
            outputs=["v_out"],
            constants={},
        )

        x = torch.randn(1, 6, 8, 256, dtype=torch.float16)
        tg = transpile_captured(type("Captured", (), {"ir_graph": graph})())
        tg.set_inputs([x.numpy()])
        got = tg.execute()[0].numpy().astype(np.float32)

        ref_graph = Graph()
        ref_x = ref_graph.input((1, 6, 8, 256), dtype=Graph.FP16)
        ref_out = ref_graph.permute(ref_graph.rope(ref_x, 10000.0, position_offset=0), (0, 2, 1, 3))
        ref_graph.set_input(ref_x, x.numpy())
        ref_graph.execute()
        ref = ref_out.numpy().astype(np.float32)

        np.testing.assert_allclose(ref, got, atol=1e-2)



if __name__ == "__main__":
    unittest.main()
