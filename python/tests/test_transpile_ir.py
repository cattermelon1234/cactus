import unittest

import numpy as np
import torch
import torch.nn as nn

from src.transpile.capture_pytorch import capture_model
from src.transpile.cleanup_passes import run_cleanup_passes
from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.graph_ir import IRValue
from src.transpile.graph_ir import verify_ir
from src.transpile.lower import transpile_captured
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


class TestTranspileIR(unittest.TestCase):
    def test_toy_graph_transpiles_and_matches(self):
        model = Toy().eval()
        x = torch.randn(2, 4, dtype=torch.float16)
        y = torch.randn(2, 4, dtype=torch.float16)

        captured = capture_model(model, (x, y))
        run_cleanup_passes(captured.ir_graph)

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
        run_cleanup_passes(captured.ir_graph)

        self.assertIn("aten.square.default", {captured.ir_graph.nodes[node_id].op for node_id in captured.ir_graph.order})

        with self.assertRaises(NotImplementedError):
            transpile_captured(captured)

    def test_not_equal_transpiles_and_matches(self):
        model = NotEqualToy().eval()
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)
        y = torch.tensor([[1, 0, 3], [0, 5, 7]], dtype=torch.float16)

        captured = capture_model(model, (x, y))
        run_cleanup_passes(captured.ir_graph)

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
        run_cleanup_passes(captured.ir_graph)

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
        run_cleanup_passes(captured.ir_graph)
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

        sdpa_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "scaled_dot_product_attention"]
        self.assertEqual(len(sdpa_nodes), 1)
        self.assertEqual(sdpa_nodes[0].attrs.get("window_size"), 7)
        self.assertEqual(sdpa_nodes[0].meta.get("attention_layer_type"), "sliding_attention")

        optimize_graph(captured.ir_graph)
        attention_nodes = [captured.ir_graph.nodes[node_id] for node_id in captured.ir_graph.order if captured.ir_graph.nodes[node_id].op == "attention"]
        self.assertEqual(len(attention_nodes), 1)
        self.assertEqual(attention_nodes[0].attrs.get("window_size"), 7)
        self.assertEqual(attention_nodes[0].meta.get("window_size_source"), "import_attr")


if __name__ == "__main__":
    unittest.main()
