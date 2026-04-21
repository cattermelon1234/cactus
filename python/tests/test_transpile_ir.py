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
from src.transpile.importers import UnsupportedImportError


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

    def test_capture_fails_strictly_for_unsupported_import_op(self):
        model = UnsupportedToy().eval()
        x = torch.randn(2, 4, dtype=torch.float16)

        with self.assertRaises(UnsupportedImportError):
            capture_model(model, (x,))

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


if __name__ == "__main__":
    unittest.main()
