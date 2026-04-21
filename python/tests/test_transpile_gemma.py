from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transpile.capture_pytorch import capture_model
from src.transpile.cleanup_passes import run_cleanup_passes
from src.transpile.lower import transpile_captured
from src.transpile.model_adapters import canonicalize_model_interface
from src.transpile.optimize_graph import optimize_graph


def _require_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_GEMMA_TRANSPILER_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_GEMMA_TRANSPILER_TEST=1 to run this Hugging Face Gemma transpile smoke test."
        )


def _require_full_model_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_GEMMA_FULL_MODEL_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_GEMMA_FULL_MODEL_TEST=1 to run the full Gemma model capture/transpile smoke tests."
        )


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise unittest.SkipTest(f"transformers is not available: {exc}") from exc
    return AutoModelForCausalLM, AutoTokenizer


class GemmaLayerNormWrapper(torch.nn.Module):
    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class GemmaMLPWrapper(torch.nn.Module):
    def __init__(self, mlp: torch.nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GemmaPostAttentionLayerNormWrapper(torch.nn.Module):
    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class GemmaLinearWrapper(torch.nn.Module):
    def __init__(self, linear: torch.nn.Module):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class GemmaSelfAttentionWrapper(torch.nn.Module):
    def __init__(self, self_attn: torch.nn.Module, rotary_emb: torch.nn.Module):
        super().__init__()
        self.self_attn = self_attn
        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        position_embeddings = self.rotary_emb(
            x,
            position_ids,
            layer_type=self.self_attn.layer_type,
        )
        hidden_states, _ = self.self_attn(x, position_embeddings=position_embeddings)
        return hidden_states


class GemmaDecoderLayerWrapper(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, rotary_emb: torch.nn.Module):
        super().__init__()
        self.layer = layer
        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        position_embeddings = self.rotary_emb(
            x,
            position_ids,
            layer_type=self.layer.self_attn.layer_type,
        )
        hidden_states = self.layer(x, position_embeddings=position_embeddings)
        return hidden_states


class GemmaFullModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = canonicalize_model_interface(model, task="causal_lm_logits").module

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


class TestTranspileGemma(unittest.TestCase):
    model_id = os.environ.get("CACTUS_GEMMA_HF_MODEL_ID", "google/gemma-3-270m-it")

    @classmethod
    def setUpClass(cls) -> None:
        _require_opt_in()
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()

        token = os.environ.get("HF_TOKEN")
        common_kwargs = {}
        if token:
            common_kwargs["token"] = token

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id, **common_kwargs)
            cls.model = AutoModelForCausalLM.from_pretrained(
                cls.model_id,
                torch_dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=True,
                **common_kwargs,
            ).eval()
        except Exception as exc:
            raise unittest.SkipTest(
                f"Could not load {cls.model_id}. You may need HF auth via HF_TOKEN or local cache. "
                f"This test is intended to match local converted weights such as ./weights/gemma-3-270m-it. "
                f"Original error: {exc}"
            ) from exc

    def _print_ir_summary(self, captured, *, max_nodes: int | None = None) -> None:
        print("\nIR inputs:", captured.ir_graph.inputs)
        print("IR outputs:", captured.ir_graph.outputs)
        print("IR node count:", len(captured.ir_graph.order))
        print("IR order:")
        node_ids = captured.ir_graph.order
        if max_nodes is not None:
            node_ids = node_ids[:max_nodes]
        for node_id in node_ids:
            node = captured.ir_graph.nodes[node_id]
            print(" ", node_id, node.op, node.inputs, node.outputs, node.attrs)
        if max_nodes is not None and len(captured.ir_graph.order) > max_nodes:
            remaining = len(captured.ir_graph.order) - max_nodes
            print(f" ... ({remaining} more nodes omitted)")

    def setUp(self) -> None:
        print(f"\n=== Running {self.id()} ===")

    def _assert_max_abs_diff(self, ref: torch.Tensor, got: np.ndarray, *, atol: float) -> None:
        ref_np = ref.detach().float().cpu().numpy()
        got_np = got.astype(np.float32)
        abs_diff = np.abs(ref_np - got_np)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        print("max abs diff:", float(max_diff))
        print("mean abs diff:", float(mean_diff))
        self.assertLessEqual(float(max_diff), atol)

    def _assert_capture_succeeds(
        self,
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        *,
        max_nodes: int | None = None,
    ) -> None:
        captured = capture_model(module, args)
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured, max_nodes=max_nodes)
        self.assertGreater(len(captured.ir_graph.order), 0)

    def test_transpile_gemma_input_layernorm(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaLayerNormWrapper(layer.input_layernorm).eval()

        x = torch.randn(2, self.model.config.hidden_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=7e-2)

    def test_transpile_gemma_post_attention_layernorm(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaPostAttentionLayerNormWrapper(layer.post_attention_layernorm).eval()

        x = torch.randn(2, self.model.config.hidden_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=7e-2)

    def test_transpile_gemma_mlp(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaMLPWrapper(layer.mlp).eval()

        x = torch.randn(2, self.model.config.hidden_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=2e-1)

    def test_transpile_gemma_gate_proj(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaLinearWrapper(layer.mlp.gate_proj).eval()

        x = torch.randn(2, self.model.config.hidden_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_up_proj(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaLinearWrapper(layer.mlp.up_proj).eval()

        x = torch.randn(2, self.model.config.hidden_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_down_proj(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaLinearWrapper(layer.mlp.down_proj).eval()

        intermediate_size = layer.mlp.down_proj.in_features
        x = torch.randn(2, intermediate_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_q_proj(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaLinearWrapper(layer.self_attn.q_proj).eval()

        x = torch.randn(2, self.model.config.hidden_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_k_proj(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaLinearWrapper(layer.self_attn.k_proj).eval()

        x = torch.randn(2, self.model.config.hidden_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_v_proj(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaLinearWrapper(layer.self_attn.v_proj).eval()

        x = torch.randn(2, self.model.config.hidden_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_o_proj(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaLinearWrapper(layer.self_attn.o_proj).eval()

        out_features = layer.self_attn.o_proj.in_features
        x = torch.randn(2, out_features, dtype=torch.float16)

        captured = capture_model(module, (x,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_capture_only_gemma_self_attention(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaSelfAttentionWrapper(layer.self_attn, self.model.model.rotary_emb).eval()

        x = torch.randn(1, 2, self.model.config.hidden_size, dtype=torch.float16)
        position_ids = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)

        self._assert_capture_succeeds(module, (x, position_ids))

    def test_capture_only_gemma_decoder_layer(self) -> None:
        layer = self.model.model.layers[0]
        module = GemmaDecoderLayerWrapper(layer, self.model.model.rotary_emb).eval()

        x = torch.randn(1, 2, self.model.config.hidden_size, dtype=torch.float16)
        position_ids = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)

        self._assert_capture_succeeds(module, (x, position_ids))

    def test_capture_only_gemma_full_model(self) -> None:
        _require_full_model_opt_in()
        module = GemmaFullModelWrapper(self.model).eval()
        input_ids = torch.randint(0, self.model.config.vocab_size, (1, 2), dtype=torch.long)
        self._assert_capture_succeeds(module, (input_ids,), max_nodes=120)

    def test_transpile_gemma_full_model(self) -> None:
        _require_full_model_opt_in()
        module = GemmaFullModelWrapper(self.model).eval()
        input_ids = torch.randint(0, self.model.config.vocab_size, (1, 2), dtype=torch.long)

        captured = capture_model(module, (input_ids,))
        run_cleanup_passes(captured.ir_graph)
        self._print_ir_summary(captured, max_nodes=120)
        optimize_graph(captured.ir_graph)
        run_cleanup_passes(captured.ir_graph)

        tg = transpile_captured(captured)
        tg.set_inputs([input_ids.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(input_ids)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=3e-1)


if __name__ == "__main__":
    unittest.main()
