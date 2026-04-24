from __future__ import annotations

import os
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transpile.capture_pytorch import capture_model
from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
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
        kwargs = {"position_embeddings": position_embeddings}
        parameters = inspect.signature(self.self_attn.forward).parameters
        if "position_ids" in parameters:
            kwargs["position_ids"] = position_ids
        if "shared_kv_states" in parameters:
            kwargs["shared_kv_states"] = {}
        if "past_key_values" in parameters:
            kwargs["past_key_values"] = None
        hidden_states, _ = self.self_attn(x, **kwargs)
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
        kwargs = {"position_embeddings": position_embeddings}
        parameters = inspect.signature(self.layer.forward).parameters
        if "position_ids" in parameters:
            kwargs["position_ids"] = position_ids
        if "shared_kv_states" in parameters:
            kwargs["shared_kv_states"] = {}
        if "past_key_values" in parameters:
            kwargs["past_key_values"] = None
        if "per_layer_input" in parameters:
            kwargs["per_layer_input"] = None
        hidden_states = self.layer(x, **kwargs)
        return hidden_states


class GemmaFullModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = canonicalize_model_interface(model, task="causal_lm_logits").module

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


class Gemma4FirstBlockCheckpointWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, checkpoint_name: str):
        super().__init__()
        adapter = canonicalize_model_interface(model, task="causal_lm_logits").module
        if not hasattr(adapter, "debug_first_block"):
            raise ValueError("checkpoint wrapper requires a Gemma4 adapter with debug_first_block()")
        self.adapter = adapter
        self.checkpoint_name = checkpoint_name

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        checkpoints = self.adapter.debug_first_block(input_ids)
        if self.checkpoint_name not in checkpoints:
            available = ", ".join(sorted(checkpoints.keys()))
            raise KeyError(f"unknown checkpoint {self.checkpoint_name!r}; available: {available}")
        return checkpoints[self.checkpoint_name]


class TestTranspileGemma(unittest.TestCase):
    # model_id = os.environ.get("CACTUS_GEMMA_HF_MODEL_ID", "google/gemma-3-270m-it")
    model_id = os.environ.get("CACTUS_GEMMA_HF_MODEL_ID", "google/gemma-4-E2B")

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

    def _adapter(self):
        return canonicalize_model_interface(self.model, task="causal_lm_logits")

    def _backbone(self):
        module = self._adapter().module
        backbone = getattr(module, "backbone", None)
        if backbone is None:
            raise unittest.SkipTest("canonicalized model does not expose a backbone")
        return backbone

    def _layer0(self):
        return self._backbone().layers[0]

    def _hidden_size(self) -> int:
        return int(self._backbone().config.hidden_size)

    def _vocab_size(self) -> int:
        backbone = self._backbone()
        vocab_size = getattr(backbone.config, "vocab_size", None)
        if vocab_size is None:
            text_cfg = getattr(self.model.config, "text_config", None)
            vocab_size = getattr(text_cfg, "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(self.model, "vocab_size", None)
        if vocab_size is None:
            raise unittest.SkipTest("could not determine vocab size for this model family")
        return int(vocab_size)

    def _assert_max_abs_diff(self, ref: torch.Tensor, got: np.ndarray, *, atol: float) -> None:
        ref_np = ref.detach().float().cpu().numpy()
        got_np = got.astype(np.float32)
        abs_diff = np.abs(ref_np - got_np)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        print("max abs diff:", float(max_diff))
        print("mean abs diff:", float(mean_diff))
        self.assertLessEqual(float(max_diff), atol)

    def _assert_finite(self, tensor: torch.Tensor | np.ndarray, *, label: str) -> None:
        array = tensor.detach().float().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor, dtype=np.float32)
        finite_ratio = float(np.isfinite(array).mean()) if array.size else 1.0
        print(f"{label} finite ratio:", finite_ratio)
        self.assertTrue(np.isfinite(array).all(), f"{label} contains non-finite values")

    def _first_block_input_ids(self) -> torch.Tensor:
        return torch.tensor([[2, 818, 5279, 529, 7001, 563]], dtype=torch.long)

    def _assert_capture_succeeds(
        self,
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        *,
        max_nodes: int | None = None,
    ) -> None:
        captured = capture_model(module, args)
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured, max_nodes=max_nodes)
        self.assertGreater(len(captured.ir_graph.order), 0)

    def test_transpile_gemma_input_layernorm(self) -> None:
        layer = self._layer0()
        module = GemmaLayerNormWrapper(layer.input_layernorm).eval()

        x = torch.randn(2, self._hidden_size(), dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=7e-2)

    def test_transpile_gemma_post_attention_layernorm(self) -> None:
        layer = self._layer0()
        module = GemmaPostAttentionLayerNormWrapper(layer.post_attention_layernorm).eval()

        x = torch.randn(2, self._hidden_size(), dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=7e-2)

    def test_transpile_gemma_mlp(self) -> None:
        layer = self._layer0()
        module = GemmaMLPWrapper(layer.mlp).eval()

        x = torch.randn(2, self._hidden_size(), dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=2e-1)

    def test_transpile_gemma_gate_proj(self) -> None:
        layer = self._layer0()
        module = GemmaLinearWrapper(layer.mlp.gate_proj).eval()

        x = torch.randn(2, self._hidden_size(), dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_up_proj(self) -> None:
        layer = self._layer0()
        module = GemmaLinearWrapper(layer.mlp.up_proj).eval()

        x = torch.randn(2, self._hidden_size(), dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_down_proj(self) -> None:
        layer = self._layer0()
        module = GemmaLinearWrapper(layer.mlp.down_proj).eval()

        intermediate_size = layer.mlp.down_proj.in_features
        x = torch.randn(2, intermediate_size, dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_q_proj(self) -> None:
        layer = self._layer0()
        module = GemmaLinearWrapper(layer.self_attn.q_proj).eval()

        x = torch.randn(2, self._hidden_size(), dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_k_proj(self) -> None:
        layer = self._layer0()
        module = GemmaLinearWrapper(layer.self_attn.k_proj).eval()

        x = torch.randn(2, self._hidden_size(), dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_v_proj(self) -> None:
        layer = self._layer0()
        module = GemmaLinearWrapper(layer.self_attn.v_proj).eval()

        x = torch.randn(2, self._hidden_size(), dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_transpile_gemma_o_proj(self) -> None:
        layer = self._layer0()
        module = GemmaLinearWrapper(layer.self_attn.o_proj).eval()

        out_features = layer.self_attn.o_proj.in_features
        x = torch.randn(2, out_features, dtype=torch.float16)

        captured = capture_model(module, (x,))
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured)

        tg = transpile_captured(captured)
        tg.set_inputs([x.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(x)

        self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=5e-2)

    def test_capture_only_gemma_self_attention(self) -> None:
        backbone = self._backbone()
        layer = backbone.layers[0]
        module = GemmaSelfAttentionWrapper(layer.self_attn, backbone.rotary_emb).eval()

        x = torch.randn(1, 2, self._hidden_size(), dtype=torch.float16)
        position_ids = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)

        self._assert_capture_succeeds(module, (x, position_ids))

    def test_capture_only_gemma_decoder_layer(self) -> None:
        backbone = self._backbone()
        layer = backbone.layers[0]
        module = GemmaDecoderLayerWrapper(layer, backbone.rotary_emb).eval()

        x = torch.randn(1, 2, self._hidden_size(), dtype=torch.float16)
        position_ids = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)

        self._assert_capture_succeeds(module, (x, position_ids))

    def test_capture_only_gemma_full_model(self) -> None:
        _require_full_model_opt_in()
        module = GemmaFullModelWrapper(self.model).eval()
        input_ids = torch.randint(0, self._vocab_size(), (1, 2), dtype=torch.long)
        self._assert_capture_succeeds(module, (input_ids,), max_nodes=120)

    # def test_transpile_gemma_full_model(self) -> None:
    #     _require_full_model_opt_in()
    #     module = GemmaFullModelWrapper(self.model).eval()
    #     input_ids = torch.randint(0, self.model.config.vocab_size, (1, 2), dtype=torch.long)
    #
    #     captured = capture_model(module, (input_ids,))
    #     run_cleanup_passes(captured.ir_graph)
    #     self._print_ir_summary(captured, max_nodes=120)
    #     optimize_graph(captured.ir_graph)
    #     run_cleanup_passes(captured.ir_graph)
    #
    #     tg = transpile_captured(captured)
    #     tg.set_inputs([input_ids.cpu().numpy()])
    #     outputs = tg.execute()
    #
    #     with torch.no_grad():
    #         ref = module(input_ids)
    #
    #     self._assert_max_abs_diff(ref, outputs[0].numpy(), atol=3e-1)
    #
    def _assert_gemma4_first_block_checkpoint_matches(self, checkpoint_name: str, *, atol: float) -> None:
        adapter = canonicalize_model_interface(self.model, task="causal_lm_logits")
        if adapter.family != "gemma4":
            raise unittest.SkipTest(f"first-block checkpoint tests are Gemma4-specific, got family={adapter.family}")

        module = Gemma4FirstBlockCheckpointWrapper(self.model, checkpoint_name).eval()
        input_ids = self._first_block_input_ids()

        captured = capture_model(module, (input_ids,))
        canonicalize_exported_graph(captured.ir_graph)
        optimize_graph(captured.ir_graph)
        canonicalize_exported_graph(captured.ir_graph)
        self._print_ir_summary(captured, max_nodes=80)

        tg = transpile_captured(captured)
        tg.set_inputs([input_ids.cpu().numpy()])
        outputs = tg.execute()

        with torch.no_grad():
            ref = module(input_ids)

        got = outputs[0].numpy()
        self._assert_finite(ref, label=f"{checkpoint_name} ref")
        self._assert_finite(got, label=f"{checkpoint_name} transpiled")
        self._assert_max_abs_diff(ref, got, atol=atol)

    def test_transpile_gemma4_first_block_pre_attn_norm(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("pre_attn_norm", atol=8e-2)

    def test_transpile_gemma4_first_block_attn_o_proj(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("attn_o_proj", atol=3e-1)

    def test_transpile_gemma4_first_block_post_attn_norm(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("post_attn_norm", atol=3e-1)

    def test_transpile_gemma4_first_block_after_attention_residual(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("after_attention_residual", atol=3e-1)

    def test_transpile_gemma4_first_block_pre_ffn_norm(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("pre_ffn_norm", atol=3e-1)

    def test_transpile_gemma4_first_block_mlp_down(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("mlp_down", atol=3e-1)

    def test_transpile_gemma4_first_block_post_ffn_norm(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("post_ffn_norm", atol=3e-1)

    def test_transpile_gemma4_first_block_after_ffn_residual(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("after_ffn_residual", atol=3e-1)

    def test_transpile_gemma4_first_block_layer_scalar_out(self) -> None:
        self._assert_gemma4_first_block_checkpoint_matches("layer_scalar_out", atol=3e-1)


if __name__ == "__main__":
    unittest.main()
