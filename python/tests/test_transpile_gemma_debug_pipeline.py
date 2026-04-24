from __future__ import annotations

import copy
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transpile.capture_pytorch import capture_model
from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
from src.transpile.lower import transpile_captured
from src.transpile.model_adapters import canonicalize_model_interface
from src.transpile.optimize_graph import optimize_graph
from src.transpile.optimize_graph import summarize_detected_gold_patterns


def _require_debug_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_GEMMA_DEBUG_PIPELINE_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_GEMMA_DEBUG_PIPELINE_TEST=1 to run the Gemma debug pipeline integration test."
        )


def _require_full_model_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_GEMMA_FULL_MODEL_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_GEMMA_FULL_MODEL_TEST=1 to run the full-model Gemma next-token test."
        )


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise unittest.SkipTest(f"transformers is not available: {exc}") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _resolve_local_model_path(model_id: str) -> str:
    explicit = Path(model_id)
    if explicit.exists():
        return str(explicit)

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + model_id.replace("/", "--"))
        / "snapshots"
    )
    if not snapshots_dir.exists():
        raise unittest.SkipTest(f"no local Hugging Face snapshot found for {model_id!r}")
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        raise unittest.SkipTest(f"no local Hugging Face snapshots found for {model_id!r}")
    return str(snapshots[-1])


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
            raise ValueError("Gemma4 debug pipeline test requires debug_first_block()")
        self.adapter = adapter
        self.checkpoint_name = checkpoint_name

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        checkpoints = self.adapter.debug_first_block(input_ids)
        if self.checkpoint_name not in checkpoints:
            available = ", ".join(sorted(checkpoints.keys()))
            raise KeyError(f"unknown checkpoint {self.checkpoint_name!r}; available: {available}")
        return checkpoints[self.checkpoint_name]


def _graph_summary(graph, label: str, *, max_nodes: int = 100) -> None:
    op_counts = Counter(graph.nodes[node_id].op for node_id in graph.order)
    semantic_counts = Counter(
        graph.nodes[node_id].op
        for node_id in graph.order
        if graph.nodes[node_id].kind == "semantic"
    )
    print(f"\n=== {label} ===")
    print(f"node_count={len(graph.order)}")
    print("top_ops=" + ", ".join(f"{op}={count}" for op, count in op_counts.most_common(12)))
    if semantic_counts:
        print("semantic_ops=" + ", ".join(f"{op}={count}" for op, count in semantic_counts.most_common()))
    unsupported = graph.meta.get("canonical_unsupported_op_counts", {})
    if isinstance(unsupported, dict) and unsupported:
        print("unsupported_ops=" + ", ".join(f"{op}={count}" for op, count in unsupported.items()))
    patterns = graph.meta.get("detected_gold_patterns", ())
    if patterns:
        pattern_counts = Counter(pattern.name for pattern in patterns)
        print("patterns=" + ", ".join(f"{name}={count}" for name, count in pattern_counts.most_common()))
    for node_id in graph.order[:max_nodes]:
        node = graph.nodes[node_id]
        print(f"  {node_id}: op={node.op} kind={node.kind} inputs={node.inputs} outputs={node.outputs} attrs={node.attrs}")
    if len(graph.order) > max_nodes:
        print(f"  ... ({len(graph.order) - max_nodes} more nodes omitted)")


def _max_abs_diff(ref: torch.Tensor, got: np.ndarray) -> float:
    ref_np = ref.detach().float().cpu().numpy()
    got_np = got.astype(np.float32)
    return float(np.max(np.abs(ref_np - got_np)))


def _mean_abs_diff(ref: torch.Tensor, got: np.ndarray) -> float:
    ref_np = ref.detach().float().cpu().numpy()
    got_np = got.astype(np.float32)
    return float(np.mean(np.abs(ref_np - got_np)))


def _topk_ids(logits: np.ndarray, k: int) -> list[int]:
    return [int(index) for index in np.argsort(logits)[-k:][::-1]]


class TestGemmaDebugPipeline(unittest.TestCase):
    model_id = os.environ.get("CACTUS_GEMMA_HF_MODEL_ID", "google/gemma-4-E2B")

    @classmethod
    def setUpClass(cls) -> None:
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        local_path = _resolve_local_model_path(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        cls.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()

    def _adapter_module(self):
        return canonicalize_model_interface(self.model, task="causal_lm_logits").module

    def _first_block_input_ids(self) -> torch.Tensor:
        return torch.tensor([[2, 818, 5279, 529, 7001, 563]], dtype=torch.long)

    def test_gemma4_debug_pipeline_first_block(self) -> None:
        _require_debug_opt_in()

        adapter = canonicalize_model_interface(self.model, task="causal_lm_logits")
        if adapter.family != "gemma4":
            raise unittest.SkipTest(f"expected gemma4 adapter, got family={adapter.family}")

        input_ids = self._first_block_input_ids()
        representative = Gemma4FirstBlockCheckpointWrapper(self.model, "after_ffn_residual").eval()
        captured = capture_model(representative, (input_ids,))

        raw_graph = copy.deepcopy(captured.ir_graph)
        cleanup_graph = copy.deepcopy(captured.ir_graph)
        canonicalize_exported_graph(cleanup_graph)
        optimized_graph = copy.deepcopy(cleanup_graph)
        optimize_graph(optimized_graph)

        _graph_summary(raw_graph, "Gemma4 First Block Raw IR")
        _graph_summary(cleanup_graph, "Gemma4 First Block After Cleanup")
        _graph_summary(optimized_graph, "Gemma4 First Block After Fusion")
        print("pattern_summary=" + str(summarize_detected_gold_patterns(optimized_graph)))

        optimized_ops = Counter(optimized_graph.nodes[node_id].op for node_id in optimized_graph.order)
        self.assertGreater(optimized_ops.get("attention_block", 0), 0)
        self.assertGreater(optimized_ops.get("rms_norm", 0), 0)

        adapter_module = adapter.module
        with torch.no_grad():
            ref_checkpoints = adapter_module.debug_first_block(input_ids)

        tolerances = {
            "pre_attn_norm": 1.0,
            "attn_o_proj": 3e-1,
            "post_attn_norm": 3e-1,
            "after_attention_residual": 3e-1,
            "pre_ffn_norm": 3e-1,
            "mlp_down": 3e-1,
            "post_ffn_norm": 3e-1,
            "after_ffn_residual": 3e-1,
            "per_layer_input_proj": 3e-1,
            "post_per_layer_input_norm": 3e-1,
            "layer_scalar_out": 3e-1,
        }

        print("\nCheckpoint diffs:")
        for checkpoint_name, ref_tensor in ref_checkpoints.items():
            module = Gemma4FirstBlockCheckpointWrapper(self.model, checkpoint_name).eval()
            checkpoint_capture = capture_model(module, (input_ids,))
            canonicalize_exported_graph(checkpoint_capture.ir_graph)
            optimize_graph(checkpoint_capture.ir_graph)
            tg = transpile_captured(checkpoint_capture)
            tg.set_inputs([input_ids.cpu().numpy()])
            got = tg.execute()[0].numpy().astype(np.float32)

            max_diff = _max_abs_diff(ref_tensor, got)
            mean_diff = _mean_abs_diff(ref_tensor, got)
            print(
                f"  {checkpoint_name}: shape={tuple(got.shape)} "
                f"max_abs_diff={max_diff:.6f} mean_abs_diff={mean_diff:.6f}"
            )
            self.assertLessEqual(max_diff, tolerances.get(checkpoint_name, 3e-1), checkpoint_name)

    def test_gemma4_full_model_next_token_matches_hf(self) -> None:
        _require_debug_opt_in()
        _require_full_model_opt_in()

        input_ids = self._first_block_input_ids()
        capture_module = GemmaFullModelWrapper(self.model).eval()
        captured = capture_model(capture_module, (input_ids,))

        raw_graph = copy.deepcopy(captured.ir_graph)
        canonicalize_exported_graph(captured.ir_graph)
        optimize_graph(captured.ir_graph)

        _graph_summary(raw_graph, "Gemma4 Full Model Raw IR", max_nodes=12)
        _graph_summary(captured.ir_graph, "Gemma4 Full Model After Fusion", max_nodes=100)

        tg = transpile_captured(captured)
        tg.set_inputs([input_ids.cpu().numpy()])
        transpiled_logits = tg.execute()[0].numpy().astype(np.float32)

        with torch.no_grad():
            hf_logits = self.model(input_ids=input_ids, use_cache=False, return_dict=False)[0].detach().float().cpu().numpy()

        hf_next = int(np.argmax(hf_logits[0, -1]))
        transpiled_next = int(np.argmax(transpiled_logits[0, -1]))
        hf_top5 = _topk_ids(hf_logits[0, -1], 5)
        transpiled_top5 = _topk_ids(transpiled_logits[0, -1], 5)

        print("\nFinal inference:")
        print(f"  input_ids={input_ids.tolist()}")
        print(f"  hf_next={hf_next} top5={hf_top5}")
        print(f"  transpiled_next={transpiled_next} top5={transpiled_top5}")
        print(f"  logits_max_abs_diff={float(np.max(np.abs(hf_logits - transpiled_logits))):.6f}")
        print(f"  logits_mean_abs_diff={float(np.mean(np.abs(hf_logits - transpiled_logits))):.6f}")

        self.assertEqual(transpiled_next, hf_next)


if __name__ == "__main__":
    unittest.main()
