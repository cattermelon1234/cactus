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


SUPPORTED_MODEL_IDS = (
    "google/gemma-4-E2B",
    "google/gemma-3-270m-it",
    "google/gemma-2b-it",
    "google/gemma-2-2b",
    "Qwen/Qwen3.5-2B",
)


def _require_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_LOCAL_MODEL_MATRIX_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_LOCAL_MODEL_MATRIX_TEST=1 to run the local model transpiler matrix test."
        )


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise unittest.SkipTest(f"transformers is not available: {exc}") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _resolve_local_snapshot(model_id: str) -> str | None:
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
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return str(snapshots[-1])


def _discover_local_models() -> list[str]:
    model_ids: list[str] = []
    for model_id in SUPPORTED_MODEL_IDS:
        if _resolve_local_snapshot(model_id) is not None:
            model_ids.append(model_id)
    return model_ids


class FullModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.adapter = canonicalize_model_interface(model, task="causal_lm_logits").module
        self.module = self.adapter

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.module(input_ids)

    def get_transpile_metadata(self):
        if hasattr(self.adapter, "get_transpile_metadata"):
            return self.adapter.get_transpile_metadata()
        return {}


def _print_graph_summary(graph, label: str, *, max_nodes: int = 10) -> None:
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
    patterns = graph.meta.get("detected_gold_patterns", ())
    if patterns:
        pattern_counts = Counter(pattern.name for pattern in patterns)
        print("patterns=" + ", ".join(f"{name}={count}" for name, count in pattern_counts.most_common()))
    for node_id in graph.order[:max_nodes]:
        node = graph.nodes[node_id]
        print(f"  {node_id}: op={node.op} kind={node.kind} attrs={node.attrs}")
    if len(graph.order) > max_nodes:
        print(f"  ... ({len(graph.order) - max_nodes} more nodes omitted)")


class TestLocalModelTranspilerMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_opt_in()
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        cls.AutoModelForCausalLM = AutoModelForCausalLM
        cls.AutoTokenizer = AutoTokenizer
        cls.available_model_ids = _discover_local_models()
        if not cls.available_model_ids:
            raise unittest.SkipTest("no supported local Hugging Face snapshots found for the model matrix test")

    def test_local_model_matrix_next_token_matches_hf(self) -> None:
        prompt = os.environ.get("CACTUS_LOCAL_MODEL_MATRIX_PROMPT", "The capital of France is")

        for model_id in self.available_model_ids:
            with self.subTest(model_id=model_id):
                snapshot = _resolve_local_snapshot(model_id)
                self.assertIsNotNone(snapshot)

                tokenizer = self.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
                model = self.AutoModelForCausalLM.from_pretrained(
                    snapshot,
                    torch_dtype=torch.float16,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                ).eval()
                adapter = canonicalize_model_interface(model, task="causal_lm_logits")
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

                print(f"\n\n### Model: {model_id}")
                print(f"family={adapter.family} prompt_len={int(input_ids.shape[1])}")

                captured = capture_model(FullModelWrapper(model).eval(), (input_ids,))
                raw_graph = copy.deepcopy(captured.ir_graph)
                _print_graph_summary(raw_graph, f"{model_id} Raw IR")

                canonicalize_exported_graph(captured.ir_graph)
                optimize_graph(captured.ir_graph)
                _print_graph_summary(captured.ir_graph, f"{model_id} After Canonicalize+Optimize")
                print("pattern_summary=" + str(summarize_detected_gold_patterns(captured.ir_graph)))

                tg = transpile_captured(captured)
                tg.set_inputs([input_ids.cpu().numpy()])
                transpiled_logits = tg.execute()[0].numpy().astype(np.float32)

                with torch.no_grad():
                    hf_logits = model(input_ids=input_ids, use_cache=False, return_dict=False)[0].detach().float().cpu().numpy()

                hf_next = int(np.argmax(hf_logits[0, -1]))
                transpiled_next = int(np.argmax(transpiled_logits[0, -1]))
                max_abs_diff = float(np.max(np.abs(hf_logits - transpiled_logits)))
                mean_abs_diff = float(np.mean(np.abs(hf_logits - transpiled_logits)))

                print(
                    f"next_token: hf={hf_next} transpiled={transpiled_next} "
                    f"logits_max_abs_diff={max_abs_diff:.6f} logits_mean_abs_diff={mean_abs_diff:.6f}"
                )

                self.assertEqual(transpiled_next, hf_next, model_id)


if __name__ == "__main__":
    unittest.main()
