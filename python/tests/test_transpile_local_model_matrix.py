from __future__ import annotations

import json
import os
import re
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transpile.capture_pytorch import capture_model
from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
from src.transpile.lower import transpile_captured
from src.transpile.model_adapters import canonicalize_model_interface
from src.transpile.optimize_graph import optimize_graph


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


def _load_test_prompt() -> str:
    prompt_file = os.environ.get("CACTUS_LOCAL_MODEL_MATRIX_PROMPT_FILE")
    if prompt_file:
        return Path(prompt_file).read_text().strip()
    return os.environ.get("CACTUS_LOCAL_MODEL_MATRIX_PROMPT", "The capital of France is")


def _artifacts_root() -> Path:
    raw_root = os.environ.get("CACTUS_LOCAL_MODEL_MATRIX_ARTIFACTS_DIR")
    if raw_root:
        return Path(raw_root)
    return Path(__file__).resolve().parent / "artifacts" / "local_model_matrix"


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "model"


def _serialize_json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _serialize_json_compatible(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_compatible(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if hasattr(value, "__dataclass_fields__"):
        return _serialize_json_compatible(asdict(value))
    return repr(value)


def _graph_to_dict(graph) -> dict[str, object]:
    return {
        "meta": _serialize_json_compatible(graph.meta),
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "constants": {
            value_id: _serialize_json_compatible(constant)
            for value_id, constant in graph.constants.items()
        },
        "values": {
            value_id: _serialize_json_compatible(asdict(value))
            for value_id, value in graph.values.items()
        },
        "nodes": [
            _serialize_json_compatible(asdict(graph.nodes[node_id]))
            for node_id in graph.order
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
        prompt = _load_test_prompt()
        self.assertTrue(prompt, "prompt must not be empty")

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
                artifact_dir = _artifacts_root() / _slugify(model_id)

                print(f"\n\n### Model: {model_id}")
                print(f"family={adapter.family} prompt_len={int(input_ids.shape[1])} artifacts={artifact_dir}")

                captured = capture_model(FullModelWrapper(model).eval(), (input_ids,))
                _write_json(
                    artifact_dir / "raw_ir.json",
                    {
                        "model_id": model_id,
                        "family": adapter.family,
                        "prompt": prompt,
                        "snapshot": snapshot,
                        "graph": _graph_to_dict(captured.ir_graph),
                    },
                )

                canonicalize_exported_graph(captured.ir_graph)
                optimize_graph(captured.ir_graph)
                _write_json(
                    artifact_dir / "optimized_ir.json",
                    {
                        "model_id": model_id,
                        "family": adapter.family,
                        "prompt": prompt,
                        "snapshot": snapshot,
                        "graph": _graph_to_dict(captured.ir_graph),
                    },
                )

                tg = transpile_captured(captured)
                tg.set_inputs([input_ids.cpu().numpy()])
                transpiled_logits = tg.execute()[0].numpy().astype(np.float32)

                with torch.no_grad():
                    hf_logits = model(input_ids=input_ids, use_cache=False, return_dict=False)[0].detach().float().cpu().numpy()

                hf_next = int(np.argmax(hf_logits[0, -1]))
                transpiled_next = int(np.argmax(transpiled_logits[0, -1]))
                max_abs_diff = float(np.max(np.abs(hf_logits - transpiled_logits)))
                mean_abs_diff = float(np.mean(np.abs(hf_logits - transpiled_logits)))
                _write_json(
                    artifact_dir / "result.json",
                    {
                        "model_id": model_id,
                        "family": adapter.family,
                        "prompt": prompt,
                        "snapshot": snapshot,
                        "prompt_len": int(input_ids.shape[1]),
                        "hf_next_token_id": hf_next,
                        "transpiled_next_token_id": transpiled_next,
                        "logits_max_abs_diff": max_abs_diff,
                        "logits_mean_abs_diff": mean_abs_diff,
                    },
                )

                print(
                    f"next_token: hf={hf_next} transpiled={transpiled_next} "
                    f"logits_max_abs_diff={max_abs_diff:.6f} logits_mean_abs_diff={mean_abs_diff:.6f}"
                )

                self.assertEqual(transpiled_next, hf_next, model_id)


if __name__ == "__main__":
    unittest.main()
