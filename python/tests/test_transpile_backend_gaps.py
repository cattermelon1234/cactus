from __future__ import annotations

import json
import os
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transpile.capture_pytorch import capture_model
from src.transpile.canonicalize.cleanup import COMPILER_SUPPORTED_OPS
from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.graph_ir import IRValue
from src.transpile.model_adapters import canonicalize_model_interface
from src.transpile.ops import OPS


SUPPORTED_MODEL_IDS = (
    "google/gemma-4-E2B",
    "google/gemma-3-270m-it",
    "google/gemma-2b-it",
    "google/gemma-2-2b",
    "Qwen/Qwen3.5-2B",
)

WRAPPER_POLICIES = {
    "addmm": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically to matmul + add",
    },
    "attention_block": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically to attention + optional gate + output projection",
    },
    "chunk": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically to repeated slice ops",
    },
    "expand": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically by materializing ones and using broadcast multiply",
    },
    "getitem": {
        "generic_wrapper": True,
        "all_cases": False,
        "note": "structural tuple/list projection is handled, but this is not a general tensor indexing kernel",
    },
    "linear": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically to matmul with reshape/bias handling",
    },
    "negate": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically to scalar_multiply(-1)",
    },
    "ones": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "compiler-supported constant materialization",
    },
    "scalar_subtract_reverse": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically to negate + scalar_add",
    },
    "softplus": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically with a stable relu/log/exp composition",
    },
    "split_with_sizes": {
        "generic_wrapper": True,
        "all_cases": True,
        "note": "lowered generically to repeated slice ops",
    },
}

DIRECT_CANONICAL_BACKEND_OPS = {
    op.name
    for op in OPS
    if op.backend_op is not None and op.name not in WRAPPER_POLICIES
}
DIRECT_CANONICAL_BACKEND_OPS.update(COMPILER_SUPPORTED_OPS)
DIRECT_CANONICAL_BACKEND_OPS.update({"gated_deltanet_prefill", "gated_deltanet_decode"})


def _require_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_LOCAL_MODEL_BACKEND_GAP_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_LOCAL_MODEL_BACKEND_GAP_TEST=1 to run the backend-gap report."
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
    return [model_id for model_id in SUPPORTED_MODEL_IDS if _resolve_local_snapshot(model_id) is not None]


def _load_test_prompt() -> str:
    prompt_file = os.environ.get("CACTUS_LOCAL_MODEL_MATRIX_PROMPT_FILE")
    if prompt_file:
        return Path(prompt_file).read_text().strip()
    return os.environ.get("CACTUS_LOCAL_MODEL_MATRIX_PROMPT", "The capital of France is")


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "model"


def _artifacts_root() -> Path:
    raw_root = os.environ.get("CACTUS_LOCAL_MODEL_BACKEND_GAP_ARTIFACTS_DIR")
    if raw_root:
        return Path(raw_root)
    return Path(__file__).resolve().parent / "artifacts" / "local_model_backend_gaps"


def _local_matrix_artifacts_root() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "local_model_matrix"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _deserialize_ir_graph(payload: dict[str, object]) -> IRGraph:
    graph_payload = payload["graph"]
    values = {
        value_id: IRValue(**value_payload)
        for value_id, value_payload in graph_payload["values"].items()
    }
    nodes = {
        node_payload["id"]: IRNode(**node_payload)
        for node_payload in graph_payload["nodes"]
    }
    return IRGraph(
        values=values,
        nodes=nodes,
        order=list(graph_payload["order"]) if "order" in graph_payload else [node["id"] for node in graph_payload["nodes"]],
        inputs=list(graph_payload["inputs"]),
        outputs=list(graph_payload["outputs"]),
        constants=dict(graph_payload["constants"]),
        meta=dict(graph_payload["meta"]),
    )


class FullModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.adapter = canonicalize_model_interface(model, task="causal_lm_logits").module
        self.module = self.adapter

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.module(input_ids)


def _classify_op(op: str) -> dict[str, object]:
    if op in DIRECT_CANONICAL_BACKEND_OPS:
        return {
            "direct_backend": True,
            "generic_wrapper": False,
            "all_cases": True,
            "note": "direct backend/compiler support",
        }
    policy = WRAPPER_POLICIES.get(op)
    if policy is not None:
        return {
            "direct_backend": False,
            **policy,
        }
    return {
        "direct_backend": False,
        "generic_wrapper": False,
        "all_cases": False,
        "note": "no generic lowering path is registered",
    }


def _summarize_graph_backend_gaps(graph) -> dict[str, object]:
    op_counts = Counter(graph.nodes[node_id].op for node_id in graph.order)
    gap_summary: dict[str, object] = {}
    important_missing: list[str] = []

    for op, count in sorted(op_counts.items()):
        classification = _classify_op(op)
        if classification["direct_backend"]:
            continue
        gap_summary[op] = {
            "count": count,
            **classification,
        }
        if not classification["generic_wrapper"] or not classification["all_cases"]:
            important_missing.append(op)

    return {
        "canonical_op_counts": dict(sorted(op_counts.items())),
        "backend_gap_ops": gap_summary,
        "important_missing_ops": important_missing,
    }


class TestTranspileBackendGaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_opt_in()
        cls.local_matrix_dirs = sorted(path for path in _local_matrix_artifacts_root().iterdir() if path.is_dir())
        if not cls.local_matrix_dirs:
            raise unittest.SkipTest("no local model matrix artifacts found for backend-gap analysis")

        try:
            AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        except unittest.SkipTest:
            cls.AutoModelForCausalLM = None
            cls.AutoTokenizer = None
            cls.available_model_ids = []
        else:
            cls.AutoModelForCausalLM = AutoModelForCausalLM
            cls.AutoTokenizer = AutoTokenizer
            cls.available_model_ids = _discover_local_models()

    def test_prefusion_backend_gap_report(self) -> None:
        prompt = _load_test_prompt()
        self.assertTrue(prompt, "prompt must not be empty")

        aggregate_counts: Counter[str] = Counter()
        aggregate_important: set[str] = set()

        for model_dir in self.local_matrix_dirs:
            with self.subTest(model_dir=model_dir.name):
                raw_ir_path = model_dir / "raw_ir.json"
                payload = json.loads(raw_ir_path.read_text())
                model_id = str(payload.get("model_id", model_dir.name))
                graph = _deserialize_ir_graph(payload)
                canonicalize_exported_graph(graph)
                summary = _summarize_graph_backend_gaps(graph)

                artifact_path = _artifacts_root() / model_dir.name / "prefusion_backend_gap_report.json"
                _write_json(
                    artifact_path,
                    {
                        "model_id": model_id,
                        "prompt": payload.get("prompt", prompt),
                        "snapshot": payload.get("snapshot"),
                        "source_artifact": str(raw_ir_path),
                        **summary,
                    },
                )

                gap_ops = summary["backend_gap_ops"]
                important_missing = summary["important_missing_ops"]
                for op, details in gap_ops.items():
                    aggregate_counts[op] += int(details["count"])
                aggregate_important.update(important_missing)

                formatted = ", ".join(
                    f"{op}={details['count']} [{details['note']}]"
                    for op, details in gap_ops.items()
                ) or "none"
                print(f"\nmodel={model_id}")
                print(f"prefusion_backend_gap_ops={formatted}")
                print(f"important_missing_ops={sorted(important_missing)}")

                self.assertTrue(artifact_path.exists())

        print("\naggregate_prefusion_backend_gap_ops=" + ", ".join(
            f"{op}={count}" for op, count in sorted(aggregate_counts.items())
        ))
        print(f"aggregate_important_missing_ops={sorted(aggregate_important)}")
