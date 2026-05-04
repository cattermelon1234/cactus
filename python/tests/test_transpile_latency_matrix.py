from __future__ import annotations

import json
import inspect
import os
import re
import statistics
import sys
import time
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transpile import model_adapters as _model_adapters
from src.cactus import cactus_destroy
from src.cactus import cactus_init
from src.cactus import cactus_prefill
from src.cactus import cactus_reset
from src.downloads import get_weights_dir
from src.transpile.capture_pytorch import capture_model
from src.transpile.lower import transpile_captured
from src.transpile.model_adapters import canonicalize_model_interface


SUPPORTED_MODEL_IDS = (
    "google/gemma-4-E2B",
    "google/gemma-3-270m-it",
    "google/gemma-2b-it",
    "google/gemma-2-2b",
    "Qwen/Qwen3.5-2B",
)


def _patch_model_adapter_mask_builder_for_latency_test() -> None:
    original = _model_adapters._call_mask_builder

    def _patched(builder, **kwargs):
        signature = inspect.signature(builder)
        if "input_embeds" in signature.parameters and "inputs_embeds" in kwargs:
            kwargs = dict(kwargs)
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        if "cache_position" in signature.parameters and "cache_position" not in kwargs:
            input_embeds = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
            if isinstance(input_embeds, torch.Tensor):
                kwargs = dict(kwargs)
                kwargs["cache_position"] = torch.arange(
                    input_embeds.shape[1],
                    device=input_embeds.device,
                )
        return builder(**kwargs)

    _model_adapters._call_mask_builder = _patched


_patch_model_adapter_mask_builder_for_latency_test()


def _require_opt_in() -> None:
    if os.environ.get("CACTUS_RUN_LOCAL_MODEL_LATENCY_TEST") != "1":
        raise unittest.SkipTest(
            "Set CACTUS_RUN_LOCAL_MODEL_LATENCY_TEST=1 to run the native-vs-transpiled latency benchmark."
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


def _resolve_native_weights_dir(model_id: str) -> Path | None:
    weights_dir = get_weights_dir(model_id)
    if (weights_dir / "config.txt").exists():
        return weights_dir
    return None


def _load_test_prompt() -> str:
    prompt_file = os.environ.get("CACTUS_LOCAL_MODEL_MATRIX_PROMPT_FILE")
    if prompt_file:
        return Path(prompt_file).read_text().strip()
    return os.environ.get("CACTUS_LOCAL_MODEL_MATRIX_PROMPT", "The capital of France is")


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "model"


def _artifacts_root() -> Path:
    raw_root = os.environ.get("CACTUS_LOCAL_MODEL_LATENCY_ARTIFACTS_DIR")
    if raw_root:
        return Path(raw_root)
    return Path(__file__).resolve().parent / "artifacts" / "local_model_latency_matrix"


def _should_enforce_slowdown_threshold() -> bool:
    if os.environ.get("CACTUS_LOCAL_MODEL_LATENCY_ENFORCE_THRESHOLD") == "1":
        return True
    return "CACTUS_LOCAL_MODEL_LATENCY_MAX_SLOWDOWN" in os.environ


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


def _build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _encode_messages(tokenizer, messages: list[dict[str, str]]) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if isinstance(encoded, torch.Tensor):
                return encoded
            if isinstance(encoded, dict):
                input_ids = encoded.get("input_ids")
                if isinstance(input_ids, torch.Tensor):
                    return input_ids
        except Exception:
            pass
    return tokenizer(messages[-1]["content"], return_tensors="pt")["input_ids"]


def _benchmark_transpiled_graph(tg, input_ids: torch.Tensor, iterations: int) -> dict[str, object]:
    timings_ms: list[float] = []
    tg.set_inputs([input_ids.cpu().numpy()])
    warmup = tg.execute()[0].numpy()
    assert warmup.shape[1] == input_ids.shape[1]

    for _ in range(iterations):
        tg.set_inputs([input_ids.cpu().numpy()])
        start = time.perf_counter()
        output = tg.execute()[0].numpy()
        end = time.perf_counter()
        assert output.shape[1] == input_ids.shape[1]
        timings_ms.append((end - start) * 1000.0)

    mean_ms = statistics.mean(timings_ms)
    return {
        "iterations": iterations,
        "timings_ms": timings_ms,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings_ms),
        "tokens": int(input_ids.shape[1]),
        "tokens_per_second": (int(input_ids.shape[1]) * 1000.0) / mean_ms if mean_ms > 0 else 0.0,
    }


def _benchmark_native_prefill(snapshot: str, messages_json: str, iterations: int) -> dict[str, object]:
    handle = cactus_init(snapshot, None, False)
    if not handle:
        raise RuntimeError(f"cactus_init failed for snapshot={snapshot}")

    timings_ms: list[float] = []
    reported_total_ms: list[float] = []
    reported_tps: list[float] = []
    prefill_tokens: list[int] = []
    try:
        cactus_reset(handle)
        warmup = json.loads(cactus_prefill(handle, messages_json, None, None))
        if not warmup.get("success", False):
            raise RuntimeError(f"native prefill warmup failed: {warmup}")

        for _ in range(iterations):
            cactus_reset(handle)
            start = time.perf_counter()
            raw = cactus_prefill(handle, messages_json, None, None)
            end = time.perf_counter()
            response = json.loads(raw)
            if not response.get("success", False):
                raise RuntimeError(f"native prefill failed: {response}")
            timings_ms.append((end - start) * 1000.0)
            reported_total_ms.append(float(response["total_time_ms"]))
            reported_tps.append(float(response["prefill_tps"]))
            prefill_tokens.append(int(response["prefill_tokens"]))
    finally:
        cactus_destroy(handle)

    mean_wall_ms = statistics.mean(timings_ms)
    mean_prefill_tokens = statistics.mean(prefill_tokens)
    return {
        "iterations": iterations,
        "timings_ms": timings_ms,
        "mean_ms": mean_wall_ms,
        "median_ms": statistics.median(timings_ms),
        "reported_total_time_ms": reported_total_ms,
        "reported_prefill_tps": reported_tps,
        "prefill_tokens": prefill_tokens,
        "mean_prefill_tokens": mean_prefill_tokens,
        "tokens_per_second_wall": (mean_prefill_tokens * 1000.0) / mean_wall_ms if mean_wall_ms > 0 else 0.0,
    }


class TestTranspileLatencyMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_opt_in()
        AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        cls.AutoModelForCausalLM = AutoModelForCausalLM
        cls.AutoTokenizer = AutoTokenizer
        cls.available_model_ids = [
            model_id
            for model_id in _discover_local_models()
            if _resolve_native_weights_dir(model_id) is not None
        ]
        if not cls.available_model_ids:
            raise unittest.SkipTest("no supported local Hugging Face snapshots found for latency benchmarking")

    def test_native_models_and_transpiled_graphs_have_similar_prefill_latency(self) -> None:
        prompt = _load_test_prompt()
        self.assertTrue(prompt, "prompt must not be empty")
        iterations = int(os.environ.get("CACTUS_LOCAL_MODEL_LATENCY_ITERS", "3"))
        slowdown_threshold = float(os.environ.get("CACTUS_LOCAL_MODEL_LATENCY_MAX_SLOWDOWN", "3.0"))
        enforce_threshold = _should_enforce_slowdown_threshold()

        for model_id in self.available_model_ids:
            with self.subTest(model_id=model_id):
                snapshot = _resolve_local_snapshot(model_id)
                self.assertIsNotNone(snapshot)
                native_weights_dir = _resolve_native_weights_dir(model_id)
                self.assertIsNotNone(native_weights_dir)

                tokenizer = self.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
                messages = _build_messages(prompt)
                messages_json = json.dumps(messages)
                input_ids = _encode_messages(tokenizer, messages)

                try:
                    model = self.AutoModelForCausalLM.from_pretrained(
                        snapshot,
                        torch_dtype=torch.float16,
                        device_map=None,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                    ).eval()
                except Exception as exc:
                    raise unittest.SkipTest(f"transformers cannot load {model_id}: {exc}") from exc

                try:
                    captured = capture_model(FullModelWrapper(model).eval(), (input_ids,))
                except Exception as exc:
                    raise unittest.SkipTest(f"torch.export capture failed for {model_id}: {exc}") from exc
                tg = transpile_captured(captured)

                transpiled_metrics = _benchmark_transpiled_graph(tg, input_ids, iterations)
                try:
                    native_metrics = _benchmark_native_prefill(str(native_weights_dir), messages_json, iterations)
                except Exception as exc:
                    raise unittest.SkipTest(f"native prefill failed for {model_id}: {exc}") from exc

                transpiled_tps = float(transpiled_metrics["tokens_per_second"])
                native_tps = float(native_metrics["tokens_per_second_wall"])
                slowdown = (native_tps / transpiled_tps) if transpiled_tps > 0 else float("inf")
                artifact_path = _artifacts_root() / _slugify(model_id) / "latency_result.json"

                payload = {
                    "model_id": model_id,
                    "prompt": prompt,
                    "snapshot": snapshot,
                    "native_weights_dir": str(native_weights_dir),
                    "input_tokens_transpiled": int(input_ids.shape[1]),
                    "native": native_metrics,
                    "transpiled": transpiled_metrics,
                    "slowdown_native_over_transpiled": slowdown,
                    "slowdown_threshold": slowdown_threshold,
                    "threshold_enforced": enforce_threshold,
                }
                _write_json(artifact_path, payload)

                print(f"\nmodel={model_id}")
                print(
                    "native_mean_ms="
                    f"{native_metrics['mean_ms']:.2f} native_tps={native_tps:.2f} "
                    f"transpiled_mean_ms={transpiled_metrics['mean_ms']:.2f} "
                    f"transpiled_tps={transpiled_tps:.2f} slowdown={slowdown:.2f}x"
                )

                self.assertTrue(artifact_path.exists())
                self.assertGreater(native_tps, 0.0)
                self.assertGreater(transpiled_tps, 0.0)
                if enforce_threshold:
                    self.assertLessEqual(
                        slowdown,
                        slowdown_threshold,
                        msg=(
                            f"{model_id} native/transpiled slowdown {slowdown:.2f}x exceeds "
                            f"threshold {slowdown_threshold:.2f}x"
                        ),
                    )
