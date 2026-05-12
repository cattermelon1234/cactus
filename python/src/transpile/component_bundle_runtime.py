from __future__ import annotations

from dataclasses import dataclass
import ctypes
import json
from collections.abc import Mapping
from pathlib import Path
import re
import struct
import time
from typing import Any

import numpy as np
import torch

from src.cactus import _lib
from src.cactus import cactus_node_t
from src.graph import Graph
from src.graph import Tensor
from src.tensor_io import CACTUS_MAGIC
from src.tensor_io import FLAG_INTERLEAVED
from src.tensor_io import align_offset
from src.transpile.audio_preprocess import generic_log_mel_features as _generic_log_mel_features
from src.transpile.audio_preprocess import load_audio_waveform as _load_audio_waveform
from src.transpile.audio_preprocess import prepare_cactus_audio_features
from src.transpile.parakeet_tdt_local import greedy_decode_parakeet_tdt_token_ids
from src.transpile.parakeet_tdt_local import load_parakeet_tdt_config
from src.transpile.parakeet_tdt_local import prepare_parakeet_tdt_audio_features


_HEADER_SIZE = 84
_PRECISION_TO_DTYPE = {
    Graph.INT8: np.int8,
    Graph.FP16: np.float16,
    Graph.FP32: np.float32,
    Graph.INT4: np.uint8,
}


@dataclass
class LoadedTensorFile:
    path: Path
    precision: int
    shape: tuple[int, ...]
    data: np.memmap
    scales: np.memmap | None
    group_size: int
    num_groups: int
    is_interleaved: bool
    original_n: int


@dataclass
class LoadedComponentGraph:
    component: str
    graph: Graph
    runtime_inputs: list[Tensor]
    outputs: list[Tensor]
    bound_constant_bindings: list[dict[str, object]]
    bound_tensor_files: list[object]

    def set_input(self, index: int, data: Any, *, dtype: int | None = None) -> None:
        if index < 0 or index >= len(self.runtime_inputs):
            raise IndexError(
                f"runtime input index out of range: {index} (have {len(self.runtime_inputs)})"
            )
        self.graph.set_input(self.runtime_inputs[index], data, dtype=dtype)

    def set_inputs(self, inputs: list[Any] | tuple[Any, ...]) -> None:
        if len(inputs) != len(self.runtime_inputs):
            raise ValueError(
                f"expected {len(self.runtime_inputs)} runtime inputs, got {len(inputs)}"
            )
        for index, value in enumerate(inputs):
            self.set_input(index, value)

    def execute(self) -> list[Tensor]:
        self.graph.execute()
        return self.outputs


def load_component_bundle_manifest(bundle_dir_or_manifest: str | Path) -> tuple[Path, dict[str, object]]:
    path = Path(bundle_dir_or_manifest).expanduser().resolve()
    if path.is_dir():
        candidate = path / "manifest.json" if path.name == "components" else path / "components" / "manifest.json"
        if not candidate.exists():
            candidate = path / "manifest.json"
        path = candidate
    if not path.exists():
        raise FileNotFoundError(f"component bundle manifest not found: {path}")
    manifest = json.loads(path.read_text())
    bundle_root = path.parent.parent if path.parent.name == "components" else path.parent
    return bundle_root, manifest


def load_saved_component_graph(
    *,
    bundle_root: str | Path,
    component_entry: dict[str, object],
    weights_dir: str | Path | None = None,
) -> LoadedComponentGraph:
    root = Path(bundle_root).expanduser().resolve()
    graph_relpath = component_entry.get("graph")
    if not isinstance(graph_relpath, str) or not graph_relpath:
        raise ValueError(f"component entry is missing graph path: {component_entry}")

    graph_path = (root / graph_relpath).resolve()
    graph = Graph.load(graph_path)

    runtime_inputs = [
        graph._tensor_from_node(int(node_id))
        for node_id in component_entry.get("runtime_input_node_ids", [])
    ]
    outputs = [
        graph._tensor_from_node(int(node_id))
        for node_id in component_entry.get("output_node_ids", [])
    ]
    bound_constant_bindings = list(component_entry.get("bound_constant_bindings") or [])
    bound_tensor_files = _rebind_bound_constants(
        graph=graph,
        bundle_root=root,
        bindings=bound_constant_bindings,
        weights_dir=weights_dir,
    )
    return LoadedComponentGraph(
        component=str(component_entry.get("component", "unknown")),
        graph=graph,
        runtime_inputs=runtime_inputs,
        outputs=outputs,
        bound_constant_bindings=bound_constant_bindings,
        bound_tensor_files=bound_tensor_files,
    )


def load_saved_component_graphs(
    bundle_dir_or_manifest: str | Path,
    *,
    weights_dir: str | Path | None = None,
) -> tuple[dict[str, LoadedComponentGraph], dict[str, object]]:
    bundle_root, manifest = load_component_bundle_manifest(bundle_dir_or_manifest)
    loaded: dict[str, LoadedComponentGraph] = {}
    for component_entry in manifest.get("components", []):
        if not isinstance(component_entry, dict):
            continue
        component_name = str(component_entry.get("component", "")).strip()
        if not component_name:
            continue
        loaded[component_name] = load_saved_component_graph(
            bundle_root=bundle_root,
            component_entry=component_entry,
            weights_dir=weights_dir,
        )
    return loaded, manifest


def run_transpiled_bundle(
    bundle_dir_or_manifest: str | Path,
    *,
    audio_file: str | Path | None = None,
    prompt: str | None = None,
    input_ids: str | list[int] | tuple[int, ...] | None = None,
    weights_dir: str | Path | None = None,
    torch_dtype: torch.dtype = torch.float16,
) -> dict[str, object]:
    component_graphs, manifest = load_saved_component_graphs(
        bundle_dir_or_manifest,
        weights_dir=weights_dir,
    )
    family = str(manifest.get("family", "") or "")
    task = str(manifest.get("task", "") or "")
    if family == "parakeet_tdt" and task == "tdt_transcription":
        if audio_file is None:
            raise ValueError("audio_file is required for Parakeet TDT component bundles")
        return _run_parakeet_tdt_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
        )
    if task in {"causal_lm_logits", "multimodal_causal_lm_logits"}:
        return _run_causal_lm_logits_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            prompt=prompt,
            input_ids=input_ids,
        )
    if task == "encoder_hidden_states":
        if audio_file is None:
            inputs_meta = manifest.get("inputs")
            if isinstance(inputs_meta, dict):
                stored_audio = inputs_meta.get("audio_file")
                if isinstance(stored_audio, str) and stored_audio:
                    audio_file = stored_audio
        if audio_file is None:
            raise ValueError("audio_file is required for encoder_hidden_states bundles")
        return _run_encoder_hidden_states_bundle(
            component_graphs=component_graphs,
            manifest=manifest,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
        )
    raise NotImplementedError(
        f"saved transpiled bundle execution is not implemented for family={family!r} task={task!r}"
    )


def execute_loaded_component_pipeline(
    components: list[LoadedComponentGraph],
    *,
    initial_store: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    store: dict[str, np.ndarray] = {}
    for key, value in initial_store.items():
        store[key] = _to_numpy(value)

    outputs_by_component: dict[str, list[np.ndarray]] = {}
    for component in components:
        runtime_inputs = []
        input_names = component_input_names(component)
        for input_name in input_names:
            if input_name not in store:
                raise KeyError(
                    f"component {component.component} is missing pipeline input {input_name!r}"
                )
            runtime_inputs.append(store[input_name])
        component.set_inputs(runtime_inputs)
        raw_outputs = component.execute()
        numpy_outputs = [output.numpy().copy() for output in raw_outputs]
        output_names = component_output_names(component)
        if len(numpy_outputs) != len(output_names):
            raise ValueError(
                f"component {component.component} produced {len(numpy_outputs)} outputs, "
                f"expected {len(output_names)}"
            )
        for output_name, value in zip(output_names, numpy_outputs, strict=True):
            store[output_name] = value
        outputs_by_component[component.component] = numpy_outputs
    return store, outputs_by_component


def component_input_names(component: LoadedComponentGraph) -> tuple[str, ...]:
    return tuple(str(value) for value in getattr(component, "_input_names", ()))


def component_output_names(component: LoadedComponentGraph) -> tuple[str, ...]:
    return tuple(str(value) for value in getattr(component, "_output_names", ()))


def _run_parakeet_tdt_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    audio_file: str | Path,
    torch_dtype: torch.dtype,
) -> dict[str, object]:
    if "audio_encoder" not in component_graphs or "decoder" not in component_graphs:
        raise ValueError("Parakeet TDT component bundle must include audio_encoder and decoder graphs")

    inputs_meta = manifest.get("inputs") or {}
    input_shapes = inputs_meta.get("input_shapes") if isinstance(inputs_meta, dict) else {}
    if not isinstance(input_shapes, dict):
        input_shapes = {}
    expected_shape = input_shapes.get("input_features")
    if not (isinstance(expected_shape, list) and len(expected_shape) == 3):
        raise ValueError("Parakeet TDT bundle manifest is missing inputs.input_shapes.input_features")

    model_source = str(manifest.get("model_source", "") or "")
    config = load_parakeet_tdt_config(model_source)
    preprocess_start = time.perf_counter()
    input_features, active_frames = prepare_parakeet_tdt_audio_features(
        audio_file=audio_file,
        expected_frames=int(expected_shape[1]),
        expected_mels=int(expected_shape[2]),
        torch_dtype=torch_dtype,
    )
    preprocess_end = time.perf_counter()

    _attach_component_io_names(manifest, component_graphs)
    encoder_start = time.perf_counter()
    store, _ = execute_loaded_component_pipeline(
        [component_graphs["audio_encoder"]],
        initial_store={"input_features": input_features},
    )
    encoder_end = time.perf_counter()
    encoder_hidden_states = np.asarray(store["encoder_hidden_states"])
    batch_size = int(encoder_hidden_states.shape[0])
    if batch_size != 1:
        raise ValueError("Parakeet TDT saved bundle runtime currently expects batch size 1")

    state_dtype = np.float16 if torch_dtype == torch.float16 else np.float32
    initial_states: list[np.ndarray] = []
    for _ in range(config.predictor_num_layers):
        state_shape = (batch_size, config.predictor_hidden_dim)
        initial_states.append(np.zeros(state_shape, dtype=state_dtype))
        initial_states.append(np.zeros(state_shape, dtype=state_dtype))

    decoder_component = component_graphs["decoder"]
    decoder_steps = 0

    def _step(
        frame: np.ndarray,
        token_id: int,
        state_values: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        nonlocal decoder_steps
        decoder_steps += 1
        input_store: dict[str, object] = {
            "encoder_frame": np.ascontiguousarray(frame),
            "token_ids": np.full((batch_size,), token_id, dtype=np.int64),
        }
        for index in range(config.predictor_num_layers):
            input_store[f"state_h_{index}"] = np.ascontiguousarray(state_values[index * 2])
            input_store[f"state_c_{index}"] = np.ascontiguousarray(state_values[index * 2 + 1])

        runtime_inputs = [input_store[name] for name in component_input_names(decoder_component)]
        decoder_component.set_inputs(runtime_inputs)
        outputs = decoder_component.execute()
        logits = outputs[0].numpy().astype(np.float32, copy=False)
        next_states = tuple(output.numpy() for output in outputs[1:])
        return logits, next_states

    decoder_start = time.perf_counter()
    emitted = greedy_decode_parakeet_tdt_token_ids(
        config=config,
        encoder_hidden_states=encoder_hidden_states,
        initial_states=tuple(initial_states),
        step=_step,
    )
    decoder_end = time.perf_counter()
    total_end = decoder_end

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "audio_file": str(Path(audio_file).expanduser().resolve()),
        "preprocess_ms": (preprocess_end - preprocess_start) * 1000.0,
        "encoder_ms": (encoder_end - encoder_start) * 1000.0,
        "decoder_ms": (decoder_end - decoder_start) * 1000.0,
        "total_ms": (total_end - preprocess_start) * 1000.0,
        "decoder_steps": decoder_steps,
        "active_feature_frames": active_frames,
        "token_ids": emitted,
        "transcript": _decode_parakeet_tdt_token_ids(config.vocabulary, emitted),
        "encoder_hidden_shape": list(encoder_hidden_states.shape),
        "component_order": list(manifest.get("component_order", [])),
    }


def _run_causal_lm_logits_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    prompt: str | None,
    input_ids: str | list[int] | tuple[int, ...] | None,
) -> dict[str, object]:
    if "decoder" not in component_graphs:
        raise ValueError("causal LM component bundle must include a decoder graph")

    token_ids, tokenizer = _resolve_causal_lm_input_ids(
        manifest=manifest,
        prompt=prompt,
        input_ids=input_ids,
    )
    if not token_ids:
        raise ValueError("causal LM bundle input token ids are empty")

    _attach_component_io_names(manifest, component_graphs)
    decoder = component_graphs["decoder"]
    runtime_inputs = component_input_names(decoder)
    if runtime_inputs and runtime_inputs != ("input_ids",):
        raise ValueError(
            "causal LM bundle runner currently expects decoder logical input ('input_ids',), "
            f"got {runtime_inputs!r}"
        )

    input_array = np.asarray([token_ids], dtype=np.int64)
    start = time.perf_counter()
    decoder.set_inputs([input_array])
    outputs = decoder.execute()
    end = time.perf_counter()
    if not outputs:
        raise RuntimeError("causal LM decoder graph produced no outputs")

    logits = outputs[0].numpy()
    if logits.ndim != 3:
        raise RuntimeError(f"expected logits with shape [batch, seq, vocab], got {list(logits.shape)}")
    next_token_id = int(np.argmax(logits[0, -1]))
    decoded = None
    if tokenizer is None:
        try:
            tokenizer = _load_bundle_tokenizer(manifest)
        except Exception:
            tokenizer = None
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode([next_token_id])

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "component_order": list(manifest.get("component_order", [])),
        "input_ids": token_ids,
        "input_shape": list(input_array.shape),
        "output_shape": list(logits.shape),
        "decoder_ms": (end - start) * 1000.0,
        "total_ms": (end - start) * 1000.0,
        "next_token_id": next_token_id,
        "next_token": decoded,
    }


def _run_encoder_hidden_states_bundle(
    *,
    component_graphs: dict[str, LoadedComponentGraph],
    manifest: dict[str, object],
    audio_file: str | Path,
    torch_dtype: torch.dtype,
) -> dict[str, object]:
    component_name = "audio_encoder" if "audio_encoder" in component_graphs else "encoder"
    if component_name not in component_graphs:
        raise ValueError("encoder_hidden_states bundle must include an audio_encoder or encoder graph")

    inputs_meta = manifest.get("inputs") or {}
    input_shapes = inputs_meta.get("input_shapes") if isinstance(inputs_meta, dict) else {}
    if not isinstance(input_shapes, dict):
        input_shapes = {}
    expected_shape = input_shapes.get("input_features")
    if not (isinstance(expected_shape, list) and len(expected_shape) == 3):
        raise ValueError("encoder bundle manifest is missing inputs.input_shapes.input_features")

    preprocess_start = time.perf_counter()
    input_features, active_frames = _prepare_generic_audio_encoder_features(
        audio_file=audio_file,
        manifest=manifest,
        expected_shape=expected_shape,
        torch_dtype=torch_dtype,
    )
    preprocess_end = time.perf_counter()

    _attach_component_io_names(manifest, component_graphs)
    encoder = component_graphs[component_name]
    encoder_start = time.perf_counter()
    encoder.set_inputs([input_features])
    outputs = encoder.execute()
    encoder_end = time.perf_counter()
    if not outputs:
        raise RuntimeError("encoder graph produced no outputs")
    hidden = outputs[0].numpy()

    return {
        "bundle_model_id": str(manifest.get("model_id", "") or ""),
        "family": str(manifest.get("family", "") or ""),
        "task": str(manifest.get("task", "") or ""),
        "audio_file": str(Path(audio_file).expanduser().resolve()),
        "component_order": list(manifest.get("component_order", [])),
        "active_feature_frames": active_frames,
        "input_shape": list(input_features.shape),
        "encoder_hidden_shape": list(hidden.shape),
        "preprocess_ms": (preprocess_end - preprocess_start) * 1000.0,
        "encoder_ms": (encoder_end - encoder_start) * 1000.0,
        "total_ms": (encoder_end - preprocess_start) * 1000.0,
    }


def _prepare_generic_audio_encoder_features(
    *,
    audio_file: str | Path,
    manifest: dict[str, object],
    expected_shape: list[object],
    torch_dtype: torch.dtype,
) -> tuple[np.ndarray, int]:
    family = str(manifest.get("family", "") or "")
    inputs_meta = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    sample_rate = int(inputs_meta.get("sample_rate", 16000) if isinstance(inputs_meta, dict) else 16000)
    batch = int(expected_shape[0])
    if batch != 1:
        raise ValueError("saved audio encoder bundle runtime currently expects batch size 1")

    if family == "whisper":
        expected_mels = int(expected_shape[1])
        expected_frames = int(expected_shape[2])
        try:
            features, active_frames = prepare_cactus_audio_features(
                audio_file,
                model_type="whisper",
                expected_frames=expected_frames,
                expected_mels=expected_mels,
                torch_dtype=torch_dtype,
                layout="mels_frames",
            )
            return np.ascontiguousarray(features.detach().cpu().numpy()), active_frames
        except Exception:
            pass
    else:
        expected_frames = int(expected_shape[1])
        expected_mels = int(expected_shape[2])

    waveform = _load_audio_waveform(audio_file, target_sample_rate=sample_rate)
    features, feature_length = _generic_log_mel_features(
        waveform,
        sample_rate=sample_rate,
        num_mels=expected_mels,
        n_fft=400,
        hop_length=160,
        frame_length=400,
        preemphasis=None,
    )
    active_frames = min(feature_length, expected_frames)
    features = features[:active_frames, :]
    if expected_frames > active_frames:
        features = np.pad(features, ((0, expected_frames - active_frames), (0, 0)), mode="constant")
    if family == "whisper":
        features = np.ascontiguousarray(features.T)
    features = np.ascontiguousarray(features, dtype=np.float16 if torch_dtype == torch.float16 else np.float32)
    return np.expand_dims(features, axis=0), active_frames


def _resolve_causal_lm_input_ids(
    *,
    manifest: dict[str, object],
    prompt: str | None,
    input_ids: str | list[int] | tuple[int, ...] | None,
) -> tuple[list[int], object | None]:
    if input_ids is not None:
        return _parse_input_ids(input_ids), None

    if prompt is None:
        inputs_meta = manifest.get("inputs")
        if isinstance(inputs_meta, dict):
            stored_ids = inputs_meta.get("input_ids")
            parsed = _parse_nested_manifest_input_ids(stored_ids)
            if parsed:
                return parsed, None
            stored_prompt = inputs_meta.get("prompt")
            if isinstance(stored_prompt, str) and stored_prompt:
                prompt = stored_prompt
    if not prompt:
        raise ValueError("provide --input-ids or --prompt for causal LM component bundles")

    tokenizer = _load_bundle_tokenizer(manifest)
    token_ids = _tokenize_bundle_prompt(tokenizer, prompt)
    return token_ids, tokenizer


def _parse_input_ids(input_ids: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(input_ids, str):
        parsed = [int(part.strip()) for part in input_ids.split(",") if part.strip()]
    else:
        parsed = [int(value) for value in input_ids]
    if not parsed:
        raise ValueError("input_ids was provided but no token ids were parsed")
    return parsed


def _parse_nested_manifest_input_ids(value: object) -> list[int] | None:
    if isinstance(value, list) and value:
        if all(isinstance(item, int) for item in value):
            return [int(item) for item in value]
        first = value[0]
        if isinstance(first, list) and all(isinstance(item, int) for item in first):
            return [int(item) for item in first]
    return None


def _patch_missing_lzma_backport() -> str | None:
    try:
        import importlib.util
        import sys

        if importlib.util.find_spec("_lzma") is not None:
            return None
        if importlib.util.find_spec("backports.lzma") is None:
            return None
        import backports.lzma as backports_lzma  # type: ignore

        sys.modules.setdefault("lzma", backports_lzma)
        return "using backports.lzma because this Python build is missing _lzma"
    except Exception:
        return None


def _load_bundle_tokenizer(manifest: dict[str, object]):
    _patch_missing_lzma_backport()
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"transformers is required to tokenize --prompt: {exc}") from exc

    model_source = str(manifest.get("model_source", "") or manifest.get("model_id", "") or "")
    if not model_source:
        raise ValueError("bundle manifest is missing model_source/model_id; provide --input-ids instead")
    return AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=True,
        trust_remote_code=True,
    )


def _tokenize_bundle_prompt(tokenizer: object, prompt: str) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            encoded = apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(value) for value in ids]
        except Exception:
            pass

    encoded = tokenizer(prompt, return_tensors=None)  # type: ignore[operator]
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _attach_component_io_names(
    manifest: dict[str, object],
    component_graphs: dict[str, LoadedComponentGraph],
) -> None:
    family = str(manifest.get("family", "") or "")
    task = str(manifest.get("task", "") or "")
    config = None
    if family == "parakeet_tdt" and task == "tdt_transcription":
        config = load_parakeet_tdt_config(str(manifest.get("model_source", "") or ""))

    for component_entry in manifest.get("components", []):
        if not isinstance(component_entry, dict):
            continue
        name = str(component_entry.get("component", "")).strip()
        if not name or name not in component_graphs:
            continue
        component = component_graphs[name]
        logical_inputs = tuple(str(value) for value in component_entry.get("logical_inputs", []))
        logical_outputs = tuple(str(value) for value in component_entry.get("logical_outputs", []))
        if not logical_inputs or not logical_outputs:
            logical_inputs, logical_outputs = _infer_legacy_component_io_names(
                family=family,
                task=task,
                component_name=name,
                config=config,
            )
        component._input_names = logical_inputs
        component._output_names = logical_outputs


def _infer_legacy_component_io_names(
    *,
    family: str,
    task: str,
    component_name: str,
    config,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if task in {"causal_lm_logits", "multimodal_causal_lm_logits"} and component_name == "decoder":
        return ("input_ids",), ("logits",)
    if task == "encoder_hidden_states" and component_name in {"audio_encoder", "encoder"}:
        return ("input_features",), ("encoder_hidden_states",)
    if family == "parakeet_tdt" and task == "tdt_transcription":
        if component_name == "audio_encoder":
            return ("input_features",), ("encoder_hidden_states",)
        if component_name == "decoder":
            predictor_layers = 0 if config is None else int(config.predictor_num_layers)
            state_inputs: list[str] = []
            state_outputs: list[str] = []
            for index in range(predictor_layers):
                state_inputs.extend((f"state_h_{index}", f"state_c_{index}"))
                state_outputs.extend((f"state_h_{index}", f"state_c_{index}"))
            return (
                ("encoder_frame", "token_ids", *state_inputs),
                ("step_logits", *state_outputs),
            )
    raise ValueError(
        f"component bundle manifest is missing logical IO names for family={family!r} task={task!r} component={component_name!r}"
    )


def _rebind_bound_constants(
    *,
    graph: Graph,
    bundle_root: Path,
    bindings: list[dict[str, object]],
    weights_dir: str | Path | None,
) -> list[object]:
    loaded: list[object] = []
    for binding in bindings:
        node_id = int(binding["node_id"])
        raw_path = str(binding["path"])
        tensor_path = _resolve_bound_tensor_path(
            raw_path,
            bundle_root=bundle_root,
            weights_dir=weights_dir,
        )
        tensor = graph._tensor_from_node(node_id)
        binding_format = str(binding.get("format", "tensor_io") or "tensor_io")
        if binding_format == "npy":
            constant_array = np.load(tensor_path, mmap_mode="r")
            precision = int(binding.get("precision", tensor.dtype))
            graph.set_external_input(tensor, int(constant_array.ctypes.data), dtype=precision)
            loaded.append(constant_array)
            continue

        tensor_file = _open_cactus_tensor_file(tensor_path)
        graph.set_external_input(tensor, int(tensor_file.data.ctypes.data), dtype=tensor_file.precision)
        if tensor_file.scales is not None and tensor_file.group_size > 0:
            rc = _lib.cactus_graph_set_grouped_scales(
                graph.h,
                cactus_node_t(tensor.id),
                int(tensor_file.group_size),
                int(tensor_file.num_groups),
                tensor_file.scales.ctypes.data_as(ctypes.c_void_p),
            )
            if rc != 0:
                raise RuntimeError("graph_set_grouped_scales failed")
        if tensor_file.is_interleaved:
            graph.set_interleaved(tensor, True, tensor_file.original_n)
        loaded.append(tensor_file)
    return loaded


def _resolve_bound_tensor_path(
    raw_path: str,
    *,
    bundle_root: Path,
    weights_dir: str | Path | None,
) -> Path:
    candidates: list[Path] = []
    explicit = Path(raw_path).expanduser()
    if explicit.is_absolute():
        candidates.append(explicit)
    else:
        candidates.append((bundle_root / explicit).resolve())
        candidates.append(explicit.resolve())

    if weights_dir is not None:
        weights_root = Path(weights_dir).expanduser().resolve()
        raw_parts = Path(raw_path).parts
        for index in range(len(raw_parts)):
            candidates.append(weights_root.joinpath(*raw_parts[index:]))
        candidates.append(weights_root / Path(raw_path).name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"could not resolve bound tensor file {raw_path!r} from bundle_root={bundle_root}"
        + ("" if weights_dir is None else f" weights_dir={Path(weights_dir).expanduser().resolve()}")
    )


def _open_cactus_tensor_file(path: str | Path) -> LoadedTensorFile:
    tensor_path = Path(path).expanduser().resolve()
    with tensor_path.open("rb") as handle:
        header = handle.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE:
        raise RuntimeError(f"tensor file is too small for a Cactus header: {tensor_path}")

    magic = header[:4]
    if magic != CACTUS_MAGIC:
        raise RuntimeError(f"tensor file is missing the CACT header: {tensor_path}")

    flags = struct.unpack_from("<I", header, 4)[0]
    alignment = max(1, int(struct.unpack_from("<I", header, 8)[0]))
    ndim = int(struct.unpack_from("<I", header, 12)[0])
    dims = struct.unpack_from("<QQQQ", header, 16)
    shape = tuple(int(dim) for dim in dims[:ndim] if int(dim) > 0)
    precision = int(struct.unpack_from("<I", header, 48)[0])
    byte_size = int(struct.unpack_from("<Q", header, 52)[0])
    scales_bytes = int(struct.unpack_from("<Q", header, 60)[0])
    group_size = int(struct.unpack_from("<I", header, 68)[0])
    num_groups = int(struct.unpack_from("<I", header, 72)[0])
    original_n = int(struct.unpack_from("<Q", header, 76)[0])

    dtype = _PRECISION_TO_DTYPE.get(precision)
    if dtype is None:
        raise RuntimeError(f"unsupported tensor precision {precision} in {tensor_path}")

    aligned_header = align_offset(_HEADER_SIZE, alignment)
    scales_offset = aligned_header if scales_bytes > 0 else 0
    data_offset = (
        align_offset(scales_offset + scales_bytes, alignment)
        if scales_bytes > 0
        else aligned_header
    )

    data_element_count = byte_size // np.dtype(dtype).itemsize
    data = np.memmap(tensor_path, mode="r", dtype=dtype, offset=data_offset, shape=(data_element_count,))
    scales = None
    if scales_bytes > 0:
        scales = np.memmap(
            tensor_path,
            mode="r",
            dtype=np.float16,
            offset=scales_offset,
            shape=(scales_bytes // np.dtype(np.float16).itemsize,),
        )
    return LoadedTensorFile(
        path=tensor_path,
        precision=precision,
        shape=shape,
        data=data,
        scales=scales,
        group_size=group_size,
        num_groups=num_groups,
        is_interleaved=bool(flags & FLAG_INTERLEAVED),
        original_n=original_n,
    )


def _decode_parakeet_tdt_token_ids(vocabulary: tuple[str, ...], token_ids: list[int]) -> str:
    pieces: list[str] = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= len(vocabulary):
            continue
        piece = vocabulary[token_id]
        if piece.startswith("<|") and piece.endswith("|>"):
            continue
        pieces.append(piece)
    text = "".join(pieces).replace("▁", " ")
    return re.sub(r"\s+", " ", text).strip()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, torch.Tensor):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    raise TypeError(f"unsupported runtime value type: {type(value).__name__}")

