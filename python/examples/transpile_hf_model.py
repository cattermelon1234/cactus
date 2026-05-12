from __future__ import annotations

import argparse
import builtins
import copy
import gc
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import Graph
from src.transpile.capture_pytorch import capture_model
from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import verify_ir
from src.transpile.lower import TranspiledGraph
from src.transpile.lower import _lower_constant_value
from src.transpile.lower import _lower_input_value
from src.transpile.lower import _lower_ir_node
from src.transpile.model_adapters import canonicalize_model_interface
from src.transpile.optimize_graph import FusionConfig
from src.transpile.optimize_graph import optimize_graph


@dataclass
class PreparedInputs:
    names: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]
    metadata: dict[str, object]


def _transformers_supports_model_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _candidate_external_site_packages() -> list[Path]:
    candidates: list[Path] = []
    major_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    pyenv_versions = Path.home() / ".pyenv" / "versions"
    if pyenv_versions.exists():
        for version_dir in sorted(pyenv_versions.iterdir(), reverse=True):
            site_packages = version_dir / "lib" / major_minor / "site-packages"
            if site_packages.exists():
                candidates.append(site_packages)
    return candidates


def _ensure_transformers_supports_model_type(model_type: str) -> str | None:
    normalized = str(model_type or "").strip().lower()
    target_module = {
        "gemma4": "transformers.models.gemma4.modeling_gemma4",
    }.get(normalized)
    if not target_module:
        return None
    if _transformers_supports_model_module(target_module):
        return None

    for site_packages in _candidate_external_site_packages():
        candidate = site_packages / Path(*target_module.split("."))
        if not candidate.with_suffix(".py").exists() and not candidate.is_dir():
            continue
        sys.path.insert(0, str(site_packages))
        for module_name in list(sys.modules):
            root_name = module_name.split(".", 1)[0]
            if root_name in {"transformers", "huggingface_hub", "tokenizers"}:
                del sys.modules[module_name]
        if _transformers_supports_model_module(target_module):
            return str(site_packages)
        try:
            sys.path.remove(str(site_packages))
        except ValueError:
            pass
    return None


def _resolve_local_snapshot(model_id_or_path: str) -> str | None:
    explicit = Path(model_id_or_path)
    if explicit.exists():
        return str(explicit)

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + model_id_or_path.replace("/", "--"))
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return str(snapshots[-1])


def _snapshot_has_model_weights(path: str | Path) -> bool:
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return False
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((root / name).exists() for name in candidates)


def _validate_weights_dir(weights_dir: str | None, *, model_id: str) -> Path | None:
    if not weights_dir:
        return None

    root = Path(weights_dir).resolve()
    if not root.exists():
        raise RuntimeError(
            f"weights_dir does not exist: {root}\n"
            "\n"
            f"Create the folder first with:\n"
            f"  cactus convert {model_id} {root}\n"
        )

    manifest_path = root / "weights_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"weights_dir is missing weights_manifest.json: {manifest_path}\n"
            "\n"
            f"Re-convert with the current converter:\n"
            f"  cactus convert {model_id} {root}\n"
        )

    return root


def _serialize_json_compatible(value: Any) -> Any:
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
        return {
            field.name: _serialize_json_compatible(getattr(value, field.name))
            for field in fields(value)
        }
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__module__}.{type(value).__name__}>"


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
            value_id: _serialize_json_compatible(value)
            for value_id, value in graph.values.items()
        },
        "nodes": [
            _serialize_json_compatible(graph.nodes[node_id])
            for node_id in graph.order
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported torch dtype: {name}")
    return mapping[normalized]


class TranspileWrapper(torch.nn.Module):
    def __init__(self, adapter_module: torch.nn.Module, *, weights_dir: str | None = None):
        super().__init__()
        self.adapter = adapter_module
        self.weights_dir = weights_dir

    def forward(self, *bound_inputs: torch.Tensor) -> torch.Tensor:
        return self.adapter(*bound_inputs)

    def get_transpile_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        provider = getattr(self.adapter, "get_transpile_metadata", None)
        if callable(provider):
            provided = provider()
            if isinstance(provided, dict):
                metadata.update(provided)
        graph_meta: dict[str, object] = {}
        base_graph = metadata.get("graph", {})
        if isinstance(base_graph, dict):
            graph_meta.update(base_graph)
        if self.weights_dir:
            graph_meta["weights_dir"] = self.weights_dir
        metadata["graph"] = graph_meta
        return metadata


def _patch_transformers_torchvision_probe() -> str | None:
    has_torchvision = importlib.util.find_spec("torchvision") is not None
    has_lzma = importlib.util.find_spec("_lzma") is not None

    if not has_torchvision or has_lzma:
        return None

    class _InterpolationModeStub:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    class _TorchvisionFunctionalStub:
        InterpolationMode = _InterpolationModeStub

        def __getattr__(self, name: str):
            raise RuntimeError(
                "torchvision functionality is unavailable because this Python build is missing _lzma; "
                f"attempted to access torchvision.transforms.functional.{name}"
            )

    # Some Transformers multimodal image/video modules reference `F.InterpolationMode`
    # or `tvF.InterpolationMode` in annotations even when torchvision probing is disabled.
    # Installing a tiny builtins-level stub lets those modules import so the PIL fallback
    # classes can be selected instead.
    builtins.F = _TorchvisionFunctionalStub()
    builtins.tvF = builtins.F

    import transformers.utils as tf_utils  # type: ignore
    import transformers.utils.import_utils as tf_import_utils  # type: ignore

    @lru_cache
    def _disabled() -> bool:
        return False

    tf_import_utils.is_torchvision_available = _disabled
    tf_import_utils.is_torchvision_v2_available = _disabled
    tf_utils.is_torchvision_available = _disabled
    tf_utils.is_torchvision_v2_available = _disabled
    return "disabled torchvision import checks because this Python build is missing _lzma"


def _patch_torch_flex_attention_compat() -> str | None:
    try:
        import torch.nn.attention.flex_attention as flex_attention  # type: ignore
    except Exception:
        return None

    if hasattr(flex_attention, "AuxRequest"):
        return None

    class _AuxRequest:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    flex_attention.AuxRequest = _AuxRequest  # type: ignore[attr-defined]
    return "installed torch flex_attention AuxRequest compatibility stub"


def _load_local_torch_state_dict(model_source: str) -> dict[str, torch.Tensor] | None:
    root = Path(model_source)
    if not root.exists() or not root.is_dir():
        return None

    safetensors_path = root / "model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file  # type: ignore

            return load_file(str(safetensors_path))
        except Exception:
            return None

    pytorch_path = root / "pytorch_model.bin"
    if pytorch_path.exists():
        try:
            loaded = torch.load(str(pytorch_path), map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(str(pytorch_path), map_location="cpu")
        return loaded if isinstance(loaded, dict) else None

    return None


def _remap_gemma4_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if "audio_tower" in new_key:
            new_key = re.sub(r"subsample_conv_projection\.layer(\d+)\.", r"subsample_conv_projection.conv_\1.", new_key)
            new_key = new_key.replace("audio_tower.layers.", "audio_tower.conformer.")
            new_key = new_key.replace(".feed_forward1.", ".ffw_layer_start.")
            new_key = new_key.replace(".feed_forward2.", ".ffw_layer_end.")
            new_key = re.sub(r"\.self_attn\.(q_proj|k_proj|v_proj)\.", r".attention.attn.\1.", new_key)
            new_key = new_key.replace(".self_attn.per_dim_scale", ".attention.attn.per_dim_scale")
            new_key = new_key.replace(".self_attn.relative_k_proj.", ".attention.attn.relative_position_embedding.pos_proj.")
            new_key = new_key.replace(".self_attn.post.", ".attention.post.")
            new_key = new_key.replace(".norm_pre_attn.", ".attention.pre_attn_norm.")
            new_key = new_key.replace(".norm_post_attn.", ".attention.post_norm.")
            new_key = new_key.replace(".norm_out.", ".norm.")
        new_key = new_key.replace(".linear.weight", ".weight")
        remapped[new_key] = value
        if new_key.endswith(".attention.attn.per_dim_scale"):
            remapped[new_key.replace(".per_dim_scale", ".per_dim_key_scale")] = value
    if "lm_head.weight" not in remapped:
        tied_embedding = remapped.get("model.language_model.embed_tokens.weight")
        if tied_embedding is not None:
            remapped["lm_head.weight"] = tied_embedding
    return remapped


def _repair_gemma4_checkpoint_weights(model: torch.nn.Module, model_source: str) -> dict[str, object]:
    raw_state_dict = _load_local_torch_state_dict(model_source)
    if raw_state_dict is None:
        return {"applied": False, "reason": "no local checkpoint state_dict"}

    remapped_state_dict = _remap_gemma4_checkpoint_state_dict(raw_state_dict)
    load_result = model.load_state_dict(remapped_state_dict, strict=False)
    return {
        "applied": True,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _load_config_json(model_id_or_path: str) -> dict[str, object]:
    local_snapshot = _resolve_local_snapshot(model_id_or_path)
    config_source = Path(local_snapshot) / "config.json" if local_snapshot else None
    if config_source is not None and config_source.exists():
        return json.loads(config_source.read_text())
    explicit = Path(model_id_or_path) / "config.json"
    if explicit.exists():
        return json.loads(explicit.read_text())
    return {}


def _load_optional_json(model_id_or_path: str, filename: str) -> dict[str, object]:
    local_snapshot = _resolve_local_snapshot(model_id_or_path)
    candidate = Path(local_snapshot) / filename if local_snapshot else None
    if candidate is not None and candidate.exists():
        return json.loads(candidate.read_text())
    explicit = Path(model_id_or_path) / filename
    if explicit.exists():
        return json.loads(explicit.read_text())
    return {}


def _load_gemma4_tokenizer_fallback(source_candidates: list[str]) -> object | None:
    try:
        from transformers import PreTrainedTokenizerFast  # type: ignore
    except Exception:
        return None

    for source in source_candidates:
        root = Path(source)
        if not root.exists() or not root.is_dir():
            continue
        tokenizer_json = root / "tokenizer.json"
        tokenizer_config_path = root / "tokenizer_config.json"
        if not tokenizer_json.exists() or not tokenizer_config_path.exists():
            continue

        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            bos_token=tokenizer_config.get("bos_token"),
            eos_token=tokenizer_config.get("eos_token"),
            unk_token=tokenizer_config.get("unk_token"),
            pad_token=tokenizer_config.get("pad_token"),
            mask_token=tokenizer_config.get("mask_token"),
            padding_side=str(tokenizer_config.get("padding_side", "right")),
            additional_special_tokens=list(tokenizer_config.get("extra_special_tokens", []) or []),
        )
        model_max_length = tokenizer_config.get("model_max_length")
        if isinstance(model_max_length, int):
            tokenizer.model_max_length = model_max_length

        chat_template_path = root / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()

        for token_attr in (
            "image_token",
            "audio_token",
            "boi_token",
            "eoi_token",
            "boa_token",
            "eoa_token",
        ):
            token_value = tokenizer_config.get(token_attr)
            if isinstance(token_value, str):
                setattr(tokenizer, token_attr, token_value)
                setattr(tokenizer, f"{token_attr}_id", tokenizer.convert_tokens_to_ids(token_value))
        return tokenizer

    return None


def _load_gemma4_processor_fallback(
    *,
    source_candidates: list[str],
    common_kwargs: dict[str, object],
) -> object | None:
    try:
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor  # type: ignore
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor  # type: ignore
        from transformers.models.gemma4.processing_gemma4 import Gemma4Processor  # type: ignore
    except Exception:
        return None

    tokenizer = None
    processor_config: dict[str, object] = {}
    for source in source_candidates:
        try:
            try:
                from transformers import AutoTokenizer  # type: ignore

                tokenizer = AutoTokenizer.from_pretrained(source, **common_kwargs)
            except Exception:
                tokenizer = _load_gemma4_tokenizer_fallback([source])
            processor_config = _load_optional_json(source, "processor_config.json")
            if tokenizer is not None and processor_config:
                break
        except Exception:
            continue
    if tokenizer is None or not processor_config:
        return None

    feature_config = dict(processor_config.get("feature_extractor", {}) or {})
    image_config = dict(processor_config.get("image_processor", {}) or {})
    feature_config.pop("feature_extractor_type", None)
    image_config.pop("image_processor_type", None)

    feature_extractor = Gemma4AudioFeatureExtractor(**feature_config)
    image_processor = Gemma4ImageProcessor(**image_config)

    processor_kwargs = {}
    for key in ("image_seq_length", "audio_seq_length", "audio_ms_per_token"):
        if key in processor_config:
            processor_kwargs[key] = processor_config[key]

    return Gemma4Processor(
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        tokenizer=tokenizer,
        **processor_kwargs,
    )


def _infer_task_from_config(model_id_or_path: str) -> str:
    config = _load_config_json(model_id_or_path)
    architectures = [str(value) for value in config.get("architectures", []) if isinstance(value, str)]
    model_type = str(config.get("model_type", "") or "").lower()

    if any("CausalLM" in value for value in architectures):
        return "causal_lm_logits"
    if any("CTC" in value for value in architectures):
        return "ctc_logits"
    if model_type == "whisper":
        return "encoder_hidden_states"
    if any("ConditionalGeneration" in value and "Whisper" in value for value in architectures):
        return "encoder_hidden_states"

    lowered_id = model_id_or_path.lower()
    if "whisper" in lowered_id:
        return "encoder_hidden_states"
    if "ctc" in lowered_id:
        return "ctc_logits"
    if any(token in lowered_id for token in ("qwen", "gemma", "llama", "mistral")):
        return "causal_lm_logits"

    raise RuntimeError(
        f"Could not infer transpile task for {model_id_or_path}.\n"
        "\n"
        "Pass one explicitly with --task, for example:\n"
        "  --task causal_lm_logits\n"
        "  --task ctc_logits\n"
        "  --task encoder_hidden_states\n"
    )


def _normalize_audio_samples(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.uint8:
        return ((samples.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max))
        if scale <= 0:
            scale = 1.0
        return (samples.astype(np.float32) / scale).astype(np.float32)
    return samples.astype(np.float32)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def _mel_filter_bank(num_mels: int, sample_rate: int, n_fft: int) -> np.ndarray:
    try:
        from transformers.audio_utils import mel_filter_bank as transformers_mel_filter_bank  # type: ignore

        return transformers_mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=num_mels,
            min_frequency=0.0,
            max_frequency=sample_rate / 2.0,
            sampling_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).astype(np.float32)
    except Exception:
        pass

    num_freq_bins = n_fft // 2 + 1
    mel_points = np.linspace(
        _hz_to_mel(np.array([0.0], dtype=np.float32))[0],
        _hz_to_mel(np.array([sample_rate / 2.0], dtype=np.float32))[0],
        num_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)

    filters = np.zeros((num_mels, num_freq_bins), dtype=np.float32)
    for idx in range(1, num_mels + 1):
        left = max(0, min(num_freq_bins - 1, int(bins[idx - 1])))
        center = max(left + 1, min(num_freq_bins - 1, int(bins[idx])))
        right = max(center + 1, min(num_freq_bins, int(bins[idx + 1])))
        for freq in range(left, center):
            filters[idx - 1, freq] = float(freq - left) / float(max(center - left, 1))
        for freq in range(center, right):
            filters[idx - 1, freq] = float(right - freq) / float(max(right - center, 1))
    return filters


def _load_audio_waveform(audio_file: str | Path, *, target_sample_rate: int) -> np.ndarray:
    sample_rate, waveform = wavfile.read(str(audio_file))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = _normalize_audio_samples(np.asarray(waveform))

    if int(sample_rate) != int(target_sample_rate):
        gcd = int(np.gcd(int(sample_rate), int(target_sample_rate)))
        waveform = resample_poly(
            waveform,
            int(target_sample_rate) // gcd,
            int(sample_rate) // gcd,
        ).astype(np.float32)

    return waveform.astype(np.float32)


def _generic_log_mel_features(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    num_mels: int,
    n_fft: int,
    hop_length: int,
    frame_length: int,
    preemphasis: float | None = None,
) -> tuple[np.ndarray, int]:
    waveform_tensor = torch.from_numpy(waveform).to(torch.float32).unsqueeze(0)
    if preemphasis is not None and abs(preemphasis) > 0.0:
        waveform_tensor = torch.cat(
            [
                waveform_tensor[:, :1],
                waveform_tensor[:, 1:] - float(preemphasis) * waveform_tensor[:, :-1],
            ],
            dim=1,
        )

    window = torch.hann_window(frame_length, periodic=False, dtype=torch.float32)
    stft = torch.stft(
        waveform_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window=window,
        return_complex=True,
        pad_mode="constant",
    )
    magnitudes = torch.view_as_real(stft)
    magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
    power = magnitudes.pow(2)

    mel_filters = torch.from_numpy(_mel_filter_bank(num_mels, sample_rate, n_fft)).to(torch.float32)
    mel_spec = power.transpose(1, 2) @ mel_filters
    mel_spec = torch.log(mel_spec + np.float32(2**-24))

    feature_lengths = max(1, int(waveform.shape[0] // hop_length))
    attention_mask = (
        torch.arange(mel_spec.shape[1], dtype=torch.long, device=mel_spec.device).unsqueeze(0) < feature_lengths
    )
    mask = attention_mask.unsqueeze(-1).to(torch.float32)
    masked = mel_spec * mask
    mean = masked.sum(dim=1) / float(feature_lengths)
    centered = masked - mean.unsqueeze(1)
    denom = float(max(1, feature_lengths - 1))
    variance = (centered.pow(2) * mask).sum(dim=1) / denom
    mel_spec = (mel_spec - mean.unsqueeze(1)) / (torch.sqrt(variance).unsqueeze(1) + np.float32(1e-5))
    mel_spec = mel_spec * mask
    return mel_spec[0].cpu().numpy().astype(np.float32), feature_lengths


def _resolve_audio_sample_rate(processor: object) -> int:
    for attr_name in ("feature_extractor", "tokenizer"):
        child = getattr(processor, attr_name, None)
        sample_rate = getattr(child, "sampling_rate", None)
        if isinstance(sample_rate, int) and sample_rate > 0:
            return sample_rate
    sample_rate = getattr(processor, "sampling_rate", None)
    if isinstance(sample_rate, int) and sample_rate > 0:
        return sample_rate
    return 16000


def _infer_fallback_audio_input_names(config: dict[str, object], task: str) -> tuple[str, ...]:
    model_type = str(config.get("model_type", "") or "").lower()
    if model_type == "whisper":
        return ("input_features",)
    if task == "encoder_hidden_states":
        return ("input_features",)
    audio_cfg = config.get("audio_config")
    if isinstance(audio_cfg, dict) and any(key in audio_cfg for key in ("features", "input_feat_size", "num_mel_bins")):
        return ("input_features", "attention_mask") if task == "ctc_logits" else ("input_features",)
    encoder_cfg = config.get("encoder")
    if isinstance(encoder_cfg, dict) and any(key in encoder_cfg for key in ("feat_in", "num_mel_bins")):
        return ("input_features", "attention_mask") if task == "ctc_logits" else ("input_features",)
    return ("input_values", "attention_mask") if task == "ctc_logits" else ("input_values",)


def _resolve_encoder_module(model: torch.nn.Module) -> torch.nn.Module | None:
    get_encoder = getattr(model, "get_encoder", None)
    if callable(get_encoder):
        encoder = get_encoder()
        if isinstance(encoder, torch.nn.Module):
            return encoder
    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, torch.nn.Module):
        return encoder
    model_attr = getattr(model, "model", None)
    if model_attr is not None:
        encoder = getattr(model_attr, "encoder", None)
        if isinstance(encoder, torch.nn.Module):
            return encoder
    return None


def _infer_expected_input_feature_frames(model: torch.nn.Module) -> int | None:
    config = getattr(model, "config", None)
    max_source_positions = getattr(config, "max_source_positions", None)
    if not isinstance(max_source_positions, int) or max_source_positions <= 0:
        return None

    encoder = _resolve_encoder_module(model)
    if encoder is None:
        return None

    stride_product = 1
    found_conv = False
    for child in encoder.children():
        if isinstance(child, torch.nn.Conv1d):
            stride = child.stride[0] if isinstance(child.stride, tuple) else child.stride
            stride_product *= int(stride)
            found_conv = True

    if not found_conv:
        return None
    return int(max_source_positions) * stride_product


def _prepare_fallback_audio_inputs(
    *,
    input_names: tuple[str, ...],
    config: dict[str, object],
    preprocessor_config: dict[str, object],
    model: torch.nn.Module,
    task: str,
    audio_file: str,
    torch_dtype: torch.dtype,
) -> PreparedInputs:
    if not input_names:
        input_names = _infer_fallback_audio_input_names(config, task)
    sample_rate = int(preprocessor_config.get("sampling_rate", config.get("sampling_rate", 16000)) or 16000)
    num_mels = int(preprocessor_config.get("feature_size", config.get("num_mel_bins", config.get("feature_size", 80))) or 80)
    encoder_cfg = config.get("encoder_config")
    if isinstance(encoder_cfg, dict):
        num_mels = int(encoder_cfg.get("num_mel_bins", encoder_cfg.get("feat_in", num_mels)) or num_mels)
    hop_length = int(preprocessor_config.get("hop_length", config.get("hop_length", 160)) or 160)
    n_fft = int(preprocessor_config.get("n_fft", config.get("n_fft", 400)) or 400)
    frame_length = int(preprocessor_config.get("win_length", preprocessor_config.get("frame_length", config.get("frame_length", n_fft))) or n_fft)
    preemphasis = preprocessor_config.get("preemphasis")
    if preemphasis is not None:
        preemphasis = float(preemphasis)
    waveform = _load_audio_waveform(audio_file, target_sample_rate=sample_rate)

    tensors: list[torch.Tensor] = []
    if input_names and input_names[0] == "input_features":
        features, feature_length = _generic_log_mel_features(
            waveform,
            sample_rate=sample_rate,
            num_mels=num_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            frame_length=frame_length,
            preemphasis=preemphasis,
        )
        expected_frames = _infer_expected_input_feature_frames(model)
        attention_mask: np.ndarray | None = None
        target_frames = feature_length
        if isinstance(expected_frames, int) and expected_frames > 0:
            target_frames = expected_frames

        active_frames = min(feature_length, target_frames)
        features = features[:active_frames, :]
        if target_frames > active_frames:
            pad_width = target_frames - active_frames
            features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
            attention_mask = np.zeros((target_frames,), dtype=np.bool_)
            attention_mask[:active_frames] = True

        tensors.append(torch.from_numpy(features).unsqueeze(0).to(dtype=torch_dtype))
        if attention_mask is not None and len(input_names) > 1 and input_names[1] == "attention_mask":
            tensors.append(torch.from_numpy(attention_mask).unsqueeze(0))
    else:
        input_values = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch_dtype)
        tensors.append(input_values)
        if len(input_names) > 1 and input_names[1] == "attention_mask":
            tensors.append(torch.ones_like(input_values, dtype=torch.float32))

    return PreparedInputs(
        names=input_names[: len(tensors)],
        tensors=tuple(tensors),
        metadata={
            "audio_file": str(Path(audio_file).resolve()),
            "sample_rate": sample_rate,
            "fallback_audio_preprocessor": True,
            "input_shapes": {
                name: list(tensor.shape)
                for name, tensor in zip(input_names, tensors)
            },
        },
    )


def _prepare_audio_inputs(
    processor: object | None,
    *,
    input_names: tuple[str, ...],
    config: dict[str, object],
    preprocessor_config: dict[str, object],
    model: torch.nn.Module,
    task: str,
    audio_file: str,
    torch_dtype: torch.dtype,
) -> PreparedInputs:
    if processor is None:
        return _prepare_fallback_audio_inputs(
            input_names=input_names,
            config=config,
            preprocessor_config=preprocessor_config,
            model=model,
            task=task,
            audio_file=audio_file,
            torch_dtype=torch_dtype,
        )

    sample_rate = _resolve_audio_sample_rate(processor)
    waveform = _load_audio_waveform(audio_file, target_sample_rate=sample_rate)
    batch = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    preferred_keys = tuple(input_names) + tuple(
        key for key in ("input_features", "input_values", "attention_mask") if key not in input_names
    )
    tensor_keys = [key for key, value in batch.items() if isinstance(value, torch.Tensor)]
    ordered_keys = [key for key in preferred_keys if key in tensor_keys]
    ordered_keys.extend(key for key in tensor_keys if key not in ordered_keys)
    if not ordered_keys:
        raise RuntimeError(f"processor did not return tensor inputs for audio file: {audio_file}")

    tensors: list[torch.Tensor] = []
    for key in ordered_keys:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            continue
        if torch.is_floating_point(value):
            value = value.to(dtype=torch_dtype)
        tensors.append(value)

    return PreparedInputs(
        names=tuple(ordered_keys[: len(tensors)]),
        tensors=tuple(tensors),
        metadata={
            "audio_file": str(Path(audio_file).resolve()),
            "sample_rate": sample_rate,
            "fallback_audio_preprocessor": False,
            "input_shapes": {
                name: list(tensor.shape)
                for name, tensor in zip(ordered_keys, tensors)
            },
        },
    )


def _prepare_text_inputs(tokenizer: object, *, prompt: str, input_ids_text: str | None) -> PreparedInputs:
    if input_ids_text:
        token_ids = [int(part.strip()) for part in input_ids_text.split(",") if part.strip()]
        if not token_ids:
            raise ValueError("--input-ids was provided but no ids were parsed")
        input_ids = torch.tensor([token_ids], dtype=torch.long)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return PreparedInputs(
        names=("input_ids",),
        tensors=(input_ids,),
        metadata={
            "prompt": prompt,
            "input_ids": input_ids.tolist(),
        },
    )


_GEMMA4_MULTIMODAL_INPUT_ORDER = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "pixel_values",
    "pixel_position_ids",
    "input_features",
    "input_features_mask",
)


def _normalize_multimodal_prompt(
    prompt: str,
    *,
    image_token: str | None,
    num_images: int,
    audio_token: str | None,
    num_audio_segments: int,
) -> str:
    normalized = prompt.strip()
    prefixes: list[str] = []

    if image_token and num_images > 0:
        image_count = normalized.count(image_token)
        if image_count < num_images:
            prefixes.append(" ".join(image_token for _ in range(num_images - image_count)))
    if audio_token and num_audio_segments > 0:
        audio_count = normalized.count(audio_token)
        if audio_count < num_audio_segments:
            prefixes.append(" ".join(audio_token for _ in range(num_audio_segments - audio_count)))

    if prefixes:
        prefix = "\n".join(part for part in prefixes if part)
        if normalized:
            return f"{prefix}\n{normalized}"
        return prefix
    return normalized


def _build_gemma4_chat_prompt(
    *,
    prompt: str,
    image_token: str | None,
    num_images: int,
    audio_token: str | None,
    num_audio_segments: int,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
) -> str:
    result = "<bos>"
    normalized_system = system_prompt.strip()
    if enable_thinking_if_supported or normalized_system:
        result += "<|turn>system\n"
        if enable_thinking_if_supported:
            result += "<|think|>"
        result += normalized_system
        result += "<turn|>\n"

    result += "<|turn>user\n"
    if image_token and num_images > 0:
        for _ in range(num_images):
            result += f"\n\n{image_token}\n\n"
    result += prompt
    if audio_token and num_audio_segments > 0:
        result += "".join(audio_token for _ in range(num_audio_segments))
    result += "<turn|>\n"
    result += "<|turn>model\n"
    return result


def _load_image_inputs(image_files: tuple[str, ...]) -> list[object]:
    if not image_files:
        return []
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Pillow is required for --image-file: {exc}") from exc

    images: list[object] = []
    for image_file in image_files:
        path = Path(image_file).resolve()
        if not path.exists():
            raise RuntimeError(f"image_file does not exist: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _prepare_gemma4_multimodal_inputs(
    processor: object | None,
    *,
    prompt: str,
    image_files: tuple[str, ...],
    audio_file: str | None,
    torch_dtype: torch.dtype,
    system_prompt: str = "",
    enable_thinking_if_supported: bool = False,
    use_gemma4_chat_template: bool = False,
) -> PreparedInputs:
    if processor is None:
        raise RuntimeError("multimodal Gemma4 transpile requires an AutoProcessor with image and audio support")

    images = _load_image_inputs(image_files)
    audio_waveforms: list[np.ndarray] = []
    sample_rate: int | None = None
    if audio_file:
        sample_rate = _resolve_audio_sample_rate(processor)
        audio_waveforms.append(_load_audio_waveform(audio_file, target_sample_rate=sample_rate))

    image_token = getattr(processor, "image_token", None)
    audio_token = getattr(processor, "audio_token", None)
    processor_prompt = _normalize_multimodal_prompt(
        prompt,
        image_token=image_token if isinstance(image_token, str) else None,
        num_images=len(images),
        audio_token=audio_token if isinstance(audio_token, str) else None,
        num_audio_segments=len(audio_waveforms),
    )
    normalized_prompt = processor_prompt
    if use_gemma4_chat_template:
        normalized_prompt = prompt.strip()
        processor_prompt = _build_gemma4_chat_prompt(
            prompt=normalized_prompt,
            image_token=image_token if isinstance(image_token, str) else None,
            num_images=len(images),
            audio_token=audio_token if isinstance(audio_token, str) else None,
            num_audio_segments=len(audio_waveforms),
            system_prompt=system_prompt,
            enable_thinking_if_supported=enable_thinking_if_supported,
        )

    batch = processor(
        text=processor_prompt,
        images=images or None,
        audio=audio_waveforms or None,
        return_tensors="pt",
    )

    ordered_keys = [
        key
        for key in _GEMMA4_MULTIMODAL_INPUT_ORDER
        if isinstance(batch.get(key), torch.Tensor)
    ]
    if not ordered_keys:
        raise RuntimeError("Gemma4 multimodal processor did not return any tensor inputs")

    tensors: list[torch.Tensor] = []
    for key in ordered_keys:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            continue
        if torch.is_floating_point(value):
            value = value.to(dtype=torch_dtype)
        tensors.append(value)

    metadata: dict[str, object] = {
        "prompt": normalized_prompt,
        "processor_prompt": processor_prompt,
        "image_files": [str(Path(path).resolve()) for path in image_files],
        "input_shapes": {
            name: list(tensor.shape)
            for name, tensor in zip(ordered_keys, tensors)
        },
    }
    if audio_file:
        metadata["audio_file"] = str(Path(audio_file).resolve())
    if sample_rate is not None:
        metadata["sample_rate"] = sample_rate

    return PreparedInputs(
        names=tuple(ordered_keys[: len(tensors)]),
        tensors=tuple(tensors),
        metadata=metadata,
    )


def _load_model_source(model_id: str, *, local_files_only: bool) -> str:
    local_snapshot = _resolve_local_snapshot(model_id)
    if local_snapshot and _snapshot_has_model_weights(local_snapshot):
        return local_snapshot
    if local_snapshot and local_files_only:
        raise RuntimeError(
            f"Found local snapshot for {model_id}, but it is incomplete and has no model weights:\n"
            f"  {local_snapshot}\n"
            "\n"
            "Re-run without --local-files-only to let transformers download the missing weights."
        )
    return model_id


def _load_transformers_bundle(
    *,
    model_id: str,
    task: str,
    torch_dtype: torch.dtype,
    token: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
):
    config = _load_config_json(model_id)
    config_model_type = str(config.get("model_type", "") or "").lower()
    external_transformers_site_packages = _ensure_transformers_supports_model_type(config_model_type)
    patch_note = _patch_transformers_torchvision_probe()
    if patch_note:
        print(f"note={patch_note}")
    flex_patch_note = _patch_torch_flex_attention_compat()
    if flex_patch_note:
        print(f"note={flex_patch_note}")
    if external_transformers_site_packages:
        print(f"note=using external transformers install for {config_model_type}: {external_transformers_site_packages}")

    try:
        from transformers import AutoFeatureExtractor  # type: ignore
        from transformers import AutoModel  # type: ignore
        from transformers import AutoModelForCTC  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
        from transformers import AutoModelForSeq2SeqLM  # type: ignore
        from transformers import AutoModelForSpeechSeq2Seq  # type: ignore
        from transformers import AutoProcessor  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"transformers is not available: {exc}") from exc

    model_source = _load_model_source(model_id, local_files_only=local_files_only)
    source_candidates = []
    for candidate in (model_source, model_id):
        if candidate not in source_candidates:
            source_candidates.append(candidate)
    common_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if token:
        common_kwargs["token"] = token

    if task == "causal_lm_logits":
        tokenizer = None
        tokenizer_errors: list[str] = []
        for source in source_candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, **common_kwargs)
                break
            except Exception as exc:
                tokenizer_errors.append(f"{source}: {exc}")
        if tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer for {model_id}:\n"
                + "\n".join(tokenizer_errors)
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            **common_kwargs,
        ).eval()
        return model_source, tokenizer, model, config
    if task == "multimodal_causal_lm_logits":
        processor = None
        processor_errors: list[str] = []
        for source in source_candidates:
            try:
                processor = AutoProcessor.from_pretrained(source, **common_kwargs)
                break
            except Exception as exc:
                processor_errors.append(f"{source}: {exc}")
        if processor is None and config_model_type == "gemma4":
            processor = _load_gemma4_processor_fallback(
                source_candidates=source_candidates,
                common_kwargs=common_kwargs,
            )
            if processor is not None:
                print("note=using manual gemma4 processor fallback")
        if processor is None:
            processor_config_hint = ""
            if config_model_type == "gemma4":
                processor_config_hint = (
                    "\n"
                    "Gemma4 multimodal transpile needs a processor bundle, not just tokenizer/model weights.\n"
                    "Your local snapshot may be missing files such as `processor_config.json` or modality-specific\n"
                    "preprocessor configs. Use an official Gemma4 snapshot that includes the processor, or let\n"
                    "transformers download one by re-running without `--local-files-only`.\n"
                )
            raise RuntimeError(
                f"Could not load processor for {model_id}:\n"
                + "\n".join(processor_errors)
                + processor_config_hint
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            **common_kwargs,
        ).eval()
        if config_model_type == "gemma4":
            repair_result = _repair_gemma4_checkpoint_weights(model, model_source)
            if repair_result.get("applied"):
                missing = repair_result.get("missing_keys", [])
                unexpected = repair_result.get("unexpected_keys", [])
                print(
                    "note=applied gemma4 checkpoint key remap"
                    f" missing_after={len(missing)} unexpected_after={len(unexpected)}"
                )
        return model_source, processor, model, config

    processor = None
    processor_errors: list[str] = []
    missing_optional_audio_dep: str | None = None
    for source in source_candidates:
        for loader in (AutoProcessor, AutoFeatureExtractor):
            try:
                processor = loader.from_pretrained(source, **common_kwargs)
                break
            except Exception as exc:
                processor_errors.append(f"{loader.__name__}@{source}: {exc}")
                if isinstance(exc, ImportError) and "requires the librosa library" in str(exc):
                    missing_optional_audio_dep = "librosa"
                    break
        if processor is not None:
            break
        if missing_optional_audio_dep is not None:
            break
    if processor is None:
        if missing_optional_audio_dep == "librosa":
            print("note=falling back to built-in audio preprocessing because the HF feature extractor requires librosa")
        else:
            print("note=falling back to built-in audio preprocessing because no HF processor/feature extractor was available")

    if task == "ctc_logits":
        model_loaders = (AutoModelForCTC, AutoModel)
    elif task == "encoder_hidden_states":
        model_loaders = (AutoModelForSpeechSeq2Seq, AutoModelForSeq2SeqLM, AutoModel)
    else:
        raise NotImplementedError(f"unsupported generic HF task: {task}")

    load_errors: list[str] = []
    for loader in model_loaders:
        try:
            model = loader.from_pretrained(
                model_source,
                dtype=torch_dtype,
                device_map=None,
                low_cpu_mem_usage=True,
                **common_kwargs,
            ).eval()
            return model_source, processor, model, config
        except Exception as exc:
            load_errors.append(f"{loader.__name__}: {exc}")

    raise RuntimeError(
        f"Could not load model for task={task} from {model_source}.\n"
        "\n".join(load_errors)
    )


def _load_optional_tokenizer(
    *,
    model_id: str,
    model_source: str,
    token: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        return None

    source_candidates = []
    for candidate in (model_source, model_id):
        if candidate not in source_candidates:
            source_candidates.append(candidate)

    common_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if token:
        common_kwargs["token"] = token

    for source in source_candidates:
        try:
            return AutoTokenizer.from_pretrained(source, **common_kwargs)
        except Exception:
            continue
    return None


def _ctc_greedy_decode_token_ids(logits: np.ndarray, *, blank_token_id: int | None) -> list[int]:
    if logits.ndim != 3 or logits.shape[0] < 1:
        raise ValueError(f"expected CTC logits with shape [batch, time, vocab], got {list(logits.shape)}")

    raw_ids = np.argmax(logits[0], axis=-1).tolist()
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in raw_ids:
        if token_id != previous:
            collapsed.append(int(token_id))
        previous = int(token_id)

    if blank_token_id is None:
        return collapsed
    return [token_id for token_id in collapsed if int(token_id) != int(blank_token_id)]


def _decode_token_ids(tokenizer: object, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError(f"tokenizer does not expose decode(): {type(tokenizer).__name__}")
    try:
        return str(decode(token_ids, skip_special_tokens=True))
    except TypeError:
        return str(decode(token_ids))


def _count_weight_bindings(ir_graph: IRGraph) -> int:
    count = 0
    for value in ir_graph.values.values():
        if isinstance(value.meta, dict) and isinstance(value.meta.get("path"), str):
            count += 1
    return count


def _lower_preoptimized_ir(ir: IRGraph) -> TranspiledGraph:
    verify_ir(ir)
    graph = Graph()
    env: dict[str, Any] = {}
    runtime_inputs = []
    bound_constants = []

    for value_id in ir.inputs:
        value = ir.values[value_id]
        tensor = _lower_input_value(graph, value)
        env[value_id] = tensor
        runtime_inputs.append(tensor)

    for value_id, const in ir.constants.items():
        value = ir.values[value_id]
        lowered_const = _lower_constant_value(graph, value, const)
        env[value_id] = lowered_const
        if hasattr(lowered_const, "g") and hasattr(lowered_const, "id"):
            bound_constants.append(lowered_const)

    for node_id in ir.order:
        node = ir.nodes[node_id]
        outputs = _lower_ir_node(graph, node, env, ir)
        if len(outputs) != len(node.outputs):
            raise ValueError(
                f"node {node.id} produced {len(outputs)} outputs, expected {len(node.outputs)}"
            )
        for output_id, tensor in zip(node.outputs, outputs):
            env[output_id] = tensor

    outputs = [env[value_id] for value_id in ir.outputs]
    return TranspiledGraph(
        graph=graph,
        runtime_inputs=runtime_inputs,
        bound_constants=bound_constants,
        outputs=outputs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Hugging Face model, canonicalize it into a generic transpile task, "
            "capture it with the Cactus transpiler, lower it to a Cactus Graph, and "
            "optionally save artifacts or run the lowered graph."
        )
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model id or local snapshot path.",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=("auto", "causal_lm_logits", "multimodal_causal_lm_logits", "ctc_logits", "encoder_hidden_states"),
        help="Transpile task. Use auto to infer from config/model id.",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Prompt used for causal_lm_logits or multimodal_causal_lm_logits when --input-ids is not set.",
    )
    parser.add_argument(
        "--input-ids",
        default="",
        help="Optional comma-separated token ids for causal_lm_logits.",
    )
    parser.add_argument(
        "--audio-file",
        default="",
        help="Path to a WAV file for audio or multimodal tasks.",
    )
    parser.add_argument(
        "--image-file",
        action="append",
        default=[],
        help="Path to an image file for multimodal tasks. Repeat to pass multiple images.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        help="Torch dtype for model loading: float16, float32, or bfloat16.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers loaders.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the model/processor to already exist locally.",
    )
    parser.add_argument(
        "--weights-dir",
        default="",
        help="Optional converted Cactus weights directory for mmap weight binding.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="Optional directory where raw_ir.json, optimized_ir.json, graph.cactus, and result.json are saved.",
    )
    parser.add_argument(
        "--graph-filename",
        default="graph.cactus",
        help="Filename to use for Graph.save() inside --artifact-dir.",
    )
    parser.add_argument("--no-fuse-gated-deltanet", action="store_true")
    parser.add_argument("--no-fuse-rms-norm", action="store_true")
    parser.add_argument("--no-fuse-rope", action="store_true")
    parser.add_argument("--no-fuse-attention", action="store_true")
    parser.add_argument("--no-fuse-attention-block", action="store_true")
    parser.add_argument("--no-fuse-add-clipped", action="store_true")
    parser.add_argument(
        "--skip-execute",
        action="store_true",
        help="Stop after lowering instead of running the transpiled graph.",
    )
    parser.add_argument(
        "--skip-reference-compare",
        action="store_true",
        help="Run the transpiled graph but skip the follow-up PyTorch reference pass.",
    )
    args = parser.parse_args()

    image_files = tuple(str(path) for path in args.image_file if str(path).strip())
    if args.task == "auto":
        inferred_task = _infer_task_from_config(args.model_id)
        config_for_auto = _load_config_json(args.model_id)
        model_type_for_auto = str(config_for_auto.get("model_type", "") or "").lower()
        has_multimodal_config = (
            model_type_for_auto == "gemma4"
            and (
                isinstance(config_for_auto.get("vision_config"), dict)
                or isinstance(config_for_auto.get("audio_config"), dict)
            )
        )
        if image_files or (args.audio_file and has_multimodal_config and inferred_task == "causal_lm_logits"):
            task = "multimodal_causal_lm_logits"
        else:
            task = inferred_task
    else:
        task = args.task
    torch_dtype = _parse_dtype(args.torch_dtype)
    validated_weights_dir = _validate_weights_dir(args.weights_dir.strip() or None, model_id=args.model_id)
    weights_dir = str(validated_weights_dir) if validated_weights_dir is not None else None
    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else None

    model_source, processor_or_tokenizer, model, model_config = _load_transformers_bundle(
        model_id=args.model_id,
        task=task,
        torch_dtype=torch_dtype,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    preprocessor_config = _load_optional_json(model_source, "preprocessor_config.json")
    if not preprocessor_config:
        preprocessor_config = _load_optional_json(args.model_id, "preprocessor_config.json")
    auxiliary_tokenizer = None
    if task == "ctc_logits":
        auxiliary_tokenizer = _load_optional_tokenizer(
            model_id=args.model_id,
            model_source=model_source,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

    if task == "causal_lm_logits":
        prepared = _prepare_text_inputs(
            processor_or_tokenizer,
            prompt=args.prompt,
            input_ids_text=args.input_ids.strip() or None,
        )
        canonical = canonicalize_model_interface(
            model,
            task=task,
            input_names=prepared.names,
        )
    elif task == "multimodal_causal_lm_logits":
        prepared = _prepare_gemma4_multimodal_inputs(
            processor_or_tokenizer,
            prompt=args.prompt,
            image_files=image_files,
            audio_file=args.audio_file.strip() or None,
            torch_dtype=torch_dtype,
        )
        canonical = canonicalize_model_interface(
            model,
            task=task,
            input_names=prepared.names,
        )
    else:
        if not args.audio_file:
            raise RuntimeError(f"--audio-file is required for task={task}")
        canonical = canonicalize_model_interface(model, task=task)
        prepared = _prepare_audio_inputs(
            processor_or_tokenizer,
            input_names=canonical.input_names,
            config=model_config,
            preprocessor_config=preprocessor_config,
            model=model,
            task=task,
            audio_file=args.audio_file,
            torch_dtype=torch_dtype,
        )
        canonical = canonicalize_model_interface(
            model,
            task=task,
            input_names=prepared.names,
        )
    wrapper = TranspileWrapper(canonical.module, weights_dir=weights_dir).eval()

    print(f"model_id={args.model_id}")
    print(f"model_source={model_source}")
    print(f"task={task}")
    print(f"adapter_family={canonical.family}")
    print(f"adapter_module={type(canonical.module).__name__}")
    print(f"input_names={','.join(prepared.names)}")
    for name, tensor in zip(prepared.names, prepared.tensors):
        print(f"input_{name}_shape={list(tensor.shape)}")
    if weights_dir:
        print(f"weights_dir={weights_dir}")

    captured = capture_model(wrapper, prepared.tensors)
    raw_ir_graph = copy.deepcopy(captured.ir_graph)

    fusion_config = FusionConfig(
        enable_gated_deltanet=not args.no_fuse_gated_deltanet,
        enable_rms_norm=not args.no_fuse_rms_norm,
        enable_rope=not args.no_fuse_rope,
        enable_attention=not args.no_fuse_attention,
        enable_attention_block=not args.no_fuse_attention_block,
        enable_add_clipped=not args.no_fuse_add_clipped,
    )

    canonicalize_exported_graph(captured.ir_graph)
    optimize_graph(captured.ir_graph, config=fusion_config)
    tg = _lower_preoptimized_ir(captured.ir_graph)

    optimized_ir_graph = copy.deepcopy(captured.ir_graph)
    binding_count = _count_weight_bindings(optimized_ir_graph)
    op_counts = Counter(optimized_ir_graph.nodes[node_id].op for node_id in optimized_ir_graph.order)

    print(f"raw_ir_nodes={len(raw_ir_graph.order)}")
    print(f"optimized_ir_nodes={len(optimized_ir_graph.order)}")
    print(f"weight_bindings={binding_count}")
    print(
        "ops="
        f"attention:{op_counts.get('attention', 0)} "
        f"conv1d:{op_counts.get('conv1d', 0)} "
        f"conv2d:{op_counts.get('conv2d', 0)} "
        f"batch_norm:{op_counts.get('batch_norm', 0)} "
        f"layer_norm:{op_counts.get('layer_norm', 0)} "
        f"rms_norm:{op_counts.get('rms_norm', 0)} "
        f"rope:{op_counts.get('rope', 0)} "
        f"linear:{op_counts.get('linear', 0)}"
    )

    if weights_dir and binding_count == 0:
        raise RuntimeError(
            f"No weight bindings were resolved from {weights_dir}\n"
            "\n"
            "The weights folder exists, but none of the captured constants matched entries in weights_manifest.json.\n"
            "\n"
            f"Recommended fix:\n"
            f"  cactus convert {args.model_id} {weights_dir}\n"
        )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            artifact_dir / "raw_ir.json",
            {
                "model_id": args.model_id,
                "model_source": model_source,
                "task": task,
                "family": canonical.family,
                "inputs": _serialize_json_compatible(prepared.metadata),
                "graph": _graph_to_dict(raw_ir_graph),
            },
        )
        _write_json(
            artifact_dir / "optimized_ir.json",
            {
                "model_id": args.model_id,
                "model_source": model_source,
                "task": task,
                "family": canonical.family,
                "inputs": _serialize_json_compatible(prepared.metadata),
                "graph": _graph_to_dict(optimized_ir_graph),
            },
        )
        graph_path = artifact_dir / args.graph_filename
        tg.graph.save(graph_path)
        print(f"saved_raw_ir={artifact_dir / 'raw_ir.json'}")
        print(f"saved_optimized_ir={artifact_dir / 'optimized_ir.json'}")
        print(f"saved_graph={graph_path}")

    if args.skip_execute:
        return 0

    if args.skip_reference_compare:
        del model
        del wrapper
        gc.collect()

    tg.set_inputs([tensor.cpu().numpy() for tensor in prepared.tensors])
    print("execute_begin=true")
    transpiled_output = tg.execute()[0].numpy().astype(np.float32)
    print("execute_done=true")

    if args.skip_reference_compare:
        result_payload = {
            "model_id": args.model_id,
            "model_source": model_source,
            "task": task,
            "family": canonical.family,
            "inputs": _serialize_json_compatible(prepared.metadata),
            "output_shape": list(transpiled_output.shape),
            "raw_ir_nodes": len(raw_ir_graph.order),
            "optimized_ir_nodes": len(optimized_ir_graph.order),
            "weight_bindings": binding_count,
            "reference_compare_skipped": True,
        }
        print(f"output_shape={list(transpiled_output.shape)}")
        if task in {"causal_lm_logits", "multimodal_causal_lm_logits"}:
            tokenizer_like = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
            transpiled_next = int(np.argmax(transpiled_output[0, -1]))
            print(f"transpiled_next_token_id={transpiled_next}")
            result_payload["transpiled_next_token_id"] = transpiled_next
            if hasattr(tokenizer_like, "decode"):
                transpiled_token = tokenizer_like.decode([transpiled_next])
                print(f"transpiled_next_token={transpiled_token!r}")
                result_payload["transpiled_next_token"] = transpiled_token
        if artifact_dir is not None:
            _write_json(artifact_dir / "result.json", result_payload)
            print(f"saved_result={artifact_dir / 'result.json'}")
        return 0

    print("reference_begin=true")
    with torch.no_grad():
        reference_output = wrapper(*prepared.tensors).detach().float().cpu().numpy()
    print("reference_done=true")

    max_abs_diff = float(np.max(np.abs(reference_output - transpiled_output)))
    mean_abs_diff = float(np.mean(np.abs(reference_output - transpiled_output)))
    result_payload: dict[str, object] = {
        "model_id": args.model_id,
        "model_source": model_source,
        "task": task,
        "family": canonical.family,
        "inputs": _serialize_json_compatible(prepared.metadata),
        "output_shape": list(reference_output.shape),
        "raw_ir_nodes": len(raw_ir_graph.order),
        "optimized_ir_nodes": len(optimized_ir_graph.order),
        "weight_bindings": binding_count,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }

    if task == "causal_lm_logits":
        tokenizer = processor_or_tokenizer
        hf_next = int(np.argmax(reference_output[0, -1]))
        transpiled_next = int(np.argmax(transpiled_output[0, -1]))
        print(f"hf_next_token_id={hf_next}")
        print(f"transpiled_next_token_id={transpiled_next}")
        print(f"logits_max_abs_diff={max_abs_diff:.6f}")
        print(f"logits_mean_abs_diff={mean_abs_diff:.6f}")
        print(f"hf_next_token={tokenizer.decode([hf_next])!r}")
        print(f"transpiled_next_token={tokenizer.decode([transpiled_next])!r}")
        result_payload.update(
            {
                "hf_next_token_id": hf_next,
                "transpiled_next_token_id": transpiled_next,
                "hf_next_token": tokenizer.decode([hf_next]),
                "transpiled_next_token": tokenizer.decode([transpiled_next]),
            }
        )
    elif task == "ctc_logits":
        blank_token_id = getattr(model.config, "pad_token_id", None)
        if blank_token_id is None and auxiliary_tokenizer is not None:
            blank_token_id = getattr(auxiliary_tokenizer, "pad_token_id", None)
        print(f"output_shape={list(reference_output.shape)}")
        print(f"output_max_abs_diff={max_abs_diff:.6f}")
        print(f"output_mean_abs_diff={mean_abs_diff:.6f}")
        if auxiliary_tokenizer is not None:
            hf_token_ids = _ctc_greedy_decode_token_ids(reference_output, blank_token_id=blank_token_id)
            transpiled_token_ids = _ctc_greedy_decode_token_ids(transpiled_output, blank_token_id=blank_token_id)
            hf_transcript = _decode_token_ids(auxiliary_tokenizer, hf_token_ids)
            transpiled_transcript = _decode_token_ids(auxiliary_tokenizer, transpiled_token_ids)
            print(f"hf_transcript={hf_transcript!r}")
            print(f"transpiled_transcript={transpiled_transcript!r}")
            result_payload.update(
                {
                    "blank_token_id": None if blank_token_id is None else int(blank_token_id),
                    "hf_transcript_token_ids": hf_token_ids,
                    "transpiled_transcript_token_ids": transpiled_token_ids,
                    "hf_transcript": hf_transcript,
                    "transpiled_transcript": transpiled_transcript,
                }
            )
    else:
        print(f"output_shape={list(reference_output.shape)}")
        print(f"output_max_abs_diff={max_abs_diff:.6f}")
        print(f"output_mean_abs_diff={mean_abs_diff:.6f}")

    if artifact_dir is not None:
        _write_json(artifact_dir / "result.json", result_payload)
        print(f"saved_result={artifact_dir / 'result.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
