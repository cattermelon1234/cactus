from __future__ import annotations

import builtins
import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.transpile.audio_preprocess import prepare_native_gemma4_audio_features


_TORCHVISION_COMPAT_LIBRARIES: list[object] = []


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
    if not pyenv_versions.exists():
        return candidates
    for version_dir in sorted(pyenv_versions.iterdir(), reverse=True):
        site_packages = version_dir / "lib" / major_minor / "site-packages"
        if site_packages.exists():
            candidates.append(site_packages)
    return candidates


def ensure_transformers_supports_gemma4() -> str | None:
    target_module = "transformers.models.gemma4.modeling_gemma4"
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


def _patch_torchvision_missing_nms_op() -> str | None:
    """Keep mismatched torch/torchvision installs importable for processors."""

    try:
        import torchvision  # type: ignore  # noqa: F401
        return None
    except RuntimeError as exc:
        if "operator torchvision::nms does not exist" not in str(exc):
            return None
    except Exception:
        return None

    try:
        import torchvision.extension as tv_extension  # type: ignore
        if bool(getattr(tv_extension, "_HAS_OPS", False)):
            return None
    except Exception:
        pass

    try:
        library = torch.library.Library("torchvision", "DEF")
        library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_COMPAT_LIBRARIES.append(library)
        return "defined missing torchvision::nms operator for torchvision import compatibility"
    except Exception:
        return None


def patch_transformers_torchvision_probe() -> str | None:
    has_torchvision = importlib.util.find_spec("torchvision") is not None
    has_lzma = importlib.util.find_spec("_lzma") is not None

    if not has_torchvision:
        return None

    if has_lzma:
        nms_patch_note = _patch_torchvision_missing_nms_op()
        return nms_patch_note

    base_note: str | None = None
    try:
        import backports.lzma as backports_lzma  # type: ignore

        sys.modules.setdefault("lzma", backports_lzma)
        base_note = "using backports.lzma because this Python build is missing _lzma"
        nms_patch_note = _patch_torchvision_missing_nms_op()
        if nms_patch_note:
            return f"{base_note}; {nms_patch_note}"
        return base_note
    except Exception:
        pass

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


def patch_torch_flex_attention_compat() -> str | None:
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


def _gemma4_split_cactus_newline_token_merges(batch: object) -> None:
    input_ids = batch.get("input_ids") if hasattr(batch, "get") else None
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return

    expansions = {
        108: (107, 107),
        109: (107, 107, 107),
    }
    if not any(int(token) in expansions for token in input_ids.reshape(-1).tolist()):
        return

    lengths: list[list[int]] = []
    max_len = 0
    for row in input_ids.detach().cpu().tolist():
        row_lengths = [len(expansions.get(int(token), (int(token),))) for token in row]
        lengths.append(row_lengths)
        max_len = max(max_len, sum(row_lengths))

    for key in ("input_ids", "attention_mask", "token_type_ids"):
        value = batch.get(key) if hasattr(batch, "get") else None
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        expanded = torch.full(
            (value.shape[0], max_len),
            0,
            dtype=value.dtype,
            device=value.device,
        )
        for row_idx, row in enumerate(value.detach().cpu().tolist()):
            out: list[int] = []
            for token_idx, item in enumerate(row):
                if key == "input_ids":
                    out.extend(expansions.get(int(item), (int(item),)))
                else:
                    out.extend([int(item)] * lengths[row_idx][token_idx])
            expanded[row_idx, : len(out)] = torch.tensor(out, dtype=value.dtype, device=value.device)
        batch[key] = expanded


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


def _get_processor_image_attr(processor: object, name: str, default: object) -> object:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, name):
        return getattr(image_processor, name)
    if isinstance(image_processor, dict) and name in image_processor:
        return image_processor[name]
    return default


def _prepare_gemma4_native_image_tensors(
    processor: object,
    image_files: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not image_files:
        return None
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Pillow is required for Gemma4 native image preprocessing: {exc}") from exc

    patch_size = int(_get_processor_image_attr(processor, "patch_size", 16))
    pooling_kernel_size = int(_get_processor_image_attr(processor, "pooling_kernel_size", 3))
    max_soft_tokens = int(_get_processor_image_attr(processor, "max_soft_tokens", 280))
    rescale_factor = float(_get_processor_image_attr(processor, "rescale_factor", 1.0 / 255.0))
    max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size
    side_multiple = pooling_kernel_size * patch_size
    patch_dim = 3 * patch_size * patch_size
    if patch_size <= 0 or pooling_kernel_size <= 0 or max_patches <= 0:
        return None

    try:
        resample_bilinear = Image.Resampling.BILINEAR
    except AttributeError:  # pragma: no cover
        resample_bilinear = Image.BILINEAR

    pixel_batches: list[np.ndarray] = []
    position_batches: list[np.ndarray] = []
    for image_file in image_files:
        path = Path(image_file).resolve()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            target_pixels = float(max_patches * patch_size * patch_size)
            factor = float(np.sqrt(target_pixels / max(1.0, float(width * height))))
            target_h = int(np.floor(factor * height / side_multiple)) * side_multiple
            target_w = int(np.floor(factor * width / side_multiple)) * side_multiple
            if target_h == 0:
                target_h = side_multiple
            if target_w == 0:
                target_w = side_multiple
            if (target_w, target_h) != rgb.size:
                rgb = rgb.resize((target_w, target_h), resample=resample_bilinear)
            array = np.asarray(rgb, dtype=np.float32) * rescale_factor

        patch_h = target_h // patch_size
        patch_w = target_w // patch_size
        num_patches = patch_h * patch_w
        if num_patches > max_patches:
            raise RuntimeError(
                f"Gemma4 native image preprocessing produced {num_patches} patches, "
                f"but max_patches={max_patches}"
            )
        chw = np.transpose(array, (2, 0, 1))
        patches = (
            chw.reshape(3, patch_h, patch_size, patch_w, patch_size)
            .transpose(1, 3, 2, 4, 0)
            .reshape(num_patches, patch_dim)
        )

        padded_patches = np.zeros((max_patches, patch_dim), dtype=np.float32)
        padded_patches[:num_patches] = patches
        positions = np.full((max_patches, 2), -1, dtype=np.int64)
        valid_positions = np.zeros((num_patches, 2), dtype=np.int64)
        for patch_y in range(patch_h):
            row_start = patch_y * patch_w
            valid_positions[row_start : row_start + patch_w, 0] = np.arange(patch_w, dtype=np.int64)
            valid_positions[row_start : row_start + patch_w, 1] = patch_y
        positions[:num_patches] = valid_positions

        pixel_batches.append(padded_patches)
        position_batches.append(positions)

    return (
        torch.from_numpy(np.stack(pixel_batches, axis=0)),
        torch.from_numpy(np.stack(position_batches, axis=0)),
    )


def prepare_gemma4_multimodal_inputs(
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
        from src.transpile.audio_preprocess import load_audio_waveform

        audio_waveforms.append(load_audio_waveform(audio_file, target_sample_rate=sample_rate))

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
    if use_gemma4_chat_template:
        _gemma4_split_cactus_newline_token_merges(batch)
        native_image_tensors = _prepare_gemma4_native_image_tensors(processor, image_files)
        if native_image_tensors is not None:
            batch["pixel_values"], batch["pixel_position_ids"] = native_image_tensors
        if audio_file and isinstance(batch.get("input_features"), torch.Tensor):
            feature_tensor = batch["input_features"]
            expected_mels = int(feature_tensor.shape[-1])
            try:
                native_audio, native_audio_mask, native_audio_frames = prepare_native_gemma4_audio_features(
                    audio_file,
                    expected_mels=expected_mels,
                    torch_dtype=torch_dtype,
                )
            except Exception as exc:
                print(f"note=falling back to processor gemma4 audio features: {exc}")
                batch["input_features"] = feature_tensor.to(dtype=torch_dtype)
                fallback_mask = batch.get("input_features_mask")
                if isinstance(fallback_mask, torch.Tensor):
                    batch["input_features_mask"] = fallback_mask.to(dtype=torch.bool)
                    native_audio_frames = int(fallback_mask.to(dtype=torch.int32).sum().item())
                else:
                    native_audio_frames = int(feature_tensor.shape[1])
                    batch["input_features_mask"] = torch.ones(
                        (int(feature_tensor.shape[0]), native_audio_frames),
                        dtype=torch.bool,
                    )
                batch["native_audio_frames"] = native_audio_frames
            else:
                batch["input_features"] = native_audio
                batch["input_features_mask"] = native_audio_mask
                batch["native_audio_frames"] = native_audio_frames

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
    native_audio_frames = batch.get("native_audio_frames")
    if isinstance(native_audio_frames, int):
        metadata["native_audio_frames"] = native_audio_frames

    return PreparedInputs(
        names=tuple(ordered_keys[: len(tensors)]),
        tensors=tuple(tensors),
        metadata=metadata,
    )


_patch_transformers_torchvision_probe = patch_transformers_torchvision_probe
_patch_torch_flex_attention_compat = patch_torch_flex_attention_compat
_prepare_gemma4_multimodal_inputs = prepare_gemma4_multimodal_inputs
