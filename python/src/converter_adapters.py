from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .config_utils import extract_base_config
from .tensor_io import create_quantization_stats
from .tensor_io import format_config_value
from .tensor_io import print_quantization_summary
from .tensor_io import save_tensor_with_header
from .weight_adapters import ExportPlan
from .weight_adapters import PlannedTensorExport
from .weight_adapters import select_adapter


def convert_hf_model_weights_with_adapters(
    model: Any,
    output_dir: str | Path,
    precision: str = "INT8",
    args: Any | None = None,
) -> dict[str, object]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    root_config = model.config
    text_config = getattr(root_config, "text_config", None) or root_config

    adapter, ctx = select_adapter(
        root_config=root_config,
        state_keys=state_dict.keys(),
        precision=precision,
    )
    plan = adapter.build_plan(state_dict, ctx)

    quantization_stats = create_quantization_stats()
    weights_manifest: dict[str, dict[str, object]] = {}

    for export in plan.tensor_exports:
        tensor = state_dict[export.source_name]
        _save_planned_tensor(
            tensor=tensor,
            export=export,
            output_dir=output_root,
            precision=precision,
            quantization_stats=quantization_stats,
            args=args,
            model_type=ctx.detected_model_type,
            model_config=ctx.model_config,
        )
        manifest_names = (export.source_name,) + export.source_names
        if export.shard_index is None:
            for name in manifest_names:
                weights_manifest[name] = {
                    "filename": export.output_name,
                    "kind": export.kind,
                }

    manifest_path = output_root / "weights_manifest.json"
    manifest_path.write_text(json.dumps(dict(sorted(weights_manifest.items())), indent=2, sort_keys=True) + "\n")

    plan_path = output_root / "adapter_export_plan.json"
    plan_path.write_text(json.dumps(_serialize_plan(plan), indent=2, sort_keys=True) + "\n")

    print_quantization_summary(quantization_stats, args)

    model_config = extract_base_config(text_config, root_config)
    model_config["model_type"] = ctx.detected_model_type
    model_config["adapter_name"] = plan.adapter_name
    config_path = output_root / "config.txt"
    with open(config_path, "w") as fh:
        for key, value in model_config.items():
            fh.write(f"{key}={format_config_value(value)}\n")
    return model_config


def _save_planned_tensor(
    *,
    tensor: Any,
    export: PlannedTensorExport,
    output_dir: Path,
    precision: str,
    quantization_stats: dict[str, object],
    args: Any | None,
    model_type: str,
    model_config: dict[str, object],
) -> None:
    tensor_value = _to_tensor_data(tensor)
    if export.shard_index is not None and export.split_mode is not None and export.shard_count is not None:
        tensor_value = _slice_tensor_for_export(
            tensor_value,
            split_mode=export.split_mode,
            shard_index=export.shard_index,
            shard_count=export.shard_count,
            model_config=model_config,
        )
    save_tensor_with_header(
        tensor_value,
        output_dir / export.output_name,
        precision=export.precision or precision,
        transpose=export.transpose,
        stats_tracker=quantization_stats,
        args=args,
        model_type=model_type,
    )


def _to_tensor_data(tensor: Any) -> Any:
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    return tensor


def _slice_tensor_for_export(
    tensor: Any,
    *,
    split_mode: str,
    shard_index: int,
    shard_count: int,
    model_config: dict[str, object],
) -> Any:
    if torch is not None and isinstance(tensor, torch.Tensor):
        shape = tuple(int(dim) for dim in tensor.shape)
    else:
        shape = tuple(int(dim) for dim in np.asarray(tensor).shape)

    if split_mode == "dim0_equal":
        if shape[0] % shard_count != 0:
            raise ValueError(f"cannot split first dim equally: shape={shape} shard_count={shard_count}")
        chunk = shape[0] // shard_count
        start = shard_index * chunk
        end = start + chunk
        return tensor[start:end, ...]

    if split_mode == "dim1_equal":
        if len(shape) < 2 or shape[1] % shard_count != 0:
            raise ValueError(f"cannot split second dim equally: shape={shape} shard_count={shard_count}")
        chunk = shape[1] // shard_count
        start = shard_index * chunk
        end = start + chunk
        return tensor[:, start:end, ...]

    if split_mode == "dim0_sizes":
        sizes = [
            int(model_config.get("linear_q_proj_dim", 0) or 0),
            int(model_config.get("linear_k_proj_dim", 0) or 0),
            int(model_config.get("linear_v_proj_dim", 0) or 0),
        ]
        if any(size <= 0 for size in sizes[:shard_count]):
            raise ValueError(f"missing projection dims for dim0_sizes split: {sizes}")
        start = sum(sizes[:shard_index])
        end = start + sizes[shard_index]
        return tensor[start:end, ...]

    raise ValueError(f"unsupported split_mode={split_mode}")


def _serialize_plan(plan: ExportPlan) -> dict[str, object]:
    return {
        "adapter_name": plan.adapter_name,
        "tensor_exports": [asdict(item) for item in plan.tensor_exports],
        "missing": [asdict(item) for item in plan.missing],
        "matched_source_names": sorted(plan.matched_source_names),
    }
