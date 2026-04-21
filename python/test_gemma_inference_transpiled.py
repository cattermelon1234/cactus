from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.cactus import cactus_destroy
from src.cactus import cactus_init
from src.cactus import cactus_score_window
from src.transpile.capture_pytorch import capture_model
from src.transpile.cleanup_passes import run_cleanup_passes
from src.transpile.lower import transpile_captured
from src.transpile.model_adapters import canonicalize_model_interface
from src.transpile.optimize_graph import FusionConfig
from src.transpile.optimize_graph import optimize_graph


def _import_transformers():
    from transformers import AutoModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    return AutoModelForCausalLM, AutoTokenizer


def _topk(logits: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(logits)[-k:][::-1]


def _format_topk(tokenizer, logits: np.ndarray, k: int) -> list[tuple[int, str, float]]:
    ids = _topk(logits, k)
    return [(int(idx), tokenizer.decode([int(idx)]), float(logits[idx])) for idx in ids]


def _print_topk(title: str, rows: list[tuple[int, str, float]]) -> None:
    print(title)
    for token_id, text, score in rows:
        print(f"  {token_id:>7}  {text!r:<20} {score:>12.6f}")


def _time_call(fn, *, warmup: int, repeats: int):
    for _ in range(max(warmup, 0)):
        fn()

    times_ms: list[float] = []
    result = None
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    assert result is not None
    return result, times_ms


def _format_times(label: str, times_ms: list[float]) -> str:
    return (
        f"{label}: mean={np.mean(times_ms):.3f}, "
        f"min={np.min(times_ms):.3f}, max={np.max(times_ms):.3f}"
    )


def _summarize_ir_ops(ir_graph) -> tuple[Counter, Counter]:
    op_counts = Counter()
    leftover_counts = Counter()
    leftover_ops = {
        "add",
        "subtract",
        "multiply",
        "multiply_inplace",
        "divide",
        "scalar_add",
        "scalar_subtract",
        "scalar_subtract_reverse",
        "scalar_multiply",
        "scalar_divide",
        "reshape",
        "flatten",
        "transpose",
        "permute",
        "slice",
        "index",
        "gather",
        "cat",
        "unsqueeze",
        "expand",
        "type_as",
        "precision_cast",
    }
    for node_id in ir_graph.order:
        op = ir_graph.nodes[node_id].op
        op_counts[op] += 1
        if op in leftover_ops:
            leftover_counts[op] += 1
    return op_counts, leftover_counts


def _extract_checkpoint_value_ids(ir_graph) -> list[tuple[str, str]]:
    checkpoints: list[tuple[str, str]] = []
    layer_scalar_nodes = []
    for node_id in ir_graph.order:
        node = ir_graph.nodes[node_id]
        torch_op = str(node.meta.get("torch_op", ""))
        if torch_op.startswith("aten.mul_.Tensor") and node.outputs:
            layer_scalar_nodes.append(node)

    if layer_scalar_nodes:
        for index, node in enumerate(layer_scalar_nodes, start=1):
            checkpoints.append((f"layer_{index}", node.outputs[0]))

    patterns = list(ir_graph.meta.get("detected_gold_patterns", [])) if hasattr(ir_graph, "meta") else []
    order_index = {node_id: index for index, node_id in enumerate(ir_graph.order)}

    block_patterns = [
        pattern
        for pattern in patterns
        if pattern.name.startswith("decoder_block_") and pattern.anchor_node_id in ir_graph.nodes
    ]
    block_patterns.sort(key=lambda pattern: order_index.get(pattern.anchor_node_id, 10**9))
    for index, pattern in enumerate(block_patterns, start=1):
        anchor = ir_graph.nodes[pattern.anchor_node_id]
        if anchor.outputs:
            checkpoints.append((f"block_{index}", anchor.outputs[0]))

    last_linear_id = next((node_id for node_id in reversed(ir_graph.order) if ir_graph.nodes[node_id].op == "linear"), None)
    if last_linear_id is not None:
        last_linear = ir_graph.nodes[last_linear_id]
        if last_linear.inputs:
            checkpoints.append(("pre_lm_head", last_linear.inputs[0]))
    return checkpoints


def _transpile_single_output(ir_graph, output_value_id: str):
    subgraph = copy.deepcopy(ir_graph)
    subgraph.outputs = [output_value_id]
    return transpile_captured(type("Captured", (), {"ir_graph": subgraph})())


def _find_first_attention_debug_values(ir_graph) -> list[tuple[str, str]]:
    for node_id in ir_graph.order:
        node = ir_graph.nodes[node_id]
        if node.op != "attention":
            continue
        values: list[tuple[str, str]] = []
        if len(node.inputs) >= 3:
            values.append(("attn_q", node.inputs[0]))
            values.append(("attn_k", node.inputs[1]))
            values.append(("attn_v", node.inputs[2]))
        if node.outputs:
            values.append(("attn_out", node.outputs[0]))
            oproj_value = _find_first_attention_oproj_value(ir_graph, node.outputs[0])
            if oproj_value is not None:
                values.append(("attn_o_proj", oproj_value))
        return values
    return []


def _find_first_attention_oproj_value(ir_graph, attention_output_id: str) -> str | None:
    queue = [attention_output_id]
    seen_values: set[str] = set()
    passthrough_ops = {"reshape", "transpose", "permute", "type_as", "precision_cast", "contiguous", "identity"}
    while queue:
        value_id = queue.pop(0)
        if value_id in seen_values:
            continue
        seen_values.add(value_id)
        value = ir_graph.values.get(value_id)
        if value is None:
            continue
        for user_id in value.users:
            user = ir_graph.nodes[user_id]
            if user.op == "linear" and user.outputs:
                return user.outputs[0]
            if user.op in passthrough_ops and user.outputs:
                queue.extend(user.outputs)
    return None


def _find_first_block_debug_values(ir_graph) -> list[tuple[str, str]]:
    checks: list[tuple[str, str]] = []

    def first_node(predicate):
        for node_id in ir_graph.order:
            node = ir_graph.nodes[node_id]
            if predicate(node):
                return node
        return None

    def weight_match(node, fragment: str) -> bool:
        return node.op == "rms_norm" and len(node.inputs) >= 2 and fragment in node.inputs[1]

    def linear_weight_match(node, fragment: str) -> bool:
        return node.op == "linear" and len(node.inputs) >= 2 and fragment in node.inputs[1]

    post_attn_norm = first_node(lambda n: weight_match(n, "modules__0___post_attention_layernorm_weight"))
    if post_attn_norm is not None:
        checks.append(("post_attn_norm", post_attn_norm.outputs[0]))
        after_attention = first_node(lambda n: n.op in {"add", "add_clipped"} and post_attn_norm.outputs[0] in n.inputs)
        if after_attention is not None:
            checks.append(("after_attention_residual", after_attention.outputs[0]))

    pre_ffn_norm = first_node(lambda n: weight_match(n, "modules__0___pre_feedforward_layernorm_weight"))
    if pre_ffn_norm is not None:
        checks.append(("pre_ffn_norm", pre_ffn_norm.outputs[0]))

    mlp_down = first_node(lambda n: linear_weight_match(n, "modules__0___mlp_down_proj_weight"))
    if mlp_down is not None:
        checks.append(("mlp_down", mlp_down.outputs[0]))

    post_ffn_norm = first_node(lambda n: weight_match(n, "modules__0___post_feedforward_layernorm_weight"))
    if post_ffn_norm is not None:
        checks.append(("post_ffn_norm", post_ffn_norm.outputs[0]))
        after_ffn = first_node(lambda n: n.op in {"add", "add_clipped"} and post_ffn_norm.outputs[0] in n.inputs)
        if after_ffn is not None:
            checks.append(("after_ffn_residual", after_ffn.outputs[0]))

    per_layer_proj = first_node(lambda n: linear_weight_match(n, "modules__0___per_layer_projection_weight"))
    if per_layer_proj is not None:
        checks.append(("per_layer_input_proj", per_layer_proj.outputs[0]))

    post_per_layer_input_norm = first_node(lambda n: weight_match(n, "modules__0___post_per_layer_input_norm_weight"))
    if post_per_layer_input_norm is not None:
        checks.append(("post_per_layer_input_norm", post_per_layer_input_norm.outputs[0]))

    first_layer_scalar = first_node(lambda n: str(n.meta.get("torch_op", "")).startswith("aten.mul_.Tensor"))
    if first_layer_scalar is not None:
        checks.append(("layer_scalar_out", first_layer_scalar.outputs[0]))

    return checks


def _reachable_ops_for_output(ir_graph, output_value_id: str) -> Counter:
    reachable_values = {output_value_id}
    reachable_nodes: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node_id in reversed(ir_graph.order):
            node = ir_graph.nodes[node_id]
            if any(output in reachable_values for output in node.outputs):
                if node_id not in reachable_nodes:
                    reachable_nodes.add(node_id)
                    changed = True
                for input_id in node.inputs:
                    if input_id not in reachable_values:
                        reachable_values.add(input_id)
                        changed = True
    counts = Counter()
    for node_id in reachable_nodes:
        counts[ir_graph.nodes[node_id].op] += 1
    return counts


def _print_layer_debug(tokenizer, adapter_module, ir_graph, input_ids: torch.Tensor, *, max_checkpoints: int | None = None) -> None:
    if not hasattr(adapter_module, "debug_forward"):
        print("Layer debug: adapter does not expose debug checkpoints.")
        return

    with torch.no_grad():
        _, ref_checkpoints = adapter_module.debug_forward(input_ids)
    ref_arrays = [tensor.detach().float().cpu().numpy() for tensor in ref_checkpoints]
    checkpoint_value_ids = _extract_checkpoint_value_ids(ir_graph)
    if not checkpoint_value_ids:
        print("Layer debug: no decoder-block checkpoints found in IR.")
        return

    print("Layer debug:")
    limit = min(len(ref_arrays), len(checkpoint_value_ids))
    if max_checkpoints is not None:
        limit = min(limit, max_checkpoints)
    if len(ref_arrays) != len(checkpoint_value_ids):
        print(
            f"  checkpoint count mismatch: adapter={len(ref_arrays)} ir={len(checkpoint_value_ids)}; "
            f"comparing first {limit}"
        )

    first_nan_label = None
    for index in range(limit):
        label, value_id = checkpoint_value_ids[index]
        try:
            tg = _transpile_single_output(ir_graph, value_id)
            tg.set_inputs([input_ids.cpu().numpy()])
            transpiled_value = tg.execute()[0].numpy().astype(np.float32)
        except NotImplementedError as exc:
            reachable_ops = _reachable_ops_for_output(ir_graph, value_id)
            hot_ops = ", ".join(f"{op}={count}" for op, count in reachable_ops.most_common(12))
            print(f"  {label}: blocked by unsupported lowering: {exc}")
            print(f"    reachable ops: {hot_ops}")
            break
        ref_value = ref_arrays[index]
        has_nan = bool(np.isnan(transpiled_value).any())
        max_abs_diff = float(np.nanmax(np.abs(ref_value - transpiled_value))) if ref_value.shape == transpiled_value.shape else float("nan")
        mean_abs_diff = float(np.nanmean(np.abs(ref_value - transpiled_value))) if ref_value.shape == transpiled_value.shape else float("nan")
        print(
            f"  {label}: shape={tuple(transpiled_value.shape)} "
            f"nan={has_nan} max_abs_diff={max_abs_diff:.6f} mean_abs_diff={mean_abs_diff:.6f}"
        )
        if first_nan_label is None and has_nan:
            first_nan_label = label
            break

    if first_nan_label is not None:
        print(f"  first NaN checkpoint: {first_nan_label}")
    else:
        print("  no NaNs found in checked layer checkpoints")


def _print_attention_debug(ir_graph, input_ids: torch.Tensor, *, selected_labels: set[str] | None = None) -> None:
    attention_values = _find_first_attention_debug_values(ir_graph)
    if not attention_values:
        print("Attention debug: no fused attention node found.")
        return
    print("Attention debug:")
    for label, value_id in attention_values:
        if selected_labels is not None and label not in selected_labels:
            continue
        try:
            tg = _transpile_single_output(ir_graph, value_id)
            tg.set_inputs([input_ids.cpu().numpy()])
            value = tg.execute()[0].numpy().astype(np.float32)
            finite = value[np.isfinite(value)]
            if finite.size:
                min_value = float(finite.min())
                max_value = float(finite.max())
                mean_abs = float(np.mean(np.abs(finite)))
            else:
                min_value = float("nan")
                max_value = float("nan")
                mean_abs = float("nan")
            print(
                f"  {label}: shape={tuple(value.shape)} "
                f"nan={bool(np.isnan(value).any())} inf={bool(np.isinf(value).any())} "
                f"min={min_value:.6f} max={max_value:.6f} mean_abs={mean_abs:.6f}"
            )
        except NotImplementedError as exc:
            reachable_ops = _reachable_ops_for_output(ir_graph, value_id)
            hot_ops = ", ".join(f"{op}={count}" for op, count in reachable_ops.most_common(12))
            print(f"  {label}: blocked by unsupported lowering: {exc}")
            print(f"    reachable ops: {hot_ops}")
            break


def _print_first_block_debug(adapter_module, ir_graph, input_ids: torch.Tensor) -> None:
    if not hasattr(adapter_module, "debug_first_block"):
        return
    ir_values = _find_first_block_debug_values(ir_graph)
    if not ir_values:
        print("First-block debug: no checkpoints found.")
        return
    ref_values = adapter_module.debug_first_block(input_ids)
    print("First-block debug:")
    for label, value_id in ir_values:
        ref_value = ref_values.get(label)
        if ref_value is None:
            continue
        try:
            tg = _transpile_single_output(ir_graph, value_id)
            tg.set_inputs([input_ids.cpu().numpy()])
            transpiled_value = tg.execute()[0].numpy().astype(np.float32)
        except NotImplementedError as exc:
            reachable_ops = _reachable_ops_for_output(ir_graph, value_id)
            hot_ops = ", ".join(f"{op}={count}" for op, count in reachable_ops.most_common(12))
            print(f"  {label}: blocked by unsupported lowering: {exc}")
            print(f"    reachable ops: {hot_ops}")
            break
        ref_array = ref_value.detach().float().cpu().numpy()
        has_nan = bool(np.isnan(transpiled_value).any())
        max_abs_diff = float(np.nanmax(np.abs(ref_array - transpiled_value))) if ref_array.shape == transpiled_value.shape else float("nan")
        mean_abs_diff = float(np.nanmean(np.abs(ref_array - transpiled_value))) if ref_array.shape == transpiled_value.shape else float("nan")
        print(
            f"  {label}: shape={tuple(transpiled_value.shape)} "
            f"nan={has_nan} max_abs_diff={max_abs_diff:.6f} mean_abs_diff={mean_abs_diff:.6f}"
        )


def _resolve_local_cactus_weights(model_id: str) -> str | None:
    model_name = model_id.split("/")[-1]
    candidate = Path("weights") / model_name
    if candidate.exists():
        return str(candidate)
    return None


def _resolve_local_hf_snapshot(model_id: str) -> str | None:
    parts = model_id.split("/")
    if len(parts) != 2:
        return None
    namespace, model_name = parts
    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{namespace}--{model_name}"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        return None
    return str(snapshots[-1])


def _make_prompt_variants(tokenizer, prompt: str, lengths: list[int]) -> list[tuple[int, torch.Tensor, str]]:
    if not lengths:
        encoded = tokenizer(prompt, return_tensors="pt")
        return [(encoded["input_ids"].shape[1], encoded["input_ids"], prompt)]

    base = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    variants: list[tuple[int, torch.Tensor, str]] = []
    for target_len in lengths:
        tokens = list(base)
        while len(tokens) < target_len:
            tokens.extend(base[1:] if len(base) > 1 else base)
        tokens = tokens[:target_len]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        text = tokenizer.decode(tokens)
        variants.append((target_len, input_ids, text))
    return variants


def _run_case(
    *,
    tokenizer,
    model_id: str,
    model_fp16,
    prompt_text: str,
    input_ids: torch.Tensor,
    topk: int,
    warmup: int,
    repeats: int,
    max_nodes: int,
    use_local_cactus: bool,
    fusion_config: FusionConfig,
    debug_layers: bool,
    debug_max_layers: int | None,
    debug_attention_targets: set[str] | None,
) -> None:
    print("\n=== Case ===")
    print(f"Prompt length: {input_ids.shape[1]} tokens")
    print(f"Prompt: {prompt_text!r}")

    print("Capturing and transpiling graph...")
    adapter = canonicalize_model_interface(model_fp16, task="causal_lm_logits")
    print(f"Capture adapter: task={adapter.task} family={adapter.family} module={type(adapter.module).__name__}")
    capture_module = adapter.module
    captured = capture_model(capture_module, (input_ids,))
    run_cleanup_passes(captured.ir_graph)
    optimize_graph(captured.ir_graph, config=fusion_config)
    run_cleanup_passes(captured.ir_graph)

    op_counts, leftover_counts = _summarize_ir_ops(captured.ir_graph)
    print(f"Optimized IR node count: {len(captured.ir_graph.order)}")
    print(
        "Semantic op counts: "
        f"attention={op_counts.get('attention', 0)}, "
        f"rms_norm={op_counts.get('rms_norm', 0)}, "
        f"rope={op_counts.get('rope', 0)}, "
        f"add_clipped={op_counts.get('add_clipped', 0)}, "
        f"linear={op_counts.get('linear', 0)}"
    )
    rms_offsets = Counter(
        int(captured.ir_graph.nodes[node_id].meta.get("rms_weight_offset", 0.0))
        for node_id in captured.ir_graph.order
        if captured.ir_graph.nodes[node_id].op == "rms_norm"
    )
    if rms_offsets:
        print(
            "RMSNorm offset counts: "
            + ", ".join(f"offset_{offset}={count}" for offset, count in sorted(rms_offsets.items()))
        )
    mul_inplace_lowered = [
        captured.ir_graph.nodes[node_id]
        for node_id in captured.ir_graph.order
        if str(captured.ir_graph.nodes[node_id].meta.get("torch_op", "")).startswith("aten.mul_.Tensor")
    ]
    if mul_inplace_lowered:
        lowered_ops = Counter(node.op for node in mul_inplace_lowered)
        print(
            "Lowered aten.mul_ ops: "
            + ", ".join(f"{op}={count}" for op, count in sorted(lowered_ops.items()))
        )
    if leftover_counts:
        summary = ", ".join(f"{op}={count}" for op, count in leftover_counts.most_common())
        print(f"Leftover scalar/layout ops: {summary}")
    else:
        print("Leftover scalar/layout ops: none")

    if max_nodes > 0:
        for node_id in captured.ir_graph.order[:max_nodes]:
            node = captured.ir_graph.nodes[node_id]
            print(" ", node_id, node.op, node.inputs, node.outputs, node.attrs)

    if debug_layers:
        _print_layer_debug(tokenizer, capture_module, captured.ir_graph, input_ids, max_checkpoints=debug_max_layers)
        _print_attention_debug(captured.ir_graph, input_ids, selected_labels=debug_attention_targets)
        _print_first_block_debug(capture_module, captured.ir_graph, input_ids)

    tg = transpile_captured(captured)
    tg.set_inputs([input_ids.cpu().numpy()])
    transpiled_outputs, cactus_times_ms = _time_call(
        lambda: tg.execute(),
        warmup=warmup,
        repeats=repeats,
    )
    transpiled_logits = transpiled_outputs[0].numpy().astype(np.float32)

    print(f"Reusing model as HF reference: {model_id} (float32)")
    ref_module = model_fp16.float().eval()
    with torch.no_grad():
        ref_tensor, hf_times_ms = _time_call(
            lambda: ref_module(input_ids=input_ids, use_cache=False, return_dict=False)[0].detach(),
            warmup=warmup,
            repeats=repeats,
        )
        ref_logits = ref_tensor.float().cpu().numpy()

    ref_last = ref_logits[0, -1]
    tr_last = transpiled_logits[0, -1]
    abs_diff = np.abs(ref_last - tr_last)

    ref_rows = _format_topk(tokenizer, ref_last, topk)
    tr_rows = _format_topk(tokenizer, tr_last, topk)
    ref_ids = {token_id for token_id, _, _ in ref_rows}
    tr_ids = {token_id for token_id, _, _ in tr_rows}
    overlap = ref_ids & tr_ids

    print(f"Input ids: {input_ids.tolist()}")
    print(f"Logits shape: {tuple(transpiled_logits.shape)}")
    print(f"Max abs diff: {float(abs_diff.max()):.6f}")
    print(f"Mean abs diff: {float(abs_diff.mean()):.6f}")
    print(_format_times("Transpiled Cactus execute time (ms)", cactus_times_ms))
    print(_format_times("HF float32 forward time (ms)", hf_times_ms))
    print(f"Speed ratio (HF/transpiled): {float(np.mean(hf_times_ms) / np.mean(cactus_times_ms)):.3f}x")
    print(f"Top-{topk} overlap: {len(overlap)}")
    if overlap:
        overlap_rows = [(token_id, tokenizer.decode([token_id])) for token_id in sorted(overlap)]
        print("Overlap tokens:")
        for token_id, text in overlap_rows:
            print(f"  {token_id:>7}  {text!r}")

    if use_local_cactus:
        weights_dir = _resolve_local_cactus_weights(model_id)
        if weights_dir is not None:
            print(f"Comparing against handwritten Cactus runtime: {weights_dir}")
            cactus_model = cactus_init(weights_dir, None, False)
            try:
                token_list = [int(x) for x in input_ids[0].tolist()]
                score_fn = lambda: cactus_score_window(
                    cactus_model,
                    token_list,
                    max(1, len(token_list) - 1),
                    len(token_list),
                    len(token_list),
                )
                score_json, handwritten_times_ms = _time_call(
                    score_fn,
                    warmup=warmup,
                    repeats=repeats,
                )
                print(_format_times("Handwritten Cactus score_window time (ms)", handwritten_times_ms))
                try:
                    parsed = json.loads(score_json)
                    keys = ", ".join(sorted(parsed.keys())[:8])
                    print(f"Handwritten Cactus score_window keys: {keys}")
                except Exception:
                    print("Handwritten Cactus score_window returned non-JSON output")
            finally:
                cactus_destroy(cactus_model)
        else:
            print("No local converted Cactus weights folder found for handwritten-runtime comparison.")

    print()
    _print_topk(f"HF float32 top-{topk}:", ref_rows)
    print()
    _print_topk(f"Transpiled top-{topk}:", tr_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run HF Gemma and the transpiled Cactus graph on the same prompt."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="The capital of France is",
        help="Prompt to score.",
    )
    parser.add_argument(
        "--model-id",
        # default=os.environ.get("CACTUS_GEMMA_HF_MODEL_ID", "google/gemma-4-E2B"),
        # default=os.environ.get("CACTUS_GEMMA_HF_MODEL_ID", "google/gemma-3-270m-it"),
        # default=os.environ.get("CACTUS_GEMMA_HF_MODEL_ID", "google/gemma-2b-it"),
        default=os.environ.get("CACTUS_QWEN_HF_MODEL_ID", "Qwen/Qwen3.5-2B"),

        help="Hugging Face model id.",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--lengths",
        default="",
        help="Comma-separated token lengths to benchmark, e.g. 16,32,64. Uses the prompt text as a repeated seed.",
    )
    parser.add_argument(
        "--compare-handwritten-cactus",
        action="store_true",
        help="Also benchmark the handwritten Cactus runtime when a local converted weights folder exists.",
    )
    parser.add_argument("--debug-layers", action="store_true", help="Run per-layer checkpoint debug after optimization.")
    parser.add_argument("--debug-max-layers", type=int, default=4, help="Maximum number of layer checkpoints to inspect in debug mode.")
    parser.add_argument(
        "--debug-attention-targets",
        default="attn_out,attn_o_proj",
        help="Comma-separated first-attention checkpoints to inspect: attn_q,attn_k,attn_v,attn_out,attn_o_proj",
    )
    parser.add_argument("--no-fuse-rms-norm", action="store_true", help="Disable RMSNorm semantic fusion.")
    parser.add_argument("--no-fuse-rope", action="store_true", help="Disable RoPE semantic fusion.")
    parser.add_argument("--no-fuse-attention", action="store_true", help="Disable attention semantic fusion.")
    parser.add_argument("--no-fuse-add-clipped", action="store_true", help="Disable add_clipped semantic fusion.")
    args = parser.parse_args()

    lengths = [int(part) for part in args.lengths.split(",") if part.strip()] if args.lengths else []
    debug_attention_targets = {part.strip() for part in args.debug_attention_targets.split(",") if part.strip()}

    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    fusion_config = FusionConfig(
        enable_rms_norm=not args.no_fuse_rms_norm,
        enable_rope=not args.no_fuse_rope,
        enable_attention=not args.no_fuse_attention,
        enable_add_clipped=not args.no_fuse_add_clipped,
    )
    token = os.environ.get("HF_TOKEN")
    common_kwargs: dict[str, object] = {}
    if token:
        common_kwargs["token"] = token
    model_source = _resolve_local_hf_snapshot(args.model_id) or args.model_id
    local_only = model_source != args.model_id

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=False,
        local_files_only=local_only,
        **common_kwargs,
    )

    print(f"Loading model for transpilation: {args.model_id} (float16)")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
        **common_kwargs,
    ).eval()

    for _, input_ids, prompt_text in _make_prompt_variants(tokenizer, args.prompt, lengths):
        _run_case(
            tokenizer=tokenizer,
            model_id=args.model_id,
            model_fp16=model,
            prompt_text=prompt_text,
            input_ids=input_ids,
            topk=args.topk,
            warmup=args.warmup,
            repeats=args.repeats,
            max_nodes=args.max_nodes,
            use_local_cactus=args.compare_handwritten_cactus,
            fusion_config=fusion_config,
            debug_layers=args.debug_layers,
            debug_max_layers=args.debug_max_layers,
            debug_attention_targets=debug_attention_targets,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
