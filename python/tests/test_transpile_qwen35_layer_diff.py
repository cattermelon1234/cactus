from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).parent.parent))

if torch is not None and AutoModelForCausalLM is not None:
    from src.transpile.capture_pytorch import capture_model
    from src.transpile.canonicalize.cleanup import canonicalize_exported_graph
    from src.transpile.model_adapters import canonicalize_model_interface
    from src.transpile.optimize_graph import optimize_graph
    from src.transpile.lower import transpile_preoptimized_ir


_RUN_QWEN35_LAYER_DIFF = os.environ.get("CACTUS_RUN_LOCAL_QWEN35_LAYER_DIFF_TEST") == "1"
_DEFAULT_MODEL_ID = os.environ.get("CACTUS_TEST_QWEN35_MODEL_ID", "Qwen/Qwen3.5-2B")
_DEFAULT_INPUT_IDS = os.environ.get("CACTUS_TEST_QWEN35_INPUT_IDS", "151644,8948,198,9707")
_DEFAULT_WEIGHTS_DIR = os.environ.get(
    "CACTUS_TEST_QWEN35_WEIGHTS_DIR",
    str((Path(__file__).resolve().parents[2] / "weights" / "qwen3.5-2b").resolve()),
)
_DEFAULT_LAYER_INDEX = os.environ.get("CACTUS_TEST_QWEN35_LAYER_INDEX")


def _resolve_local_model_path(model_id: str) -> str:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--" + model_id.replace("/", "--"))
    )
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"could not find local snapshot for {model_id!r} under {snapshots_dir}")
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"no snapshots found for {model_id!r} under {snapshots_dir}")
    return str(snapshots[-1])


def _parse_input_ids(raw: str) -> list[int]:
    token_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not token_ids:
        raise ValueError("CACTUS_TEST_QWEN35_INPUT_IDS produced no token ids")
    return token_ids


def _summarize_ops(ir) -> str:
    counts: dict[str, int] = {}
    for node_id in ir.order:
        op_name = str(ir.nodes[node_id].op)
        counts[op_name] = counts.get(op_name, 0) + 1
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    preview = items[:8]
    return ",".join(f"{name}:{count}" for name, count in preview)


@dataclass
class DiffStats:
    label: str
    shape: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    argmax_ref: int | None
    argmax_transpiled: int | None
    raw_ir_nodes: int
    optimized_ir_nodes: int
    top_ops: str


if torch is not None and AutoModelForCausalLM is not None:
    class _WeightsBoundTranspileWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module, *, weights_dir: str | None):
            super().__init__()
            self.module = module
            self.weights_dir = weights_dir

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return self.module(*args)

        def get_transpile_metadata(self) -> dict[str, object]:
            metadata: dict[str, object] = {}
            provider = getattr(self.module, "get_transpile_metadata", None)
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


    class Qwen35CheckpointWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module, checkpoint_index: int, *, weights_dir: str | None = None):
            super().__init__()
            adapter = canonicalize_model_interface(model, task="causal_lm_logits", weights_dir=weights_dir)
            if adapter.family != "qwen3_5":
                raise ValueError(f"expected qwen3_5 adapter, got family={adapter.family}")
            self.adapter = adapter.module
            self.checkpoint_index = int(checkpoint_index)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            _, checkpoints = self.adapter.debug_forward(input_ids)
            return checkpoints[self.checkpoint_index]


    class Qwen35LogitsWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
            super().__init__()
            adapter = canonicalize_model_interface(model, task="causal_lm_logits", weights_dir=weights_dir)
            if adapter.family != "qwen3_5":
                raise ValueError(f"expected qwen3_5 adapter, got family={adapter.family}")
            self.adapter = adapter.module

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            logits, _ = self.adapter.debug_forward(input_ids)
            return logits


    class Qwen35FinalNormWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module, *, weights_dir: str | None = None):
            super().__init__()
            adapter = canonicalize_model_interface(model, task="causal_lm_logits", weights_dir=weights_dir)
            if adapter.family != "qwen3_5":
                raise ValueError(f"expected qwen3_5 adapter, got family={adapter.family}")
            self.adapter = adapter.module

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            _, checkpoints = self.adapter.debug_forward(input_ids)
            return checkpoints[-1]


    class Qwen35LayerCheckpointWrapper(torch.nn.Module):
        def __init__(
            self,
            model: torch.nn.Module,
            *,
            layer_index: int,
            checkpoint_name: str,
            weights_dir: str | None = None,
        ):
            super().__init__()
            adapter = canonicalize_model_interface(model, task="causal_lm_logits", weights_dir=weights_dir)
            if adapter.family != "qwen3_5":
                raise ValueError(f"expected qwen3_5 adapter, got family={adapter.family}")
            self.adapter = adapter.module
            self.layer_index = int(layer_index)
            self.checkpoint_name = checkpoint_name

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            checkpoints = self._debug_layer_checkpoints(input_ids)
            if self.checkpoint_name not in checkpoints:
                available = ", ".join(sorted(checkpoints.keys()))
                raise KeyError(
                    f"unknown checkpoint {self.checkpoint_name!r}; available: {available}"
                )
            return checkpoints[self.checkpoint_name]

        @staticmethod
        def _reshape_for_norm(output: torch.Tensor, *, head_dim: int) -> torch.Tensor:
            return output.reshape(-1, head_dim)

        def _linear_attention_subcheckpoints(
            self,
            layer: torch.nn.Module,
            *,
            hidden_states: torch.Tensor,
            batch_size: int,
            seq_len: int,
        ) -> dict[str, torch.Tensor]:
            checkpoints: dict[str, torch.Tensor] = {}

            mixed_qkv = layer.in_proj_qkv(hidden_states)
            checkpoints["mixer.in_proj_qkv"] = mixed_qkv

            z = layer.in_proj_z(hidden_states)
            checkpoints["mixer.in_proj_z"] = z
            z = z.reshape(batch_size, seq_len, -1, layer.head_v_dim)

            b = layer.in_proj_b(hidden_states)
            checkpoints["mixer.in_proj_b"] = b

            a = layer.in_proj_a(hidden_states)
            checkpoints["mixer.in_proj_a"] = a

            mixed_qkv = mixed_qkv.transpose(1, 2)
            checkpoints["mixer.qkv_transposed"] = mixed_qkv

            if layer.causal_conv1d_fn is not None:
                mixed_qkv = layer.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=layer.conv1d.weight.squeeze(1),
                    bias=layer.conv1d.bias,
                    activation=layer.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = torch.nn.functional.silu(
                    layer.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]]
                )
            checkpoints["mixer.conv_out"] = mixed_qkv

            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(
                mixed_qkv,
                [layer.key_dim, layer.key_dim, layer.value_dim],
                dim=-1,
            )
            checkpoints["mixer.query"] = query
            checkpoints["mixer.key"] = key
            checkpoints["mixer.value"] = value

            query = query.reshape(batch_size, seq_len, -1, layer.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, layer.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, layer.head_v_dim)

            beta = b.sigmoid()
            checkpoints["mixer.beta"] = beta

            g = -layer.A_log.float().exp() * torch.nn.functional.softplus(a.float() + layer.dt_bias)
            checkpoints["mixer.g"] = g

            if layer.num_v_heads // layer.num_k_heads > 1:
                query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
                key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
            checkpoints["mixer.query_repeated"] = query
            checkpoints["mixer.key_repeated"] = key
            checkpoints["mixer.value_reshaped"] = value
            return checkpoints

        def _mlp_subcheckpoints(
            self,
            layer: torch.nn.Module,
            *,
            hidden_states: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            checkpoints: dict[str, torch.Tensor] = {}

            gate_proj = layer.mlp.gate_proj(hidden_states)
            checkpoints["mlp.gate_proj"] = gate_proj

            gate_act = layer.mlp.act_fn(gate_proj)
            checkpoints["mlp.act"] = gate_act

            up_proj = layer.mlp.up_proj(hidden_states)
            checkpoints["mlp.up_proj"] = up_proj

            mul = gate_act * up_proj
            checkpoints["mlp.mul"] = mul

            down_proj = layer.mlp.down_proj(mul)
            checkpoints["mlp.down_proj"] = down_proj
            return checkpoints

        def _debug_layer_checkpoints(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
            backbone = self.adapter.backbone
            layer = backbone.layers[self.layer_index]
            layer_type = backbone.config.layer_types[self.layer_index]
            inputs_embeds = backbone.embed_tokens(input_ids)
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
            text_position_ids = position_ids[0]
            multimodal_position_ids = position_ids[1:]

            selected_layer_types = tuple(backbone.config.layer_types[: self.layer_index + 1])
            needs_causal_mask = any(layer_type == "full_attention" for layer_type in selected_layer_types)
            needs_linear_attn_mask = any(layer_type == "linear_attention" for layer_type in selected_layer_types)
            causal_mask = None
            position_embeddings = None
            if needs_causal_mask:
                causal_mask = self.adapter._create_causal_mask(
                    config=backbone.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=text_position_ids,
                )
                position_embeddings = backbone.rotary_emb(inputs_embeds, multimodal_position_ids)
            linear_attn_mask = backbone._update_linear_attn_mask(None, None) if needs_linear_attn_mask else None

            hidden_states = inputs_embeds
            for i in range(self.layer_index):
                prev_layer = backbone.layers[i]
                prev_layer_type = backbone.config.layer_types[i]
                hidden_states = self._run_qwen35_layer(
                    prev_layer,
                    prev_layer_type,
                    hidden_states,
                    causal_mask=causal_mask,
                    linear_attn_mask=linear_attn_mask,
                    position_ids=text_position_ids,
                    position_embeddings=position_embeddings,
                )

            return self._run_qwen35_layer_checkpoints(
                layer,
                layer_type,
                hidden_states,
                causal_mask=causal_mask,
                linear_attn_mask=linear_attn_mask,
                position_ids=text_position_ids,
                position_embeddings=position_embeddings,
            )

        def _run_qwen35_layer(
            self,
            layer: torch.nn.Module,
            layer_type: str,
            hidden_states: torch.Tensor,
            *,
            causal_mask: torch.Tensor | None,
            linear_attn_mask: torch.Tensor | None,
            position_ids: torch.Tensor,
            position_embeddings: torch.Tensor | None,
        ) -> torch.Tensor:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            if layer_type == "linear_attention":
                hidden_states = layer.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=None,
                    attention_mask=linear_attn_mask,
                )
            elif layer_type == "full_attention":
                hidden_states, _ = layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    position_embeddings=position_embeddings,
                )
            else:
                raise ValueError(f"unsupported qwen3_5 layer type: {layer_type!r}")
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        def _run_qwen35_layer_checkpoints(
            self,
            layer: torch.nn.Module,
            layer_type: str,
            hidden_states: torch.Tensor,
            *,
            causal_mask: torch.Tensor | None,
            linear_attn_mask: torch.Tensor | None,
            position_ids: torch.Tensor,
            position_embeddings: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            checkpoints: dict[str, torch.Tensor] = {}

            residual = hidden_states
            pre_attn_norm = layer.input_layernorm(hidden_states)
            checkpoints["pre_attn_norm"] = pre_attn_norm

            if layer_type == "linear_attention":
                mixer_out = layer.linear_attn(
                    hidden_states=pre_attn_norm,
                    cache_params=None,
                    attention_mask=linear_attn_mask,
                )
                checkpoints.update(
                    self._linear_attention_subcheckpoints(
                        layer.linear_attn,
                        hidden_states=pre_attn_norm,
                        batch_size=pre_attn_norm.shape[0],
                        seq_len=pre_attn_norm.shape[1],
                    )
                )
            elif layer_type == "full_attention":
                if causal_mask is None or position_embeddings is None:
                    raise RuntimeError("full_attention checkpointing requires causal_mask and position_embeddings")
                mixer_out, _ = layer.self_attn(
                    hidden_states=pre_attn_norm,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    position_embeddings=position_embeddings,
                )
            else:
                raise ValueError(f"unsupported qwen3_5 layer type: {layer_type!r}")
            checkpoints["mixer_out"] = mixer_out

            after_attention_residual = residual + mixer_out
            checkpoints["after_attention_residual"] = after_attention_residual

            post_attn_norm = layer.post_attention_layernorm(after_attention_residual)
            checkpoints["post_attn_norm"] = post_attn_norm

            checkpoints.update(self._mlp_subcheckpoints(layer, hidden_states=post_attn_norm))
            mlp_out = checkpoints["mlp.down_proj"]
            checkpoints["mlp_out"] = mlp_out

            after_ffn_residual = after_attention_residual + mlp_out
            checkpoints["after_ffn_residual"] = after_ffn_residual
            return checkpoints


class TestTranspileQwen35LayerDiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _RUN_QWEN35_LAYER_DIFF:
            raise unittest.SkipTest("set CACTUS_RUN_LOCAL_QWEN35_LAYER_DIFF_TEST=1 to run")
        if torch is None:
            raise unittest.SkipTest("torch is not available")
        if AutoModelForCausalLM is None:
            raise unittest.SkipTest("transformers is not available")

        local_model_path = _resolve_local_model_path(_DEFAULT_MODEL_ID)
        cls.model_id = _DEFAULT_MODEL_ID
        cls.model_path = local_model_path
        cls.input_ids_list = _parse_input_ids(_DEFAULT_INPUT_IDS)
        cls.input_ids = torch.tensor([cls.input_ids_list], dtype=torch.long)
        cls.inspect_layer_index = None if _DEFAULT_LAYER_INDEX is None else int(_DEFAULT_LAYER_INDEX)
        cls.weights_dir = None
        candidate_weights_dir = Path(_DEFAULT_WEIGHTS_DIR).expanduser().resolve()
        if candidate_weights_dir.exists():
            cls.weights_dir = str(candidate_weights_dir)
        else:
            raise unittest.SkipTest(
                f"Qwen3.5 converted weights dir not found: {candidate_weights_dir}"
            )

        common_kwargs: dict[str, object] = {
            "local_files_only": True,
            "torch_dtype": torch.float16,
            "device_map": None,
            "low_cpu_mem_usage": True,
        }
        token = os.environ.get("HF_TOKEN")
        if token:
            common_kwargs["token"] = token

        cls.model = AutoModelForCausalLM.from_pretrained(local_model_path, **common_kwargs).eval()

        adapter = canonicalize_model_interface(cls.model, task="causal_lm_logits", weights_dir=cls.weights_dir)
        if adapter.family != "qwen3_5":
            raise unittest.SkipTest(f"expected qwen3_5 adapter, got {adapter.family}")
        cls.layer_types = tuple(adapter.module.backbone.config.layer_types)
        cls.num_layers = int(adapter.module.backbone.config.num_hidden_layers)

        print("")
        print(f"[qwen35-layer-diff] model_id={cls.model_id}")
        print(f"[qwen35-layer-diff] model_path={cls.model_path}")
        print(f"[qwen35-layer-diff] input_ids={cls.input_ids_list}")
        print(f"[qwen35-layer-diff] num_layers={cls.num_layers}")
        print(f"[qwen35-layer-diff] inspect_layer_index={cls.inspect_layer_index}")
        print(f"[qwen35-layer-diff] weights_dir={cls.weights_dir}")

    @classmethod
    def tearDownClass(cls) -> None:
        model = getattr(cls, "model", None)
        if model is not None:
            del cls.model

    def _run_wrapper_diff(self, label: str, wrapper: torch.nn.Module) -> DiffStats:
        wrapper = wrapper.eval()
        with torch.no_grad():
            reference = wrapper(self.input_ids).detach().float().cpu().numpy()

        transpile_wrapper = _WeightsBoundTranspileWrapper(wrapper, weights_dir=self.weights_dir).eval()
        captured = capture_model(transpile_wrapper, (self.input_ids,))
        raw_ir_nodes = len(captured.ir_graph.order)
        canonicalize_exported_graph(captured.ir_graph)
        optimize_graph(captured.ir_graph)
        optimized_ir_nodes = len(captured.ir_graph.order)
        top_ops = _summarize_ops(captured.ir_graph)
        transpiled = transpile_preoptimized_ir(captured.ir_graph)
        transpiled.set_inputs([self.input_ids.cpu().numpy()])
        actual = transpiled.execute()[0].numpy().astype(np.float32)

        diff = np.abs(reference.astype(np.float32) - actual)
        max_abs_diff = float(np.max(diff))
        mean_abs_diff = float(np.mean(diff))

        argmax_ref: int | None = None
        argmax_transpiled: int | None = None
        if reference.ndim == 3:
            argmax_ref = int(np.argmax(reference[0, -1]))
            argmax_transpiled = int(np.argmax(actual[0, -1]))

        print(
            "[qwen35-layer-diff] "
            f"{label} "
            f"shape={tuple(reference.shape)} "
            f"raw_ir_nodes={raw_ir_nodes} "
            f"optimized_ir_nodes={optimized_ir_nodes} "
            f"max_abs_diff={max_abs_diff:.6f} "
            f"mean_abs_diff={mean_abs_diff:.6f} "
            f"top_ops={top_ops}"
        )
        if argmax_ref is not None and argmax_transpiled is not None:
            print(
                "[qwen35-layer-diff] "
                f"{label} "
                f"argmax_ref={argmax_ref} "
                f"argmax_transpiled={argmax_transpiled}"
            )

        return DiffStats(
            label=label,
            shape=tuple(reference.shape),
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            argmax_ref=argmax_ref,
            argmax_transpiled=argmax_transpiled,
            raw_ir_nodes=raw_ir_nodes,
            optimized_ir_nodes=optimized_ir_nodes,
            top_ops=top_ops,
        )

    def test_qwen35_layer_by_layer_debug_diff(self) -> None:
        if self.inspect_layer_index is None:
            layer_stats: list[DiffStats] = []
            for layer_index in range(self.num_layers):
                wrapper = Qwen35CheckpointWrapper(
                    self.model,
                    checkpoint_index=layer_index,
                    weights_dir=self.weights_dir,
                )
                layer_stats.append(self._run_wrapper_diff(f"layer_{layer_index}", wrapper))

            norm_wrapper = Qwen35FinalNormWrapper(self.model, weights_dir=self.weights_dir)
            norm_stats = self._run_wrapper_diff("final_norm", norm_wrapper)
            logits_stats = self._run_wrapper_diff(
                "logits",
                Qwen35LogitsWrapper(self.model, weights_dir=self.weights_dir),
            )

            worst_layer = max(layer_stats, key=lambda item: item.max_abs_diff)
            print(
                "[qwen35-layer-diff] "
                f"worst_layer={worst_layer.label} "
                f"worst_layer_max_abs_diff={worst_layer.max_abs_diff:.6f} "
                f"worst_layer_mean_abs_diff={worst_layer.mean_abs_diff:.6f}"
            )
            print(
                "[qwen35-layer-diff] "
                f"final_norm_max_abs_diff={norm_stats.max_abs_diff:.6f} "
                f"logits_max_abs_diff={logits_stats.max_abs_diff:.6f}"
            )

            self.assertEqual(len(layer_stats), self.num_layers)
            self.assertTrue(np.isfinite(logits_stats.max_abs_diff))
            self.assertTrue(np.isfinite(logits_stats.mean_abs_diff))
            return

        layer_index = int(self.inspect_layer_index)
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError(
                f"CACTUS_TEST_QWEN35_LAYER_INDEX out of range: {layer_index} (have {self.num_layers})"
            )

        layer_type = self.layer_types[layer_index]
        checkpoint_names: tuple[str, ...]
        if layer_type == "linear_attention":
            checkpoint_names = (
                "pre_attn_norm",
                "mixer.in_proj_qkv",
                "mixer.in_proj_z",
                "mixer.in_proj_b",
                "mixer.in_proj_a",
                "mixer.qkv_transposed",
                "mixer.conv_out",
                "mixer.query",
                "mixer.key",
                "mixer.value",
                "mixer.beta",
                "mixer.g",
                "mixer.query_repeated",
                "mixer.key_repeated",
                "mixer.value_reshaped",
                "mixer_out",
                "after_attention_residual",
                "post_attn_norm",
                "mlp.gate_proj",
                "mlp.act",
                "mlp.up_proj",
                "mlp.mul",
                "mlp.down_proj",
                "mlp_out",
                "after_ffn_residual",
            )
        else:
            checkpoint_names = (
                "pre_attn_norm",
                "mixer_out",
                "after_attention_residual",
                "post_attn_norm",
                "mlp.gate_proj",
                "mlp.act",
                "mlp.up_proj",
                "mlp.mul",
                "mlp.down_proj",
                "mlp_out",
                "after_ffn_residual",
            )
        stats: list[DiffStats] = []
        for checkpoint_name in checkpoint_names:
            wrapper = Qwen35LayerCheckpointWrapper(
                self.model,
                layer_index=layer_index,
                checkpoint_name=checkpoint_name,
                weights_dir=self.weights_dir,
            )
            stats.append(self._run_wrapper_diff(f"layer_{layer_index}.{checkpoint_name}", wrapper))

        logits_stats = self._run_wrapper_diff(
            "logits",
            Qwen35LogitsWrapper(self.model, weights_dir=self.weights_dir),
        )

        worst = max(stats, key=lambda item: item.max_abs_diff)
        print(
            "[qwen35-layer-diff] "
            f"layer_index={layer_index} "
            f"worst_checkpoint={worst.label} "
            f"worst_checkpoint_max_abs_diff={worst.max_abs_diff:.6f} "
            f"worst_checkpoint_mean_abs_diff={worst.mean_abs_diff:.6f} "
            f"logits_max_abs_diff={logits_stats.max_abs_diff:.6f}"
        )

        self.assertTrue(np.isfinite(logits_stats.max_abs_diff))
        self.assertTrue(np.isfinite(logits_stats.mean_abs_diff))
