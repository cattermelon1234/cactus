from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Iterable
from typing import Sequence

from .config_utils import cfg_get
from .config_utils import detect_model_type
from .weight_patterns import EMBED_NAMES
from .weight_patterns import LAYER_PREFIXES
from .weight_patterns import OUTPUT_NAMES
from .weight_patterns import OUTPUT_NORM_NAMES


@dataclass(frozen=True)
class ExportTarget:
    filename: str
    precision: str | None = None
    transpose: bool = False
    kind: str = "weight"
    source_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class DerivedSplit:
    # Supported values: "dim0_equal", "dim1_equal", "dim0_sizes"
    mode: str
    outputs: tuple[ExportTarget, ...]
    size_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExportRule:
    source_patterns: tuple[str, ...]
    primary: ExportTarget
    derived_split: DerivedSplit | None = None
    required: bool = False


@dataclass(frozen=True)
class AdapterContext:
    root_config: object
    text_config: object
    model_config: dict[str, object]
    detected_model_type: str
    precision: str


@dataclass(frozen=True)
class PlannedTensorExport:
    source_name: str
    output_name: str
    kind: str
    precision: str
    transpose: bool = False
    shard_index: int | None = None
    shard_count: int | None = None
    split_mode: str | None = None
    source_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedMissingExport:
    scope: str
    output_name: str
    source_patterns: tuple[str, ...]


@dataclass
class ExportPlan:
    adapter_name: str
    tensor_exports: list[PlannedTensorExport] = field(default_factory=list)
    missing: list[PlannedMissingExport] = field(default_factory=list)
    matched_source_names: set[str] = field(default_factory=set)


class ArchitectureAdapter:
    adapter_name = "base"
    model_types: tuple[str, ...] = ()
    layer_prefixes: tuple[str, ...] = tuple(LAYER_PREFIXES)

    @classmethod
    def matches(
        cls,
        *,
        detected_model_type: str,
        root_config: object,
        text_config: object,
        state_keys: Iterable[str],
    ) -> bool:
        del root_config, text_config, state_keys
        return detected_model_type in cls.model_types

    def global_rules(self, ctx: AdapterContext) -> Sequence[ExportRule]:
        del ctx
        return ()

    def layer_rules(self, layer_idx: int, ctx: AdapterContext) -> Sequence[ExportRule]:
        del layer_idx, ctx
        return ()

    def layer_prefix_candidates(self, layer_idx: int, ctx: AdapterContext) -> Sequence[str]:
        del ctx
        return tuple(prefix.format(i=layer_idx) for prefix in self.layer_prefixes)

    def build_plan(self, state_dict: dict[str, object], ctx: AdapterContext) -> ExportPlan:
        plan = ExportPlan(adapter_name=self.adapter_name)

        for rule in self.global_rules(ctx):
            self._apply_rule(
                plan=plan,
                state_dict=state_dict,
                scope="global",
                full_names=rule.source_patterns,
                rule=rule,
            )

        num_layers = int(ctx.model_config.get("num_layers", 0) or 0)
        for layer_idx in range(num_layers):
            prefixes = self._existing_layer_prefixes(
                state_dict=state_dict,
                layer_idx=layer_idx,
                ctx=ctx,
            )
            if not prefixes:
                plan.missing.append(
                    PlannedMissingExport(
                        scope=f"layer_{layer_idx}",
                        output_name="<no-layer-prefix>",
                        source_patterns=("<no-matching-prefix>",),
                    )
                )
                continue

            for prefix in prefixes:
                for rule in self.layer_rules(layer_idx, ctx):
                    full_names = tuple(prefix + pattern for pattern in rule.source_patterns)
                    self._apply_rule(
                        plan=plan,
                        state_dict=state_dict,
                        scope=f"layer_{layer_idx}",
                        full_names=full_names,
                        rule=rule,
                    )

        return plan

    def _existing_layer_prefixes(
        self,
        *,
        state_dict: dict[str, object],
        layer_idx: int,
        ctx: AdapterContext,
    ) -> tuple[str, ...]:
        matches: list[str] = []
        keys = tuple(state_dict.keys())
        for prefix in self.layer_prefix_candidates(layer_idx, ctx):
            if any(key.startswith(prefix) for key in keys):
                matches.append(prefix)
        return tuple(matches)

    def _apply_rule(
        self,
        *,
        plan: ExportPlan,
        state_dict: dict[str, object],
        scope: str,
        full_names: tuple[str, ...],
        rule: ExportRule,
    ) -> None:
        source_name = next((name for name in full_names if name in state_dict), None)
        if source_name is None:
            if rule.required:
                plan.missing.append(
                    PlannedMissingExport(
                        scope=scope,
                        output_name=rule.primary.filename,
                        source_patterns=full_names,
                    )
                )
            return

        plan.tensor_exports.append(
            PlannedTensorExport(
                source_name=source_name,
                output_name=rule.primary.filename,
                kind=rule.primary.kind,
                precision=rule.primary.precision or "FP16",
                transpose=rule.primary.transpose,
                source_names=rule.primary.source_names,
            )
        )
        plan.matched_source_names.add(source_name)

        if rule.derived_split is None:
            return

        for shard_index, derived_target in enumerate(rule.derived_split.outputs):
            plan.tensor_exports.append(
                PlannedTensorExport(
                    source_name=source_name,
                    output_name=derived_target.filename,
                    kind=derived_target.kind,
                    precision=derived_target.precision or rule.primary.precision or "FP16",
                    transpose=derived_target.transpose,
                    shard_index=shard_index,
                    shard_count=len(rule.derived_split.outputs),
                    split_mode=rule.derived_split.mode,
                    source_names=derived_target.source_names,
                )
            )


def _global_export_rules(*, precision: str) -> list[ExportRule]:
    rules: list[ExportRule] = []
    for name in EMBED_NAMES:
        rules.append(
            ExportRule(
                source_patterns=(name,),
                primary=ExportTarget(
                    filename="token_embeddings.weights",
                    precision=precision,
                    kind="embedding",
                ),
            )
        )
    for name in OUTPUT_NORM_NAMES:
        rules.append(
            ExportRule(
                source_patterns=(name,),
                primary=ExportTarget(
                    filename="output_norm.weights",
                    precision=precision,
                ),
            )
        )
    for name in OUTPUT_NAMES:
        rules.append(
            ExportRule(
                source_patterns=(name,),
                primary=ExportTarget(
                    filename="output_weight.weights",
                    precision=precision,
                ),
            )
        )
    return rules


def _decoder_layer_rules(layer_idx: int, *, precision: str, include_qk_norm: bool) -> list[ExportRule]:
    rules = [
        ExportRule(
            source_patterns=("input_layernorm.weight", "ln_1.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_input_norm.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("self_attn.q_proj.weight", "attn.q_proj.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_attn_q.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("self_attn.k_proj.weight", "attn.k_proj.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_attn_k.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("self_attn.v_proj.weight", "attn.v_proj.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_attn_v.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("self_attn.o_proj.weight", "attn.o_proj.weight", "attn.c_proj.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_attn_output.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("mlp.gate_proj.weight", "feed_forward.w1.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_ffn_gate.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("mlp.up_proj.weight", "feed_forward.w3.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_ffn_up.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("mlp.down_proj.weight", "feed_forward.w2.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_ffn_down.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("post_attention_layernorm.weight", "ln_2.weight"),
            primary=ExportTarget(f"layer_{layer_idx}_post_attn_norm.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("pre_feedforward_layernorm.weight",),
            primary=ExportTarget(f"layer_{layer_idx}_pre_ffn_norm.weights", precision=precision),
        ),
        ExportRule(
            source_patterns=("post_feedforward_layernorm.weight",),
            primary=ExportTarget(f"layer_{layer_idx}_post_ffn_norm.weights", precision=precision),
        ),
    ]
    if include_qk_norm:
        rules.extend(
            [
                ExportRule(
                    source_patterns=("self_attn.q_norm.weight", "self_attn.q_layernorm.weight"),
                    primary=ExportTarget(f"layer_{layer_idx}_attn_q_norm.weights", precision=precision),
                ),
                ExportRule(
                    source_patterns=("self_attn.k_norm.weight", "self_attn.k_layernorm.weight"),
                    primary=ExportTarget(f"layer_{layer_idx}_attn_k_norm.weights", precision=precision),
                ),
            ]
        )
    return rules


class GenericDecoderAdapter(ArchitectureAdapter):
    adapter_name = "generic_decoder"
    model_types = ("llama", "smol", "mistral", "phi", "bert", "qwen", "gemma")

    def global_rules(self, ctx: AdapterContext) -> Sequence[ExportRule]:
        return _global_export_rules(precision=ctx.precision)

    def layer_rules(self, layer_idx: int, ctx: AdapterContext) -> Sequence[ExportRule]:
        include_qk_norm = bool(ctx.detected_model_type in ("gemma", "qwen"))
        return _decoder_layer_rules(layer_idx, precision=ctx.precision, include_qk_norm=include_qk_norm)


class GemmaAdapter(GenericDecoderAdapter):
    adapter_name = "gemma"
    model_types = ("gemma",)


class LlamaLikeAdapter(GenericDecoderAdapter):
    adapter_name = "llama_like"
    model_types = ("llama", "smol")


class QwenAdapter(GenericDecoderAdapter):
    adapter_name = "qwen"
    model_types = ("qwen", "qwen3_5")

    def layer_rules(self, layer_idx: int, ctx: AdapterContext) -> Sequence[ExportRule]:
        rules = list(_decoder_layer_rules(layer_idx, precision=ctx.precision, include_qk_norm=True))
        rules.extend(
            [
                ExportRule(
                    source_patterns=("linear_attn.in_proj_qkv.weight",),
                    primary=ExportTarget(
                        f"layer_{layer_idx}_linear_attn_qkv.weights",
                        precision=ctx.precision,
                    ),
                    derived_split=DerivedSplit(
                        mode="dim0_sizes",
                        outputs=(
                            ExportTarget(f"layer_{layer_idx}_linear_attn_q.weights", precision=ctx.precision),
                            ExportTarget(f"layer_{layer_idx}_linear_attn_k.weights", precision=ctx.precision),
                            ExportTarget(f"layer_{layer_idx}_linear_attn_v.weights", precision=ctx.precision),
                        ),
                        size_keys=("linear_q_proj_dim", "linear_k_proj_dim", "linear_v_proj_dim"),
                    ),
                ),
                ExportRule(
                    source_patterns=("linear_attn.out_proj.weight",),
                    primary=ExportTarget(f"layer_{layer_idx}_linear_attn_output.weights", precision=ctx.precision),
                ),
                ExportRule(
                    source_patterns=("linear_attn.norm.weight",),
                    primary=ExportTarget(f"layer_{layer_idx}_linear_attn_norm.weights", precision=ctx.precision),
                ),
                ExportRule(
                    source_patterns=("linear_attn.conv1d.weight",),
                    primary=ExportTarget(f"layer_{layer_idx}_linear_attn_conv1d.weights", precision=ctx.precision),
                ),
            ]
        )
        return rules


class GPT2Adapter(ArchitectureAdapter):
    adapter_name = "gpt2"
    model_types = ("gpt2",)
    layer_prefixes = ("transformer.h.{i}.",)

    def global_rules(self, ctx: AdapterContext) -> Sequence[ExportRule]:
        return _global_export_rules(precision=ctx.precision)

    def layer_rules(self, layer_idx: int, ctx: AdapterContext) -> Sequence[ExportRule]:
        return (
            ExportRule(
                source_patterns=("ln_1.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_input_norm.weights", precision=ctx.precision),
            ),
            ExportRule(
                source_patterns=("attn.c_attn.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_attn_qkv.weights", precision=ctx.precision),
                derived_split=DerivedSplit(
                    mode="dim1_equal",
                    outputs=(
                        ExportTarget(f"layer_{layer_idx}_attn_q.weights", precision=ctx.precision),
                        ExportTarget(f"layer_{layer_idx}_attn_k.weights", precision=ctx.precision),
                        ExportTarget(f"layer_{layer_idx}_attn_v.weights", precision=ctx.precision),
                    ),
                ),
            ),
            ExportRule(
                source_patterns=("attn.c_proj.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_attn_output.weights", precision=ctx.precision),
            ),
            ExportRule(
                source_patterns=("mlp.c_fc.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_mlp_fc1.weights", precision=ctx.precision),
            ),
            ExportRule(
                source_patterns=("mlp.c_proj.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_mlp_fc2.weights", precision=ctx.precision),
            ),
            ExportRule(
                source_patterns=("ln_2.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_post_attn_norm.weights", precision=ctx.precision),
            ),
        )


class WhisperAdapter(ArchitectureAdapter):
    adapter_name = "whisper"
    model_types = ("whisper",)
    layer_prefixes = ("encoder.layers.{i}.", "decoder.layers.{i}.")

    def global_rules(self, ctx: AdapterContext) -> Sequence[ExportRule]:
        return (
            ExportRule(
                source_patterns=("decoder.embed_tokens.weight",),
                primary=ExportTarget("token_embeddings.weights", precision=ctx.precision, kind="embedding"),
            ),
            ExportRule(
                source_patterns=("proj_out.weight",),
                primary=ExportTarget("output_weight.weights", precision=ctx.precision),
            ),
        )

    def layer_rules(self, layer_idx: int, ctx: AdapterContext) -> Sequence[ExportRule]:
        del ctx
        return (
            ExportRule(
                source_patterns=("self_attn.q_proj.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_self_attn_q.weights", precision="FP16"),
            ),
            ExportRule(
                source_patterns=("self_attn.k_proj.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_self_attn_k.weights", precision="FP16"),
            ),
            ExportRule(
                source_patterns=("self_attn.v_proj.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_self_attn_v.weights", precision="FP16"),
            ),
            ExportRule(
                source_patterns=("self_attn.out_proj.weight",),
                primary=ExportTarget(f"layer_{layer_idx}_self_attn_output.weights", precision="FP16"),
            ),
        )


DEFAULT_ADAPTERS: tuple[type[ArchitectureAdapter], ...] = (
    GemmaAdapter,
    QwenAdapter,
    LlamaLikeAdapter,
    GPT2Adapter,
    WhisperAdapter,
    GenericDecoderAdapter,
)


def build_adapter_context(root_config: object, *, precision: str) -> AdapterContext:
    text_config = cfg_get(root_config, "text_config", None)
    cfg = text_config if text_config is not None else root_config
    detected_model_type = detect_model_type(cfg, root_config)
    model_config = {
        "num_layers": int(
            cfg_get(cfg, "num_hidden_layers", cfg_get(cfg, "num_layers", 0) or 0)
        ),
        "linear_q_proj_dim": int(cfg_get(cfg, "linear_q_proj_dim", 0) or 0),
        "linear_k_proj_dim": int(cfg_get(cfg, "linear_k_proj_dim", 0) or 0),
        "linear_v_proj_dim": int(cfg_get(cfg, "linear_v_proj_dim", 0) or 0),
    }
    return AdapterContext(
        root_config=root_config,
        text_config=cfg,
        model_config=model_config,
        detected_model_type=detected_model_type,
        precision=precision,
    )


def select_adapter(
    *,
    root_config: object,
    state_keys: Iterable[str],
    precision: str,
    registry: Sequence[type[ArchitectureAdapter]] = DEFAULT_ADAPTERS,
) -> tuple[ArchitectureAdapter, AdapterContext]:
    ctx = build_adapter_context(root_config, precision=precision)
    keys = tuple(state_keys)
    for adapter_cls in registry:
        if adapter_cls.matches(
            detected_model_type=ctx.detected_model_type,
            root_config=root_config,
            text_config=ctx.text_config,
            state_keys=keys,
        ):
            return adapter_cls(), ctx
    return GenericDecoderAdapter(), ctx
