from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldPattern:
    name: str
    source_models: tuple[str, ...]
    semantic_ops: tuple[str, ...]
    description: str


GOLD_PATTERNS: tuple[GoldPattern, ...] = (
    GoldPattern(
        name="decoder_attention_gqa",
        source_models=("model_gemma.cpp", "model_qwen.cpp", "model_lfm2.cpp", "model_gemma4.cpp"),
        semantic_ops=("rms_norm", "rope", "attention", "attention_int8_hybrid"),
        description=(
            "Q/K/V projections from the same normalized hidden state; Q and K optionally get per-head RMSNorm, "
            "Q/K are rotary-embedded, then Cactus attention consumes native Q-head and KV-head counts."
        ),
    ),
    GoldPattern(
        name="gated_mlp_gelu",
        source_models=("model_gemma.cpp", "model_gemma4.cpp"),
        semantic_ops=("matmul", "gelu", "multiply"),
        description="Two projection branches from the same normalized input: GELU(gate) * up, then a down projection.",
    ),
    GoldPattern(
        name="gated_mlp_silu",
        source_models=("model_qwen.cpp", "model_lfm2.cpp"),
        semantic_ops=("matmul", "silu", "multiply"),
        description="Two projection branches from the same normalized input: SiLU(gate) * up, then a down projection.",
    ),
    GoldPattern(
        name="decoder_block_post_attn_norm",
        source_models=("model_gemma.cpp", "model_gemma4.cpp", "model_gemma3n.cpp"),
        semantic_ops=("rms_norm", "attention", "add_clipped"),
        description=(
            "Pre-norm attention block with post-attention RMSNorm before the residual add, followed by a second "
            "pre-FFN RMSNorm and a post-FFN RMSNorm before the final residual add."
        ),
    ),
    GoldPattern(
        name="decoder_block_simple_residual",
        source_models=("model_qwen.cpp", "model_lfm2.cpp"),
        semantic_ops=("rms_norm", "attention", "add"),
        description=(
            "Pre-norm attention block with a direct residual add after attention, followed by a second RMSNorm and "
            "a gated MLP branch with another direct residual add."
        ),
    ),
    GoldPattern(
        name="gemma4_partial_rope_attention",
        source_models=("model_gemma4.cpp",),
        semantic_ops=("rms_norm", "rope", "concat", "attention"),
        description=(
            "Gemma4 attention where only a prefix of the head dimension is rotary-embedded, optionally with shared "
            "K/V heads and sliding-window attention."
        ),
    ),
    GoldPattern(
        name="sliding_window_attention_mask",
        source_models=("model_gemma.cpp", "model_gemma4.cpp"),
        semantic_ops=("attention",),
        description=(
            "Exported PyTorch may build a boolean mask with diff/cumsum/comparisons, but the handwritten Cactus "
            "models encode the same behavior directly as an attention `window_size`."
        ),
    ),
)


def gold_patterns_by_name() -> dict[str, GoldPattern]:
    return {pattern.name: pattern for pattern in GOLD_PATTERNS}
