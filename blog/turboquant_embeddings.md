---
title: "TurboQuant-H: Hadamard Rotation for 2-Bit Embedding Quantization"
description: "TurboQuant compresses KV cache to 1-3 bits. We introduce TurboQuant-H, a simplified offline variant using Hadamard rotation and per-group Lloyd-Max codebooks, applied to per-layer embedding tables where it matters most: models where embeddings are 60%+ of total weight storage."
keywords: ["TurboQuant-H", "TurboQuant", "vector quantization", "embeddings", "2-bit", "on-device AI", "Gemma 4", "Gemma 3n", "per-layer embeddings", "AltUp", "Hadamard", "memory reduction"]
author: "Karen Mosoyan, Noah Cylich, Henry Ndubuaku"
date: 2026-04-21
tags: ["quantization", "embeddings", "TurboQuant-H", "Gemma", "on-device", "memory optimization"]
---

# Extending TurboQuant To Embeddings Can Reduce Memory & Storage by 40% In Embedding-Dominated Models Like Gemma-270m, Gemma 3n & Gemma 4n

*By Karen Mosoyan, Noah Cylich, Henry Ndubuaku*

## Abstract

TurboQuant (Zandieh et al., ICLR 2026) compresses KV cache vectors to 1-3 bits via random orthogonal rotation, optimal scalar quantization, and QJL bias correction. We introduce **TurboQuant-H**, a simplified offline variant that replaces random rotation with Hadamard rotation, uses per-group Lloyd-Max codebooks, and drops the QJL correction stage. We apply TurboQuant-H to per-layer input (PLI) embedding tables in Gemma 4 E2B, where embeddings constitute 60.6% of total model weight. On Gemma 4 E2B, TurboQuant-H compresses PLI weights from 2,496 MB to 624 MB (4x) at 2.125 effective bits per dimension, reducing total LLM storage by 40% (4,790 MB → 2,918 MB) with a perplexity increase of 0.06 (1.85 → 1.91) and no measured speed regression.

## 1. Introduction

TurboQuant (ICLR 2026) compresses KV cache vectors to 1-3 bits with near-zero quality loss. The technique is elegant: rotate vectors with a random orthogonal matrix, exploit the resulting Beta distribution to apply optimal scalar quantizers per coordinate, then correct inner product bias with a 1-bit QJL residual. The paper demonstrates quality neutrality at 3.5 bits and marginal degradation at 2.5 bits on Llama-3.1-8B and Ministral-7B.

But TurboQuant was designed for KV cache, vectors generated at runtime during inference. There's a catch: mobile devices and wearables need small models, which we found to significantly degrade when KV cache goes below INT4. We in fact keep KV cache at INT8 on Cactus to ensure correctness. This makes applying TurboQuant to Cactus KV workloads tricky.

However, with the emergence of per-layer embedding architectures (each layer has its own embedding lookup), these embeddings dominate the parameter count of models like the Gemma E-series. For instance, Gemma E2B has 2.3B effective parameters but 5.1B total, because the per-layer embeddings alone account for the difference. That bloats memory and storage footprint by more than 2x. There is a need to re-visit embedding quantisation.

## 2. Background: The Embedding Problem

Most quantization research focuses on linear layer weights and activations. Embeddings are treated as untouchable lookup tables, typically kept at FP16 or at best INT8 while everything else goes to INT4. This made sense when embeddings were a small fraction of total parameters. That assumption broke with modern architectures.

### 2.1 Gemma-270m: More Embedding Than Model

Gemma-3-270m has a 256K vocabulary with a 1152-dimensional embedding. That's 295M parameters just in the embedding table, for a model nominally described as 270M parameters. The embedding table is larger than all other weights combined. At INT8, the embedding alone is 281 MB. The attention and FFN weights for all layers fit in roughly 200 MB.

### 2.2 Per-Layer Embeddings Multiply the Problem

Gemma 4 E2B uses AltUp, a technique where each of the 35 transformer layers gets its own embedding projection from the 262K-token vocabulary. Instead of one shared embedding table, you have a shared table plus a per-layer table. The numbers on Cactus's current INT4 weights:

| Component | Size | % of Model |
|---|---|---|
| `token_embeddings` (shared) | 408 MB | 8.7% |
| `embed_tokens_per_layer` (35 layers) | 2,496 MB | 52.1% |
| **Total embedding storage** | **2,904 MB** | **60.6%** |
| All other weights (attention, FFN, norms, encoders) | 1,886 MB | 39.4% |
| **Total model** | **4,790 MB** | 100% |

The per-layer embedding table is 2.5 GB. More than half the model. This is not unique to Gemma 4. The AltUp design pattern, where per-layer vocabulary projections replace a single shared embedding, is becoming standard for models that need large vocabularies (262K tokens for multilingual coverage) without proportionally large hidden dimensions. Gemma 3n uses the same architecture. Any model that follows this pattern will be embedding-dominated.

## 3. TurboQuant-H

TurboQuant-H shares the core insight from TurboQuant; rotation concentrates coordinates into a well-behaved distribution, enabling aggressive scalar quantization — but simplifies the pipeline for offline weight quantization.

### 3.1 Comparison with TurboQuant

| | TurboQuant (Zandieh et al.) | TurboQuant-H (this work) |
|---|---|---|
| **Target** | KV cache (runtime activations) | Embedding weight tables (offline) |
| **Rotation** | Random orthogonal matrix via QR of Gaussian, $O(d^2)$ | Hadamard matrix, $O(N \log N)$, symmetric = self-inverse |
| **Quantizer** | Per-coordinate scalar quantizer (precomputed for Beta distribution) | Per-group Lloyd-Max codebook (trained on actual weight distribution) |
| **Codebook** | Implicit (quantization levels derived from Beta CDF) | Explicit FP16 centroids per group (0.125 bits overhead at group-128) |
| **Bias correction** | Two-stage: MSE quantizer at $b-1$ bits + 1-bit QJL residual | Single-stage: no QJL correction |
| **When it runs** | Every forward pass during inference | Once during weight conversion |
| **Bit width** | 2.5-bit and 3.5-bit | 2-bit (+0.125 codebook overhead ≈ 2.125 effective) |

### 3.2 Formal Description

Let $\mathbf{E} \in \mathbb{R}^{V \times D}$ be the PLI embedding matrix with vocabulary size $V = 262{,}144$ and embedding dimension $D = 8{,}190$. We partition each row into $\lceil D/G \rceil$ groups of $G = 128$ contiguous elements.

**Quantization** (offline, during weight conversion):

For each group $g$ with elements $\mathbf{x}_g \in \mathbb{R}^G$:

$$\hat{\mathbf{x}}_g = \mathbf{H}_G \cdot \mathbf{x}_g$$

where $\mathbf{H}_G$ is the $G \times G$ normalized Hadamard matrix satisfying $\mathbf{H}_G^T \mathbf{H}_G = \mathbf{I}$ and $\mathbf{H}_G = \mathbf{H}_G^T$.

Train a Lloyd-Max codebook $\mathcal{C}_g = \{c_1, c_2, c_3, c_4\}$ (at $b = 2$ bits, 4 centroids) for each group by minimizing:

$$\mathcal{C}_g^* = \arg\min_{\mathcal{C}} \sum_{i=1}^{G} \min_{c \in \mathcal{C}} \left( \hat{x}_{g,i} - c \right)^2$$

Quantize each rotated element to its nearest centroid:

$$q_{g,i} = \arg\min_{j \in \{1,\ldots,2^b\}} \left| \hat{x}_{g,i} - c_j \right|$$

Store the 2-bit indices $q_{g,i}$ and FP16 codebook $\mathcal{C}_g$ per group.

**Dequantization** (at inference):

$$\tilde{\mathbf{x}}_g = \mathbf{H}_G \cdot \text{scatter}(\mathcal{C}_g, \mathbf{q}_g)$$

where $\text{scatter}(\mathcal{C}_g, \mathbf{q}_g)_i = c_{q_{g,i}}$ maps indices back to centroids. Since $\mathbf{H}_G$ is symmetric and orthogonal, the inverse rotation is the same forward transform. No transpose is needed.

**Effective bit rate:**

$$b_{\text{eff}} = b + \frac{2^b \cdot 16}{G} = 2 + \frac{4 \cdot 16}{128} = 2 + 0.125 = 2.125 \text{ bits/element}$$

The 0.125-bit codebook overhead is negligible at group size 128.

### 3.3 The Quantization Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION (offline, cactus convert)            │
│                                                                      │
│  PLI Matrix E ∈ ℝ^{262K × 8190}                                      │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────┐                                             │
│  │ Partition into       │  Each row → ⌈8190/128⌉ = 64 groups         │
│  │ groups of G=128      │  of 128 contiguous elements                │
│  └──────────┬──────────┘                                             │
│             ▼                                                        │
│  ┌─────────────────────┐                                             │
│  │ Hadamard rotation    │  x̂_g = H_128 · x_g                         │
│  │ per group            │  O(G log G) butterfly, concentrates        │
│  │                      │  coordinates for uniform quantization      │
│  └──────────┬──────────┘                                             │
│             ▼                                                        │
│  ┌─────────────────────┐                                             │
│  │ Lloyd-Max codebook   │  Train 4 centroids (2-bit) per group       │
│  │ per group            │  C_g = {c₁, c₂, c₃, c₄} in FP16            │
│  └──────────┬──────────┘                                             │
│             ▼                                                        │
│  ┌─────────────────────┐                                             │
│  │ Quantize by          │  q_{g,i} = argmin_j |x̂_{g,i} - c_j|        │
│  │ proximity            │  Store 2-bit indices per element           │
│  └──────────┬──────────┘                                             │
│             ▼                                                        │
│  Output: 2-bit index tensor + FP16 codebooks (per group)             │
│  Effective: 2.125 bits/element                                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    DEQUANTIZATION (inference, per token)             │
│                                                                      │
│  Token IDs                                                           │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────┐                                             │
│  │ Gather 2-bit indices│  Look up row from compressed table          │
│  │ + codebook per group│  4× less memory bandwidth than INT8         │
│  └──────────┬──────────┘                                             │
│             ▼                                                        │
│  ┌─────────────────────┐                                             │
│  │ Scatter codebook    │  Replace 2-bit indices with FP16            │
│  │ values              │  centroid values from C_g                   │
│  └──────────┬──────────┘                                             │
│             ▼                                                        │
│  ┌─────────────────────┐                                             │
│  │ Hadamard rotation   │  x̃_g = H_128 · scatter(C_g, q_g)            │
│  │ (same as forward,   │  H is symmetric: H = H^T = H^{-1}           │
│  │  no transpose)      │  O(G log G) butterfly per group             │
│  └──────────┬──────────┘                                             │
│             ▼                                                        │
│  FP16 embedding → feed to transformer layer                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.4 Design Decisions

**Why Hadamard instead of random orthogonal?** The Hadamard matrix is deterministic, $O(N \log N)$ to apply via the butterfly factorization (same structure as the FFT), and its own inverse ($\mathbf{H} = \mathbf{H}^T = \mathbf{H}^{-1}$). For offline weight quantization we don't need the data-oblivious guarantees of a random rotation, we have full access to the weight data at conversion time. The Hadamard rotation still concentrates coordinates, which is all we need to make low-bit scalar quantization work.

**Why no QJL correction?** TurboQuant's second stage exists because MSE-optimal quantizers introduce multiplicative bias in inner product estimation. At 1-bit, $\mathbb{E}[\langle \mathbf{y}, Q(\mathbf{x}) \rangle] = \frac{2}{\pi} \langle \mathbf{y}, \mathbf{x} \rangle$, a 36% shrinkage. The QJL residual corrects this at the cost of 1 additional bit per dimension. But we're quantizing at 2 bits with a trained codebook, not a precomputed scalar quantizer. The per-group Lloyd-Max codebook already minimizes distortion over the actual weight distribution, and the Hadamard rotation ensures the codebook sees well-spread inputs. At 2 bits with group-128, the inner product bias is small enough that the downstream perplexity impact is negligible (PPL 1.91 vs 1.85). Adding QJL would cost an extra bit per dimension for a correction that isn't needed at this operating point.

**Why per-group codebooks instead of per-coordinate?** TurboQuant can use a single precomputed quantizer because random rotation makes all coordinates identically distributed (each follows $\text{Beta}(\frac{1}{2}, \frac{d-1}{2})$ on the unit sphere). Hadamard rotation concentrates coordinates but doesn't make them identically distributed, there are structured patterns from the butterfly network. Per-group codebooks adapt to these patterns. We tried joint codebooks across groups; per-group was consistently better.

**Why group size 128?** We swept group sizes from 32 to 512:

| Group size | Codebook overhead (bits/elem) | Hadamard cost | Quality |
|---|---|---|---|
| 32 | 0.500 | fastest | degraded (high overhead eats bit budget) |
| 64 | 0.250 | fast | good |
| **128** | **0.125** | **fast** | **best (sweet spot)** |
| 256 | 0.063 | moderate | slightly worse (less uniform within group) |
| 512 | 0.031 | slow | worse (distribution spreads too thin) |

Group-128 gives 0.125 bits overhead, a fast butterfly ($7 \times 128/8 = 112$ NEON FMAs per group), and the best quality.

## 4. Results

### 4.1 Perplexity

Evaluated on Gemma 4 E2B. Evaluation set: 128 self-generated WildChat-1M completions from our `trajectories.jsonl` calibration set, completion-only NLL, 24,438 scored tokens.

| Variant | Avg bits | PPL |
|---|---|---|
| HuggingFace BF16 | 16 | 1.2892 |
| Cactus default (INT4 linears + INT8 PLI + INT8 token-emb) | ~6.3 | 1.8547 |
| **Cactus + TurboQuant-H PLI** | **~3.8** | **1.9111** |

Perplexity moves from 1.85 to 1.91. A $\Delta$PPL of 0.06 on a 24K-token eval set, within noise for practical use. No measured speed regression.

### 4.2 Disk Footprint

| Variant | Size (MB) | Factor |
|---|---|---|
| HuggingFace FP16 snapshot | ~10,240 | 1.00× |
| Cactus default (INT4 linears + INT8 PLI + INT8 emb) | ~4,790 | 0.47× |
| **Cactus + TurboQuant-H PLI** | **2,918** | **0.29×** |

The PLI table specifically: **2,496 MB → 624 MB, a 4× reduction.**

Total LLM weight reduction: **40%** from the Cactus baseline. Including the vision and audio encoders (untouched by this change), the overall model reduction is **30%**.

For Gemma 4 E2B, that's the difference between a 4.8 GB model and a 2.9 GB model. On a 4 GB RAM Android device, that's the difference between fitting and not fitting.

For Gemma-270m the situation is even more extreme: the embedding table (295M params) is larger than all other weights combined, so the same technique would cut total model size roughly in half.

## 5. Inference Path

### 5.1 Before (Cactus default)

```
token_ids → gather from INT8 table → dequantize to FP16 → feed to transformer
```

### 5.2 After (TurboQuant-H)

```
token_ids → gather 2-bit indices → scatter codebook → Hadamard rotation → FP16 → transformer
```

### 5.3 Overhead Analysis

The Hadamard rotation on 128-element groups is a butterfly network with $\log_2(128) = 7$ stages. On ARM NEON (128-bit SIMD, 8 FP16 lanes), each stage processes the group in $128/8 = 16$ vector operations, giving $7 \times 16 = 112$ fused multiply-add instructions per group.

For a single PLI embedding row of 8,190 elements:

$$\text{Groups per row} = \lceil 8190 / 128 \rceil = 64$$

$$\text{Total FMAs} = 64 \times 112 = 7{,}168$$

At 2 FLOPS per FMA on a 2 GHz A15 NEON unit, this completes in under 2 microseconds per embedding lookup. The gather from a 4× smaller table more than compensates in memory bandwidth savings.

## 6. Related Work

**TurboQuant** (Zandieh et al., ICLR 2026) introduced data-oblivious vector quantization with random rotation and QJL correction for KV cache compression, achieving near-optimal distortion rates within a constant factor of $\approx 2.7$ from information-theoretic lower bounds.

**QuIP#** (Tseng et al., 2024) uses random orthogonal rotation (via the randomized Hadamard transform) for weight quantization of linear layers, but does not address embedding tables or per-layer embeddings.

**GPTQ, AWQ** focus on linear layer weight quantization with calibration data. These methods do not handle embedding tables, which are pure lookup operations with no gradient flow during inference.

To our knowledge, TurboQuant-H is the first application of rotation-based vector quantization specifically to per-layer embedding tables, which is where the technique yields the largest storage benefit due to the embedding-dominated weight distribution in AltUp architectures.

## 7. Limitations and Future Work

- TurboQuant-H currently applies only to PLI tables. Extending to shared `token_embeddings` is straightforward but has a smaller relative impact (408 MB vs 2,496 MB).
- The 0.06 PPL increase, while negligible in our eval, should be validated on downstream tasks (MMLU, IFEval, GPQA) across Gemma-270m, Gemma 3n, and Gemma 4 E4B.
- At 1-bit (with QJL correction), the theoretical compression reaches 8× on PLI tables. We have not yet evaluated whether the quality tradeoff is acceptable at this extreme.
- The per-group codebook design means codebook storage scales as $O(V \cdot D / G \cdot 2^b \cdot 16)$. At 2-bit with group-128 this is negligible, but at higher bit widths or smaller group sizes it could become significant.

## Try It

Run Gemma 4 today on Cactus:

```bash
brew install cactus-compute/cactus/cactus
cactus run google/gemma-4-E2B-it
```

TurboQuant-H PLI weights will ship in an upcoming release. If you're working on embedding quantization or have thoughts on extending this to the shared token embedding table, open an issue on [GitHub](https://github.com/cactus-compute/cactus).

## Citation

If you use TurboQuant-H in your research, please cite:

```bibtex
@article{turboquant-h,
  title     = {TurboQuant-H: Extending TurboQuant to Embeddings for 40\%
               Storage Reduction in Embedding-Dominated Models},
  author    = {Mosoyan, Karen and Cylich, Noah and Ndubuaku, Henry},
  year      = {2026},
  url       = {https://github.com/cactus-compute/cactus},
  note      = {Cactus Compute Technical Report}
}
```

## References

- Zandieh, Daliri, Hadian, Mirrokni. [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874). ICLR 2026.
- Tseng et al. QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks. 2024.
- Google Gemma Team. Gemma 3n / Gemma 4 Technical Reports. 2026.

## See Also

- [Gemma 4 on Cactus](/blog/gemma4.md) — Day-one multimodal support with vision, audio, and hybrid inference
- [LFM-2.5-350m on Cactus](/blog/lfm2.5_350m.md) — INT8 quantization deep dive and zero-copy loading
- [Cactus Engine API](/docs/cactus_engine.md) — Full C API reference
