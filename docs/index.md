---
title: "Cactus Documentation"
description: "Complete documentation index for Cactus, the energy-efficient AI inference engine for phones, wearables, Macs, and ARM devices."
keywords: ["Cactus", "documentation", "on-device AI", "mobile inference", "API reference", "SDK"]
---

# Cactus Documentation

Cactus is an energy-efficient AI inference engine for running LLMs, vision models, and speech models on mobile devices, Macs, and ARM chips like Raspberry Pi.

## Core API References

| Document | Description |
|----------|-------------|
| [Cactus Engine API](cactus_engine.md) | C FFI for chat completion, streaming, tool calling, transcription, embeddings, RAG, vision, VAD, and cloud handoff |
| [Cactus Graph API](cactus_graph.md) | Computational graph framework for tensor operations, matrix multiplication, attention, normalization, and activation functions |
| [Cactus Index API](cactus_index.md) | On-device vector database with cosine similarity search for RAG applications |

## SDK References

| SDK | Language | Platforms |
|-----|----------|-----------|
| [Python](/python/) | Python | Mac, Linux |
| [Swift](/apple/) | Swift | iOS, macOS, tvOS, watchOS, Android |
| [Kotlin/Android](/android/) | Kotlin | Android, iOS (via KMP) |
| [Flutter](/flutter/) | Dart | iOS, macOS, Android |
| [Rust](/rust/) | Rust | Mac, Linux |
| [React Native](https://github.com/cactus-compute/cactus-react-native) | JavaScript | iOS, Android |

## Guides

| Document | Description |
|----------|-------------|
| [Fine-tuning Guide](finetuning.md) | Train LoRA fine-tunes with Unsloth and deploy them to iOS and Android via Cactus |
| [Runtime Compatibility](compatibility.md) | How runtime versions map to model weight versions on HuggingFace |
| [Contributing](/CONTRIBUTING.md) | Code style, PR guidelines, and how to contribute to Cactus |

## External Links

- [GitHub Repository](https://github.com/cactus-compute/cactus)
- [Website](https://cactuscompute.com/)
- [HuggingFace Weights](https://huggingface.co/Cactus-Compute)
- [Reddit Community](https://www.reddit.com/r/cactuscompute/)
- [iOS Demo App](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo App](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)

## Blog

| Post | Author | Description |
|------|--------|-------------|
| [Hybrid Transcription](/blog/hybrid_transcription.md) | Roman Shemet | Sub-150ms transcription with cloud-level accuracy using on-device/cloud hybrid inference |
| [LFM2 24B Review](/blog/lfm2_24b_a2b.md) | Noah Cylich & Henry Ndubuaku | Running LFM2-24B MoE locally on Mac for coding use cases |
