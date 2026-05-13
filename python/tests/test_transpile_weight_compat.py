from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tensor_io import save_tensor_with_header
from src.transpile.weight_binding import WeightBinding
from src.transpile.weight_compat import _open_cactus_tensor_file
from src.transpile.weight_compat import ensure_binding_compatible
from src.transpile.weight_compat import ensure_embedding_binding_compatible


def _write_grouped_int8_embedding(path: Path, rows: int, cols: int) -> None:
    rng = np.random.default_rng(1234)
    tensor = rng.standard_normal((rows, cols), dtype=np.float32)
    save_tensor_with_header(tensor, path, precision="INT8", model_type="generic")


def test_token_embedding_binding_upgrades_to_cq4(tmp_path: Path) -> None:
    source = tmp_path / "token_embeddings.weights"
    _write_grouped_int8_embedding(source, rows=8, cols=128)

    legacy_fp16 = tmp_path / "token_embeddings.fp16.weights"
    legacy_fp16.write_bytes(b"legacy")

    binding = WeightBinding(
        path=str(source),
        kind="embedding",
        source_name="model.embed_tokens.weight",
    )
    compat = ensure_embedding_binding_compatible(binding)

    assert compat.path.endswith(".cq4.weights")
    assert compat.kind == "embedding"
    assert not legacy_fp16.exists()

    opened = _open_cactus_tensor_file(compat.path)
    assert opened.precision == 6
    assert opened.shape == (8, 128)
    assert opened.scales is not None


def test_per_layer_embedding_binding_upgrades_to_cq2(tmp_path: Path) -> None:
    source = tmp_path / "embed_tokens_per_layer.weights"
    _write_grouped_int8_embedding(source, rows=8, cols=256)

    binding = WeightBinding(
        path=str(source),
        kind="embedding",
        source_name="model.language_model.embed_tokens_per_layer.weight",
    )
    compat = ensure_embedding_binding_compatible(binding)

    assert compat.path.endswith(".cq2.weights")

    opened = _open_cactus_tensor_file(compat.path)
    assert opened.precision == 4
    assert opened.shape == (8, 256)
    assert opened.scales is not None


def test_gemma4_per_layer_projection_binding_upgrades_legacy_int4(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    source = tmp_path / "per_layer_model_proj.weights"
    tensor = rng.standard_normal((8, 128), dtype=np.float32)
    save_tensor_with_header(tensor, source, precision="INT4", model_type="generic")

    binding = WeightBinding(
        path=str(source),
        kind="weight",
        source_name="module.model.model.language_model.per_layer_model_projection.weight",
    )
    compat = ensure_binding_compatible(binding, source_tensor=tensor)

    assert compat.path.endswith(".cq4.weights")
    opened = _open_cactus_tensor_file(compat.path)
    assert opened.precision == 6
    assert opened.shape == (8, 128)
    assert opened.scales is not None


def test_decoder_binding_prefers_existing_cq4_companion(tmp_path: Path) -> None:
    rng = np.random.default_rng(4321)
    source = tmp_path / "layer_0_attn_q.weights"
    tensor = rng.standard_normal((8, 128), dtype=np.float32)
    save_tensor_with_header(tensor, source, precision="INT4", model_type="generic")

    companion = tmp_path / "layer_0_attn_q.cq4.weights"
    save_tensor_with_header(tensor, companion, precision="FP16", model_type="generic")

    binding = WeightBinding(
        path=str(source),
        kind="weight",
        source_name="module.backbone.layers.0.self_attn.q_proj.weight",
    )
    compat = ensure_binding_compatible(binding, source_tensor=0)

    assert compat.path == str(companion)
