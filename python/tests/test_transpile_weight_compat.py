from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tensor_io import save_tensor_with_header
from src.transpile.weight_binding import WeightBinding
from src.transpile.weight_binding import resolve_weight_binding
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


def test_resolve_weight_binding_maps_lfm_decoder_weights(tmp_path: Path) -> None:
    for filename in (
        "output_norm.weights",
        "layer_0_input_norm.weights",
        "layer_0_conv_in_proj.weights",
        "layer_0_conv_depthwise.weights",
        "layer_0_conv_out_proj.weights",
        "layer_0_post_attn_norm.weights",
        "layer_0_ffn_gate.weights",
        "layer_0_ffn_up.weights",
        "layer_0_ffn_down.weights",
        "layer_2_attn_q.weights",
        "layer_2_attn_k.weights",
        "layer_2_attn_v.weights",
        "layer_2_attn_output.weights",
        "layer_2_attn_q_norm.weights",
        "layer_2_attn_k_norm.weights",
    ):
        (tmp_path / filename).write_bytes(b"")

    output_norm = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="adapter.model.model.norm.weight",
    )
    assert output_norm is not None
    assert output_norm.path.endswith("output_norm.weights")

    conv_in_proj = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="adapter.model.model.language_model.layers.0.conv.in_proj.weight",
    )
    assert conv_in_proj is not None
    assert conv_in_proj.path.endswith("layer_0_conv_in_proj.weights")

    ffn_gate = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="adapter.model.model.language_model.layers.0.feed_forward.w1.weight",
    )
    assert ffn_gate is not None
    assert ffn_gate.path.endswith("layer_0_ffn_gate.weights")

    attn_q_norm = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="adapter.model.model.language_model.layers.2.self_attn.q_layernorm.weight",
    )
    assert attn_q_norm is not None
    assert attn_q_norm.path.endswith("layer_2_attn_q_norm.weights")


def test_resolve_weight_binding_maps_flattened_whisper_names(tmp_path: Path) -> None:
    for filename in (
        "encoder_conv1_weight.weights",
        "encoder_conv1_bias.bias",
        "encoder_position_embeddings.weights",
        "decoder_token_embeddings.weights",
        "decoder.layer_0_self_attn_q.weights",
        "decoder.layer_0_self_attn_q.bias",
        "decoder.layer_0_encoder_attn_output.weights",
        "decoder.layer_0_encoder_attn_output.bias",
        "decoder.layer_0_final_norm.weights",
        "decoder.layer_0_final_norm.bias",
        "decoder.layer_0_mlp_fc1.weights",
        "decoder.layer_0_mlp_fc1.bias",
    ):
        (tmp_path / filename).write_bytes(b"")

    conv1 = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="v_module_encoder_conv1_weight",
    )
    assert conv1 is not None
    assert conv1.path.endswith("encoder_conv1_weight.weights")

    decoder_embed = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="v_module_decoder_embed_tokens_weight",
    )
    assert decoder_embed is not None
    assert decoder_embed.path.endswith("decoder_token_embeddings.weights")

    self_attn_q = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="v_module_decoder_layers_0_self_attn_q_proj_weight",
    )
    assert self_attn_q is not None
    assert self_attn_q.path.endswith("decoder.layer_0_self_attn_q.weights")

    encoder_attn_out_bias = resolve_weight_binding(
        weights_dir=str(tmp_path),
        source_name="v_module_decoder_layers_0_encoder_attn_out_proj_bias",
    )
    assert encoder_attn_out_bias is not None
    assert encoder_attn_out_bias.path.endswith("decoder.layer_0_encoder_attn_output.bias")
