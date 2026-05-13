from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converter_adapters import convert_hf_model_weights_with_adapters
from src.weight_adapters import select_adapter


class _FakeConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeModel:
    def __init__(self, state_dict: dict[str, torch.Tensor], config: _FakeConfig):
        self._state_dict = state_dict
        self.config = config

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state_dict


class TestConverterAdapters(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is required for converter adapter tests")
    def test_selects_gemma_adapter_from_config(self) -> None:
        config = _FakeConfig(model_type="gemma", num_hidden_layers=1)
        adapter, ctx = select_adapter(
            root_config=config,
            state_keys=("model.layers.0.self_attn.q_proj.weight",),
            precision="FP16",
        )
        self.assertEqual(adapter.adapter_name, "gemma")
        self.assertEqual(ctx.detected_model_type, "gemma")

    @unittest.skipIf(torch is None, "torch is required for converter adapter tests")
    def test_exports_basic_decoder_weights_and_manifest(self) -> None:
        state_dict = {
            "model.embed_tokens.weight": torch.ones(8, 4, dtype=torch.float16),
            "model.norm.weight": torch.ones(4, dtype=torch.float16),
            "lm_head.weight": torch.ones(8, 4, dtype=torch.float16),
            "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4, dtype=torch.float16),
            "model.layers.0.self_attn.k_proj.weight": torch.ones(4, 4, dtype=torch.float16),
            "model.layers.0.self_attn.v_proj.weight": torch.ones(4, 4, dtype=torch.float16),
            "model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4, dtype=torch.float16),
            "model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4, dtype=torch.float16),
            "model.layers.0.mlp.up_proj.weight": torch.ones(8, 4, dtype=torch.float16),
            "model.layers.0.mlp.down_proj.weight": torch.ones(4, 8, dtype=torch.float16),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(4, dtype=torch.float16),
            "model.layers.0.pre_feedforward_layernorm.weight": torch.ones(4, dtype=torch.float16),
            "model.layers.0.post_feedforward_layernorm.weight": torch.ones(4, dtype=torch.float16),
        }
        model = _FakeModel(
            state_dict=state_dict,
            config=_FakeConfig(
                model_type="gemma",
                num_hidden_layers=1,
                hidden_size=4,
                num_attention_heads=1,
                num_key_value_heads=1,
                intermediate_size=8,
                max_position_embeddings=16,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = convert_hf_model_weights_with_adapters(model, tmpdir, precision="FP16")
            root = Path(tmpdir)
            self.assertEqual(config["adapter_name"], "gemma")
            self.assertTrue((root / "token_embeddings.weights").exists())
            self.assertTrue((root / "output_norm.weights").exists())
            self.assertTrue((root / "layer_0_attn_q.weights").exists())
            self.assertTrue((root / "weights_manifest.json").exists())
            self.assertTrue((root / "adapter_export_plan.json").exists())
            manifest = (root / "weights_manifest.json").read_text()
            self.assertIn("model.layers.0.self_attn.q_proj.weight", manifest)


if __name__ == "__main__":
    unittest.main()
