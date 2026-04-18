import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.graph import Graph, Tensor


class TestGemmaLikeGraph(unittest.TestCase):

    def _rebind_tensor(self, graph, tensor):
        return Tensor(graph, tensor.id, tensor.shape, tensor.dtype)

    def _build_tiny_gemma_like_graph(self, seq_len=3, vocab_size=16, hidden_dim=8, num_heads=2, ffn_dim=16):
        head_dim = hidden_dim // num_heads
        g = Graph()

        tokens = g.input((seq_len,), dtype=Graph.FP32)
        embedding_table = g.input((vocab_size, hidden_dim))
        output_norm_weight = g.input((hidden_dim,))

        input_norm_weight = g.input((hidden_dim,))
        attn_q_norm_weight = g.input((head_dim,))
        attn_k_norm_weight = g.input((head_dim,))
        attn_q_weight = g.input((hidden_dim, hidden_dim))
        attn_k_weight = g.input((hidden_dim, hidden_dim))
        attn_v_weight = g.input((hidden_dim, hidden_dim))
        attn_output_weight = g.input((hidden_dim, hidden_dim))

        post_attention_norm_weight = g.input((hidden_dim,))
        pre_ffn_norm_weight = g.input((hidden_dim,))
        post_ffn_norm_weight = g.input((hidden_dim,))
        ffn_gate_weight = g.input((ffn_dim, hidden_dim))
        ffn_up_weight = g.input((ffn_dim, hidden_dim))
        ffn_down_weight = g.input((hidden_dim, ffn_dim))

        hidden = g.embedding_from_tensor(embedding_table, tokens)
        hidden = g.scalar_multiply(hidden, math.sqrt(hidden_dim))

        normalized_input = g.rms_norm(hidden, input_norm_weight, eps=1e-5)

        q_proj = g.matmul(normalized_input, attn_q_weight, pretransposed_rhs=True)
        k_proj = g.matmul(normalized_input, attn_k_weight, pretransposed_rhs=True)
        v_proj = g.matmul(normalized_input, attn_v_weight, pretransposed_rhs=True)

        q_proj = g.reshape(q_proj, (seq_len * num_heads, head_dim))
        q_proj = g.rms_norm(q_proj, attn_q_norm_weight, eps=1e-5)
        q_proj = g.reshape(q_proj, (seq_len, hidden_dim))

        k_proj = g.reshape(k_proj, (seq_len * num_heads, head_dim))
        k_proj = g.rms_norm(k_proj, attn_k_norm_weight, eps=1e-5)
        k_proj = g.reshape(k_proj, (seq_len, hidden_dim))

        q_proj_4d = g.reshape(q_proj, (1, seq_len, num_heads, head_dim))
        k_proj_4d = g.reshape(k_proj, (1, seq_len, num_heads, head_dim))
        v_proj_4d = g.reshape(v_proj, (1, seq_len, num_heads, head_dim))

        q_proj_4d = g.rope(q_proj_4d, 10000.0, position_offset=0)
        k_proj_4d = g.rope(k_proj_4d, 10000.0, position_offset=0)

        attn_output_4d = g.attention(
            q_proj_4d,
            k_proj_4d,
            v_proj_4d,
            scale=1.0 / math.sqrt(head_dim),
            is_causal=True,
            position_offset=0,
            window_size=0,
        )
        attn_output = g.reshape(attn_output_4d, (seq_len, hidden_dim))
        attn_output = g.matmul(attn_output, attn_output_weight, pretransposed_rhs=True)

        normalized_attn = g.rms_norm(attn_output, post_attention_norm_weight, eps=1e-5)
        after_attention = g.add_clipped(hidden, normalized_attn)

        pre_mlp_norm = g.rms_norm(after_attention, pre_ffn_norm_weight, eps=1e-5)
        gate_output = g.matmul(pre_mlp_norm, ffn_gate_weight, pretransposed_rhs=True)
        up_output = g.matmul(pre_mlp_norm, ffn_up_weight, pretransposed_rhs=True)
        gate_gelu = g.gelu(gate_output)
        gated = g.multiply(gate_gelu, up_output)
        mlp_output = g.matmul(gated, ffn_down_weight, pretransposed_rhs=True)

        normalized_mlp = g.rms_norm(mlp_output, post_ffn_norm_weight, eps=1e-5)
        hidden = g.add_clipped(after_attention, normalized_mlp)

        final_hidden = g.rms_norm(hidden, output_norm_weight, eps=1e-5)
        logits = g.matmul(final_hidden, embedding_table, pretransposed_rhs=True)
        probs = g.softmax(logits, axis=1)

        inputs = {
            "tokens": tokens,
            "embedding_table": embedding_table,
            "output_norm_weight": output_norm_weight,
            "input_norm_weight": input_norm_weight,
            "attn_q_norm_weight": attn_q_norm_weight,
            "attn_k_norm_weight": attn_k_norm_weight,
            "attn_q_weight": attn_q_weight,
            "attn_k_weight": attn_k_weight,
            "attn_v_weight": attn_v_weight,
            "attn_output_weight": attn_output_weight,
            "post_attention_norm_weight": post_attention_norm_weight,
            "pre_ffn_norm_weight": pre_ffn_norm_weight,
            "post_ffn_norm_weight": post_ffn_norm_weight,
            "ffn_gate_weight": ffn_gate_weight,
            "ffn_up_weight": ffn_up_weight,
            "ffn_down_weight": ffn_down_weight,
        }
        return g, inputs, logits, probs

    def _make_inputs(self, vocab_size=16, hidden_dim=8, num_heads=2, ffn_dim=16):
        head_dim = hidden_dim // num_heads
        rng = np.random.default_rng(1234)
        return {
            "tokens": np.array([1.0, 3.0, 5.0], dtype=np.float32),
            "embedding_table": rng.normal(0.0, 0.2, size=(vocab_size, hidden_dim)).astype(np.float16),
            "output_norm_weight": rng.normal(1.0, 0.05, size=(hidden_dim,)).astype(np.float16),
            "input_norm_weight": rng.normal(1.0, 0.05, size=(hidden_dim,)).astype(np.float16),
            "attn_q_norm_weight": rng.normal(1.0, 0.05, size=(head_dim,)).astype(np.float16),
            "attn_k_norm_weight": rng.normal(1.0, 0.05, size=(head_dim,)).astype(np.float16),
            "attn_q_weight": rng.normal(0.0, 0.15, size=(hidden_dim, hidden_dim)).astype(np.float16),
            "attn_k_weight": rng.normal(0.0, 0.15, size=(hidden_dim, hidden_dim)).astype(np.float16),
            "attn_v_weight": rng.normal(0.0, 0.15, size=(hidden_dim, hidden_dim)).astype(np.float16),
            "attn_output_weight": rng.normal(0.0, 0.15, size=(hidden_dim, hidden_dim)).astype(np.float16),
            "post_attention_norm_weight": rng.normal(1.0, 0.05, size=(hidden_dim,)).astype(np.float16),
            "pre_ffn_norm_weight": rng.normal(1.0, 0.05, size=(hidden_dim,)).astype(np.float16),
            "post_ffn_norm_weight": rng.normal(1.0, 0.05, size=(hidden_dim,)).astype(np.float16),
            "ffn_gate_weight": rng.normal(0.0, 0.12, size=(ffn_dim, hidden_dim)).astype(np.float16),
            "ffn_up_weight": rng.normal(0.0, 0.12, size=(ffn_dim, hidden_dim)).astype(np.float16),
            "ffn_down_weight": rng.normal(0.0, 0.12, size=(hidden_dim, ffn_dim)).astype(np.float16),
        }

    def _bind_inputs(self, graph, tensor_map, values):
        for name, tensor in tensor_map.items():
            dtype = Graph.FP32 if name == "tokens" else None
            graph.set_input(tensor, values[name], dtype=dtype)

    def test_tiny_gemma_like_inference_runs(self):
        g, inputs, logits, probs = self._build_tiny_gemma_like_graph()
        values = self._make_inputs()

        self._bind_inputs(g, inputs, values)
        g.execute()

        logits_out = logits.numpy()
        probs_out = probs.numpy()

        self.assertEqual(logits_out.shape, (3, 16))
        self.assertEqual(probs_out.shape, (3, 16))
        np.testing.assert_allclose(probs_out.sum(axis=1), np.ones((3,), dtype=np.float16), atol=5e-2)

        top_ids = np.argmax(probs_out.astype(np.float32), axis=1)
        self.assertEqual(top_ids.shape, (3,))
        self.assertTrue(np.all((top_ids >= 0) & (top_ids < 16)))

    def test_tiny_gemma_like_save_load_roundtrip(self):
        g, inputs, logits, probs = self._build_tiny_gemma_like_graph()
        values = self._make_inputs()

        self._bind_inputs(g, inputs, values)
        g.execute()

        expected_logits = logits.numpy()
        expected_probs = probs.numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny_gemma_like.cg"
            g.save(path)

            loaded = Graph.load(path)
            loaded_inputs = {
                name: self._rebind_tensor(loaded, tensor) for name, tensor in inputs.items()
            }
            loaded_logits = self._rebind_tensor(loaded, logits)
            loaded_probs = self._rebind_tensor(loaded, probs)

            self._bind_inputs(loaded, loaded_inputs, values)
            loaded.execute()

            np.testing.assert_allclose(loaded_logits.numpy(), expected_logits, atol=1e-2)
            np.testing.assert_allclose(loaded_probs.numpy(), expected_probs, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
