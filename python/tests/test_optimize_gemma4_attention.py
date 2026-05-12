from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.graph_ir import IRValue
from src.transpile.optimize_graph import normalize_gemma4_decoder_attention_semantics


def test_normalize_gemma4_full_attention_uses_sequence_window_compat() -> None:
    graph = IRGraph(
        values={
            "query": IRValue(id="query", shape=(1, 8, 800, 512), dtype="fp16"),
            "key": IRValue(id="key", shape=(1, 1, 800, 512), dtype="fp16"),
            "value": IRValue(id="value", shape=(1, 1, 800, 512), dtype="fp16"),
            "full_attention_mask": IRValue(
                id="full_attention_mask",
                shape=(1, 1, 800, 800),
                dtype="bool",
            ),
            "proj": IRValue(id="proj", shape=(1536, 4096), dtype="fp16"),
            "out": IRValue(id="out", shape=(1, 800, 1536), dtype="fp16", producer="attn"),
        },
        nodes={
            "attn": IRNode(
                id="attn",
                op="attention_block",
                inputs=["query", "key", "value", "full_attention_mask", "proj"],
                outputs=["out"],
                attrs={
                    "has_mask": True,
                    "is_causal": False,
                    "window_size": 0,
                    "attention_output_shape": (1, 800, 4096),
                },
                meta={"attention_layer_type": "full_attention"},
            )
        },
        order=["attn"],
        inputs=["query", "key", "value", "full_attention_mask"],
        outputs=["out"],
        constants={},
        meta={
            "adapter_family": "gemma4",
            "component": "decoder",
            "input_names": ("query", "key", "value", "full_attention_mask"),
        },
    )

    changed = normalize_gemma4_decoder_attention_semantics(graph)

    assert changed is True
    assert graph.nodes["attn"].inputs == ["query", "key", "value", "proj"]
    assert graph.nodes["attn"].attrs["has_mask"] is False
    assert graph.nodes["attn"].attrs["is_causal"] is True
    assert graph.nodes["attn"].attrs["window_size"] == 800
    assert graph.nodes["attn"].meta["gemma4_full_attention_window_compat"] is True


def test_normalize_gemma4_sliding_attention_elides_runtime_mask() -> None:
    graph = IRGraph(
        values={
            "query": IRValue(id="query", shape=(1, 8, 800, 256), dtype="fp16"),
            "key": IRValue(id="key", shape=(1, 8, 800, 256), dtype="fp16"),
            "value": IRValue(id="value", shape=(1, 8, 800, 256), dtype="fp16"),
            "sliding_attention_mask": IRValue(
                id="sliding_attention_mask",
                shape=(1, 1, 800, 800),
                dtype="bool",
            ),
            "proj": IRValue(id="proj", shape=(1536, 2048), dtype="fp16"),
            "out": IRValue(id="out", shape=(1, 800, 1536), dtype="fp16", producer="attn"),
        },
        nodes={
            "attn": IRNode(
                id="attn",
                op="attention_block",
                inputs=["query", "key", "value", "sliding_attention_mask", "proj"],
                outputs=["out"],
                attrs={
                    "has_mask": True,
                    "is_causal": False,
                    "window_size": 0,
                    "attention_output_shape": (1, 800, 2048),
                },
                meta={"attention_layer_type": "sliding_attention"},
            )
        },
        order=["attn"],
        inputs=["query", "key", "value", "sliding_attention_mask"],
        outputs=["out"],
        constants={},
        meta={
            "adapter_family": "gemma4",
            "component": "decoder",
            "sliding_window": 512,
            "input_names": ("query", "key", "value", "sliding_attention_mask"),
        },
    )

    changed = normalize_gemma4_decoder_attention_semantics(graph)

    assert changed is True
    assert graph.nodes["attn"].inputs == ["query", "key", "value", "proj"]
    assert graph.nodes["attn"].attrs["has_mask"] is False
    assert graph.nodes["attn"].attrs["is_causal"] is True
    assert graph.nodes["attn"].attrs["window_size"] == 512
