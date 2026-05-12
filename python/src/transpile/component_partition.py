from __future__ import annotations

import copy
from collections import Counter

from src.transpile.graph_ir import IRGraph
from src.transpile.graph_ir import IRNode
from src.transpile.graph_ir import IRValue
from src.transpile.graph_ir import verify_ir


COMPONENT_AUDIO_ENCODER = "audio_encoder"
COMPONENT_VISION_ENCODER = "vision_encoder"
COMPONENT_LM_ENCODER = "lm_encoder"
COMPONENT_DECODER = "decoder"
COMPONENT_UNSPECIFIED = "unspecified"

COMPONENT_ORDER = (
    COMPONENT_AUDIO_ENCODER,
    COMPONENT_VISION_ENCODER,
    COMPONENT_LM_ENCODER,
    COMPONENT_DECODER,
)


def _normalized_name_candidates(
    *,
    value_id: str | None = None,
    source_name: str | None = None,
    meta: dict[str, object] | None = None,
) -> tuple[str, ...]:
    names: list[str] = []

    def _add(value: object) -> None:
        if not isinstance(value, str):
            return
        normalized = value.strip().lower()
        if normalized and normalized not in names:
            names.append(normalized)

    if value_id is not None:
        _add(value_id)
    if source_name is not None:
        _add(source_name)
    if isinstance(meta, dict):
        _add(meta.get("torch_name"))
        for key in ("module_paths", "import_hint_providers"):
            raw_paths = meta.get(key)
            if isinstance(raw_paths, (tuple, list)):
                for path in raw_paths:
                    _add(path)
    return tuple(names)


def _contains_any(name: str, needles: tuple[str, ...]) -> bool:
    return any(needle in name for needle in needles)


def classify_component_name_candidates(
    candidates: tuple[str, ...],
    *,
    family: str,
    task: str,
) -> str | None:
    if not candidates:
        return None

    for name in candidates:
        if _contains_any(
            name,
            (
                "pixel_values",
                "image_features",
                "vision_tower",
                "vision_model",
                "vision_encoder",
                "vision_backbone",
                "embed_vision",
                "image_tower",
                "visual_encoder",
                "visual.",
            ),
        ):
            return COMPONENT_VISION_ENCODER

    for name in candidates:
        if _contains_any(
            name,
            (
                "input_features",
                "audio_features",
                "speech_features",
                "mel_features",
                "audio_tower",
                "audio_encoder",
                "speech_encoder",
                "speech_model.encoder",
                "acoustic_encoder",
                "feature_encoder",
                "encoder.pre_encode",
                "encoder.subsampling",
                "encoder.layers",
                "encoder.conformer",
                "conformer.",
                "subsample_conv_projection",
            ),
        ):
            return COMPONENT_AUDIO_ENCODER

    for name in candidates:
        if _contains_any(
            name,
            (
                "lm_head",
                "transformer.h",
                "decoder.layers",
                "decoder.layer",
                "decoder.block",
                "language_model.layers",
                "language_model.model.layers",
                "model.language_model.layers",
                "text_model.layers",
                "model.layers",
                "backbone.layers",
                "adapter_backbone_layers_slice",
                "decoder.prediction",
                "dec_rnn",
                "joint.",
                "jointnet",
                "ctc_head",
                "classifier",
                "output_projection",
                "output_layer",
            ),
        ):
            return COMPONENT_DECODER

    for name in candidates:
        if task == "multimodal_causal_lm_logits" and _contains_any(
            name,
            (
                "input_ids",
                "inputs_embeds",
                "text_embeds",
                "position_ids",
                "get_input_embeddings",
                "embed_tokens",
                "embed_text",
                "embed_tokens_per_layer",
                "per_layer_model_projection",
                "per_layer_input",
                "placeholder",
                "multimodal_projector",
                "multi_modal_projector",
                "connector",
                "merge",
                "masked_scatter",
                "create_causal_mask",
                "create_masks_for_generate",
                "token_type",
                "attention_mask",
                "pixel_position",
                "multimodal_backbone",
                "adapter_model_model_language_model_embed_tokens",
                "adapter_backbone_embed_tokens_per_layer",
                "adapter_backbone_per_layer_model_projection",
            ),
        ):
            return COMPONENT_LM_ENCODER

    for name in candidates:
        if task in {"ctc_logits", "encoder_hidden_states"} and name.startswith("encoder."):
            return COMPONENT_AUDIO_ENCODER

    if family == "gemma4" and task == "multimodal_causal_lm_logits":
        for name in candidates:
            if _contains_any(
                name,
                (
                    "input_ids",
                    "token_type_ids",
                    "attention_mask",
                    "input_features_mask",
                ),
            ):
                return COMPONENT_LM_ENCODER

    if family == "parakeet_tdt":
        for name in candidates:
            if _contains_any(
                name,
                (
                    "decoder.prediction",
                    "dec_rnn",
                    "joint.",
                    "jointnet",
                    "predictor_",
                    "tdt_joint",
                ),
            ):
                return COMPONENT_DECODER
            if name.startswith("encoder."):
                return COMPONENT_AUDIO_ENCODER

    return None


def classify_value_component(value_id: str, value: IRValue, *, family: str, task: str) -> str | None:
    return classify_component_name_candidates(
        _normalized_name_candidates(
            value_id=value_id,
            source_name=value.meta.get("source_name") if isinstance(value.meta, dict) else None,
            meta=value.meta,
        ),
        family=family,
        task=task,
    )


def classify_node_component(node: IRNode, *, family: str, task: str) -> str | None:
    return classify_component_name_candidates(
        _normalized_name_candidates(meta=node.meta),
        family=family,
        task=task,
    )


def _task_default_component(*, family: str, task: str) -> str:
    if task == "multimodal_causal_lm_logits":
        return COMPONENT_DECODER
    if task == "causal_lm_logits":
        return COMPONENT_DECODER
    if task in {"ctc_logits", "encoder_hidden_states"}:
        return COMPONENT_AUDIO_ENCODER
    if family == "parakeet_tdt":
        return COMPONENT_AUDIO_ENCODER
    return COMPONENT_UNSPECIFIED


def _rebuild_users(graph: IRGraph) -> None:
    for value in graph.values.values():
        value.users = []
    for value_id in graph.inputs:
        graph.values[value_id].producer = None
    for value_id in graph.constants:
        graph.values[value_id].producer = None
    for node_id in graph.order:
        node = graph.nodes[node_id]
        for output_id in node.outputs:
            graph.values[output_id].producer = node_id
        for input_id in node.inputs:
            graph.values[input_id].users.append(node_id)


def annotate_ir_components(graph: IRGraph) -> None:
    family = str(graph.meta.get("adapter_family", "") or "").lower()
    task = str(graph.meta.get("task", "") or "").lower()
    default_component = _task_default_component(family=family, task=task)

    for value_id, value in graph.values.items():
        component = classify_value_component(value_id, value, family=family, task=task)
        if component is not None:
            value.meta["component"] = component

    for node_id in graph.order:
        node = graph.nodes[node_id]
        component = classify_node_component(node, family=family, task=task)
        if component is not None:
            node.meta["component"] = component
            for output_id in node.outputs:
                graph.values[output_id].meta["component"] = component

    for _ in range(8):
        changed = False
        for node_id in graph.order:
            node = graph.nodes[node_id]
            component = node.meta.get("component")
            if component not in COMPONENT_ORDER:
                input_components = {
                    graph.values[input_id].meta.get("component")
                    for input_id in node.inputs
                    if graph.values[input_id].meta.get("component") in COMPONENT_ORDER
                }
                if len(input_components) == 1:
                    component = next(iter(input_components))
                    node.meta["component"] = component
                    changed = True
            component = node.meta.get("component")
            if component in COMPONENT_ORDER:
                for output_id in node.outputs:
                    if graph.values[output_id].meta.get("component") != component:
                        graph.values[output_id].meta["component"] = component
                        changed = True

        for node_id in reversed(graph.order):
            node = graph.nodes[node_id]
            component = node.meta.get("component")
            if component in COMPONENT_ORDER:
                continue
            output_components: set[str] = set()
            for output_id in node.outputs:
                value = graph.values[output_id]
                for user_id in value.users:
                    user_component = graph.nodes[user_id].meta.get("component")
                    if user_component in COMPONENT_ORDER:
                        output_components.add(user_component)
            if len(output_components) == 1:
                component = next(iter(output_components))
                node.meta["component"] = component
                for output_id in node.outputs:
                    graph.values[output_id].meta["component"] = component
                changed = True

        if not changed:
            break

    for value in graph.values.values():
        component = value.meta.get("component")
        if component not in COMPONENT_ORDER and default_component != COMPONENT_UNSPECIFIED:
            value.meta["component"] = default_component

    for node_id in graph.order:
        node = graph.nodes[node_id]
        component = node.meta.get("component")
        if component not in COMPONENT_ORDER and default_component != COMPONENT_UNSPECIFIED:
            node.meta["component"] = default_component
            for output_id in node.outputs:
                graph.values[output_id].meta["component"] = default_component

    counts = Counter(
        str(graph.nodes[node_id].meta.get("component", COMPONENT_UNSPECIFIED))
        for node_id in graph.order
    )
    graph.meta["component_counts"] = dict(sorted(counts.items()))


def summarize_ir_components(graph: IRGraph) -> dict[str, int]:
    annotate_ir_components(graph)
    counts = graph.meta.get("component_counts", {})
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items()}
    return {}


def _clone_value_for_subgraph(value: IRValue, *, as_input: bool) -> IRValue:
    cloned = copy.deepcopy(value)
    if as_input:
        cloned.producer = None
    cloned.users = []
    return cloned


def extract_component_subgraphs(graph: IRGraph) -> dict[str, IRGraph]:
    annotate_ir_components(graph)
    component_graphs: dict[str, IRGraph] = {}

    for component in COMPONENT_ORDER:
        node_ids = [
            node_id
            for node_id in graph.order
            if graph.nodes[node_id].meta.get("component") == component
        ]
        if not node_ids:
            continue

        node_id_set = set(node_ids)
        inputs: list[str] = []
        outputs: list[str] = []
        constants: dict[str, object] = {}
        values: dict[str, IRValue] = {}
        nodes: dict[str, IRNode] = {}

        for node_id in node_ids:
            node = copy.deepcopy(graph.nodes[node_id])
            nodes[node_id] = node
            for input_id in node.inputs:
                if input_id in graph.constants:
                    constants[input_id] = graph.constants[input_id]
                    if input_id not in values:
                        values[input_id] = _clone_value_for_subgraph(graph.values[input_id], as_input=True)
                    continue
                producer = graph.values[input_id].producer
                if producer not in node_id_set and input_id not in inputs:
                    inputs.append(input_id)
                    values[input_id] = _clone_value_for_subgraph(graph.values[input_id], as_input=True)
            for output_id in node.outputs:
                if output_id not in values:
                    values[output_id] = _clone_value_for_subgraph(graph.values[output_id], as_input=False)

        for node_id in node_ids:
            node = graph.nodes[node_id]
            for output_id in node.outputs:
                value = graph.values[output_id]
                if output_id in graph.outputs or any(user_id not in node_id_set for user_id in value.users):
                    if output_id not in outputs:
                        outputs.append(output_id)

        component_graph = IRGraph(
            values=values,
            nodes=nodes,
            order=list(node_ids),
            inputs=inputs,
            outputs=outputs,
            constants=constants,
            meta={
                **copy.deepcopy(graph.meta),
                "component": component,
            },
        )
        _rebuild_users(component_graph)
        verify_ir(component_graph)
        component_graphs[component] = component_graph

    return component_graphs
