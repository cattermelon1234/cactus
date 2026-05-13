"""Compatibility wrapper for v2 Cactus FFI bindings."""

from cactus.bindings import cactus as _cactus
from cactus.bindings.cactus import *  # noqa: F401,F403

_lib = _cactus._lib
_err = _cactus._err
cactus_graph_t = _cactus.cactus_graph_t
cactus_node_t = _cactus.cactus_node_t
cactus_tensor_info_t = _cactus.cactus_tensor_info_t
cactus_preprocess_audio_features = _cactus.cactus_preprocess_audio_features
