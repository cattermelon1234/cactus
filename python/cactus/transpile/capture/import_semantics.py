from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cactus.transpile.capture.graph_ir import IRGraph
from cactus.transpile.capture.graph_ir import IRNode
from cactus.transpile.capture.graph_ir import verify_ir

def apply_import_semantics(graph: IRGraph) -> IRGraph:
    return graph
