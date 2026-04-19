from dataclasses import dataclass
from dataclasses import field


@dataclass
class IRValue:
    id: str 
    shape: tuple[int, ...] | None = None 
    dtype: str | None = None 
    producer: str | None = None
    users: list[str] = field(default_factory=list)

@dataclass 
class IRNode:
    id: str
    op: str 
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, object] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)
    kind: str = "generic"

@dataclass 
class IRGraph:
    values: dict[str, IRValue] 
    nodes: dict[str, IRNode]
    order: list[str]
    inputs: list[str]
    outputs: list[str]
    constants: dict[str, object] = field(default_factory=dict)

    def add_node(self, node: IRNode) -> None:
        self.nodes[node.id] = node
        for output in node.outputs:
            self.values[output] = IRValue(id=output, producer=node.id)

    def add_value(self, value: IRValue) -> None:
        self.values[value.id] = value
