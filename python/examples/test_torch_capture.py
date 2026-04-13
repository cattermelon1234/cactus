import torch
import torch.nn as nn

from src.transpile.capture_pytorch import (
    capture_model_with_fallback,
    dump_graph,
    get_dtype,
    get_shape,
)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h)
        y = self.fc2(h)
        return y


def main():
    model = TinyBlock().eval()
    x = torch.randn(2, 8)

    captured = capture_model_with_fallback(model, args=(x,))

    print("strict:", captured.strict)
    print("graph_module type:", type(captured.graph_module).__name__)
    print("state_dict keys:", list(captured.state_dict.keys()))
    print()

    print("=== FX Graph ===")
    print(captured.graph)
    print()

    print("=== Node Walk ===")
    for i, node in enumerate(captured.graph.nodes):
        print(f"[{i}] op={node.op} name={node.name} target={node.target}")
        print(f"    args={node.args}")
        print(f"    kwargs={node.kwargs}")
        print(f"    shape={get_shape(node)} dtype={get_dtype(node)}")
        print()

    print("=== Full Dump ===")
    print(dump_graph(captured))


if __name__ == "__main__":
    main()
