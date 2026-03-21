import numpy as np

from src.graph import Graph


def test_graph_bridge_pow_abs_smoke():
    g = Graph()

    a = g.input((4,))
    y = a.pow(2.0).abs()

    g.set_input(a, np.array([-2.0, -1.0, 2.0, 3.0], dtype=np.float16))
    g.execute()

    out = y.numpy()
    expected = np.array([4.0, 1.0, 4.0, 9.0], dtype=np.float16)
    print("out:", out)

    assert np.allclose(out, expected, atol=1e-2)


def test_graph_bridge_subtract_operator():
    g = Graph()
    a = g.input((4,))
    b = g.input((4,))
    y = a - b

    g.set_input(a, np.array([5, 6, 7, 8], dtype=np.float16))
    g.set_input(b, np.array([1, 2, 3, 4], dtype=np.float16))
    g.execute()

    out = y.numpy()
    expected = np.array([4, 4, 4, 4], dtype=np.float16)
    print("out:", out)
    assert np.allclose(out, expected, atol=1e-2)


def test_graph_bridge_composed_inference_graph():
    g = Graph()

    a = g.input((2, 2))
    b = g.input((2, 2))

    diff = a - b
    summ = a + b
    mixed = (diff * summ) / b
    feat = mixed.abs().pow(2.0).view((4,))
    out_node = g.cat([feat, feat], axis=0)

    a_data = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float16)
    b_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    g.set_input(a, a_data)
    g.set_input(b, b_data)
    g.execute()

    out = out_node.numpy()
    base = (((a_data - b_data) * (a_data + b_data)) / b_data)
    expected = np.concatenate([np.abs(base).reshape(4) ** 2, np.abs(base).reshape(4) ** 2]).astype(np.float16)
    print("complex out:", out)

    assert out.shape == (8,)
    assert np.allclose(out, expected, atol=1e-2)


test_graph_bridge_pow_abs_smoke()
test_graph_bridge_subtract_operator()
test_graph_bridge_composed_inference_graph()
