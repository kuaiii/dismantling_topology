import networkx as nx
import pytest

from autosnd.metrics import evaluate_order, validate_order


def _naive_trajectory(graph, order, steps):
    residual = graph.copy()
    values = []
    original_n = graph.number_of_nodes()
    for node in order[:steps]:
        residual.remove_node(node)
        largest = max((len(part) for part in nx.connected_components(residual)), default=0)
        values.append(largest / original_n)
    return tuple(values)


def test_reverse_dsu_matches_naive_deletion():
    graph = nx.barbell_graph(5, 2)
    order = list(range(graph.number_of_nodes()))
    metrics = evaluate_order(graph, order, rate=0.30)
    expected = _naive_trajectory(graph, order, metrics.steps)
    assert metrics.trajectory == expected
    assert metrics.anc_gcc == pytest.approx(sum(expected) / len(expected))


@pytest.mark.parametrize(
    "order,message",
    [
        ([0, 1], "exactly"),
        ([0, 1, 1], "duplicate"),
        ([0, 1, 9], "permutation"),
    ],
)
def test_order_validation_is_strict(order, message):
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match=message):
        validate_order(graph, order)


def test_empty_graph_has_zero_anc():
    metrics = evaluate_order(nx.Graph(), [], rate=1.0)
    assert metrics.anc_gcc == 0.0
    assert metrics.trajectory == ()
