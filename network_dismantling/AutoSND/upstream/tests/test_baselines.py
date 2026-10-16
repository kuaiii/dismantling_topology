import networkx as nx

from autosnd.baselines import BASELINES, get_baseline


EXPECTED = {
    "HDA",
    "CoreHD",
    "DC",
    "CI",
    "GND",
    "MinSum",
    "BPD",
    "KCore",
    "CLUC",
}


def test_registry_exposes_only_paper_native_methods():
    assert set(BASELINES) == EXPECTED
    assert get_baseline("corehd") is BASELINES["CoreHD"]


def test_every_baseline_returns_deterministic_full_order_without_mutation():
    graph = nx.barbell_graph(5, 1)
    graph.add_node(99)
    original_nodes = set(graph.nodes())
    original_edges = set(graph.edges())
    for spec in BASELINES.values():
        first, runtime = spec.run_timed(graph)
        second = spec.run(graph)
        assert first == second, spec.name
        assert runtime >= 0, spec.name
        assert len(first) == graph.number_of_nodes(), spec.name
        assert set(first) == original_nodes, spec.name
        assert set(graph.nodes()) == original_nodes, spec.name
        assert set(graph.edges()) == original_edges, spec.name
