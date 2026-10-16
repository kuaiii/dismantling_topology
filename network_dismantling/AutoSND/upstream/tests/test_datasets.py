from importlib.resources import files

import networkx as nx

from autosnd.datasets import (
    dataset_info,
    load_graph,
    load_manifest,
    main12_names,
)


EXPECTED_MAIN12 = (
    "CEnew",
    "Collaboration",
    "condmat",
    "crime",
    "email",
    "Grid",
    "hamster",
    "HepPh",
    "Yeast",
    "HI-II-14",
    "Digg",
    "Gnutella31",
)


def test_main12_is_exact_and_proxy_is_separate():
    assert main12_names() == EXPECTED_MAIN12
    data_root = files("autosnd.data")
    actual = {path.name for path in data_root.joinpath("main12").iterdir()}
    expected = {dataset_info(name).file for name in EXPECTED_MAIN12}
    assert actual == expected
    assert {path.name for path in data_root.joinpath("proxy").iterdir()} == {
        "Powerlaw_500.edgelist"
    }
    assert data_root.joinpath("NOTICE.md").is_file()


def test_all_bundled_graphs_match_manifest():
    manifest = load_manifest()
    for name, expected in manifest["datasets"].items():
        graph = load_graph(name, verify=True)
        assert graph.number_of_nodes() == expected["nodes"]
        assert graph.number_of_edges() == expected["edges"]
        assert set(graph.nodes()) == set(range(expected["nodes"]))
        assert nx.is_connected(graph)
