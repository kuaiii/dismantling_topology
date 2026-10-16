import networkx as nx
import pytest

from autosnd.candidate import (
    ROOT_HDA_CODE,
    execute_candidate,
    extract_candidate_code,
    validate_candidate_source,
)


def test_extracts_python_fence():
    response = "Explanation\n```python\ndef degree_order(G):\n    return list(G.nodes())\n```"
    assert extract_candidate_code(response).startswith("def degree_order")


@pytest.mark.parametrize(
    "code,match",
    [
        ("import os\ndef degree_order(G): return []", "top-level|Import"),
        ("def degree_order(G):\n    return open('x')", "not allowed"),
        ("x = 1\ndef degree_order(G):\n    return []", "top-level"),
        ("def degree_order(G=[]):\n    return []", "default"),
    ],
)
def test_safety_guard_rejects_unsafe_or_ambiguous_code(code, match):
    with pytest.raises(ValueError, match=match):
        validate_candidate_source(code)


def test_root_candidate_executes_in_timeout_worker():
    graph = nx.path_graph(10)
    result = execute_candidate(ROOT_HDA_CODE, graph, rate=0.30, timeout_seconds=10)
    assert result.valid
    assert len(result.order) == 10
    assert result.anc_gcc is not None


def test_large_order_does_not_deadlock_the_windows_spawn_pipe():
    graph = nx.path_graph(5000)
    code = "def degree_order(G):\n    return list(G.nodes())\n"
    result = execute_candidate(code, graph, rate=0.30, timeout_seconds=15)
    assert result.valid, result.error
    assert len(result.order) == 5000


def test_generator_work_is_included_in_candidate_runtime():
    graph = nx.path_graph(100)
    generator_code = """def degree_order(G):
    for node in G.nodes():
        value = 0
        for index in range(50000):
            value += index
        yield node
"""
    list_code = """def degree_order(G):
    order = []
    for node in G.nodes():
        value = 0
        for index in range(50000):
            value += index
        order.append(node)
    return order
"""
    generated = execute_candidate(
        generator_code, graph, rate=0.30, timeout_seconds=15
    )
    materialized = execute_candidate(
        list_code, graph, rate=0.30, timeout_seconds=15
    )
    assert generated.valid, generated.error
    assert materialized.valid, materialized.error
    assert generated.runtime_seconds is not None
    assert materialized.runtime_seconds is not None
    assert generated.runtime_seconds >= 0.25 * materialized.runtime_seconds


def test_timeout_records_censored_runtime():
    graph = nx.path_graph(5)
    code = """def degree_order(G):
    while True:
        pass
"""
    result = execute_candidate(code, graph, rate=0.30, timeout_seconds=0.1)
    assert not result.valid
    assert result.runtime_censored
    assert result.runtime_seconds == pytest.approx(0.1)
