from autosnd.search import _policy_violation, select_stage3_seeds
from autosnd.stage2 import EvidenceRecord, StructureDescription


def _record(candidate_id, runtime):
    code = "def degree_order(G):\n    return list(G.nodes())\n"
    return EvidenceRecord(
        candidate_id=candidate_id,
        code=code,
        valid=True,
        anc_gcc=0.10,
        runtime_seconds=runtime,
        error=None,
        structure=StructureDescription(
            family="other",
            signals=(),
            neighbor_scope=0,
            update_scope="static",
            expensive_patterns=(),
            neighbors_calls=0,
        ),
    )


def test_equal_quality_ties_do_not_penalize_fast_candidates_by_id():
    records = [
        _record("a", 100.0),
        _record("b", 90.0),
        _record("c", 2.0),
        _record("d", 1.0),
    ]
    seeds = select_stage3_seeds(
        records,
        {"quality": 1, "speed": 1, "balanced": 1, "repair": 1},
    )
    assert seeds["quality"] == ("d",)
    assert seeds["speed"] == ("d",)
    assert seeds["balanced"] == ("d",)


def test_fixed_policy_guard_ignores_comments_and_allows_bounded_two_hop():
    policy = object()
    comment_only = """def degree_order(G):
    # betweenness_centrality and unbounded two-hop are policy words only.
    return list(G.nodes())
"""
    bounded_two_hop = """def degree_order(G):
    nodes = list(G.nodes())
    for node in nodes:
        for neighbor in list(G.neighbors(node))[:4]:
            list(G.neighbors(neighbor))[:4]
    return nodes
"""
    forbidden_call = """def degree_order(G):
    values = betweenness_centrality(G)
    return sorted(G.nodes(), key=lambda node: values[node])
"""
    assert _policy_violation(comment_only, policy) is None
    assert _policy_violation(bounded_two_hop, policy) is None
    assert "betweenness_centrality" in _policy_violation(forbidden_call, policy)
