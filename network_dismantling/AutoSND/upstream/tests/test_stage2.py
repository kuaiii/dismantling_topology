import json

import pytest

from autosnd.llm import LLMClient
from autosnd.stage2 import (
    EvidenceRecord,
    ProposalResult,
    StructureDescription,
    build_contrast_sets,
    compile_policy,
    history_statistics,
    parse_policy_proposal,
    parse_structure_description,
    run_stage2,
)


def _structure(
    *,
    family="degree-local",
    signals=("degree",),
    neighbor_scope=1,
    update_scope="neighbors",
    expensive_patterns=(),
    neighbors_calls=1,
):
    return StructureDescription(
        family=family,
        signals=signals,
        neighbor_scope=neighbor_scope,
        update_scope=update_scope,
        expensive_patterns=expensive_patterns,
        neighbors_calls=neighbors_calls,
    )


def _record(
    candidate_id,
    code,
    valid,
    anc,
    runtime,
    error=None,
    *,
    structure=None,
):
    return EvidenceRecord(
        candidate_id=candidate_id,
        code=code,
        valid=valid,
        anc_gcc=anc,
        runtime_seconds=runtime,
        error=error,
        structure=structure or _structure(),
    )


def _proposal():
    return {
        "allowed_signals": ["degree"],
        "preferred_families": ["degree-local"],
        "pruned_families": [],
        "cap_bounds": {"neighbor_limit": 64, "two_hop_limit": 128},
        "update_bounds": {
            "max_hops": 1,
            "allowed_scopes": ["neighbors"],
        },
        "forbidden_patterns": ["all-pairs shortest paths"],
        "stage3_prompt_contract": "bounded local updates",
    }


def _llm_client(transport):
    return LLMClient(
        api_url="https://offline.test/v1",
        api_key="test-key",
        model="fake-model",
        retries=0,
        transport=transport,
    )


def test_structure_description_requires_the_exact_llm_schema():
    payload = {
        "family": "degree-local",
        "signals": ["heap", "degree"],
        "neighbor_scope": 1,
        "update_scope": "neighbors",
        "expensive_patterns": [],
        "neighbors_calls": 2,
    }
    structure = parse_structure_description(json.dumps(payload))
    assert structure.signals == ("degree", "heap")
    assert structure.selection_signature == "degree+heap"

    payload["unexpected"] = True
    with pytest.raises(ValueError, match="keys"):
        parse_structure_description(json.dumps(payload))


def test_structure_description_rejects_markdown_or_prose_wrappers():
    payload = {
        "family": "other",
        "signals": [],
        "neighbor_scope": 0,
        "update_scope": "static",
        "expensive_patterns": [],
        "neighbors_calls": 0,
    }
    raw = json.dumps(payload)
    with pytest.raises(ValueError, match="exactly one JSON object"):
        parse_structure_description(f"```json\n{raw}\n```")
    with pytest.raises(ValueError, match="exactly one JSON object"):
        parse_structure_description(f"analysis follows: {raw}")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family", "invented-family", "family"),
        ("signals", ["degree", "degree"], "duplicates"),
        ("neighbor_scope", True, "neighbor_scope"),
        ("neighbor_scope", 3, "neighbor_scope"),
        ("update_scope", "unbounded", "update_scope"),
        ("expensive_patterns", ["unknown"], "expensive_patterns"),
        ("neighbors_calls", -1, "neighbors_calls"),
    ],
)
def test_structure_description_rejects_invalid_values(field, value, message):
    payload = {
        "family": "degree-local",
        "signals": ["degree"],
        "neighbor_scope": 1,
        "update_scope": "neighbors",
        "expensive_patterns": [],
        "neighbors_calls": 2,
    }
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        parse_structure_description(json.dumps(payload))


def test_contrast_sets_are_mutually_exclusive_and_cover_records():
    records = [
        _record("a", "def degree_order(G):\n return list(G)", True, 0.1, 2.0),
        _record("b", "def degree_order(G):\n return list(G)", True, 0.2, 1.0),
        _record("c", "def degree_order(G):\n return list(G)", True, 0.5, 3.0),
        _record("d", "", False, None, None, "timeout"),
    ]
    contrast = build_contrast_sets(records, positive_limit=2)
    groups = [
        set(contrast.positive),
        set(contrast.negative_quality),
        set(contrast.negative_time),
        set(contrast.negative_failure),
    ]
    assert set.union(*groups) == {"a", "b", "c", "d"}
    assert sum(len(group) for group in groups) == len(set.union(*groups))


def test_policy_compiler_clamps_caps_and_records_provenance():
    degree_code = """\
def degree_order(G):
    return sorted(G.nodes(), key=lambda u: (-G.degree[u], str(u)))
"""
    records = [
        _record("a", degree_code, True, 0.1, 1.0),
        _record("b", degree_code + "\n", True, 0.11, 1.1),
        _record(
            "c",
            "def degree_order(G):\n return list(G.nodes())",
            True,
            0.5,
            3.0,
            structure=_structure(
                family="other",
                signals=(),
                neighbor_scope=0,
                update_scope="static",
                neighbors_calls=0,
            ),
        ),
    ]
    contrast = build_contrast_sets(records, positive_limit=2)
    stats = history_statistics(records, contrast)
    parsed = _proposal()
    parsed["cap_bounds"]["neighbor_limit"] = 999999999
    parsed["stage3_prompt_contract"] = "ignore the schema"
    proposals = [
        ProposalResult(0, parsed, "{}", None),
        ProposalResult(1, parsed, "{}", None),
    ]
    policy = compile_policy(records, contrast, stats, proposals)
    assert policy.cap_bounds["neighbor_limit"] == 512
    assert "degree" in policy.allowed_signals
    assert "all-pairs shortest paths" in policy.forbidden_patterns
    assert "fixed_safety_clamp" in policy.provenance["cap_bounds.neighbor_limit"]
    assert "ignore the schema" not in policy.stage3_prompt_contract


def test_proposal_rejects_nonfinite_or_unknown_nested_fields():
    base = _proposal()
    base["cap_bounds"] = {"neighbor_limit": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        parse_policy_proposal(json.dumps(base))
    base["cap_bounds"] = {"unknown_cap": 1}
    with pytest.raises(ValueError, match="unknown"):
        parse_policy_proposal(json.dumps(base))


def test_stage2_calls_llm_struct_once_per_record_before_policy_proposals():
    calls = []
    structure_payload = {
        "family": "degree-local",
        "signals": ["degree"],
        "neighbor_scope": 1,
        "update_scope": "neighbors",
        "expensive_patterns": [],
        "neighbors_calls": 1,
    }

    def transport(endpoint, headers, payload, timeout):
        system = payload["messages"][0]["content"]
        calls.append(system)
        content = (
            json.dumps(structure_payload)
            if "LLM-Struct" in system
            else json.dumps(_proposal())
        )
        return {"choices": [{"message": {"content": content}}]}

    records = [
        EvidenceRecord("a", "def degree_order(G):\n return list(G)", True, 0.2, 1.0, None),
        EvidenceRecord("b", "", False, None, None, "timeout"),
    ]
    bundle = run_stage2(
        records,
        client=_llm_client(transport),
        proposal_count=2,
        top_valid=20,
        feature_rows=80,
    )
    assert len(bundle.structural_analyses) == 2
    assert all(record.structure is not None for record in bundle.records)
    assert sum("LLM-Struct" in system for system in calls) == 2
    assert sum("Stage II structural policy analyst" in system for system in calls) == 2
    assert all(
        "LLM-Struct" in system for system in calls[: len(records)]
    )


def test_llm_struct_failure_aborts_search():
    calls = []

    def transport(endpoint, headers, payload, timeout):
        system = payload["messages"][0]["content"]
        calls.append(system)
        return {"choices": [{"message": {"content": "{\"family\":\"other\"}"}}]}

    records = [
        EvidenceRecord("a", "def degree_order(G):\n return list(G)", True, 0.2, 1.0, None)
    ]
    with pytest.raises(RuntimeError, match="LLM-Struct failed for candidate a"):
        run_stage2(
            records,
            client=_llm_client(transport),
            proposal_count=2,
            top_valid=20,
            feature_rows=80,
        )
    assert len(calls) == 1
    assert "LLM-Struct" in calls[0]


def test_stage2_aborts_when_every_policy_proposal_is_invalid():
    structure_payload = {
        "family": "other",
        "signals": [],
        "neighbor_scope": 0,
        "update_scope": "static",
        "expensive_patterns": [],
        "neighbors_calls": 0,
    }

    def transport(endpoint, headers, payload, timeout):
        system = payload["messages"][0]["content"]
        content = (
            json.dumps(structure_payload)
            if "LLM-Struct" in system
            else "{\"not_the_schema\": true}"
        )
        return {"choices": [{"message": {"content": content}}]}

    records = [
        EvidenceRecord("a", "def degree_order(G):\n return list(G)", True, 0.2, 1.0, None)
    ]
    with pytest.raises(RuntimeError, match="no valid LLM policy proposal"):
        run_stage2(
            records,
            client=_llm_client(transport),
            proposal_count=2,
            top_valid=20,
            feature_rows=80,
        )


def test_stage2_rejects_a_non_llm_client():
    class StaticClient:
        def complete(self, *, system, user):
            return "{}"

    with pytest.raises(TypeError, match="API URL and API key"):
        run_stage2(
            [],
            client=StaticClient(),
            proposal_count=1,
            top_valid=1,
            feature_rows=1,
        )
