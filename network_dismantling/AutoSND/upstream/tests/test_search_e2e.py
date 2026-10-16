import json
from dataclasses import replace

from autosnd.config import load_config
from autosnd.llm import LLMClient
from autosnd.search import run_search


CANDIDATES = [
    """def degree_order(G):
    return sorted(G.nodes(), key=lambda u: (-G.degree[u], u))
""",
    """def degree_order(G):
    return sorted(G.nodes(), key=lambda u: (G.degree[u], u), reverse=True)
""",
    """def degree_order(G):
    return sorted(G.nodes(), reverse=True)
""",
    """def degree_order(G):
    return sorted(G.nodes())
""",
    """def degree_order(G):
    nodes = list(G.nodes())
    return nodes[1:] + nodes[:1]
""",
    """def degree_order(G):
    nodes = sorted(G.nodes())
    return nodes[::2] + nodes[1::2]
""",
    """def degree_order(G):
    nodes = sorted(G.nodes(), reverse=True)
    return nodes[::2] + nodes[1::2]
""",
    """def degree_order(G):
    nodes = list(G.nodes())
    return nodes[-1:] + nodes[:-1]
""",
]


def test_offline_fake_llm_completes_all_three_stages(tmp_path):
    state = {
        "candidate": 0,
        "structure": 0,
        "proposal": 0,
    }

    def transport(endpoint, headers, payload, timeout):
        system = payload["messages"][0]["content"]
        if "LLM-Struct" in system:
            state["structure"] += 1
            structure = {
                "family": "degree-local",
                "signals": ["degree"],
                "neighbor_scope": 1,
                "update_scope": "neighbors",
                "expensive_patterns": [],
                "neighbors_calls": 1,
            }
            content = json.dumps(structure)
        elif "Stage II structural policy analyst" in system:
            state["proposal"] += 1
            proposal = {
                "allowed_signals": ["degree", "heap"],
                "preferred_families": ["degree-local"],
                "pruned_families": [],
                "cap_bounds": {"neighbor_limit": 64, "two_hop_limit": 128},
                "update_bounds": {"max_hops": 1},
                "forbidden_patterns": ["all-pairs shortest paths"],
                "stage3_prompt_contract": "bounded local updates",
            }
            content = json.dumps(proposal)
        else:
            code = CANDIDATES[state["candidate"]]
            state["candidate"] += 1
            content = f"```python\n{code}```"
        return {"choices": [{"message": {"content": content}}]}

    config = load_config("configs/smoke.yaml")
    config = replace(
        config,
        output_dir=tmp_path / "run",
        candidate_timeout_seconds=15,
    )
    client = LLMClient(
        api_url="https://offline.test/v1",
        api_key="never-write-this-key",
        model="fake-model",
        transport=transport,
    )
    result = run_search(config, client)
    assert len(result.stage1_records) == 4
    assert result.root_record.valid
    assert len(result.stage2.structural_analyses) == 4
    assert len(result.stage2.proposals) == 2
    assert len(result.stage3_records) == 4
    assert state == {
        "candidate": 8,
        "structure": 4,
        "proposal": 2,
    }
    assert result.final_quality.valid
    assert result.final_speed.valid
    assert (tmp_path / "run" / "final" / "AutoSND-Q.py").is_file()
    manifest_path = tmp_path / "run" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"] == "fake-model"
    assert manifest["configured_model"] == config.model
    assert manifest["llm_request"]["provider_seed"] is None
    assert manifest["schema_version"] == 2
    assert manifest["budgets"] == {
        "B1": 4,
        "B2": 2,
        "B3": 4,
    }
    archived_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "run").rglob("*")
        if path.is_file()
    )
    assert "never-write-this-key" not in archived_text
    assert "offline.test" not in archived_text
