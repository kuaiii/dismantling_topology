"""Offline full-search integration test: mock only LLM transport, run real workers."""
import json
from pathlib import Path
import tempfile


CODES = [
    "def degree_order(G):\n    return sorted(G.nodes(), key=lambda u: (-G.degree[u], u))\n",
    "def degree_order(G):\n    return sorted(G.nodes(), key=lambda u: (G.degree[u], u), reverse=True)\n",
    "def degree_order(G):\n    return sorted(G.nodes(), reverse=True)\n",
    "def degree_order(G):\n    return sorted(G.nodes())\n",
    "def degree_order(G):\n    nodes = list(G.nodes())\n    return nodes[1:] + nodes[:1]\n",
    "def degree_order(G):\n    nodes = sorted(G.nodes())\n    return nodes[::2] + nodes[1::2]\n",
    "def degree_order(G):\n    nodes = sorted(G.nodes(), reverse=True)\n    return nodes[::2] + nodes[1::2]\n",
    "def degree_order(G):\n    nodes = list(G.nodes())\n    return nodes[-1:] + nodes[:-1]\n",
]


def fake_client():
    from network_dismantling.AutoSND.upstream.src.autosnd.llm import LLMClient
    state = {"candidate": 0, "structure": 0, "proposal": 0}
    def transport(endpoint, headers, payload, timeout):
        system = payload["messages"][0]["content"]
        if "LLM-Struct" in system:
            state["structure"] += 1
            content = json.dumps({"family": "degree-local", "signals": ["degree"],
                                  "neighbor_scope": 1, "update_scope": "neighbors",
                                  "expensive_patterns": [], "neighbors_calls": 1})
        elif "Stage II structural policy analyst" in system:
            state["proposal"] += 1
            content = json.dumps({"allowed_signals": ["degree", "heap"],
                                  "preferred_families": ["degree-local"], "pruned_families": [],
                                  "cap_bounds": {"neighbor_limit": 64, "two_hop_limit": 128},
                                  "update_bounds": {"max_hops": 1},
                                  "forbidden_patterns": ["all-pairs shortest paths"],
                                  "stage3_prompt_contract": "bounded local updates"})
        else:
            content = "```python\n" + CODES[state["candidate"]] + "```"
            state["candidate"] += 1
        return {"choices": [{"message": {"content": content}}]}
    client = LLMClient(api_url="https://offline-test.invalid/v1", api_key="offline-test-secret",
                       model="offline-test-model", transport=transport)
    return client, state


def main():
    import networkx as nx
    import network_dismantling.AutoSND.upstream.src.autosnd.search as search
    from network_dismantling.unified_interface import dismantle, METHOD_REGISTRY

    assert {"NIRM", "AutoSND"} <= set(METHOD_REGISTRY)
    def forbidden_proxy(*args, **kwargs):
        raise AssertionError("Input-graph search must never load Powerlaw_500 or another bundled graph")
    real_loader = search.load_graph
    real_execution = search.execute_candidate
    observations = []
    def observed_execution(code, graph, **kwargs):
        observations.append(graph.number_of_nodes())
        return real_execution(code, graph, **kwargs)
    search.load_graph = forbidden_proxy
    search.execute_candidate = observed_execution
    try:
        with tempfile.TemporaryDirectory() as directory:
            histories = []
            for size, selection in ((10, "quality"), (13, "speed")):
                G = nx.relabel_nodes(nx.cycle_graph(size - 1), lambda node: "node_" + str(node))
                G.add_node("isolated")
                before_nodes, before_edges = set(G.nodes()), set(G.edges())
                client, counts = fake_client()
                info = {}
                observations.clear()
                sequence = dismantle(G, "AutoSND", stop_condition=2,
                                     selection=selection, llm_client=client,
                                     candidate_timeout_seconds=30,
                                     output_dir=directory, result_info=info)
                assert len(sequence) == size and set(sequence) == set(G.nodes())
                assert len(set(sequence)) == size
                assert set(G.nodes()) == before_nodes and set(G.edges()) == before_edges
                assert counts == {"candidate": 8, "structure": 4, "proposal": 2}
                assert len(observations) == 9 and set(observations) == {size}
                run = Path(info["run_dir"])
                manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
                assert manifest["proxy_dataset"] is None
                assert manifest["evaluation_graph"]["source"] == "supplied_graph"
                assert manifest["evaluation_graph"]["nodes"] == size
                assert manifest["evaluation_rate"] == 1.0
                assert info["status"] == "complete" and Path(info["selected_code_path"]).is_file()
                stage1 = (run / "stage1_records.jsonl").read_text(encoding="utf-8")
                assert f"Target network: n={size}," in stage1
                selection_record = json.loads((run / "final/selection.json").read_text(encoding="utf-8"))
                assert "supplied-graph" in selection_record["selection_scope"]
                text = "\n".join(path.read_text(encoding="utf-8") for path in run.rglob("*") if path.is_file())
                assert "offline-test-secret" not in text and "offline-test.invalid" not in text
                histories.append((run, manifest["evaluation_graph"]["sha256"]))
                print(f"PASS: {size}-node target, complete three stages, selection={selection}, no proxy loading")
            assert histories[0][0] != histories[1][0] and histories[0][1] != histories[1][1]
            empty_info = {}
            assert dismantle(nx.Graph(), "AutoSND", result_info=empty_info) == []
            assert empty_info["search_performed"] is False
    finally:
        search.load_graph = real_loader
        search.execute_candidate = real_execution
    print("ALL OFFLINE SEARCH TESTS PASSED. No network request or paid API call was made.")


if __name__ == "__main__":
    main()
