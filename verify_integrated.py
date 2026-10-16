"""Verify the two bundled methods with real local dependencies and an offline LLM."""
import argparse
import importlib.abc
import inspect
import json
from pathlib import Path
import sys
import tempfile


class RejectStandaloneAutoSND(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0].lower() == "autosnd":
            raise ImportError("Standalone AutoSND is disabled during portability verification")
        return None


def check_order(graph, sequence):
    assert len(sequence) == graph.number_of_nodes(), "Wrong sequence length"
    assert len(set(sequence)) == len(sequence), "Duplicate node"
    assert set(sequence) == set(graph.nodes()), "Original node IDs were not restored"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "nirm", "autosnd", "interface"], default="all")
    parser.add_argument("--in-process", action="store_true", help="Offline fixture check without Windows IPC; does not verify timeout/spawn")
    args = parser.parse_args()
    sys.meta_path.insert(0, RejectStandaloneAutoSND())
    import networkx as nx
    from network_dismantling.unified_interface import dismantle, METHOD_REGISTRY
    from network_dismantling import integration_adapters as adapters

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "docs/integration_manifest.json").read_text(encoding="utf-8"))
    baseline = set(manifest["original_method_names"])
    assert set(METHOD_REGISTRY) == baseline | {"NIRM", "AutoSND"}
    assert str(inspect.signature(dismantle)).startswith("(G:")
    assert set(inspect.signature(dismantle).parameters) == {"G", "method", "stop_condition", "kwargs"}
    G = nx.relabel_nodes(nx.karate_club_graph(), lambda node: "node_" + str(node))
    before = G.copy()
    check_order(G, dismantle(G, "degree", stop_condition=3))
    print("PASS: original interface and all original method names preserved; only NIRM and AutoSND added")
    assert adapters.NIRM_ROOT.is_relative_to(root)
    assert adapters.AUTOSND_ROOT.is_relative_to(root)
    print("PASS: bundled paths resolve inside this downloaded folder")

    if args.mode in ("all", "nirm"):
        for graph in (G, nx.Graph(), nx.empty_graph(1), nx.empty_graph(4),
                      nx.disjoint_union(nx.path_graph(6), nx.path_graph(3))):
            check_order(graph, dismantle(graph, "NIRM", stop_condition=3))
        print("PASS: real NIRM checkpoint inference, string IDs, empty graph, isolates, disconnected graph")
        print("NIRM MODEL:", adapters.NIRM_ROOT / "Model.py")
        print("NIRM CHECKPOINT:", adapters.NIRM_ROOT / "checkpoints/NIRM_onepass.pkl")

    if args.mode in ("all", "autosnd"):
        from network_dismantling.AutoSND.upstream.src.autosnd.datasets import load_graph
        from network_dismantling.AutoSND.upstream.src.autosnd.config import load_config
        proxy = load_graph("Powerlaw_500", verify=True)
        assert proxy.number_of_nodes() == 500
        assert load_config("configs/smoke.yaml").stage1.candidates == 4
        print("PASS: bundled AutoSND data and config resources, no standalone autosnd import")
        from network_dismantling.AutoSND.upstream.src.autosnd import candidate, search
        original_candidate_execution = candidate.execute_candidate
        original_search_execution = search.execute_candidate
        if args.in_process:
            def synchronous_fixture_execution(code, graph, *, rate, timeout_seconds):
                candidate.validate_candidate_source(code)
                class FixtureConnection:
                    def send(self, payload): self.payload = payload
                    def close(self): pass
                connection = FixtureConnection()
                candidate._candidate_worker(code, graph, connection)
                status, payload = connection.payload
                if status == "error":
                    return candidate.CandidateExecution(False, (), None, None, str(payload))
                order, runtime = payload
                from network_dismantling.AutoSND.upstream.src.autosnd.metrics import evaluate_order
                metrics = evaluate_order(graph, order, rate=rate)
                return candidate.CandidateExecution(True, tuple(order), metrics.anc_gcc, runtime, None)
            candidate.execute_candidate = synchronous_fixture_execution
            search.execute_candidate = synchronous_fixture_execution
            print("IN-PROCESS FIXTURE MODE: Windows spawning and timeout behavior are not verified")
        try:
            from autosnd_search_test import main as offline_search_test
            offline_search_test()
        finally:
            candidate.execute_candidate = original_candidate_execution
            search.execute_candidate = original_search_execution
    assert set(G.nodes()) == set(before.nodes()) and set(G.edges()) == set(before.edges())
    print("VERIFIED:", root)
    print("No real API calls were made.")


if __name__ == "__main__":
    main()
