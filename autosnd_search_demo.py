"""Single call: input graph -> AutoSND search -> selected dismantling order."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="Optional edge-list text; default: karate-club graph")
    parser.add_argument("--nodes", help="Optional list of all nodes, one ID per line")
    parser.add_argument("--selection", choices=["quality", "speed"], default="quality")
    parser.add_argument("--config", default=None)
    parser.add_argument("--evaluation-rate", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--stop", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.stop < 0:
        parser.error("--stop must be nonnegative")
    import networkx as nx
    from network_dismantling.unified_interface import dismantle
    if args.graph:
        G = nx.read_edgelist(args.graph, data=False, comments="#", encoding="utf-8-sig")
        if args.nodes:
            G.add_nodes_from(line.strip() for line in Path(args.nodes).read_text(encoding="utf-8-sig").splitlines() if line.strip())
    else:
        G = nx.karate_club_graph()
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    details = {}
    started = time.perf_counter()
    sequence = dismantle(G, method="AutoSND", stop_condition=args.stop,
                         config_path=args.config, selection=args.selection,
                         evaluation_rate=args.evaluation_rate,
                         candidate_timeout_seconds=args.timeout,
                         output_dir=args.output_dir, result_info=details)
    H = G.copy()
    largest = max((len(c) for c in nx.connected_components(H)), default=0)
    removed_to_stop = 0
    if largest > args.stop:
        for removed_to_stop, node in enumerate(sequence, start=1):
            H.remove_node(node)
            if max((len(c) for c in nx.connected_components(H)), default=0) <= args.stop:
                break
    report = {"details": details, "sequence": sequence, "removed_to_stop": removed_to_stop,
              "interface_seconds": time.perf_counter() - started}
    if details.get("run_dir"):
        result_path = Path(details["run_dir"]) / "returned_result.json"
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("RESULT:", result_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
