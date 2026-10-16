"""Run an original or newly bundled method through the same public interface."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="degree", help="For example: degree, NIRM, AutoSND")
    parser.add_argument("--graph", help="UTF-8 edge list: two node IDs per line")
    parser.add_argument("--stop", type=int, default=3)
    args = parser.parse_args()
    if args.stop < 0:
        parser.error("--stop must be nonnegative")

    import networkx as nx
    from network_dismantling.unified_interface import dismantle

    G = (nx.read_edgelist(args.graph, data=False, encoding="utf-8-sig")
         if args.graph else nx.karate_club_graph())
    order = dismantle(G, method=args.method, stop_condition=args.stop)
    print(json.dumps({"method": args.method, "nodes": len(G), "order": order},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
