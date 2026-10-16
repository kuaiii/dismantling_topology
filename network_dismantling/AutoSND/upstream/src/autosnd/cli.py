"""Command-line interface for AutoSND."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from .baselines import BASELINES, get_baseline
from .config import api_environment, load_config, provider_seed_environment
from .datasets import dataset_info, load_graph, main12_names
from .llm import LLMClient
from .metrics import evaluate_order
from .search import run_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autosnd",
        description="Minimal AutoSND library for network dismantling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets = subparsers.add_parser("datasets", help="list the bundled main-12 graphs")
    datasets.add_argument("--verify", action="store_true", help="load and hash every graph")

    doctor = subparsers.add_parser("doctor", help="check configuration, data, and API settings")
    doctor.add_argument("--config", default="configs/smoke.yaml")
    doctor.add_argument(
        "--offline",
        action="store_true",
        help="skip AUTOSND_API_URL/AUTOSND_API_KEY checks",
    )

    baseline = subparsers.add_parser("baseline", help="run a native ND baseline")
    baseline.add_argument("--method", required=True, choices=tuple(BASELINES))
    baseline.add_argument("--datasets", nargs="+", required=True)
    baseline.add_argument("--rate", type=float, default=1.0)
    baseline.add_argument("--output", type=Path)

    search = subparsers.add_parser("search", help="run the three-stage AutoSND search")
    search.add_argument("--config", default="configs/smoke.yaml")
    search.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = _dispatch(args)
    except (KeyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    raise SystemExit(code)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "datasets":
        return _datasets(args.verify)
    if args.command == "doctor":
        return _doctor(Path(args.config), args.offline)
    if args.command == "baseline":
        return _baseline(args.method, args.datasets, args.rate, args.output)
    if args.command == "search":
        config = load_config(args.config)
        if args.output:
            from dataclasses import replace

            config = replace(config, output_dir=args.output)
        api_url, api_key = api_environment()
        client = LLMClient(
            api_url=api_url,
            api_key=api_key,
            model=config.model,
            seed=provider_seed_environment(),
        )
        result = run_search(config, client)
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "AutoSND-Q": result.final_quality.candidate_id,
                    "AutoSND-S": result.final_speed.candidate_id,
                },
                indent=2,
            )
        )
        return 0
    raise ValueError(f"unsupported command: {args.command}")


def _datasets(verify: bool) -> int:
    rows = []
    for name in main12_names():
        info = dataset_info(name)
        if verify:
            load_graph(name, verify=True)
        rows.append(
            {
                "name": name,
                "nodes": info.nodes,
                "edges": info.edges,
                "verified": verify,
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


def _doctor(config_path: Path, offline: bool) -> int:
    config = load_config(config_path)
    proxy = load_graph(config.proxy_dataset, verify=True)
    first = main12_names()[0]
    load_graph(first, verify=True)
    api = "skipped"
    if not offline:
        api_environment()
        api = "configured"
    print(
        json.dumps(
            {
                "config": str(config_path),
                "model": config.model,
                "proxy": {
                    "name": config.proxy_dataset,
                    "nodes": proxy.number_of_nodes(),
                    "edges": proxy.number_of_edges(),
                },
                "main12_count": len(main12_names()),
                "api": api,
            },
            indent=2,
        )
    )
    return 0


def _baseline(
    method: str,
    datasets: list[str],
    rate: float,
    output: Path | None,
) -> int:
    spec = get_baseline(method)
    rows = []
    for name in datasets:
        graph = load_graph(name, verify=True)
        order, runtime = spec.run_timed(graph, rate=rate)
        metrics = evaluate_order(graph, order, rate=rate)
        rows.append(
            {
                "method": spec.name,
                "fidelity": spec.fidelity,
                "dataset": dataset_info(name).name,
                "rate": rate,
                "anc_gcc": metrics.anc_gcc,
                "runtime_seconds": runtime,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            }
        )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(str(output.resolve()))
    else:
        print(json.dumps(rows, indent=2))
    return 0
