"""The nine ND-native baselines."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..metrics import validate_order
from . import algorithms

OrderFunction = Callable[..., list[Any]]


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    function: OrderFunction
    fidelity: str

    def run(self, graph: nx.Graph, *, rate: float | None = None) -> list[Any]:
        """Return a strict full permutation without mutating ``graph``."""
        order, _ = self.run_timed(graph, rate=rate)
        return order

    def run_timed(
        self,
        graph: nx.Graph,
        *,
        rate: float | None = None,
    ) -> tuple[list[Any], float]:
        """Return a validated order and algorithm-only runtime."""
        before_nodes = set(graph.nodes())
        before_edges = set(graph.edges())
        working_graph = graph.copy()
        started = time.perf_counter()
        order = self.function(working_graph, rate=rate)
        runtime_seconds = time.perf_counter() - started
        validate_order(graph, order)
        if set(graph.nodes()) != before_nodes or set(graph.edges()) != before_edges:
            raise RuntimeError(f"{self.name} mutated its input graph")
        return order, runtime_seconds


BASELINES: dict[str, BaselineSpec] = {
    "HDA": BaselineSpec("HDA", algorithms.hda_original_order, "native reimplementation"),
    "CoreHD": BaselineSpec(
        "CoreHD", algorithms.corehd_fast_order, "native reimplementation"
    ),
    "DC": BaselineSpec("DC", algorithms.dc_order, "native reimplementation"),
    "CI": BaselineSpec("CI", algorithms.ci_order, "native reimplementation"),
    "GND": BaselineSpec(
        "GND", algorithms.gnd_order, "clean-room GND-style approximation"
    ),
    "MinSum": BaselineSpec(
        "MinSum", algorithms.minsum_order, "clean-room Min-Sum-inspired approximation"
    ),
    "BPD": BaselineSpec(
        "BPD", algorithms.bpd_order, "clean-room BPD-inspired approximation"
    ),
    "KCore": BaselineSpec(
        "KCore", algorithms.kcore_order, "native reimplementation"
    ),
    "CLUC": BaselineSpec(
        "CLUC", algorithms.cluc_order, "native reimplementation"
    ),
}


def get_baseline(name: str) -> BaselineSpec:
    aliases = {key.casefold(): key for key in BASELINES}
    try:
        return BASELINES[aliases[name.casefold()]]
    except KeyError as exc:
        raise KeyError(
            f"unknown baseline {name!r}; choose one of: {', '.join(BASELINES)}"
        ) from exc
