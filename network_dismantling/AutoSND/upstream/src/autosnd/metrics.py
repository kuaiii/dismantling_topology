"""Strict full-permutation validation and ANC (GCC) evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx


@dataclass(frozen=True)
class DismantlingMetrics:
    anc_gcc: float
    rate: float
    steps: int
    trajectory: tuple[float, ...]


def validate_order(graph: nx.Graph, order: Iterable[Any]) -> list[Any]:
    sequence = list(order)
    expected = set(graph.nodes())
    actual = set(sequence)
    if len(sequence) != len(expected):
        raise ValueError(
            f"order must contain exactly {len(expected)} nodes; got {len(sequence)}"
        )
    if len(actual) != len(sequence):
        raise ValueError("order contains duplicate nodes")
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"order is not a node permutation; missing={len(missing)}, extra={len(extra)}"
        )
    return sequence


def evaluate_order(
    graph: nx.Graph,
    order: Iterable[Any],
    *,
    rate: float = 1.0,
) -> DismantlingMetrics:
    """Evaluate ANC (GCC) after each deletion in the requested prefix.

    GCC is normalized by the original number of nodes. ``rate=0.30`` therefore
    returns the search metric ANC@30% (GCC); ``rate=1`` returns complete-sequence
    ANC (GCC). The input must be a strict full node permutation.
    """
    if not 0 < rate <= 1:
        raise ValueError("rate must be in (0, 1]")
    sequence = validate_order(graph, order)
    n = graph.number_of_nodes()
    if n == 0:
        return DismantlingMetrics(0.0, rate, 0, ())

    steps = max(1, min(n, int(round(n * rate))))
    prefix = sequence[:steps]
    removed = set(prefix)
    nodes = list(graph.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    dsu = _DisjointSet(n)

    for node in nodes:
        if node not in removed:
            dsu.activate(index[node])
    for left, right in graph.edges():
        dsu.union(index[left], index[right])

    reverse_values: list[float] = []
    for offset in range(steps, 0, -1):
        reverse_values.append(dsu.largest / n)
        node = prefix[offset - 1]
        node_idx = index[node]
        dsu.activate(node_idx)
        for neighbor in graph.neighbors(node):
            dsu.union(node_idx, index[neighbor])

    trajectory = tuple(reversed(reverse_values))
    return DismantlingMetrics(
        anc_gcc=sum(trajectory) / len(trajectory),
        rate=rate,
        steps=steps,
        trajectory=trajectory,
    )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.component_size = [1] * size
        self.active = [False] * size
        self.largest = 0

    def activate(self, node: int) -> None:
        if self.active[node]:
            return
        self.active[node] = True
        self.parent[node] = node
        self.component_size[node] = 1
        self.largest = max(self.largest, 1)

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        if not self.active[left] or not self.active[right]:
            return
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.component_size[root_left] < self.component_size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.component_size[root_left] += self.component_size[root_right]
        self.largest = max(self.largest, self.component_size[root_left])
