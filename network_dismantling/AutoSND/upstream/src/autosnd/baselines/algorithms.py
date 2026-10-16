# -*- coding: utf-8 -*-
"""Self-contained baseline algorithms for network dismantling experiments.

All functions return a full node order. The evaluated prefix is controlled by
the caller through the removal rate, which keeps baseline code independent from
metric code.
"""

from __future__ import annotations

import heapq
import warnings
from typing import Any, Iterable, List, Mapping

import networkx as nx


def _node_key(node: Any) -> tuple[int, Any]:
    try:
        return (0, int(node))
    except (TypeError, ValueError):
        return (1, str(node))


def complete_order(graph: nx.Graph, order: Iterable[Any]) -> List[Any]:
    """Keep valid unique nodes from ``order`` and append all missing nodes."""
    nodes = set(graph.nodes())
    out: List[Any] = []
    seen = set()
    for node in order:
        if node in nodes and node not in seen:
            seen.add(node)
            out.append(node)
    out.extend([node for node in graph.nodes if node not in seen])
    return out


def score_order(graph: nx.Graph, scores: Mapping[Any, float]) -> List[Any]:
    """Return a deterministic descending-score order and append missing nodes."""
    order = [node for node, _ in sorted(scores.items(), key=lambda item: (-float(item[1]), _node_key(item[0])))]
    return complete_order(graph, order)


def _budget(graph: nx.Graph, rate: float | None) -> int:
    n = graph.number_of_nodes()
    if rate is None:
        return n
    return max(1, min(n, int(round(n * rate))))


def hda_original_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Original HDA: repeatedly rescan the residual graph for max degree."""
    h = graph.copy()
    order: List[Any] = []
    for _ in range(_budget(graph, rate)):
        if h.number_of_nodes() == 0:
            break
        node = max(h.nodes, key=lambda u: (h.degree[u], str(u)))
        order.append(node)
        h.remove_node(node)
    return complete_order(graph, order)


def dc_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Static degree centrality baseline."""
    order = sorted(graph.nodes(), key=lambda u: (graph.degree[u], str(u)), reverse=True)
    return complete_order(graph, order)


def kcore_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Static k-core baseline, tie-broken by degree."""
    if graph.number_of_edges() == 0:
        return dc_order(graph, rate)
    core = nx.core_number(nx.Graph(graph))
    order = sorted(graph.nodes(), key=lambda u: (core.get(u, 0), graph.degree[u], _node_key(u)), reverse=True)
    return complete_order(graph, order)


def cluc_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """CLUC/ClusterRank-style static local-clustering baseline."""
    if graph.number_of_nodes() == 0:
        return []
    h = nx.Graph(graph)
    clustering = nx.clustering(h)
    degree = dict(h.degree())
    scores = {}
    for node in h.nodes:
        neighbor_pressure = sum(degree[neighbor] + 1 for neighbor in h.neighbors(node))
        scores[node] = (10.0 ** (-clustering[node])) * neighbor_pressure
    return score_order(graph, scores)


def ci_order(graph: nx.Graph, rate: float | None = None, radius: int = 2) -> List[Any]:
    """Collective Influence baseline with radius 2 by default."""
    h = nx.Graph(graph)
    degree = dict(h.degree())
    scores: dict[Any, float] = {}
    for node in h.nodes:
        if degree[node] <= 1:
            scores[node] = 0.0
            continue
        lengths = nx.single_source_shortest_path_length(h, node, cutoff=radius)
        boundary = [other for other, distance in lengths.items() if distance == radius]
        scores[node] = float((degree[node] - 1) * sum(max(0, degree[other] - 1) for other in boundary))
    return score_order(graph, scores)


def corehd_fast_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Online CoreHD-fast: remove max residual degree in the current 2-core."""
    h = graph.to_undirected() if graph.is_directed() else graph
    nodes = list(h.nodes())
    alive = set(nodes)
    nbrs = {u: set(h.neighbors(u)) for u in nodes}
    deg = {u: len(nbrs[u]) for u in nodes}

    core = set(nodes)
    core_deg = {u: deg[u] for u in nodes}
    queue = [u for u in nodes if core_deg[u] < 2]
    while queue:
        u = queue.pop()
        if u not in core:
            continue
        core.remove(u)
        for v in nbrs[u]:
            if v in core:
                core_deg[v] -= 1
                if core_deg[v] < 2:
                    queue.append(v)

    version = {u: 0 for u in nodes}
    core_heap = [(-deg[u], str(u), u, version[u]) for u in core]
    all_heap = [(-deg[u], str(u), u, version[u]) for u in nodes]
    heapq.heapify(core_heap)
    heapq.heapify(all_heap)

    def peel_from_core(start_nodes: Iterable[Any]) -> None:
        q = [u for u in start_nodes if u in core and core_deg.get(u, 0) < 2]
        while q:
            x = q.pop()
            if x not in core:
                continue
            core.remove(x)
            for y in nbrs[x]:
                if y in core:
                    core_deg[y] -= 1
                    if core_deg[y] < 2:
                        q.append(y)

    order: List[Any] = []
    target = _budget(graph, rate)
    while alive and len(order) < target:
        if core:
            while core_heap:
                _, _, u, vu = heapq.heappop(core_heap)
                if u in alive and u in core and vu == version[u]:
                    break
            else:
                u = max(core, key=lambda x: (deg.get(x, 0), str(x)))
        else:
            while all_heap:
                _, _, u, vu = heapq.heappop(all_heap)
                if u in alive and vu == version[u]:
                    break
            else:
                u = max(alive, key=lambda x: (deg.get(x, 0), str(x)))

        if u not in alive:
            continue
        order.append(u)
        alive.remove(u)
        was_core = u in core
        if was_core:
            core.remove(u)
        touched = set()
        for v in list(nbrs[u]):
            if v not in alive:
                continue
            nbrs[v].discard(u)
            deg[v] = len(nbrs[v] & alive)
            version[v] += 1
            touched.add(v)
            heapq.heappush(all_heap, (-deg[v], str(v), v, version[v]))
            if v in core:
                if was_core:
                    core_deg[v] -= 1
                heapq.heappush(core_heap, (-deg[v], str(v), v, version[v]))
        nbrs[u].clear()
        deg[u] = 0
        version[u] += 1
        if touched:
            peel_from_core(touched)
    return complete_order(graph, order)


def _top_scored_nodes(scores: Mapping[Any, float], k: int) -> List[Any]:
    return [
        node
        for node, _score in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), _node_key(item[0])),
        )[:k]
    ]


def _largest_component_nodes(graph: nx.Graph) -> set[Any]:
    if graph.number_of_nodes() == 0:
        return set()
    return set(max(nx.connected_components(graph), key=lambda c: (len(c), tuple(sorted(c, key=_node_key)))))


def _component_batch_size(component_size: int) -> int:
    if component_size <= 100:
        return 1
    if component_size <= 2000:
        return max(1, component_size // 100)
    return max(8, component_size // 80)


def _dynamic_component_order(
    graph: nx.Graph,
    score_func,
    rate: float | None = None,
) -> List[Any]:
    """Batch dynamic dismantling helper for strong Python baselines."""
    h = nx.Graph(graph)
    order: List[Any] = []
    target = _budget(graph, rate)
    while h.number_of_nodes() and len(order) < target:
        component_nodes = _largest_component_nodes(h)
        if not component_nodes:
            break
        component = h.subgraph(component_nodes).copy()
        scores = score_func(component)
        if not scores:
            scores = {node: float(component.degree[node]) for node in component.nodes}
        batch = min(target - len(order), _component_batch_size(component.number_of_nodes()))
        remove_nodes = _top_scored_nodes(scores, batch)
        for node in remove_nodes:
            if node in h:
                order.append(node)
                h.remove_node(node)
    return complete_order(graph, order)


def minsum_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Min-Sum-inspired decycling and dismantling order.

    The implementation follows the decycling view of network dismantling: first
    target the maintained 2-core using local cavity-style depth pressure, then
    continue greedily on the largest residual component until a full order is
    produced.
    """

    def scores(component: nx.Graph) -> dict[Any, float]:
        if component.number_of_nodes() == 0:
            return {}
        try:
            core = nx.k_core(component, k=2)
        except nx.NetworkXError:
            core = nx.Graph()
        pool = core if core.number_of_nodes() else component
        degree = dict(component.degree())
        out: dict[Any, float] = {}
        for node in pool.nodes:
            nbr_pressure = sum(max(0, degree.get(neighbor, 0) - 1) for neighbor in component.neighbors(node))
            out[node] = 10_000.0 * max(0, degree.get(node, 0) - 1) + nbr_pressure + degree.get(node, 0)
        return out

    return _dynamic_component_order(graph, scores, rate)


def bpd_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Belief-propagation-decimation-inspired dismantling order.

    This clean-room Python version uses local non-backtracking pressure as a
    deterministic proxy for BPD occupation probability, then decimates the
    highest-pressure vertices in small batches.
    """

    def scores(component: nx.Graph) -> dict[Any, float]:
        degree = dict(component.degree())
        out: dict[Any, float] = {}
        for node in component.nodes:
            excess = max(0, degree.get(node, 0) - 1)
            cavity = sum(max(0, degree.get(neighbor, 0) - 1) for neighbor in component.neighbors(node))
            second = 0.0
            for neighbor in component.neighbors(node):
                second += sum(max(0, degree.get(two_hop, 0) - 1) for two_hop in component.neighbors(neighbor) if two_hop != node)
            out[node] = excess * (1.0 + cavity) + 0.05 * second + degree.get(node, 0)
        return out

    return _dynamic_component_order(graph, scores, rate)


def gnd_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Generalized-network-dismantling-style separator order.

    Uses spectral bisection on components of at most 2,500 nodes and removes
    high cross-cut vertices. Larger or numerically difficult components use an
    explicitly documented deterministic separator-pressure approximation.
    """

    def spectral_separator_scores(component: nx.Graph) -> dict[Any, float]:
        n = component.number_of_nodes()
        degree = dict(component.degree())
        if n < 4:
            return {node: float(degree.get(node, 0)) for node in component.nodes}
        if n <= 2500:
            try:
                import numpy as np
                import scipy.sparse.linalg as spla
            except ImportError as exc:
                raise RuntimeError(
                    "GND spectral mode requires the optional 'gnd' dependencies"
                ) from exc

            nodes = list(component.nodes())
            matrix = nx.laplacian_matrix(component, nodelist=nodes).astype(float)
            v0 = np.linspace(-1.0, 1.0, n, dtype=float)
            v0 -= float(v0.mean())
            try:
                values, vectors = spla.eigsh(
                    matrix,
                    k=2,
                    which="SM",
                    tol=1e-3,
                    maxiter=max(200, n * 2),
                    v0=v0,
                )
                idx = int(np.argsort(values)[1])
                fiedler = vectors[:, idx]
                median = float(np.median(fiedler))
                side = {nodes[i]: fiedler[i] >= median for i in range(n)}
                out: dict[Any, float] = {}
                for node in nodes:
                    cross = sum(
                        1
                        for neighbor in component.neighbors(node)
                        if side[neighbor] != side[node]
                    )
                    out[node] = 1000.0 * cross + degree.get(node, 0)
                if any(score > 0 for score in out.values()):
                    return out
            except Exception as exc:
                warnings.warn(
                    f"GND spectral step failed ({type(exc).__name__}); "
                    "using deterministic separator pressure",
                    RuntimeWarning,
                    stacklevel=2,
                )
        out = {}
        for node in component.nodes:
            nbr_degree = sum(degree.get(neighbor, 0) for neighbor in component.neighbors(node))
            out[node] = float(degree.get(node, 0) * (1 + nbr_degree))
        return out

    return _dynamic_component_order(graph, spectral_separator_scores, rate)
