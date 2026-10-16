"""Minimal three-stage AutoSND search for the ND task."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .candidate import (
    ROOT_HDA_CODE,
    execute_candidate,
    extract_candidate_code,
)
from .config import SearchConfig
from .datasets import load_graph
from .llm import LLMClient
from .stage2 import (
    CompiledPolicy,
    EvidenceRecord,
    Stage2Bundle,
    run_stage2,
)

STAGE1_SYSTEM = """\
You design deterministic network-dismantling heuristics.
Return only one Python code block defining degree_order(G).
The function must return every node exactly once and must not mutate G.
Imports, file/network/process I/O, randomness, eval, and exec are forbidden.
Use only Graph methods, Python builtins, and the available heapq/math/collections modules.
"""

STAGE3_ROLES = {
    "quality": "Prioritize lower ANC on the configured evaluation prefix (GCC) while respecting the compiled policy.",
    "speed": "Prioritize runtime among candidates that retain competitive ANC@30% (GCC).",
    "balanced": "Explore a structurally distinct quality-runtime trade-off.",
    "repair": "Preserve the useful structure but remove expensive or fragile operations.",
}
EXPLORATION_DIRECTIONS = (
    "change the node-priority logic without assuming a target signal family",
    "change how residual state is maintained and updated",
    "change the bounded local-neighborhood access pattern",
    "simplify bookkeeping and reduce repeated work",
    "try a structurally distinct deterministic mechanism",
)


@dataclass(frozen=True)
class SearchResult:
    output_dir: Path
    root_record: EvidenceRecord
    stage1_records: tuple[EvidenceRecord, ...]
    stage2: Stage2Bundle
    stage3_records: tuple[EvidenceRecord, ...]
    seeds: dict[str, tuple[str, ...]]
    final_quality: EvidenceRecord
    final_speed: EvidenceRecord


def run_search(config: SearchConfig, client: LLMClient, *, graph=None, progress=None) -> SearchResult:
    if not isinstance(client, LLMClient):
        raise TypeError(
            "AutoSND search requires a configured LLMClient with API URL and API key"
        )
    # AUTOSND_SUPPLIED_GRAPH_V2
    supplied = graph is not None
    if graph is None:
        graph = load_graph(config.proxy_dataset, verify=True)
    else:
        import networkx as nx
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a NetworkX graph")
        graph = nx.Graph(graph.to_undirected() if graph.is_directed() else graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() == 0:
        raise ValueError("AutoSND search requires a nonempty graph")
    evaluation_graph = _search_graph_metadata(graph, supplied, config.proxy_dataset)
    notify = progress if progress is not None else (lambda message: None)
    notify(f"AutoSND target: {evaluation_graph['nodes']} nodes, {evaluation_graph['edges']} edges")
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    notify("Evaluating the HDA root on the target graph")
    root = _evaluate(
        ROOT_HDA_CODE,
        candidate_id=_candidate_id(ROOT_HDA_CODE),
        graph=graph,
        config=config,
        stage="stage1",
        parent_id=None,
    )
    if not root.valid:
        raise RuntimeError(f"root HDA candidate failed: {root.error}")
    notify("Stage I: generate and evaluate candidates on the target graph")
    stage1 = _run_stage1(config, client, graph, root)
    notify("Stage II: analyze structures and compile policy from target-graph evidence")
    stage2 = run_stage2(
        stage1,
        client=client,
        proposal_count=config.stage2.proposals,
        top_valid=config.stage2.top_valid,
        feature_rows=config.stage2.feature_rows,
    )
    seed_source = stage2.records
    seeds = select_stage3_seeds(seed_source, config.stage3.seed_quotas)
    notify("Stage III: expand Q/S/B/repair branches and evaluate on the target graph")
    stage3 = _run_stage3(config, client, graph, stage2.policy, seed_source, seeds)
    quality, speed = select_final_candidates(stage3)

    _write_run(
        output_dir,
        config,
        client,
        root,
        stage1,
        stage2,
        stage3,
        seeds,
        quality,
        speed,
        evaluation_graph=evaluation_graph,
    )
    notify("Search finished; selection and candidate code saved")
    return SearchResult(
        output_dir=output_dir,
        root_record=root,
        stage1_records=tuple(stage1),
        stage2=stage2,
        stage3_records=tuple(stage3),
        seeds=seeds,
        final_quality=quality,
        final_speed=speed,
    )


def select_stage3_seeds(
    records: tuple[EvidenceRecord, ...] | list[EvidenceRecord],
    quotas: dict[str, int],
) -> dict[str, tuple[str, ...]]:
    """Apply the Q/S/B/R seed rules using persisted z_i."""
    valid = [
        record
        for record in records
        if record.valid
        and record.anc_gcc is not None
        and record.runtime_seconds is not None
    ]
    if not valid:
        raise RuntimeError("Stage III cannot start without a valid Stage I candidate")
    quality_rank = _goodness(valid, "anc_gcc")
    time_rank = _goodness(valid, "runtime_seconds")
    quality_score = {
        record.candidate_id: 0.90 * quality_rank[record.candidate_id]
        + 0.10 * time_rank[record.candidate_id]
        for record in valid
    }
    balanced_score = {
        record.candidate_id: 0.80 * quality_rank[record.candidate_id]
        + 0.20 * time_rank[record.candidate_id]
        for record in valid
    }
    quality_seeds = _top_ids(valid, quality_score, quotas["quality"])

    speed_pool = [
        record
        for record in valid
        if quality_rank[record.candidate_id] >= 0.55
    ] or valid
    speed_quality_rank = _goodness(speed_pool, "anc_gcc")
    speed_time_rank = _goodness(speed_pool, "runtime_seconds")
    speed_score = {
        record.candidate_id: 0.60 * speed_quality_rank[record.candidate_id]
        + 0.40 * speed_time_rank[record.candidate_id]
        for record in speed_pool
    }
    speed_seeds = _top_ids(speed_pool, speed_score, quotas["speed"])

    balanced_seeds = _diverse_top_ids(
        valid, balanced_score, quotas["balanced"]
    )
    runtime_threshold = _quantile(
        [float(record.runtime_seconds) for record in valid], 0.75
    )
    repair_ids = set(quality_seeds)
    repair_ids.update(
        record.candidate_id
        for record in valid
        if float(record.runtime_seconds) >= runtime_threshold
    )
    repair_pool = [
        record for record in valid if record.candidate_id in repair_ids
    ]
    repair_seeds = _top_ids(repair_pool, quality_score, quotas["repair"])
    return {
        "quality": tuple(quality_seeds),
        "speed": tuple(speed_seeds),
        "balanced": tuple(balanced_seeds),
        "repair": tuple(repair_seeds),
    }


def select_final_candidates(
    records: list[EvidenceRecord],
) -> tuple[EvidenceRecord, EvidenceRecord]:
    valid = [
        record
        for record in records
        if record.valid
        and record.anc_gcc is not None
        and record.runtime_seconds is not None
    ]
    if not valid:
        raise RuntimeError("Stage III produced no valid candidate")
    frontier = _pareto_frontier(valid)
    quality = min(
        frontier,
        key=lambda record: (
            float(record.anc_gcc),
            float(record.runtime_seconds),
            record.candidate_id,
        ),
    )
    quality_rank = _goodness(frontier, "anc_gcc")
    speed_pool = [
        record
        for record in frontier
        if quality_rank[record.candidate_id] >= 0.55
    ] or frontier
    speed = min(
        speed_pool,
        key=lambda record: (
            float(record.runtime_seconds),
            float(record.anc_gcc),
            record.candidate_id,
        ),
    )
    return quality, speed


def _run_stage1(
    config: SearchConfig,
    client: LLMClient,
    graph: Any,
    root: EvidenceRecord,
) -> list[EvidenceRecord]:
    graph_context = _graph_prompt_context(graph)
    tree_records = [root]
    generated: list[EvidenceRecord] = []
    seen = {root.candidate_id}
    children: Counter[str] = Counter()
    for index in range(config.stage1.candidates):
        parent = _select_parent(tree_records, children, weights=(0.80, 0.20))
        selection_score, exploration_bonus = _parent_terms(
            tree_records,
            parent,
            children,
            weights=(0.80, 0.20),
        )
        direction = EXPLORATION_DIRECTIONS[
            (config.seed + index) % len(EXPLORATION_DIRECTIONS)
        ]
        prompt = (
            f"{graph_context}\n"
            f"Stage I broad exploration, generated candidate {index + 1}/"
            f"{config.stage1.candidates}.\n"
            "Improve the parent using any simple, deterministic graph structure. "
            "Do not assume a target signal family. Keep the implementation readable "
            "and return a complete node permutation.\n\n"
            f"Initial exploration direction theta_i: {direction}.\n"
            f"Parent search score f: {selection_score:.8f}\n"
            f"Parent exploration bonus: {exploration_bonus:.8f}\n"
            f"Parent priority U: {selection_score + exploration_bonus:.8f}\n"
            f"Parent ANC@{config.evaluation_rate:.0%} (GCC): {parent.anc_gcc:.8f}\n"
            f"Parent runtime: {parent.runtime_seconds:.8f}s\n"
            f"Parent code:\n```python\n{parent.code}\n```"
        )
        record = _generate_and_evaluate(
            client,
            prompt,
            parent=parent,
            graph=graph,
            config=config,
            stage="stage1",
            seen=seen,
            parent_selection_score=selection_score,
            parent_exploration_bonus=exploration_bonus,
        )
        generated.append(record)
        tree_records.append(record)
        children[parent.candidate_id] += 1
    return generated


def _run_stage3(
    config: SearchConfig,
    client: LLMClient,
    graph: Any,
    policy: CompiledPolicy,
    stage1: tuple[EvidenceRecord, ...],
    seeds: dict[str, tuple[str, ...]],
) -> list[EvidenceRecord]:
    graph_context = _graph_prompt_context(graph)
    by_id = {record.candidate_id: record for record in stage1}
    seen = set(by_id)
    records: list[EvidenceRecord] = []
    branch_nodes: dict[str, list[EvidenceRecord]] = {
        branch: [by_id[candidate_id] for candidate_id in candidate_ids]
        for branch, candidate_ids in seeds.items()
    }
    children: dict[str, Counter[str]] = {
        branch: Counter() for branch in ("quality", "speed", "balanced", "repair")
    }
    weights = {
        "quality": (0.90, 0.10),
        "speed": (0.60, 0.40),
        "balanced": (0.80, 0.20),
        "repair": (0.90, 0.10),
    }
    for branch in ("quality", "speed", "balanced", "repair"):
        budget = config.stage3.branch_budgets[branch]
        for index in range(budget):
            pool = [
                record for record in branch_nodes[branch] if record.valid
            ]
            if not pool:
                pool = [record for record in stage1 if record.valid]
            branch_children = children[branch]
            parent = _select_parent(pool, branch_children, weights=weights[branch])
            selection_score, exploration_bonus = _parent_terms(
                pool,
                parent,
                branch_children,
                weights=weights[branch],
            )
            prompt = (
                f"{graph_context}\n"
                f"Stage III {branch} branch, expansion {index + 1}/{budget}.\n"
                f"{STAGE3_ROLES[branch]}\n\n"
                f"{policy.stage3_prompt_contract}\n\n"
                f"Parent branch score f_b: {selection_score:.8f}\n"
                f"Parent exploration bonus: {exploration_bonus:.8f}\n"
                f"Parent priority U: {selection_score + exploration_bonus:.8f}\n"
                f"Parent ANC@{config.evaluation_rate:.0%} (GCC): {parent.anc_gcc:.8f}\n"
                f"Parent runtime: {parent.runtime_seconds:.8f}s\n"
                f"Parent code:\n```python\n{parent.code}\n```"
            )
            record = _generate_and_evaluate(
                client,
                prompt,
                parent=parent,
                graph=graph,
                config=config,
                stage=f"stage3:{branch}",
                seen=seen,
                parent_selection_score=selection_score,
                parent_exploration_bonus=exploration_bonus,
                policy=policy,
            )
            records.append(record)
            branch_nodes[branch].append(record)
            branch_children[parent.candidate_id] += 1
    return records


def _generate_and_evaluate(
    client: LLMClient,
    prompt: str,
    *,
    parent: EvidenceRecord,
    graph: Any,
    config: SearchConfig,
    stage: str,
    seen: set[str],
    parent_selection_score: float,
    parent_exploration_bonus: float,
    policy: CompiledPolicy | None = None,
) -> EvidenceRecord:
    response: str | None = None
    try:
        response = client.complete(system=STAGE1_SYSTEM, user=prompt)
        code = extract_candidate_code(response)
    except Exception as exc:
        error_id = _unique_record_id("generation-error", seen)
        return EvidenceRecord(
            candidate_id=error_id,
            code="",
            valid=False,
            anc_gcc=None,
            runtime_seconds=None,
            error=f"generation error: {type(exc).__name__}: {exc}",
            stage=stage,
            parent_id=parent.candidate_id,
            parent_selection_score=parent_selection_score,
            parent_exploration_bonus=parent_exploration_bonus,
            prompt=prompt,
            raw_response=response,
        )
    candidate_id = _candidate_id(code)
    if candidate_id in seen:
        duplicate_id = _unique_record_id(f"{candidate_id}-duplicate", seen)
        return EvidenceRecord(
            candidate_id=duplicate_id,
            code=code,
            valid=False,
            anc_gcc=None,
            runtime_seconds=None,
            error=f"duplicate candidate: {candidate_id}",
            stage=stage,
            parent_id=parent.candidate_id,
            parent_selection_score=parent_selection_score,
            parent_exploration_bonus=parent_exploration_bonus,
            prompt=prompt,
            raw_response=response,
        )
    seen.add(candidate_id)
    violation = _policy_violation(code, policy)
    if violation:
        return EvidenceRecord(
            candidate_id=candidate_id,
            code=code,
            valid=False,
            anc_gcc=None,
            runtime_seconds=None,
            error=f"fixed Stage III guard: {violation}",
            stage=stage,
            parent_id=parent.candidate_id,
            parent_selection_score=parent_selection_score,
            parent_exploration_bonus=parent_exploration_bonus,
            prompt=prompt,
            raw_response=response,
        )
    return replace(
        _evaluate(
            code,
            candidate_id=candidate_id,
            graph=graph,
            config=config,
            stage=stage,
            parent_id=parent.candidate_id,
        ),
        parent_selection_score=parent_selection_score,
        parent_exploration_bonus=parent_exploration_bonus,
        prompt=prompt,
        raw_response=response,
    )


def _evaluate(
    code: str,
    *,
    candidate_id: str,
    graph: Any,
    config: SearchConfig,
    stage: str,
    parent_id: str | None,
) -> EvidenceRecord:
    execution = execute_candidate(
        code,
        graph,
        rate=config.evaluation_rate,
        timeout_seconds=config.candidate_timeout_seconds,
    )
    return EvidenceRecord(
        candidate_id=candidate_id,
        code=code,
        valid=execution.valid,
        anc_gcc=execution.anc_gcc,
        runtime_seconds=execution.runtime_seconds,
        error=execution.error,
        runtime_censored=execution.runtime_censored,
        stage=stage,
        parent_id=parent_id,
    )


def _select_parent(
    records: list[EvidenceRecord],
    children: Counter[str],
    *,
    weights: tuple[float, float],
) -> EvidenceRecord:
    valid = [
        record
        for record in records
        if record.valid
        and record.anc_gcc is not None
        and record.runtime_seconds is not None
    ]
    if not valid:
        raise RuntimeError("no valid parent candidate")
    quality = _goodness(valid, "anc_gcc")
    runtime = _goodness(valid, "runtime_seconds")
    alpha, beta = weights
    return max(
        valid,
        key=lambda record: (
            alpha * quality[record.candidate_id]
            + beta * runtime[record.candidate_id]
            + 0.10 / (1 + children[record.candidate_id]),
            -children[record.candidate_id],
            record.candidate_id,
        ),
    )


def _parent_terms(
    records: list[EvidenceRecord],
    parent: EvidenceRecord,
    children: Counter[str],
    *,
    weights: tuple[float, float],
) -> tuple[float, float]:
    valid = [
        record
        for record in records
        if record.valid
        and record.anc_gcc is not None
        and record.runtime_seconds is not None
    ]
    quality = _goodness(valid, "anc_gcc")
    runtime = _goodness(valid, "runtime_seconds")
    alpha, beta = weights
    score = (
        alpha * quality[parent.candidate_id]
        + beta * runtime[parent.candidate_id]
    )
    bonus = 0.10 / (1 + children[parent.candidate_id])
    return score, bonus


def _goodness(records: list[EvidenceRecord], field: str) -> dict[str, float]:
    values = {
        record.candidate_id: float(getattr(record, field)) for record in records
    }
    unique = sorted(set(values.values()))
    if len(unique) == 1:
        return {candidate_id: 1.0 for candidate_id in values}
    rank = {
        value: 1.0 - index / (len(unique) - 1)
        for index, value in enumerate(unique)
    }
    return {
        candidate_id: rank[value] for candidate_id, value in values.items()
    }


def _top_ids(
    records: list[EvidenceRecord],
    scores: dict[str, float],
    limit: int,
) -> list[str]:
    ranked = sorted(
        records,
        key=lambda record: (
            -scores[record.candidate_id],
            float(record.anc_gcc),
            float(record.runtime_seconds),
            record.candidate_id,
        ),
    )
    return [record.candidate_id for record in ranked[:limit]]


def _diverse_top_ids(
    records: list[EvidenceRecord],
    scores: dict[str, float],
    limit: int,
) -> list[str]:
    groups: dict[str, deque[EvidenceRecord]] = defaultdict(deque)
    for record in sorted(
        records,
        key=lambda item: (
            -scores[item.candidate_id],
            float(item.anc_gcc),
            float(item.runtime_seconds),
            item.candidate_id,
        ),
    ):
        if record.structure is None:
            raise RuntimeError(
                f"candidate {record.candidate_id} has no LLM-Struct description"
            )
        structure = record.structure
        groups[structure.selection_signature].append(record)
    signatures = sorted(
        groups,
        key=lambda signature: (
            -scores[groups[signature][0].candidate_id],
            signature,
        ),
    )
    selected: list[str] = []
    while signatures and len(selected) < limit:
        next_round: list[str] = []
        for signature in signatures:
            if len(selected) >= limit:
                break
            if groups[signature]:
                selected.append(groups[signature].popleft().candidate_id)
            if groups[signature]:
                next_round.append(signature)
        signatures = next_round
    return selected


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pareto_frontier(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    return [
        candidate
        for candidate in records
        if not any(
            other.candidate_id != candidate.candidate_id
            and float(other.anc_gcc) <= float(candidate.anc_gcc)
            and float(other.runtime_seconds) <= float(candidate.runtime_seconds)
            and (
                float(other.anc_gcc) < float(candidate.anc_gcc)
                or float(other.runtime_seconds) < float(candidate.runtime_seconds)
            )
            for other in records
        )
    ]


def _candidate_id(code: str) -> str:
    normalized = "\n".join(line.rstrip() for line in code.strip().splitlines()) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _policy_violation(
    code: str,
    policy: CompiledPolicy | None,
) -> str | None:
    if policy is None:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    forbidden_calls = {
        "all_pairs_shortest_path",
        "all_pairs_shortest_path_length",
        "betweenness_centrality",
        "edge_betweenness_centrality",
        "floyd_warshall",
        "floyd_warshall_numpy",
        "floyd_warshall_predecessor_and_distance",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else None
        )
        if name in forbidden_calls:
            return f"forbidden global graph routine {name}()"
    return None


def _unique_record_id(prefix: str, seen: set[str]) -> str:
    index = 1
    candidate = f"{prefix}-{index:04d}"
    while candidate in seen:
        index += 1
        candidate = f"{prefix}-{index:04d}"
    seen.add(candidate)
    return candidate


def _write_run(
    output_dir: Path,
    config: SearchConfig,
    client: LLMClient,
    root: EvidenceRecord,
    stage1: list[EvidenceRecord],
    stage2: Stage2Bundle,
    stage3: list[EvidenceRecord],
    seeds: dict[str, tuple[str, ...]],
    quality: EvidenceRecord,
    speed: EvidenceRecord,
    evaluation_graph=None,
) -> None:
    manifest = {
        "schema_version": 2,
        "task": "network_dismantling",
        "model": client.model,
        "configured_model": config.model,
        "seed": config.seed,
        "llm_request": {
            "model": client.model,
            "temperature": client.temperature,
            "timeout_seconds": client.timeout_seconds,
            "retries": client.retries,
            "provider_seed": client.seed,
        },
        "proxy_dataset": None if evaluation_graph and evaluation_graph["source"] == "supplied_graph" else config.proxy_dataset,
        "evaluation_graph": evaluation_graph,
        "evaluation_rate": config.evaluation_rate,
        "candidate_timeout_seconds": config.candidate_timeout_seconds,
        "budgets": {
            "B1": config.stage1.candidates,
            "B2": config.stage2.proposals,
            "B3": config.stage3.candidates,
        },
        "api_key_recorded": False,
        "api_url_recorded": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "root_record.json", _record_dict(root))
    _write_jsonl(output_dir / "stage1_records.jsonl", stage1)
    _write_json(output_dir / "stage2_bundle.json", stage2.to_dict())
    _write_json(output_dir / "stage3_seeds.json", seeds)
    _write_jsonl(output_dir / "stage3_records.jsonl", stage3)
    frontier = _pareto_frontier(
        [
            record
            for record in stage3
            if record.valid
            and record.anc_gcc is not None
            and record.runtime_seconds is not None
        ]
    )
    _write_json(
        output_dir / "stage3_pareto_frontier.json",
        [_record_dict(record) for record in frontier],
    )
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "AutoSND-Q.py").write_text(quality.code, encoding="utf-8")
    (final_dir / "AutoSND-S.py").write_text(speed.code, encoding="utf-8")
    _write_json(
        final_dir / "selection.json",
        {
            "quality": _record_dict(quality),
            "speed": _record_dict(speed),
            "selection_scope": "Stage III supplied-graph Pareto frontier only" if evaluation_graph and evaluation_graph["source"] == "supplied_graph" else "Stage III proxy Pareto frontier only",
            "selector_config": {
                "quality": f"lexicographic (ANC@{config.evaluation_rate:.0%} (GCC), runtime, candidate_id)",
                "speed_quality_rank_threshold": 0.55,
                "speed": "lexicographic (runtime, ANC@30% (GCC), candidate_id)",
                "rank_ties": "equal metric values receive equal unique-value rank",
                "frontier_candidate_ids": [
                    record.candidate_id for record in frontier
                ],
            },
        },
    )


def _record_dict(record: EvidenceRecord) -> dict[str, Any]:
    return asdict(record)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[EvidenceRecord]) -> None:
    path.write_text(
        "".join(
            json.dumps(_record_dict(record), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _search_graph_metadata(graph, supplied, dataset):
    nodes = list(graph.nodes())
    index = {node: offset for offset, node in enumerate(nodes)}
    edges = sorted(tuple(sorted((index[left], index[right]))) for left, right in graph.edges())
    encoded = json.dumps({"nodes": [repr(node) for node in nodes], "edges": edges},
                         ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return {"source": "supplied_graph" if supplied else "bundled_proxy",
            "dataset": None if supplied else dataset, "nodes": len(nodes),
            "edges": len(edges), "sha256": hashlib.sha256(encoded).hexdigest()}


def _graph_prompt_context(graph):
    import networkx as nx
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    components = [len(component) for component in nx.connected_components(graph)]
    return (f"Target network: n={n}, m={m}, degree_min={min(degrees, default=0)}, "
            f"degree_max={max(degrees, default=0)}, mean_degree={2*m/n if n else 0:.6f}, "
            f"isolates={sum(degree == 0 for degree in degrees)}, "
            f"components={len(components)}, largest_component={max(components, default=0)}. "
            "All candidate feedback below was measured on this target network.")
