"""Stage II: compile execution evidence into a structural policy.

The Stage II data flow is

    E1 -> per-candidate z_i -> contrast set C1 -> statistics T1
       -> fixed-schema proposals -> deterministic policy Pi.

Stage II never generates or re-evaluates a candidate.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Any

from .llm import LLMClient

SIGNALS = (
    "degree",
    "frontier",
    "weak_tie",
    "two_hop",
    "boundary",
    "redundancy",
    "phase",
    "heap",
)
FAMILIES = (
    "component-refresh",
    "two-hop-boundary",
    "redundancy-aware",
    "heap-local-update",
    "degree-local",
    "other",
)
EXPENSIVE_PATTERNS = (
    "component_refresh",
    "global_recompute",
    "unbounded_two_hop",
)
PRIOR_FORBIDDEN = (
    "file or process I/O",
    "network access",
    "dynamic code execution",
    "unseeded randomness",
    "mutation of the input graph",
)
FIXED_PRUNED_FAMILIES = (
    "external-io",
    "randomized-unseeded",
    "global-all-pairs",
)
CAP_RANGES = {
    "neighbor_limit": (1, 512),
    "two_hop_limit": (1, 1024),
    "update_radius": (0, 2),
    "component_refresh_interval": (1, 500),
}
CAP_DEFAULTS = {
    "neighbor_limit": 64,
    "two_hop_limit": 128,
    "update_radius": 1,
    "component_refresh_interval": 20,
}
PROPOSAL_SCHEMA = {
    "allowed_signals": [],
    "preferred_families": [],
    "pruned_families": [],
    "cap_bounds": {},
    "update_bounds": {},
    "forbidden_patterns": [],
    "stage3_prompt_contract": "",
}
STRUCTURE_SCHEMA = {
    "family": "",
    "signals": [],
    "neighbor_scope": 0,
    "update_scope": "",
    "expensive_patterns": [],
    "neighbors_calls": 0,
}


@dataclass(frozen=True)
class StructureDescription:
    family: str
    signals: tuple[str, ...]
    neighbor_scope: int
    update_scope: str
    expensive_patterns: tuple[str, ...]
    neighbors_calls: int

    @property
    def signature(self) -> str:
        signal_part = "+".join(self.signals) if self.signals else "none"
        return f"{self.family}|{signal_part}|n{self.neighbor_scope}|{self.update_scope}"

    @property
    def selection_signature(self) -> str:
        """g(z): the eight structural signal flags."""
        return "+".join(self.signals) if self.signals else "plain"


@dataclass(frozen=True)
class EvidenceRecord:
    candidate_id: str
    code: str
    valid: bool
    anc_gcc: float | None
    runtime_seconds: float | None
    error: str | None
    runtime_censored: bool = False
    stage: str = "stage1"
    parent_id: str | None = None
    structure: StructureDescription | None = None
    parent_selection_score: float | None = None
    parent_exploration_bonus: float | None = None
    prompt: str | None = None
    raw_response: str | None = None


@dataclass(frozen=True)
class ContrastSets:
    positive: tuple[str, ...]
    negative_quality: tuple[str, ...]
    negative_time: tuple[str, ...]
    negative_failure: tuple[str, ...]


@dataclass(frozen=True)
class ProposalResult:
    index: int
    parsed: dict[str, Any] | None
    raw_response: str
    error: str | None
    system_prompt: str = ""
    user_prompt: str = ""


@dataclass(frozen=True)
class StructureAnalysisResult:
    index: int
    candidate_id: str
    structure: StructureDescription
    raw_response: str
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class CompiledPolicy:
    allowed_signals: tuple[str, ...]
    discouraged_signals: tuple[str, ...]
    preferred_families: tuple[str, ...]
    pruned_families: tuple[str, ...]
    cap_bounds: dict[str, int]
    update_bounds: dict[str, Any]
    forbidden_patterns: tuple[str, ...]
    stage3_prompt_contract: str
    provenance: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Stage2Bundle:
    records: tuple[EvidenceRecord, ...]
    structural_analyses: tuple[StructureAnalysisResult, ...]
    contrast_sets: ContrastSets
    history_statistics: dict[str, Any]
    proposals: tuple[ProposalResult, ...]
    policy: CompiledPolicy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_structure(
    record: EvidenceRecord,
    *,
    client: LLMClient,
    index: int,
    total: int,
) -> tuple[EvidenceRecord, StructureAnalysisResult]:
    """Apply LLM-Struct to one archived candidate."""
    if not isinstance(client, LLMClient):
        raise TypeError(
            "LLM-Struct requires a configured LLMClient with API URL and API key"
        )
    system_prompt = (
        "You are AutoSND LLM-Struct. Analyze exactly one archived candidate "
        "program and return only its structural description as JSON. "
        "Do not generate or modify candidate code."
    )
    candidate_payload = {
        "candidate_id": record.candidate_id,
        "valid": record.valid,
        "anc_gcc": record.anc_gcc,
        "runtime_seconds": record.runtime_seconds,
        "runtime_censored": record.runtime_censored,
        "error": record.error,
        "code": record.code,
    }
    user_prompt = (
        f"Candidate {index}/{total}. Return exactly one JSON object matching "
        "the schema. Use only the enumerated values shown below; do not add "
        "fields or Markdown.\n\n"
        f"SCHEMA:\n{json.dumps(STRUCTURE_SCHEMA, ensure_ascii=False)}\n"
        f"ALLOWED_FAMILIES:\n{json.dumps(FAMILIES)}\n"
        f"ALLOWED_SIGNALS:\n{json.dumps(SIGNALS)}\n"
        f"ALLOWED_UPDATE_SCOPES:\n"
        f"{json.dumps(('static', 'neighbors', 'two_hop', 'global'))}\n"
        f"ALLOWED_EXPENSIVE_PATTERNS:\n{json.dumps(EXPENSIVE_PATTERNS)}\n\n"
        "If the code is empty or cannot be analyzed, return the neutral structure "
        '{"family":"other","signals":[],"neighbor_scope":0,'
        '"update_scope":"static","expensive_patterns":[],"neighbors_calls":0}.\n\n'
        f"CANDIDATE:\n{json.dumps(candidate_payload, ensure_ascii=False)}"
    )
    try:
        raw = client.complete(system=system_prompt, user=user_prompt)
        structure = parse_structure_description(raw)
    except Exception as exc:
        raise RuntimeError(
            f"LLM-Struct failed for candidate {record.candidate_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    analysis = StructureAnalysisResult(
        index=index,
        candidate_id=record.candidate_id,
        structure=structure,
        raw_response=raw,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return replace(record, structure=structure), analysis


def parse_structure_description(raw: str) -> StructureDescription:
    parsed = _parse_json_object(raw, label="LLM-Struct response")
    if set(parsed) != set(STRUCTURE_SCHEMA):
        raise ValueError("LLM-Struct keys do not match the fixed structure schema")
    family = parsed["family"]
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError("family must be one of the allowed structural families")
    signals = parsed["signals"]
    if not isinstance(signals, list) or not all(
        isinstance(item, str) and item in SIGNALS for item in signals
    ):
        raise ValueError("signals must contain only allowed structural signals")
    if len(signals) != len(set(signals)):
        raise ValueError("signals must not contain duplicates")
    neighbor_scope = parsed["neighbor_scope"]
    if (
        isinstance(neighbor_scope, bool)
        or not isinstance(neighbor_scope, int)
        or not 0 <= neighbor_scope <= 2
    ):
        raise ValueError("neighbor_scope must be an integer in [0, 2]")
    update_scope = parsed["update_scope"]
    if update_scope not in {"static", "neighbors", "two_hop", "global"}:
        raise ValueError("update_scope is not allowed")
    expensive_patterns = parsed["expensive_patterns"]
    if not isinstance(expensive_patterns, list) or not all(
        isinstance(item, str) and item in EXPENSIVE_PATTERNS
        for item in expensive_patterns
    ):
        raise ValueError(
            "expensive_patterns must contain only allowed expensive patterns"
        )
    if len(expensive_patterns) != len(set(expensive_patterns)):
        raise ValueError("expensive_patterns must not contain duplicates")
    neighbors_calls = parsed["neighbors_calls"]
    if (
        isinstance(neighbors_calls, bool)
        or not isinstance(neighbors_calls, int)
        or neighbors_calls < 0
    ):
        raise ValueError("neighbors_calls must be a non-negative integer")
    return StructureDescription(
        family=family,
        signals=tuple(item for item in SIGNALS if item in set(signals)),
        neighbor_scope=neighbor_scope,
        update_scope=update_scope,
        expensive_patterns=tuple(
            item for item in EXPENSIVE_PATTERNS if item in set(expensive_patterns)
        ),
        neighbors_calls=neighbors_calls,
    )


def build_contrast_sets(
    records: list[EvidenceRecord],
    *,
    positive_limit: int = 20,
) -> ContrastSets:
    valid = [
        record
        for record in records
        if record.valid
        and record.anc_gcc is not None
        and record.runtime_seconds is not None
        and math.isfinite(record.anc_gcc)
        and math.isfinite(record.runtime_seconds)
    ]
    layers = _pareto_layers(valid)
    positive: list[EvidenceRecord] = []
    for layer in layers:
        remaining = positive_limit - len(positive)
        if remaining <= 0:
            break
        positive.extend(
            sorted(
                layer,
                key=lambda item: (
                    item.anc_gcc,
                    item.runtime_seconds,
                    item.candidate_id,
                ),
            )[:remaining]
        )
    positive_ids = {record.candidate_id for record in positive}
    remainder = [record for record in valid if record.candidate_id not in positive_ids]
    quality_rank = _goodness_ranks(valid, "anc_gcc")
    time_rank = _goodness_ranks(valid, "runtime_seconds")
    negative_quality: list[str] = []
    negative_time: list[str] = []
    for record in remainder:
        if quality_rank[record.candidate_id] <= time_rank[record.candidate_id]:
            negative_quality.append(record.candidate_id)
        else:
            negative_time.append(record.candidate_id)
    valid_ids = {record.candidate_id for record in valid}
    failures = [
        record.candidate_id
        for record in records
        if record.candidate_id not in valid_ids
    ]
    return ContrastSets(
        positive=tuple(record.candidate_id for record in positive),
        negative_quality=tuple(negative_quality),
        negative_time=tuple(negative_time),
        negative_failure=tuple(failures),
    )


def history_statistics(
    records: list[EvidenceRecord],
    contrast: ContrastSets,
) -> dict[str, Any]:
    categories = {
        "positive": set(contrast.positive),
        "negative_quality": set(contrast.negative_quality),
        "negative_time": set(contrast.negative_time),
        "negative_failure": set(contrast.negative_failure),
    }
    categories["negative"] = set().union(
        categories["negative_quality"],
        categories["negative_time"],
        categories["negative_failure"],
    )
    by_id = {record.candidate_id: record for record in records}
    feature_members: dict[str, set[str]] = defaultdict(set)
    for record in records:
        structure = _require_structure(record)
        tokens = _feature_tokens(structure)
        for pair in combinations(structure.signals, 2):
            tokens.add("pair:" + "+".join(pair))
        for token in tokens:
            feature_members[token].add(record.candidate_id)

    feature_stats: dict[str, Any] = {}
    for token, members in sorted(feature_members.items()):
        if token.startswith("pair:") and len(members) < 2:
            continue
        groups: dict[str, Any] = {}
        for category, category_ids in categories.items():
            selected = sorted(members & category_ids)
            denominator = len(category_ids)
            valid_selected = [
                by_id[item]
                for item in selected
                if by_id[item].valid
                and by_id[item].anc_gcc is not None
                and by_id[item].runtime_seconds is not None
            ]
            groups[category] = {
                "count": len(selected),
                "rate": len(selected) / denominator if denominator else 0.0,
                "mean_anc_gcc": _mean(
                    [float(item.anc_gcc) for item in valid_selected]
                ),
                "mean_runtime_seconds": _mean(
                    [float(item.runtime_seconds) for item in valid_selected]
                ),
            }
        feature_stats[token] = groups

    return {
        "group_sizes": {key: len(value) for key, value in categories.items()},
        "feature_statistics": feature_stats,
        "scope_statistics": _scope_statistics(records, categories),
        "failure_types": dict(
            Counter(
                _failure_type(record.error)
                for record in records
                if record.candidate_id in categories["negative_failure"]
            )
        ),
    }


def run_stage2(
    records: list[EvidenceRecord],
    *,
    client: LLMClient,
    proposal_count: int,
    top_valid: int,
    feature_rows: int,
) -> Stage2Bundle:
    if not isinstance(client, LLMClient):
        raise TypeError(
            "Stage II requires a configured LLMClient with API URL and API key"
        )
    if not records:
        raise RuntimeError("Stage II requires at least one Stage I record")
    tagged: list[EvidenceRecord] = []
    structural_analyses: list[StructureAnalysisResult] = []
    for index, record in enumerate(records, start=1):
        tagged_record, analysis = analyze_structure(
            record,
            client=client,
            index=index,
            total=len(records),
        )
        tagged.append(tagged_record)
        structural_analyses.append(analysis)
    contrast = build_contrast_sets(tagged, positive_limit=top_valid)
    statistics_payload = history_statistics(tagged, contrast)
    prompt = build_policy_prompt(
        tagged,
        contrast,
        statistics_payload,
        top_valid=top_valid,
        feature_rows=feature_rows,
    )
    proposals: list[ProposalResult] = []
    for index in range(proposal_count):
        raw = ""
        system_prompt = (
            "You are the AutoSND Stage II structural policy analyst. "
            "Use only the supplied Stage I evidence. Do not generate candidate code."
        )
        user_prompt = f"Independent proposal {index + 1}/{proposal_count}.\n\n{prompt}"
        try:
            raw = client.complete(
                system=system_prompt,
                user=user_prompt,
            )
            parsed = parse_policy_proposal(raw)
            proposals.append(
                ProposalResult(
                    index,
                    parsed,
                    raw,
                    None,
                    system_prompt,
                    user_prompt,
                )
            )
        except Exception as exc:
            proposals.append(
                ProposalResult(
                    index,
                    None,
                    raw,
                    f"{type(exc).__name__}: {exc}",
                    system_prompt,
                    user_prompt,
                )
            )
    policy = compile_policy(tagged, contrast, statistics_payload, proposals)
    return Stage2Bundle(
        records=tuple(tagged),
        structural_analyses=tuple(structural_analyses),
        contrast_sets=contrast,
        history_statistics=statistics_payload,
        proposals=tuple(proposals),
        policy=policy,
    )


def build_policy_prompt(
    records: list[EvidenceRecord],
    contrast: ContrastSets,
    statistics_payload: dict[str, Any],
    *,
    top_valid: int,
    feature_rows: int,
) -> str:
    by_id = {record.candidate_id: record for record in records}
    positive_rows = [
        _record_summary(by_id[candidate_id], contrast_role="positive")
        for candidate_id in contrast.positive[:top_valid]
    ]
    negative_roles = {
        candidate_id: "negative_quality"
        for candidate_id in contrast.negative_quality
    }
    negative_roles.update(
        {
            candidate_id: "negative_time"
            for candidate_id in contrast.negative_time
        }
    )
    negative_roles.update(
        {
            candidate_id: "negative_failure"
            for candidate_id in contrast.negative_failure
        }
    )
    contrast_ids = _round_robin_ids(
        (
            contrast.negative_quality,
            contrast.negative_time,
            contrast.negative_failure,
        ),
        feature_rows,
    )
    negative_rows = [
        _record_summary(
            by_id[candidate_id],
            contrast_role=negative_roles[candidate_id],
        )
        for candidate_id in contrast_ids
    ]
    feature_rows_payload = [
        _record_summary(record, contrast_role="archive")
        for record in records[:feature_rows]
    ]
    evidence = {
        "positive_pareto_rows": positive_rows,
        "negative_rows": negative_rows,
        "feature_rows_in_log_order": feature_rows_payload,
        "history_statistics": statistics_payload,
    }
    return (
        "Infer structural guidance for later degree_order(G) generation. "
        "Prefer evidence-supported local signals and bounded updates; distinguish "
        "quality, runtime, and failure evidence. Return one JSON object matching "
        "the exact schema below. No Markdown and no candidate code.\n\n"
        f"SCHEMA:\n{json.dumps(PROPOSAL_SCHEMA, ensure_ascii=False)}\n\n"
        f"EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}"
    )


def _parse_json_object(raw: str, *, label: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label} must be exactly one JSON object: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def parse_policy_proposal(raw: str) -> dict[str, Any]:
    parsed = _parse_json_object(raw, label="proposal")
    if set(parsed) != set(PROPOSAL_SCHEMA):
        raise ValueError("proposal keys do not match the fixed Stage II schema")
    for key in (
        "allowed_signals",
        "preferred_families",
        "pruned_families",
        "forbidden_patterns",
    ):
        if not isinstance(parsed[key], list) or not all(
            isinstance(item, str) for item in parsed[key]
        ):
            raise ValueError(f"{key} must be a list of strings")
    for key in ("cap_bounds", "update_bounds"):
        if not isinstance(parsed[key], dict):
            raise ValueError(f"{key} must be an object")
    if set(parsed["cap_bounds"]) - set(CAP_RANGES):
        raise ValueError("cap_bounds contains an unknown field")
    for key, value in parsed["cap_bounds"].items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"cap_bounds.{key} must be a finite number")
    if set(parsed["update_bounds"]) - {"max_hops", "allowed_scopes"}:
        raise ValueError("update_bounds contains an unknown field")
    if "max_hops" in parsed["update_bounds"]:
        value = parsed["update_bounds"]["max_hops"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("update_bounds.max_hops must be a finite number")
    if "allowed_scopes" in parsed["update_bounds"]:
        scopes = parsed["update_bounds"]["allowed_scopes"]
        if not isinstance(scopes, list) or not all(
            isinstance(scope, str) for scope in scopes
        ):
            raise ValueError("update_bounds.allowed_scopes must be a list of strings")
    if not isinstance(parsed["stage3_prompt_contract"], str):
        raise ValueError("stage3_prompt_contract must be a string")
    return parsed


def compile_policy(
    records: list[EvidenceRecord],
    contrast: ContrastSets,
    statistics_payload: dict[str, Any],
    proposals: list[ProposalResult],
) -> CompiledPolicy:
    """Deterministically merge Pi_E with the fixed safety envelope Pi_prior."""
    valid_proposals = [
        result.parsed for result in proposals if result.parsed is not None
    ]
    if not valid_proposals:
        raise RuntimeError("Stage II produced no valid LLM policy proposal")
    threshold = len(valid_proposals) // 2 + 1
    feature_stats = statistics_payload["feature_statistics"]
    allowed_signals: set[str] = set()
    preferred_families: set[str] = set()
    evidence_avoid: set[str] = set()
    provenance: dict[str, list[str]] = defaultdict(list)

    for signal in SIGNALS:
        stat = feature_stats.get(f"signal:{signal}")
        if not stat:
            continue
        delta = stat["positive"]["rate"] - stat["negative"]["rate"]
        if stat["positive"]["count"] >= 2 and delta >= 0.10:
            allowed_signals.add(signal)
            provenance[f"allowed_signals.{signal}"].append("stage1_statistics")
        if stat["negative"]["count"] >= 2 and delta <= -0.15:
            evidence_avoid.add(signal)
            provenance[f"avoid_signals.{signal}"].append("stage1_statistics")
    for family in FAMILIES:
        stat = feature_stats.get(f"family:{family}")
        if not stat:
            continue
        delta = stat["positive"]["rate"] - stat["negative"]["rate"]
        if stat["positive"]["count"] >= 2 and delta >= 0.10:
            preferred_families.add(family)
            provenance[f"preferred_families.{family}"].append("stage1_statistics")

    for signal, votes in _proposal_votes(valid_proposals, "allowed_signals").items():
        if signal in SIGNALS and votes >= threshold and signal not in evidence_avoid:
            allowed_signals.add(signal)
            provenance[f"allowed_signals.{signal}"].append("stage2_proposal_majority")
    for family, votes in _proposal_votes(
        valid_proposals, "preferred_families"
    ).items():
        if family in FAMILIES and votes >= threshold:
            preferred_families.add(family)
            provenance[f"preferred_families.{family}"].append(
                "stage2_proposal_majority"
            )

    if not allowed_signals:
        allowed_signals.add("degree")
        provenance["allowed_signals.degree"].append("fixed_safety_envelope")
    if not preferred_families:
        preferred_families.add("degree-local")
        provenance["preferred_families.degree-local"].append(
            "fixed_safety_envelope"
        )

    pruned = set(FIXED_PRUNED_FAMILIES)
    for family in FIXED_PRUNED_FAMILIES:
        provenance[f"pruned_families.{family}"].append("fixed_safety_envelope")
    caps = dict(CAP_DEFAULTS)
    for key in caps:
        numeric = [
            proposal["cap_bounds"].get(key)
            for proposal in valid_proposals
            if isinstance(proposal["cap_bounds"].get(key), (int, float))
            and not isinstance(proposal["cap_bounds"].get(key), bool)
            and math.isfinite(float(proposal["cap_bounds"].get(key)))
        ]
        if numeric:
            lower, upper = CAP_RANGES[key]
            value = int(round(statistics.median(float(item) for item in numeric)))
            caps[key] = min(upper, max(lower, value))
            provenance[f"cap_bounds.{key}"].extend(
                ("stage2_proposals", "fixed_safety_clamp")
            )
        else:
            provenance[f"cap_bounds.{key}"].append("base_policy")

    positive_ids = set(contrast.positive)
    positive_records = [
        record for record in records if record.candidate_id in positive_ids
    ]
    evidence_max_hops = min(
        2,
        max(
            (
                _require_structure(record).neighbor_scope
                for record in positive_records
            ),
            default=1,
        ),
    )
    proposed_hops = [
        proposal["update_bounds"].get("max_hops")
        for proposal in valid_proposals
        if isinstance(proposal["update_bounds"].get("max_hops"), (int, float))
        and not isinstance(proposal["update_bounds"].get("max_hops"), bool)
        and math.isfinite(float(proposal["update_bounds"].get("max_hops")))
    ]
    if proposed_hops:
        max_hops = min(
            evidence_max_hops,
            max(0, min(2, int(round(statistics.median(proposed_hops))))),
        )
        provenance["update_bounds.max_hops"].extend(
            ("stage1_positive_evidence", "stage2_proposals", "fixed_safety_clamp")
        )
    else:
        max_hops = evidence_max_hops
        provenance["update_bounds.max_hops"].extend(
            ("stage1_positive_evidence", "fixed_safety_clamp")
        )
    evidence_scopes = {
        _require_structure(record).update_scope for record in positive_records
    } or {"neighbors"}
    scope_votes: Counter[str] = Counter()
    for proposal in valid_proposals:
        raw_scopes = proposal["update_bounds"].get("allowed_scopes", [])
        if isinstance(raw_scopes, list):
            scope_votes.update(
                {
                    str(scope)
                    for scope in raw_scopes
                    if str(scope) in {"static", "neighbors", "two_hop", "global"}
                }
            )
    voted_scopes = {
        scope for scope, votes in scope_votes.items() if votes >= threshold
    }
    allowed_scopes = sorted((voted_scopes & evidence_scopes) or evidence_scopes)
    update_bounds = {
        "max_hops": max_hops,
        "allowed_scopes": allowed_scopes,
    }
    provenance["update_bounds.allowed_scopes"].append("stage1_positive_evidence")
    if voted_scopes & evidence_scopes:
        provenance["update_bounds.allowed_scopes"].append(
            "stage2_proposal_majority"
        )

    forbidden = set(PRIOR_FORBIDDEN)
    for item in PRIOR_FORBIDDEN:
        provenance[f"forbidden_patterns.{item}"].append("fixed_safety_envelope")
    known_forbidden = {
        "global betweenness": "per-step global betweenness",
        "all-pairs shortest paths": "all-pairs shortest paths",
        "per-step component refresh": "per-step full component refresh",
        "unbounded two-hop": "unbounded two-hop expansion",
    }
    forbidden_votes = _proposal_votes(valid_proposals, "forbidden_patterns")
    for raw, votes in forbidden_votes.items():
        normalized = raw.strip().casefold()
        if normalized in known_forbidden and votes >= threshold:
            canonical = known_forbidden[normalized]
            forbidden.add(canonical)
            provenance[f"forbidden_patterns.{canonical}"].append(
                "stage2_proposal_majority"
            )

    allowed = tuple(item for item in SIGNALS if item in allowed_signals)
    preferred = tuple(item for item in FAMILIES if item in preferred_families)
    pruned_tuple = tuple(sorted(pruned))
    forbidden_tuple = tuple(sorted(forbidden))
    contract = _stage3_contract(
        allowed,
        preferred,
        pruned_tuple,
        caps,
        update_bounds,
        forbidden_tuple,
        tuple(sorted(evidence_avoid)),
    )
    provenance["stage3_prompt_contract"].append("deterministic_compiler")
    return CompiledPolicy(
        allowed_signals=allowed,
        discouraged_signals=tuple(sorted(evidence_avoid)),
        preferred_families=preferred,
        pruned_families=pruned_tuple,
        cap_bounds=caps,
        update_bounds=update_bounds,
        forbidden_patterns=forbidden_tuple,
        stage3_prompt_contract=contract,
        provenance={key: tuple(value) for key, value in sorted(provenance.items())},
    )


def _stage3_contract(
    allowed: tuple[str, ...],
    preferred: tuple[str, ...],
    pruned: tuple[str, ...],
    caps: dict[str, int],
    update_bounds: dict[str, Any],
    forbidden: tuple[str, ...],
    avoid_signals: tuple[str, ...],
) -> str:
    return "\n".join(
        (
            "Compiled AutoSND Stage II policy (evidence-conditioned):",
            f"- Prefer families: {', '.join(preferred)}.",
            f"- Allowed structural signals: {', '.join(allowed)}.",
            f"- Evidence-associated signals to avoid: {', '.join(avoid_signals) or 'none'}.",
            f"- Pruned unsafe families: {', '.join(pruned)}.",
            f"- Caps: {json.dumps(caps, sort_keys=True)}.",
            f"- Update bounds: {json.dumps(update_bounds, sort_keys=True)}.",
            f"- Forbidden patterns: {', '.join(forbidden)}.",
            "- Return one deterministic, complete degree_order(G) implementation.",
        )
    )


def _feature_tokens(structure: StructureDescription) -> set[str]:
    tokens = {f"family:{structure.family}"}
    tokens.update(f"signal:{signal}" for signal in structure.signals)
    tokens.update(f"expensive:{item}" for item in structure.expensive_patterns)
    tokens.add(f"neighbor_scope:{structure.neighbor_scope}")
    tokens.add(f"update_scope:{structure.update_scope}")
    return tokens


def _require_structure(record: EvidenceRecord) -> StructureDescription:
    if record.structure is None:
        raise RuntimeError(
            f"candidate {record.candidate_id} has no LLM-Struct description"
        )
    return record.structure


def _pareto_layers(records: list[EvidenceRecord]) -> list[list[EvidenceRecord]]:
    remaining = list(records)
    layers: list[list[EvidenceRecord]] = []
    while remaining:
        layer = [
            candidate
            for candidate in remaining
            if not any(
                _dominates(other, candidate)
                for other in remaining
                if other.candidate_id != candidate.candidate_id
            )
        ]
        if not layer:
            break
        layers.append(layer)
        layer_ids = {candidate.candidate_id for candidate in layer}
        remaining = [
            candidate
            for candidate in remaining
            if candidate.candidate_id not in layer_ids
        ]
    return layers


def _dominates(left: EvidenceRecord, right: EvidenceRecord) -> bool:
    return (
        left.anc_gcc <= right.anc_gcc
        and left.runtime_seconds <= right.runtime_seconds
        and (
            left.anc_gcc < right.anc_gcc
            or left.runtime_seconds < right.runtime_seconds
        )
    )


def _goodness_ranks(
    records: list[EvidenceRecord], field: str
) -> dict[str, float]:
    values = {record.candidate_id: float(getattr(record, field)) for record in records}
    sorted_unique = sorted(set(values.values()))
    if len(sorted_unique) == 1:
        return {candidate_id: 1.0 for candidate_id in values}
    lookup = {
        value: 1.0 - index / (len(sorted_unique) - 1)
        for index, value in enumerate(sorted_unique)
    }
    return {candidate_id: lookup[value] for candidate_id, value in values.items()}


def _record_summary(
    record: EvidenceRecord,
    *,
    contrast_role: str,
) -> dict[str, Any]:
    structure = _require_structure(record)
    return {
        "candidate_id": record.candidate_id,
        "contrast_role": contrast_role,
        "valid": record.valid,
        "anc_gcc": record.anc_gcc,
        "runtime_seconds": record.runtime_seconds,
        "runtime_censored": record.runtime_censored,
        "error": record.error,
        "z_i": asdict(structure),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _scope_statistics(
    records: list[EvidenceRecord],
    categories: dict[str, set[str]],
) -> dict[str, Any]:
    by_id = {record.candidate_id: record for record in records}
    output: dict[str, Any] = {}
    for category, candidate_ids in categories.items():
        structures = [
            _require_structure(by_id[candidate_id])
            for candidate_id in candidate_ids
            if candidate_id in by_id
        ]
        neighbor_scopes = [item.neighbor_scope for item in structures]
        neighbor_calls = [item.neighbors_calls for item in structures]
        output[category] = {
            "neighbor_scope": _range_summary(neighbor_scopes),
            "neighbors_calls": _range_summary(neighbor_calls),
            "update_scope_counts": dict(
                sorted(Counter(item.update_scope for item in structures).items())
            ),
        }
    return output


def _range_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _round_robin_ids(
    groups: tuple[tuple[str, ...], ...],
    limit: int,
) -> list[str]:
    queues = [list(group) for group in groups]
    selected: list[str] = []
    while any(queues) and len(selected) < limit:
        for queue_items in queues:
            if queue_items and len(selected) < limit:
                selected.append(queue_items.pop(0))
    return selected


def _failure_type(error: str | None) -> str:
    lowered = (error or "unknown").casefold()
    for label in ("timeout", "syntax", "memory", "permutation", "duplicate"):
        if label in lowered:
            return label
    return "runtime"


def _proposal_votes(
    proposals: list[dict[str, Any]], field: str
) -> Counter[str]:
    votes: Counter[str] = Counter()
    for proposal in proposals:
        seen = {
            str(item).strip()
            for item in proposal[field]
            if isinstance(item, str) and str(item).strip()
        }
        votes.update(seen)
    return votes
