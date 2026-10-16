"""Typed configuration loading for AutoSND."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


BRANCHES = ("quality", "speed", "balanced", "repair")


@dataclass(frozen=True)
class Stage1Config:
    candidates: int


@dataclass(frozen=True)
class Stage2Config:
    proposals: int
    top_valid: int
    feature_rows: int


@dataclass(frozen=True)
class Stage3Config:
    candidates: int
    seed_quotas: dict[str, int]
    branch_budgets: dict[str, int]


@dataclass(frozen=True)
class SearchConfig:
    seed: int
    model: str
    proxy_dataset: str
    evaluation_rate: float
    candidate_timeout_seconds: float
    output_dir: Path
    stage1: Stage1Config
    stage2: Stage2Config
    stage3: Stage3Config


def load_config(path: str | Path) -> SearchConfig:
    config_path = Path(path)
    raw = yaml.safe_load(_read_config_text(config_path))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    stage1 = _mapping(raw, "stage1")
    stage2 = _mapping(raw, "stage2")
    stage3 = _mapping(raw, "stage3")
    quotas = _integer_map(_mapping(stage3, "seed_quotas"), "stage3.seed_quotas")
    budgets = _integer_map(
        _mapping(stage3, "branch_budgets"), "stage3.branch_budgets"
    )
    _require_branches(quotas, "stage3.seed_quotas")
    _require_branches(budgets, "stage3.branch_budgets")

    model = os.environ.get("AUTOSND_MODEL", str(raw.get("model", "")).strip())
    if not model:
        raise ValueError("set model in YAML or AUTOSND_MODEL")
    config = SearchConfig(
        seed=int(raw.get("seed", 42)),
        model=model,
        proxy_dataset=str(raw.get("proxy_dataset", "Powerlaw_500")),
        evaluation_rate=float(raw.get("evaluation_rate", 0.30)),
        candidate_timeout_seconds=float(raw.get("candidate_timeout_seconds", 90)),
        output_dir=Path(str(raw.get("output_dir", "runs/default"))),
        stage1=Stage1Config(candidates=int(stage1.get("candidates", 0))),
        stage2=Stage2Config(
            proposals=int(stage2.get("proposals", 0)),
            top_valid=int(stage2.get("top_valid", 20)),
            feature_rows=int(stage2.get("feature_rows", 80)),
        ),
        stage3=Stage3Config(
            candidates=int(stage3.get("candidates", 0)),
            seed_quotas=quotas,
            branch_budgets=budgets,
        ),
    )
    _validate(config)
    return config


def api_environment() -> tuple[str, str]:
    url = os.environ.get("AUTOSND_API_URL", "").strip()
    key = os.environ.get("AUTOSND_API_KEY", "").strip()
    missing = [
        name
        for name, value in (
            ("AUTOSND_API_URL", url),
            ("AUTOSND_API_KEY", key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("missing required environment variable(s): " + ", ".join(missing))
    return url, key


def provider_seed_environment() -> int | None:
    """Return an optional provider-side seed, independent of the search seed."""
    raw = os.environ.get("AUTOSND_PROVIDER_SEED", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError("AUTOSND_PROVIDER_SEED must be an integer") from exc


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _read_config_text(config_path: Path) -> str:
    if config_path.is_file():
        return config_path.read_text(encoding="utf-8")
    if config_path.as_posix() in {"configs/smoke.yaml", "configs/paper.yaml"}:
        return (
            files(__package__ + ".defaults")
            .joinpath(config_path.name)
            .read_text(encoding="utf-8")
        )
    raise FileNotFoundError(f"configuration file not found: {config_path}")


def _integer_map(value: dict[str, Any], label: str) -> dict[str, int]:
    try:
        result = {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be integers") from exc
    if any(item < 0 for item in result.values()):
        raise ValueError(f"{label} values must be non-negative")
    return result


def _require_branches(value: dict[str, int], label: str) -> None:
    if set(value) != set(BRANCHES):
        raise ValueError(f"{label} must define exactly: {', '.join(BRANCHES)}")


def _validate(config: SearchConfig) -> None:
    if not 0 < config.evaluation_rate <= 1:
        raise ValueError("evaluation_rate must be in (0, 1]")
    if config.candidate_timeout_seconds <= 0:
        raise ValueError("candidate_timeout_seconds must be positive")
    if config.stage1.candidates <= 0 or config.stage3.candidates <= 0:
        raise ValueError("Stage I and Stage III candidate budgets must be positive")
    if config.stage2.proposals <= 0:
        raise ValueError("Stage II proposal budget must be positive")
    if sum(config.stage3.branch_budgets.values()) != config.stage3.candidates:
        raise ValueError("Stage III branch budgets must sum to stage3.candidates")
    if sum(config.stage3.seed_quotas.values()) <= 0:
        raise ValueError("Stage III must select at least one seed")
