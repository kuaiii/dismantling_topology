"""Dataset discovery, loading, and integrity checks."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Iterator

import networkx as nx


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    file: str
    nodes: int
    edges: int
    sha256_normalized_edges: str
    role: str


def _data_root() -> Any:
    return files(__package__ + ".data")


def load_manifest() -> dict[str, Any]:
    return json.loads(_data_root().joinpath("manifest.json").read_text(encoding="utf-8"))


def dataset_info(name: str) -> DatasetInfo:
    manifest = load_manifest()
    canonical = _canonical_name(name, manifest["datasets"])
    return DatasetInfo(name=canonical, **manifest["datasets"][canonical])


def main12_names() -> tuple[str, ...]:
    return tuple(load_manifest()["main12"])


def available_datasets(*, include_proxy: bool = True) -> tuple[str, ...]:
    manifest = load_manifest()
    names = list(manifest["main12"])
    if include_proxy:
        names.extend(
            name
            for name, meta in manifest["datasets"].items()
            if meta["role"] == "synthetic_search_proxy"
        )
    return tuple(names)


@contextmanager
def dataset_path(name: str) -> Iterator[Path]:
    """Yield a filesystem path, including for resources installed from a wheel."""
    info = dataset_info(name)
    resource = _dataset_resource(info)
    with as_file(resource) as path:
        yield Path(path)


def load_graph(name: str, *, verify: bool = False) -> nx.Graph:
    info = dataset_info(name)
    resource = _dataset_resource(info)
    graph = nx.Graph()
    canonical_edges: list[tuple[int, int]] = []

    with resource.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith(("#", "%")):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"{info.file}:{line_number}: expected exactly two node ids")
            try:
                left, right = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"{info.file}:{line_number}: node ids must be integers"
                ) from exc
            if left == right:
                raise ValueError(f"{info.file}:{line_number}: self-loop is not allowed")
            edge = (left, right) if left < right else (right, left)
            canonical_edges.append(edge)
            graph.add_edge(*edge)

    if len(canonical_edges) != len(set(canonical_edges)):
        raise ValueError(f"{info.name}: duplicate undirected edges are not allowed")
    if graph.number_of_nodes() != info.nodes or graph.number_of_edges() != info.edges:
        raise ValueError(
            f"{info.name}: expected ({info.nodes}, {info.edges}), got "
            f"({graph.number_of_nodes()}, {graph.number_of_edges()})"
        )
    if verify:
        digest = normalized_edge_hash(canonical_edges)
        if digest != info.sha256_normalized_edges:
            raise ValueError(
                f"{info.name}: normalized edge hash mismatch: {digest}"
            )
    return graph


def normalized_edge_hash(edges: list[tuple[int, int]]) -> str:
    unique = sorted(set(edges))
    payload = "".join(f"{left} {right}\n" for left, right in unique).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_name(name: str, datasets: dict[str, Any]) -> str:
    aliases = {key.casefold(): key for key in datasets}
    try:
        return aliases[name.casefold()]
    except KeyError as exc:
        choices = ", ".join(datasets)
        raise KeyError(f"unknown dataset {name!r}; choose one of: {choices}") from exc


def _dataset_resource(info: DatasetInfo) -> Any:
    directory = "main12" if info.role == "main12" else "proxy"
    return _data_root().joinpath(directory, info.file)
