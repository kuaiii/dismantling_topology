"""Adapters for the inspected local NIRM and AutoSND repositories.

Only imports heavy dependencies when the corresponding method is called.
Both algorithms keep their native full order, even when a threshold is passed.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
NIRM_ROOT = PACKAGE_ROOT / "NIRM" / "upstream"
AUTOSND_ROOT = PACKAGE_ROOT / "AutoSND" / "upstream"


def _convert_nirm_state(state, expected, torch):
    """Map PyG 1.7 GAT names to PyG 2.7 without ignoring missing weights."""
    converted = dict(state)
    for layer in ("conv1", "conv2", "conv3"):
        prefix = layer + "."
        for old, new in (("att_l", "att_src"), ("att_r", "att_dst")):
            old_key, new_key = prefix + old, prefix + new
            if old_key in converted and new_key in expected:
                if new_key in converted:
                    raise ValueError("Checkpoint contains both " + old_key + " and " + new_key)
                converted[new_key] = converted.pop(old_key)
        left, right, shared = (prefix + item for item in
                               ("lin_l.weight", "lin_r.weight", "lin.weight"))
        if left in converted and shared in expected:
            if right not in converted or not torch.equal(converted[left], converted[right]):
                raise ValueError("Old GAT source/destination weights are not shared: " + layer)
            if shared in converted:
                raise ValueError("Checkpoint contains both old and new linear weights")
            converted[shared] = converted.pop(left)
            converted.pop(right)
    missing = set(expected) - set(converted)
    extra = set(converted) - set(expected)
    if missing or extra:
        raise ValueError(f"NIRM checkpoint keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return converted


@lru_cache(maxsize=2)
def _load_nirm(model_file, checkpoint_file, device):
    import torch

    spec = importlib.util.spec_from_file_location("_integrated_original_nirm_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load NIRM Model.py: " + model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.NIRM()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a NIRM checkpoint with model_state_dict")
    state = _convert_nirm_state(checkpoint["model_state_dict"], model.state_dict(), torch)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _nirm_features(G, nodes):
    """The same five features and column normalization as original Test.py."""
    import networkx as nx
    import numpy as np

    def normalized(values):
        column = np.asarray(values, dtype=np.float64)
        largest = column.max() if len(column) else 0.0
        return column / largest if largest > 0 else np.zeros_like(column)

    degree = nx.degree_centrality(G)
    second = []
    for node in nodes:
        layers = dict(nx.bfs_successors(G, source=node, depth_limit=2))
        second.append(sum(len(children) for children in layers.values()))
    neighbor_degree = nx.average_neighbor_degree(G)
    clustering = nx.clustering(G)
    return np.column_stack((
        normalized([degree[node] for node in nodes]),
        normalized(second),
        normalized([neighbor_degree[node] for node in nodes]),
        [clustering[node] for node in nodes],
        np.ones(len(nodes)),
    ))


def nirm_order(G, stop_condition=1, *, nirm_root=None, model_path=None,
               device="cpu", **kwargs):
    """Original NIRM one-pass prediction; returns a full node permutation."""
    if G.number_of_nodes() == 0:
        return []
    import numpy as np
    import torch

    root = Path(nirm_root) if nirm_root is not None else NIRM_ROOT
    model_file = root / "Model.py"
    checkpoint = Path(model_path) if model_path is not None else root / "checkpoints/NIRM_onepass.pkl"
    for path in (model_file, checkpoint):
        if not path.is_file():
            raise FileNotFoundError("Required NIRM file not found: " + str(path))
    device = str(torch.device(device))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use device='cpu'")
    model = _load_nirm(str(model_file.resolve()), str(checkpoint.resolve()), device)
    nodes = list(G.nodes())
    x = torch.tensor(_nirm_features(G, nodes), dtype=torch.float32, device=device)
    # Preserve Test.py's adjacency nonzero order and unweighted topology.
    adjacency = nx_adjacency(G, nodes)
    source, target = np.nonzero(adjacency)
    edge_index = torch.tensor(np.vstack((source, target)), dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model(x, edge_index, len(nodes)).reshape(-1)
        if scores.numel() != len(nodes) or not torch.isfinite(scores).all():
            raise ValueError("NIRM returned invalid node scores")
        indices = scores.topk(len(nodes)).indices.cpu().tolist()
    return [nodes[index] for index in indices]


def nx_adjacency(G, nodes):
    import networkx as nx
    return nx.to_numpy_array(G, nodelist=nodes, weight=None)


def autosnd_search_order(G, stop_condition=1, *, config_path=None,
                         selection="quality", evaluation_rate=1.0,
                         candidate_timeout_seconds=None, output_dir=None,
                         result_info=None, llm_client=None, verbose=True,
                         **kwargs):
    """Run all three AutoSND stages on G, then return the selected full order.

    Every call starts a fresh search. No frozen candidate or proxy is required.
    llm_client is injectable for offline contract tests; normal calls use env.
    """
    import inspect
    import json
    import time
    from datetime import datetime
    from dataclasses import replace
    from uuid import uuid4
    from .AutoSND.upstream.src.autosnd.candidate import execute_candidate
    from .AutoSND.upstream.src.autosnd.config import load_config, api_environment, provider_seed_environment
    from .AutoSND.upstream.src.autosnd.llm import LLMClient
    from .AutoSND.upstream.src.autosnd.search import run_search

    if kwargs:
        raise TypeError("Unknown AutoSND parameters: " + ", ".join(sorted(kwargs)))
    if selection not in ("quality", "speed"):
        raise ValueError("selection must be 'quality' or 'speed'")
    if not 0 < evaluation_rate <= 1:
        raise ValueError("evaluation_rate must be in (0, 1]")
    if result_info is not None and not isinstance(result_info, dict):
        raise TypeError("result_info must be a dict")
    if G.number_of_nodes() == 0:
        if result_info is not None:
            result_info.update({"status": "empty_graph", "search_performed": False})
        return []
    if "graph" not in inspect.signature(run_search).parameters:
        raise RuntimeError("Bundled AutoSND search is missing input-graph support; restore the complete integrated folder")
    config_file = Path(config_path) if config_path is not None else AUTOSND_ROOT / "configs/smoke.yaml"
    config = load_config(config_file)
    timeout = config.candidate_timeout_seconds if candidate_timeout_seconds is None else float(candidate_timeout_seconds)
    if timeout <= 0:
        raise ValueError("candidate_timeout_seconds must be positive")
    if llm_client is None:
        url, key = api_environment()
        llm_client = LLMClient(api_url=url, api_key=key, model=config.model,
                               seed=provider_seed_environment())
    if not isinstance(llm_client, LLMClient):
        raise TypeError("llm_client must be an AutoSND LLMClient")
    parent = Path(output_dir) if output_dir is not None else PACKAGE_ROOT.parent / "runs" / "AutoSND"
    run_dir = parent.resolve() / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:12])
    config = replace(config, evaluation_rate=float(evaluation_rate),
                     candidate_timeout_seconds=timeout, output_dir=run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    reverse = G.graph.get("_reverse_mapping", {})
    input_record = {"nodes": list(range(len(nodes))),
                    "edges": [[index[left], index[right]] for left, right in G.edges()],
                    "node_id_repr": [repr(reverse.get(node, node)) for node in nodes]}
    (run_dir / "input_graph.json").write_text(json.dumps(input_record, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {"status": "running", "search_performed": True,
            "run_dir": str(run_dir), "selection": selection,
            "evaluation_rate": config.evaluation_rate,
            "stop_condition": stop_condition, "nodes": len(nodes),
            "edges": G.number_of_edges()}
    def save_status():
        (run_dir / "interface_result.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        if result_info is not None:
            result_info.update(info)
    save_status()
    log = (lambda message: print(message, flush=True)) if verbose else None
    started = time.perf_counter()
    try:
        search = run_search(config, llm_client, graph=G.copy(), progress=log)
        winner = search.final_quality if selection == "quality" else search.final_speed
        final = execute_candidate(winner.code, G.copy(), rate=1.0, timeout_seconds=timeout)
        if not final.valid:
            raise RuntimeError("Selected AutoSND candidate failed: " + str(final.error))
        info.update({"status": "complete", "selected_candidate_id": winner.candidate_id,
                     "selected_code_path": str(run_dir / "final" / ("AutoSND-Q.py" if selection == "quality" else "AutoSND-S.py")),
                     "selected_search_anc": winner.anc_gcc,
                     "selected_full_anc": final.anc_gcc,
                     "selected_runtime_seconds": final.runtime_seconds,
                     "total_search_and_execution_seconds": time.perf_counter() - started})
        save_status()
        if verbose:
            print("AutoSND selected:", winner.candidate_id, "records:", run_dir, flush=True)
        return list(final.order)
    except Exception as exc:
        info.update({"status": "failed", "error_type": type(exc).__name__,
                     "error": llm_client.redact(str(exc)),
                     "total_search_and_execution_seconds": time.perf_counter() - started})
        save_status()
        raise
