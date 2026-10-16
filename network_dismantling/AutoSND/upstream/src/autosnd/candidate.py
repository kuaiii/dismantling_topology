"""Candidate extraction, safety checks, and timeout-bounded execution."""

from __future__ import annotations

import ast
import collections
import heapq
import math
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .metrics import evaluate_order, validate_order

ROOT_HDA_CODE = """\
def degree_order(G):
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = max(H.nodes(), key=lambda u: (H.degree[u], str(u)))
        order.append(node)
        H.remove_node(node)
    return order
"""

_BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
}
_BANNED_MODULE_NAMES = {
    "ctypes",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
}


@dataclass(frozen=True)
class CandidateExecution:
    valid: bool
    order: tuple[Any, ...]
    anc_gcc: float | None
    runtime_seconds: float | None
    error: str | None
    runtime_censored: bool = False


def extract_candidate_code(response: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.I)
    for block in blocks:
        if "def degree_order" in block:
            return block.strip() + "\n"
    start = response.find("def degree_order")
    if start >= 0:
        return response[start:].strip() + "\n"
    raise ValueError("response does not contain def degree_order(G)")


def validate_candidate_source(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"candidate syntax error: {exc.msg}") from exc

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    degree_functions = [
        node
        for node in functions
        if isinstance(node, ast.FunctionDef) and node.name == "degree_order"
    ]
    if len(degree_functions) != 1:
        raise ValueError("candidate must define exactly one degree_order function")
    for node in tree.body:
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not isinstance(node, ast.FunctionDef) and not is_docstring:
            raise ValueError("top-level candidate code may contain functions only")
    for function in functions:
        if function.decorator_list:
            raise ValueError("function decorators are not allowed")
        if function.args.defaults or function.args.kw_defaults:
            raise ValueError("function default expressions are not allowed")

    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name):
            if node.id in _BANNED_MODULE_NAMES or node.id.startswith("__"):
                raise ValueError(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                raise ValueError(f"call {node.func.id!r} is not allowed")
    return tree


def execute_candidate(
    code: str,
    graph: nx.Graph,
    *,
    rate: float,
    timeout_seconds: float,
) -> CandidateExecution:
    try:
        validate_candidate_source(code)
    except ValueError as exc:
        return CandidateExecution(False, (), None, None, str(exc))

    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_candidate_worker,
        args=(code, graph, child_connection),
    )
    started = time.monotonic()
    process.start()
    child_connection.close()
    if not parent_connection.poll(timeout_seconds):
        process.kill()
        process.join()
        parent_connection.close()
        return CandidateExecution(
            False,
            (),
            None,
            float(timeout_seconds),
            f"candidate exceeded {timeout_seconds:g}s timeout",
            True,
        )
    try:
        status, payload = parent_connection.recv()
    except EOFError:
        return CandidateExecution(
            False,
            (),
            None,
            None,
            f"candidate worker exited with code {process.exitcode}",
        )
    finally:
        parent_connection.close()
    remaining = max(0.0, timeout_seconds - (time.monotonic() - started))
    process.join(remaining)
    if process.is_alive():
        process.kill()
        process.join()

    if status == "error":
        return CandidateExecution(False, (), None, None, str(payload))
    order, runtime = payload
    try:
        strict_order = validate_order(graph, order)
        metrics = evaluate_order(graph, strict_order, rate=rate)
    except Exception as exc:
        return CandidateExecution(
            False,
            (),
            None,
            float(runtime),
            f"{type(exc).__name__}: {exc}",
        )
    return CandidateExecution(
        True,
        tuple(strict_order),
        metrics.anc_gcc,
        float(runtime),
        None,
    )


def _candidate_worker(code: str, graph: nx.Graph, output: Any) -> None:
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "collections": collections,
        "heapq": heapq,
        "math": math,
    }
    try:
        compiled = compile(code, "<autosnd-candidate>", "exec")
        exec(compiled, namespace, namespace)
        function = namespace["degree_order"]
        working_graph = graph.copy()
        started = time.perf_counter()
        order = list(function(working_graph))
        runtime = time.perf_counter() - started
        output.send(("ok", (order, runtime)))
    except BaseException as exc:
        output.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        output.close()
