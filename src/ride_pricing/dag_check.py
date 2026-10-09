"""Parse the Airflow DAG file without importing Airflow."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

DAG_FILE = Path(__file__).resolve().parents[2] / "dags" / "nyc_taxi_demand_dag.py"
EXPECTED_DAG_ID = "nyc_taxi_demand"
EXPECTED_SCHEDULE = "0 6 * * *"
EXPECTED_ORDER = ["extract", "validate", "transform", "train", "evaluate"]


@dataclass(frozen=True)
class DagSpec:
    dag_id: str
    schedule: str
    n_tasks: int
    task_order: list[str]
    acyclic: bool
    callables: dict[str, str]


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def _constant(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _chain(node: ast.AST) -> list[str] | None:
    """Flatten a left-associative chain of `>>` between variable names."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
        left = _chain(node.left)
        right = _chain(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def _topo(nodes: list[str], edges: list[tuple[str, str]]) -> tuple[list[str], bool]:
    incoming = {node: 0 for node in nodes}
    children: dict[str, list[str]] = {node: [] for node in nodes}
    for src, dst in edges:
        if src not in incoming or dst not in incoming:
            raise ValueError(f"dependency {src} >> {dst} is not a declared task")
        children[src].append(dst)
        incoming[dst] += 1
    queue = [node for node in nodes if incoming[node] == 0]
    order: list[str] = []
    while queue:
        queue.sort()
        node = queue.pop(0)
        order.append(node)
        for child in children[node]:
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    return order, len(order) == len(nodes)


def parse_dag_source(source: str) -> DagSpec:
    tree = ast.parse(source)
    dag_id = None
    schedule = None
    variables: dict[str, tuple[str, str]] = {}
    edges: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) == "DAG":
            dag_id = _constant(_keyword(node, "dag_id"))
            schedule = _constant(_keyword(node, "schedule"))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Call) and _call_name(value.func) == "PythonOperator":
                task_id = _constant(_keyword(value, "task_id"))
                callable_node = _keyword(value, "python_callable")
                callable_name = callable_node.id if isinstance(callable_node, ast.Name) else None
                if task_id and callable_name:
                    variables[node.targets[0].id] = (task_id, callable_name)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.BinOp):
            names = _chain(node.value)
            if names and len(names) >= 2:
                for src, dst in zip(names, names[1:]):
                    edges.append((src, dst))
    if not dag_id or schedule is None:
        raise ValueError("DAG() is missing dag_id or schedule")
    if len(variables) != len(EXPECTED_ORDER):
        raise ValueError(f"expected {len(EXPECTED_ORDER)} PythonOperator tasks, found {len(variables)}")
    var_to_task = {name: pair[0] for name, pair in variables.items()}
    callables = {pair[0]: pair[1] for pair in variables.values()}
    task_edges = [(var_to_task[src], var_to_task[dst]) for src, dst in edges]
    order, acyclic = _topo(list(callables), task_edges)
    return DagSpec(
        dag_id=dag_id,
        schedule=schedule,
        n_tasks=len(callables),
        task_order=order,
        acyclic=acyclic,
        callables=callables,
    )


def parse_dag_file(path: Path | None = None) -> DagSpec:
    dag_path = DAG_FILE if path is None else path
    return parse_dag_source(dag_path.read_text(encoding="utf-8"))


def static_errors(spec: DagSpec) -> list[str]:
    errors: list[str] = []
    if spec.dag_id != EXPECTED_DAG_ID:
        errors.append(f"dag_id is {spec.dag_id}")
    if spec.schedule != EXPECTED_SCHEDULE:
        errors.append(f"schedule is {spec.schedule}")
    if spec.n_tasks != len(EXPECTED_ORDER):
        errors.append(f"n_tasks is {spec.n_tasks}")
    if not spec.acyclic:
        errors.append("graph has a cycle")
    if spec.task_order != EXPECTED_ORDER:
        errors.append(f"task_order is {spec.task_order}")
    for task_id, callable_name in spec.callables.items():
        if task_id != callable_name:
            errors.append(f"task {task_id} calls {callable_name}")
    return errors
