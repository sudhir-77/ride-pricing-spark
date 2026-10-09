"""Run the DAG callables in the statically checked order and write a run log."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ride_pricing.dag_check import EXPECTED_ORDER, parse_dag_file, static_errors


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_check(path: Path, spec_dag_id: str, order: list[str], acyclic: bool, all_succeeded: bool) -> dict:
    payload = {
        "dag_id": spec_dag_id,
        "n_tasks": len(order),
        "task_order": list(order),
        "acyclic": bool(acyclic),
        "all_succeeded": bool(all_succeeded),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_local(
    log_path: Path,
    check_path: Path,
    callables: dict[str, Callable[[], None]] | None = None,
) -> dict:
    """Execute extract -> validate -> transform -> train -> evaluate.

    `callables` overrides the real tasks. Tests pass fakes so they can check
    order and the log without starting Spark.
    """
    spec = parse_dag_file()
    errors = static_errors(spec)
    if errors:
        raise RuntimeError("DAG failed static validation: " + "; ".join(errors))
    if callables is None:
        from ride_pricing import tasks as task_mod

        callables = {name: getattr(task_mod, name) for name in EXPECTED_ORDER}
    missing = [name for name in spec.task_order if name not in callables]
    if missing:
        raise RuntimeError(f"local runner is missing callables: {missing}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    all_succeeded = True
    for name in spec.task_order:
        lines.append(f"{_stamp()} task={name} status=started")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            callables[name]()
        except Exception:
            elapsed = time.perf_counter() - started
            lines.append(f"{_stamp()} task={name} status=failed seconds={elapsed:.3f}")
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _write_check(check_path, spec.dag_id, spec.task_order, spec.acyclic, False)
            all_succeeded = False
            raise
        elapsed = time.perf_counter() - started
        lines.append(f"{_stamp()} task={name} status=success seconds={elapsed:.3f}")
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all_succeeded:
        raise RuntimeError("local runner failed")
    return _write_check(check_path, spec.dag_id, spec.task_order, spec.acyclic, True)
