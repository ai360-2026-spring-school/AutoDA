"""De-duplicating view of past planner attempts, injected into every planner
prompt so the LLM stops proposing things it just tried.

State holds an append-only ``experiment_log`` list of entries built by
``make_experiment_entry``. Before injecting into the planner, we dedup by
``(operation, args_signature)``, keeping the latest outcome.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def args_signature(args: dict[str, Any]) -> str:
    """Canonical, hashable representation of an action's args.

    Used to detect duplicate attempts. Falls back to a string repr for
    args containing non-JSON-serialisable values (which is itself fine —
    the same non-serialisable args produce the same string).
    """
    try:
        canonical = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        canonical = repr(sorted(args.items())) if isinstance(args, dict) else repr(args)
    return canonical


def args_hash(args: dict[str, Any]) -> str:
    return hashlib.sha1(args_signature(args).encode("utf-8")).hexdigest()[:10]


def make_experiment_entry(
    *,
    step: int,
    operation: str,
    kind: str,
    args: dict[str, Any],
    decision: str,
    cv_delta: float | None,
    obs: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    """Build one ``experiment_log`` row from a finished iteration."""
    obs = obs or {}
    summary = obs.get("summary") or obs.get("error") or ""
    if not summary and error:
        summary = error[:120]
    return {
        "step": int(step),
        "operation": str(operation),
        "kind": str(kind),
        "args": dict(args or {}),
        "args_signature": args_signature(args or {}),
        "decision": str(decision),
        "cv_delta": (round(float(cv_delta), 6) if cv_delta is not None else None),
        "summary": str(summary)[:160],
        "error": (str(error)[:160] if error else None),
    }


def dedup_experiments(
    entries: Iterable[dict[str, Any]], limit: int = 40
) -> list[dict[str, Any]]:
    """Dedup by (operation, args_signature) keeping the most recent outcome.

    Returns at most ``limit`` entries, newest last.
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for e in entries:
        key = (e.get("operation", ""), e.get("args_signature", ""))
        # keep the latest — entries arrive in step order
        by_key[key] = e
    # newest last
    items = sorted(by_key.values(), key=lambda x: x.get("step", 0))
    if len(items) > limit:
        items = items[-limit:]
    return items


def format_experiment_log(entries: list[dict[str, Any]]) -> str:
    """Render the deduped log into a short, planner-friendly block.

    Lines are grouped by decision (KEPT first — these are part of the
    pipeline now; REJECTED — don't repeat with same args; ERROR — note
    the message; INFO — already gathered).
    """
    if not entries:
        return "(no experiments tried yet)"

    by_decision: dict[str, list[dict[str, Any]]] = {"keep": [], "reject": [], "error": []}
    info_calls: list[dict[str, Any]] = []
    for e in entries:
        if e.get("kind") == "info":
            info_calls.append(e)
        else:
            by_decision.setdefault(e.get("decision", "error"), []).append(e)

    def _fmt(e: dict[str, Any]) -> str:
        op = e["operation"]
        args = e.get("args", {})
        try:
            args_str = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except Exception:
            args_str = repr(args)
        cv_delta = e.get("cv_delta")
        delta_str = f"  delta={cv_delta:+.4f}" if cv_delta is not None else ""
        err = e.get("error")
        err_str = f"  error={err}" if err else ""
        return f"- {op}({args_str}){delta_str}{err_str}"

    out: list[str] = []
    if by_decision["keep"]:
        out.append("KEPT (already in the pipeline — do NOT redo):")
        out.extend(_fmt(e) for e in by_decision["keep"])
    if by_decision["reject"]:
        out.append("")
        out.append("REJECTED (same args won't be reconsidered — vary args or try something else):")
        out.extend(_fmt(e) for e in by_decision["reject"])
    if by_decision["error"]:
        out.append("")
        out.append("ERRORED (don't repeat with same args; fix the args or try a different op):")
        out.extend(_fmt(e) for e in by_decision["error"])
    if info_calls:
        out.append("")
        out.append("INFO TOOL CALLS (results are in the insights ledger):")
        out.extend(_fmt(e) for e in info_calls)

    return "\n".join(out)
