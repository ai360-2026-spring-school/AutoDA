"""Build the action-catalog schema for the planner prompt from each action's
signature and docstring instead of a hand-maintained dict.

Docstring contract for every action / info-tool:

    \"\"\"<one-line summary>.

    Use when: <when this action helps>.
    Effect: <what changes / what the LLM learns>.

    Args:
        col_name: <type>. <one-line description>. (default: ...)
        another:  <type>. <description>.
    \"\"\"

The parser is intentionally tiny — Google-style ``Args:`` block, one arg per
line, optional type/description/default. Anything that doesn't match falls back
to a signature-introspected entry without docs (no crash).
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Any, Callable


_ARG_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_][\w]*)\s*:\s*(?P<rest>.*)$"
)


def _parse_docstring(fn: Callable[..., Any]) -> dict[str, Any]:
    """Pull summary + ``Use when`` / ``Effect`` lines + ``Args:`` block."""
    raw = inspect.getdoc(fn) or ""
    if not raw:
        return {"summary": "", "use_when": "", "effect": "", "args_doc": {}}

    lines = raw.split("\n")
    # First non-empty line is the summary
    summary = ""
    for ln in lines:
        if ln.strip():
            summary = ln.strip().rstrip(".")
            break

    use_when = ""
    effect = ""
    for ln in lines[1:]:
        s = ln.strip()
        if s.lower().startswith("use when:"):
            use_when = s.split(":", 1)[1].strip()
        elif s.lower().startswith("effect:"):
            effect = s.split(":", 1)[1].strip()

    # Args block
    args_doc: dict[str, str] = {}
    in_args = False
    base_indent: int | None = None
    for ln in lines:
        stripped = ln.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            base_indent = None
            continue
        if not in_args:
            continue
        # Stop at blank line followed by a non-arg block, or at another section header
        if not stripped:
            # blank line ends the args block only if the next non-blank starts a new section
            base_indent = base_indent  # noop; we just look at indent below
            continue
        if stripped.lower().rstrip(":") in {"returns", "raises", "example", "examples", "notes", "yields"}:
            in_args = False
            continue
        # arg line — accept either "name: rest" or "name (type): rest"
        m = _ARG_LINE_RE.match(stripped)
        if m:
            name = m.group("name")
            rest = m.group("rest").strip()
            args_doc[name] = rest
        # else: continuation of previous arg's description
        elif args_doc:
            last = list(args_doc.keys())[-1]
            args_doc[last] = (args_doc[last] + " " + stripped).strip()

    return {
        "summary": summary,
        "use_when": use_when,
        "effect": effect,
        "args_doc": args_doc,
    }


# Args that the framework injects automatically; never advertised to the LLM.
_FRAMEWORK_ONLY_ARGS = {"df", "target", "feature_importances", "task"}


def _signature_args(fn: Callable[..., Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return out
    for name, param in sig.parameters.items():
        if name in _FRAMEWORK_ONLY_ARGS:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = param.annotation
        ann_str = inspect.formatannotation(ann) if ann is not inspect._empty else None
        default = None if param.default is inspect._empty else param.default
        out.append({
            "name": name,
            "type": ann_str,
            "required": param.default is inspect._empty,
            "default": default,
        })
    return out


def describe_action(fn: Callable[..., Any], *, operation: str, kind: str) -> dict[str, Any]:
    """Build one catalog entry for the planner."""
    doc = _parse_docstring(fn)
    args = _signature_args(fn)
    # Attach docstring descriptions to signature args
    for a in args:
        a["doc"] = doc["args_doc"].get(a["name"], "")
    return {
        "operation": operation,
        "kind": kind,
        "summary": doc["summary"],
        "use_when": doc["use_when"],
        "effect": doc["effect"],
        "args": args,
    }


def build_catalog(
    transformers: dict[str, Callable[..., Any]],
    info_tools: dict[str, Callable[..., Any]],
) -> list[dict[str, Any]]:
    """Build the full schema list. Stable ordering: transformers, then info tools."""
    cat: list[dict[str, Any]] = []
    for name, fn in transformers.items():
        cat.append(describe_action(fn, operation=name, kind="transformer"))
    for name, fn in info_tools.items():
        cat.append(describe_action(fn, operation=name, kind="info"))
    return cat


def format_catalog(catalog: list[dict[str, Any]]) -> str:
    """Render the catalog as a compact planner-friendly text block.

    Less verbose than JSON dump while still showing every arg with its
    type and default. Designed to fit in <4000 chars for ~20 actions.
    """
    if not catalog:
        return "(no actions)"

    out: list[str] = []
    for entry in catalog:
        op = entry["operation"]
        kind = entry["kind"]
        summary = entry.get("summary") or "(no summary)"
        out.append(f"\n• {op}  [kind={kind}] — {summary}")
        if entry.get("use_when"):
            out.append(f"    use when: {entry['use_when']}")
        if entry.get("effect"):
            out.append(f"    effect:   {entry['effect']}")
        for a in entry.get("args", []):
            parts = [f"    - {a['name']}"]
            if a.get("type"):
                parts.append(f": {a['type']}")
            if not a.get("required"):
                parts.append(f"  (default: {a.get('default')!r})")
            if a.get("doc"):
                parts.append(f"  — {a['doc']}")
            out.append("".join(parts))
    return "\n".join(out)
