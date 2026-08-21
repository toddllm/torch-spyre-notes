#!/usr/bin/env python3
"""Scan torch-spyre for call sites of expensive analysis methods.

Detects call sites where the same expensive analysis method (e.g.
``get_read_writes``, ``op_read_writes``, ``iteration_space_from_op``,
``get_fill_order``, ``_build_indirect_load_subs``, ``get_reads``,
``get_writes``, ``normalize_ranges``, ``normalize_layouts``) is invoked
multiple times inside the same enclosing pass class -- a strong signal
for missing caching / result reuse.

Output JSON schema is fixed by the coordinator (see task brief).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

TORCH_SPYRE_SHA = "fea0c4be901e1383b1f700dbad8887128b0fcb27"

# Method / helper names of interest. Order matters only for logging.
TARGET_METHODS: set[str] = {
    "get_read_writes",
    "op_read_writes",
    "get_fill_order",
    "iteration_space_from_op",
    "iter_space_from_op",
    "_build_indirect_load_subs",
    "get_reads",
    "get_writes",
    "normalize_ranges",
    "normalize_layouts",
}


def _iter_python_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # Skip vendored / test fixtures we don't own; keep tests/ so we
        # spot repeated helper calls in fixture-heavy tests too.
        if "/.git/" in str(p):
            continue
        yield p


class _EnclosingScopeFinder(ast.NodeVisitor):
    """Track enclosing function/method and enclosing class for each node.

    We attach ``_enclosing_func`` and ``_enclosing_class`` string attrs
    to every visited Call node so the top-level walker can read them.
    """

    def __init__(self) -> None:
        self._func_stack: list[str] = []
        self._class_stack: list[str] = []
        self.calls: list[ast.Call] = []

    # -- generic helpers -------------------------------------------------
    def _visit_children(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    # -- scope pushes ----------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._class_stack.append(node.name)
        self._visit_children(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._func_stack.append(node.name)
        self._visit_children(node)
        self._func_stack.pop()

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._func_stack.append(node.name)
        self._visit_children(node)
        self._func_stack.pop()

    # -- call collection -------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # tag with current enclosing scopes so downstream reads see them
        node._enclosing_func = (  # type: ignore[attr-defined]
            self._func_stack[-1] if self._func_stack else None
        )
        node._enclosing_class = (  # type: ignore[attr-defined]
            self._class_stack[-1] if self._class_stack else None
        )
        self.calls.append(node)
        # recurse into args/keywords/func in case of nested calls
        self._visit_children(node)


def _call_method_name(call: ast.Call) -> str | None:
    """Return the invoked name for calls we care about.

    Handles both ``obj.get_read_writes(...)`` (Attribute) and bare
    ``op_read_writes(op)`` (Name) forms. We deliberately don't try to
    resolve imports -- name-level matching is sufficient because the
    target set is distinctive.
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _snippet(lines: list[str], lineno: int, radius: int = 1) -> str:
    """Return up to (2*radius+1) source lines around lineno (1-indexed)."""
    start = max(0, lineno - 1 - radius)
    end = min(len(lines), lineno - 1 + radius + 1)
    return "\n".join(lines[start:end])


def scan_file(path: Path) -> list[dict]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    finder = _EnclosingScopeFinder()
    finder.visit(tree)

    lines = source.splitlines()
    hits: list[dict] = []
    for call in finder.calls:
        name = _call_method_name(call)
        if name is None or name not in TARGET_METHODS:
            continue
        # skip the defining site itself when we hit a *definition*'s
        # default-arg call -- rare, but easy to guard: definitions
        # aren't Calls. Nothing to do here.
        line = call.lineno
        hits.append(
            {
                "file": str(path),
                "line": line,
                "kind": name,
                "call_kind": (
                    "attribute"
                    if isinstance(call.func, ast.Attribute)
                    else "name"
                ),
                "enclosing_func": getattr(call, "_enclosing_func", None),
                "enclosing_class": getattr(call, "_enclosing_class", None),
                "snippet": _snippet(lines, line, radius=1),
                # 3-line context centered on the call
                "context": _snippet(lines, line, radius=1),
            }
        )
    return hits


def _annotate_repeats(hits: list[dict]) -> None:
    """Add ``pass_repeat_count`` -- how many calls to the same method live
    inside the same (file, enclosing_class) pair.

    We key on (file, enclosing_class or "<module>", kind) so free-function
    passes (no class) still get counted per-file. This is the crude
    reachability signal called for in the task.
    """
    key_counts: dict[tuple[str, str, str], int] = {}
    for h in hits:
        key = (
            h["file"],
            h["enclosing_class"] or "<module>",
            h["kind"],
        )
        key_counts[key] = key_counts.get(key, 0) + 1
    for h in hits:
        key = (
            h["file"],
            h["enclosing_class"] or "<module>",
            h["kind"],
        )
        h["pass_repeat_count"] = key_counts[key]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root of the torch-spyre worktree",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Where to write the JSON result",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of hits to print to stdout (default 20)",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"--root not a directory: {root}")

    scanned = 0
    all_hits: list[dict] = []
    for path in _iter_python_files(root):
        scanned += 1
        all_hits.extend(scan_file(path))

    _annotate_repeats(all_hits)

    # Sort by (repeat count desc, file, line) so the top-N surfaces the
    # most repeated call sites first.
    all_hits.sort(
        key=lambda h: (-h["pass_repeat_count"], h["file"], h["line"])
    )

    result = {
        "scanner": "repeated_analysis",
        "torch_spyre_sha": TORCH_SPYRE_SHA,
        "scanned_files": scanned,
        "total_hits": len(all_hits),
        "hits": all_hits,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Human-readable summary
    print(
        f"repeated_analysis: scanned {scanned} files, "
        f"{len(all_hits)} hits -> {args.out}"
    )
    print(f"top {args.top} (by pass_repeat_count desc):")
    for h in all_hits[: args.top]:
        cls = h["enclosing_class"] or "<module>"
        fn = h["enclosing_func"] or "<module>"
        rel = h["file"]
        try:
            rel = str(Path(h["file"]).relative_to(root))
        except ValueError:
            pass
        print(
            f"  {rel}:{h['line']}  {h['kind']}  "
            f"[class={cls} func={fn} repeats={h['pass_repeat_count']}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
