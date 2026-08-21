#!/usr/bin/env python3
"""Scan torch-spyre for mutations of GraphLowering state.

Catalogues every write to ``graph.operations``, ``graph.buffers``,
``graph.name_to_buffer``, ``graph.removed_buffers``, ``graph.constants``,
``graph.graph_inputs``, ``graph.graph_outputs``, and
``graph.scheduler_node_map`` — plus registration/replacement helpers
(``register_buffer``, ``register_operation``, ``replace_computed_buffer_body``,
``replace_by_example``).

Output JSON follows the phase schema (see notes in scans/README.md).

Stdlib only.  Skips ``tests/``, ``examples/``, and ``docs/`` unless a hit
inside them dereferences a name we recognise as a real GraphLowering
receiver (``V.graph.*`` or ``self.graph.*``) — those cases are marked
with ``in_test_or_docs = True`` in the emitted record.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Target attributes and helper names
# ---------------------------------------------------------------------------

TARGET_ATTRS: frozenset[str] = frozenset(
    {
        "operations",
        "buffers",
        "name_to_buffer",
        "removed_buffers",
        "constants",
        "graph_inputs",
        "graph_outputs",
        "scheduler_node_map",
    }
)

# Method calls we treat as mutations when invoked on one of the target
# containers.  Anything else (e.g. ``.get``, ``.items``) is a read.
MUTATING_METHODS: dict[str, str] = {
    "append": "append",
    "extend": "append",
    "insert": "insert",
    "add": "append",           # set.add
    "update": "replace",       # dict.update / set.update
    "remove": "remove",
    "pop": "remove",
    "discard": "remove",
    "clear": "clear",
    "__setitem__": "replace",
    "__delitem__": "remove",
    "setdefault": "replace",
}

HELPER_FUNCS: dict[str, str] = {
    "register_buffer": "register",
    "register_operation": "register",
    "replace_computed_buffer_body": "replace",
    "replace_by_example": "replace",
}

# Ignore-directory prefixes (relative to --root).  Kept case-insensitive to
# be safe on macOS.
NOISY_DIR_PREFIXES: tuple[str, ...] = ("tests/", "examples/", "docs/")

# A hit inside noisy dirs is still kept if the receiver looks like the real
# GraphLowering.  Match ``V.graph`` or ``self.graph`` — that is how the
# passes reach into inductor's graph.
REAL_GRAPH_RE = re.compile(r"\b(?:V|self)\s*\.\s*graph\b")


# ---------------------------------------------------------------------------
# Hit record
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    file: str
    line: int
    kind: str
    snippet: str
    context: str
    function: str = ""
    attr: str = ""
    receiver: str = ""
    in_test_or_docs: bool = False


# ---------------------------------------------------------------------------
# AST walk
# ---------------------------------------------------------------------------


def _attr_chain(node: ast.AST) -> list[str]:
    """Return dotted names for an attribute chain, or [] if not resolvable."""
    parts: list[str] = []
    cur: ast.AST | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return []


def _ends_with_graph_target(chain: list[str]) -> tuple[bool, str, str]:
    """Return (matches, target_attr, receiver_repr) for chains ending in
    ``...graph.<target>``.
    """
    if len(chain) < 3:
        return False, "", ""
    if chain[-2] != "graph":
        return False, "", ""
    if chain[-1] not in TARGET_ATTRS:
        return False, "", ""
    receiver = ".".join(chain[:-2]) or "?"
    return True, chain[-1], receiver + ".graph"


def _mid_chain_has_graph_target(chain: list[str]) -> tuple[bool, str, str]:
    """Return (matches, target_attr, receiver_repr) for chains where
    ``graph.<target>`` appears mid-chain (e.g. ``V.graph.buffers.append``).
    """
    for i in range(len(chain) - 1):
        if chain[i] == "graph" and chain[i + 1] in TARGET_ATTRS:
            receiver = ".".join(chain[: i + 1]) if i > 0 else "graph"
            return True, chain[i + 1], receiver
    return False, "", ""


def _fmt_snippet(source_lines: list[str], line_no: int) -> tuple[str, str]:
    """Return (snippet_1line, context_3lines) for the given 1-based line."""
    idx = line_no - 1
    if idx < 0 or idx >= len(source_lines):
        return "", ""
    snippet = source_lines[idx].rstrip()
    lo = max(0, idx - 1)
    hi = min(len(source_lines), idx + 2)
    context = "\n".join(l.rstrip() for l in source_lines[lo:hi])
    return snippet, context


class Scanner(ast.NodeVisitor):
    def __init__(self, rel_path: str, source_lines: list[str], noisy: bool):
        self.rel_path = rel_path
        self.source_lines = source_lines
        self.noisy = noisy
        self.hits: list[Hit] = []
        # Stack of enclosing function names for context.
        self._func_stack: list[str] = []

    # -- function tracking --------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    # -- assignments --------------------------------------------------------

    def _record(self, node: ast.AST, kind: str, attr: str, receiver: str) -> None:
        snippet, context = _fmt_snippet(self.source_lines, node.lineno)
        # Noisy-dir filter: keep only if receiver looks like V.graph / self.graph
        in_test = self.noisy
        if in_test and not REAL_GRAPH_RE.search(snippet):
            return
        self.hits.append(
            Hit(
                file=self.rel_path,
                line=node.lineno,
                kind=kind,
                snippet=snippet.strip(),
                context=context,
                function=self._func_stack[-1] if self._func_stack else "",
                attr=attr,
                receiver=receiver,
                in_test_or_docs=in_test,
            )
        )

    def _check_assign_target(self, target: ast.AST, node: ast.AST) -> None:
        # X.graph.<attr> = ...
        if isinstance(target, ast.Attribute):
            chain = _attr_chain(target)
            matches, attr, receiver = _ends_with_graph_target(chain)
            if matches:
                self._record(node, "replace", attr, receiver)
                return
        # X.graph.<attr>[k] = ...      -> Subscript(value=Attribute(...))
        if isinstance(target, ast.Subscript):
            chain = _attr_chain(target.value)
            matches, attr, receiver = _ends_with_graph_target(chain)
            if matches:
                self._record(node, "replace", attr, receiver)
                return

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            # Also handle tuple / list unpacking targets.
            if isinstance(tgt, (ast.Tuple, ast.List)):
                for elt in tgt.elts:
                    self._check_assign_target(elt, node)
            else:
                self._check_assign_target(tgt, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # graph.operations += [...] etc.
        self._check_assign_target(node.target, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_assign_target(node.target, node)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for tgt in node.targets:
            if isinstance(tgt, ast.Subscript):
                chain = _attr_chain(tgt.value)
                matches, attr, receiver = _ends_with_graph_target(chain)
                if matches:
                    self._record(node, "remove", attr, receiver)
            elif isinstance(tgt, ast.Attribute):
                chain = _attr_chain(tgt)
                matches, attr, receiver = _ends_with_graph_target(chain)
                if matches:
                    self._record(node, "remove", attr, receiver)
        self.generic_visit(node)

    # -- calls --------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # Helper functions (positional/attribute).  Trigger on any name/attr
        # that matches, regardless of receiver — these are all narrow enough
        # to be worth flagging.
        helper_name = None
        if isinstance(func, ast.Attribute):
            helper_name = func.attr
        elif isinstance(func, ast.Name):
            helper_name = func.id
        if helper_name in HELPER_FUNCS:
            self._record(node, HELPER_FUNCS[helper_name], helper_name, "")
            self.generic_visit(node)
            return

        # Mutating methods on a graph.<attr> receiver.
        if isinstance(func, ast.Attribute) and func.attr in MUTATING_METHODS:
            chain = _attr_chain(func.value)  # chain to receiver of the call
            matches, attr, receiver = _ends_with_graph_target(chain)
            if matches:
                self._record(
                    node,
                    MUTATING_METHODS[func.attr],
                    attr,
                    f"{receiver}.{func.attr}",
                )
                self.generic_visit(node)
                return
            # Support ``X.graph.<attr>.<sub>.append(...)`` where target sits
            # mid-chain (rare, but harmless to check).
            matches, attr, receiver = _mid_chain_has_graph_target(chain)
            if matches:
                self._record(
                    node,
                    MUTATING_METHODS[func.attr],
                    attr,
                    f"{receiver}.{func.attr}",
                )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def iter_python_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # skip virtualenvs, hidden dirs, and vendored trees
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(part.startswith(".") for part in parts):
            continue
        if parts[0] in {"build", "dist", ".venv", "venv", "site-packages"}:
            continue
        yield p


def scan_file(path: Path, root: Path) -> list[Hit]:
    rel = path.relative_to(root).as_posix()
    noisy = rel.startswith(NOISY_DIR_PREFIXES) or "/tests/" in f"/{rel}"
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    scanner = Scanner(rel_path=rel, source_lines=source.splitlines(), noisy=noisy)
    scanner.visit(tree)
    return scanner.hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"ERROR: root {root} does not exist", file=sys.stderr)
        return 2

    all_hits: list[Hit] = []
    scanned = 0
    for p in iter_python_files(root):
        scanned += 1
        all_hits.extend(scan_file(p, root))

    # Sort by (file, line) for deterministic output.
    all_hits.sort(key=lambda h: (h.file, h.line))

    payload = {
        "scanner": "graph_mutations",
        "torch_spyre_sha": "fea0c4be901e1383b1f700dbad8887128b0fcb27",
        "scanned_files": scanned,
        "total_hits": len(all_hits),
        "hits": [asdict(h) for h in all_hits],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Aggregate kind counts for the summary line.
    kind_counts: dict[str, int] = {}
    attr_counts: dict[str, int] = {}
    for h in all_hits:
        kind_counts[h.kind] = kind_counts.get(h.kind, 0) + 1
        if h.attr:
            attr_counts[h.attr] = attr_counts.get(h.attr, 0) + 1

    print(f"scanned_files={scanned}  total_hits={len(all_hits)}  out={args.out}")
    print(
        "kinds: "
        + ", ".join(f"{k}={v}" for k, v in sorted(kind_counts.items(), key=lambda x: -x[1]))
    )
    print(
        "attrs: "
        + ", ".join(f"{k}={v}" for k, v in sorted(attr_counts.items(), key=lambda x: -x[1]))
    )
    print(f"top {args.top} hits:")
    for h in all_hits[: args.top]:
        marker = " [test/doc]" if h.in_test_or_docs else ""
        fn = f"::{h.function}" if h.function else ""
        print(f"  {h.file}:{h.line}{fn} [{h.kind}/{h.attr}]{marker}  {h.snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
