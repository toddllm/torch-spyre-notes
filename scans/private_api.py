#!/usr/bin/env python3
"""
Scanner: private_api

Exhaustively catalogue every Torch-Spyre reach into private upstream API.

Detects:
- Import statements of `torch._...` modules (torch._inductor, torch._dynamo,
  torch.fx.experimental, torch._C, torch._prims_common, torch._decomp, etc.)
- Attribute access on `torch` or its submodules where the attribute starts
  with an underscore (heuristic: chained attribute access rooted at `torch`
  where any segment begins with `_`).
- getattr(x, "_...") and setattr(x, "_...", ...)
- object.__setattr__(...) calls
- Access to dunder introspection attrs (__closure__, __code__, __globals__,
  __wrapped__, __dict__) on presumed non-local objects.
- Subclassing torch._... modules or classes (ClassDef bases).
- Indexed access into upstream registries (V.graph._..., lowerings[...],
  decomp table lookups, etc.).
- `# type: ignore[...]` comments (categorized by error code).
- `attr-defined` suppressions specifically.

Emits per hit: file, line, kind, private name touched, and a 3-line snippet.
Includes a hotspots section (top 10 files by hit count).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tokenize
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TORCH_SPYRE_SHA = "fea0c4be901e1383b1f700dbad8887128b0fcb27"
SCANNER_NAME = "private_api"

# Public torch dunder attributes that are legitimately part of the API
# and should NOT be flagged as private introspection (e.g. Tensor.__add__).
PUBLIC_DUNDERS = {
    "__init__", "__new__", "__repr__", "__str__", "__call__", "__enter__",
    "__exit__", "__len__", "__iter__", "__next__", "__contains__",
    "__getitem__", "__setitem__", "__delitem__", "__eq__", "__ne__",
    "__lt__", "__le__", "__gt__", "__ge__", "__hash__", "__bool__",
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
    "__mod__", "__pow__", "__matmul__", "__radd__", "__rsub__", "__rmul__",
    "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__", "__rmatmul__",
    "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__",
    "__imod__", "__ipow__", "__imatmul__", "__neg__", "__pos__", "__abs__",
    "__invert__", "__and__", "__or__", "__xor__", "__lshift__", "__rshift__",
    "__format__", "__reduce__", "__reduce_ex__", "__sizeof__", "__copy__",
    "__deepcopy__", "__getstate__", "__setstate__", "__getattr__",
    "__setattr__", "__delattr__", "__dir__", "__class__", "__subclasshook__",
    "__init_subclass__", "__index__", "__int__", "__float__", "__complex__",
    "__round__", "__ceil__", "__floor__", "__trunc__", "__reversed__",
    "__name__", "__doc__", "__module__", "__qualname__", "__file__",
    "__version__", "__all__",  # module-public metadata
    "__annotations__",  # type-hint accessor is broadly public
    "__slots__", "__mro__", "__bases__",
}

# The specific introspection dunders we DO want to catch when accessed on
# what looks like a non-local object.
INTROSPECTION_DUNDERS = {
    "__closure__", "__code__", "__globals__", "__wrapped__", "__dict__",
    "__func__", "__self__", "__defaults__", "__kwdefaults__",
}

# Regexes for line-based passes
RE_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore(?:\[([^\]]*)\])?")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_source(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def make_snippet(lines: List[str], lineno: int) -> str:
    """Return a 3-line snippet centered on lineno (1-based)."""
    lo = max(1, lineno - 1)
    hi = min(len(lines), lineno + 1)
    out = []
    for i in range(lo, hi + 1):
        prefix = ">>" if i == lineno else "  "
        out.append(f"{prefix} {i:5d}: {lines[i - 1].rstrip()}")
    return "\n".join(out)


def attr_chain(node: ast.AST) -> Optional[List[str]]:
    """
    Flatten an Attribute/Name chain into ['torch', '_dynamo', 'guards'] etc.
    Returns None if the chain root isn't a Name.
    """
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def is_torch_root(chain: List[str]) -> bool:
    return bool(chain) and chain[0] == "torch"


def has_private_segment(chain: List[str]) -> Tuple[bool, str]:
    """
    Return (True, first_private_segment) if any non-root segment starts with '_'
    and is not a benign public dunder.
    """
    for seg in chain[1:]:
        if seg.startswith("_") and seg not in PUBLIC_DUNDERS:
            return True, seg
    return False, ""


def is_string_constant(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------


class PrivateAPIVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str, lines: List[str]) -> None:
        self.rel_path = rel_path
        self.lines = lines
        self.hits: List[Dict] = []
        # Track names bound to `torch` (import torch as t) or torch submodules.
        self.torch_aliases: Dict[str, List[str]] = {"torch": ["torch"]}
        # First-pass: harvest import aliases before walking uses.
        # (visit() naturally handles both; but we want alias tracking to be
        # available when an attribute use appears syntactically before its
        # import — which shouldn't happen in valid Python, but be safe.)

    # -- utility -----------------------------------------------------------

    def _add(
        self,
        node: ast.AST,
        kind: str,
        name: str,
        extra_context: Optional[str] = None,
    ) -> None:
        lineno = getattr(node, "lineno", 0) or 0
        snippet = make_snippet(self.lines, lineno) if lineno else ""
        context = extra_context or ""
        self.hits.append(
            {
                "file": self.rel_path,
                "line": lineno,
                "kind": kind,
                "private_name": name,
                "snippet": snippet,
                "context": context,
            }
        )

    # -- imports -----------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod = alias.name
            bound = alias.asname or mod.split(".", 1)[0]
            parts = mod.split(".")
            if parts and parts[0] == "torch":
                self.torch_aliases.setdefault(bound, parts)
                if any(p.startswith("_") for p in parts[1:]):
                    self._add(
                        node, "import", mod,
                        extra_context=f"import {mod}"
                        + (f" as {alias.asname}" if alias.asname else ""),
                    )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        level = node.level or 0
        parts = mod.split(".") if mod else []
        # from torch._foo import bar
        if not level and parts and parts[0] == "torch":
            # Track imported names as roots inside torch's private surface.
            for alias in node.names:
                imported = alias.name
                bound = alias.asname or imported
                self.torch_aliases.setdefault(
                    bound, parts + [imported]
                )
            if any(p.startswith("_") for p in parts[1:]):
                names = ", ".join(a.name for a in node.names)
                self._add(
                    node, "import", mod,
                    extra_context=f"from {mod} import {names}",
                )
            else:
                # Even from a public parent, `from torch import _foo`
                # is a private reach.
                for alias in node.names:
                    if alias.name.startswith("_") and alias.name not in PUBLIC_DUNDERS:
                        self._add(
                            node, "import",
                            f"{mod}.{alias.name}" if mod else alias.name,
                            extra_context=f"from {mod} import {alias.name}",
                        )
        self.generic_visit(node)

    # -- class bases (subclassing) ----------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            chain = attr_chain(base)
            if chain is None:
                continue
            # Resolve through alias table
            root = chain[0]
            if root in self.torch_aliases:
                resolved = self.torch_aliases[root] + chain[1:]
                if is_torch_root(resolved):
                    private, seg = has_private_segment(resolved)
                    if private:
                        self._add(
                            node, "subclass",
                            ".".join(resolved),
                            extra_context=f"class {node.name}({'.'.join(chain)})",
                        )
        self.generic_visit(node)

    # -- attribute access --------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = attr_chain(node)
        if chain is not None:
            root = chain[0]
            if root in self.torch_aliases:
                resolved = self.torch_aliases[root] + chain[1:]
                if is_torch_root(resolved):
                    private, seg = has_private_segment(resolved)
                    if private:
                        self._add(
                            node, "attr",
                            ".".join(resolved),
                            extra_context=f"attr chain: {'.'.join(chain)}",
                        )
                        # Do not descend further — the whole chain is one hit.
                        return

            # Dunder introspection detection on non-local objects.
            if node.attr in INTROSPECTION_DUNDERS:
                # Skip obvious self.__x accesses inside the class body — those
                # are usually a class touching its own internals.  We still
                # flag if the object is anything but `self`.
                base_repr = _short_expr(node.value)
                if base_repr != "self":
                    self._add(
                        node, "dunder", node.attr,
                        extra_context=f"{base_repr}.{node.attr}",
                    )
        self.generic_visit(node)

    # -- call sites: getattr/setattr/object.__setattr__ -------------------

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # getattr(x, "_foo") / setattr(x, "_foo", ...)
        if isinstance(func, ast.Name) and func.id in ("getattr", "setattr"):
            if len(node.args) >= 2:
                name = is_string_constant(node.args[1])
                if name and name.startswith("_") and name not in PUBLIC_DUNDERS:
                    obj = _short_expr(node.args[0])
                    kind = "getattr" if func.id == "getattr" else "setattr"
                    self._add(
                        node, kind, name,
                        extra_context=f"{func.id}({obj}, {name!r}, ...)",
                    )
        # object.__setattr__(x, "_foo", val)
        if isinstance(func, ast.Attribute) and func.attr == "__setattr__":
            base_chain = attr_chain(func)
            if base_chain and base_chain[0] == "object":
                name_expr = node.args[1] if len(node.args) >= 2 else None
                name = is_string_constant(name_expr) if name_expr else None
                obj = _short_expr(node.args[0]) if node.args else "?"
                name_repr = repr(name) if name else "<expr>"
                self._add(
                    node, "__setattr__",
                    name or "<dynamic>",
                    extra_context=(
                        f"object.__setattr__({obj}, {name_repr}, ...)"
                    ),
                )
        self.generic_visit(node)

    # -- subscript (registry indexing) ------------------------------------

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Detect indexing into something whose chain root is torch/aliased
        # torch or a "V.graph._foo"-style compiler registry access.
        chain = attr_chain(node.value)
        if chain is not None:
            root = chain[0]
            # Case A: torch or aliased torch chain with a private segment.
            if root in self.torch_aliases:
                resolved = self.torch_aliases[root] + chain[1:]
                if is_torch_root(resolved):
                    private, seg = has_private_segment(resolved)
                    if private:
                        key = _short_expr(node.slice)
                        self._add(
                            node, "registry",
                            ".".join(resolved),
                            extra_context=(
                                f"{'.'.join(chain)}[{key}]"
                            ),
                        )
                        return
            # Case B: heuristic — chain rooted at 'V' (inductor virtual
            # environment) or ending in a private segment.
            if root == "V":
                private, seg = has_private_segment(chain)
                if private or len(chain) > 1:
                    key = _short_expr(node.slice)
                    self._add(
                        node, "registry",
                        ".".join(chain),
                        extra_context=f"{'.'.join(chain)}[{key}]",
                    )
                    return
            # Case C: a bare Name that was imported from torch._... and now
            # indexed (e.g. `lowerings[foo]` after `from torch._inductor.lowering
            # import lowerings`).
            if len(chain) == 1 and root in self.torch_aliases:
                aliased = self.torch_aliases[root]
                if any(p.startswith("_") for p in aliased[1:]):
                    key = _short_expr(node.slice)
                    self._add(
                        node, "registry",
                        ".".join(aliased),
                        extra_context=f"{root}[{key}]  # from {'.'.join(aliased)}",
                    )
                    return
        self.generic_visit(node)


def _short_expr(node: ast.AST) -> str:
    """Best-effort short repr of a small expression node for context."""
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


# ---------------------------------------------------------------------------
# Comment / token-based passes
# ---------------------------------------------------------------------------


def scan_type_ignores(
    rel_path: str, source: str, lines: List[str]
) -> List[Dict]:
    hits: List[Dict] = []
    try:
        tokens = list(tokenize.generate_tokens(iter(source.splitlines(True)).__next__))
    except tokenize.TokenizeError:
        return hits
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        m = RE_TYPE_IGNORE.search(tok.string)
        if not m:
            continue
        codes = m.group(1) or ""
        codes_norm = codes.strip()
        # Split codes and emit one hit per code, or a single hit if bare
        # `# type: ignore`.
        lineno = tok.start[0]
        snippet = make_snippet(lines, lineno)
        if not codes_norm:
            hits.append(
                {
                    "file": rel_path,
                    "line": lineno,
                    "kind": "type-ignore",
                    "private_name": "<bare>",
                    "snippet": snippet,
                    "context": tok.string.strip(),
                }
            )
            continue
        for raw_code in codes_norm.split(","):
            code = raw_code.strip()
            if not code:
                continue
            hits.append(
                {
                    "file": rel_path,
                    "line": lineno,
                    "kind": "type-ignore",
                    "private_name": code,
                    "snippet": snippet,
                    "context": tok.string.strip(),
                }
            )
    return hits


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # Skip vendored/third-party trees or virtualenvs if any.
        parts = set(p.parts)
        if any(seg in parts for seg in (".venv", "venv", "node_modules")):
            continue
        yield p


def scan_file(root: Path, path: Path) -> List[Dict]:
    source = read_source(path)
    if source is None:
        return []
    lines = source.splitlines()
    rel_path = str(path.relative_to(root))
    hits: List[Dict] = []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # Still run comment scan on parseable-line-by-line basis.
        hits.extend(scan_type_ignores(rel_path, source, lines))
        return hits
    visitor = PrivateAPIVisitor(rel_path, lines)
    visitor.visit(tree)
    hits.extend(visitor.hits)
    hits.extend(scan_type_ignores(rel_path, source, lines))
    return hits


def dedupe(hits: List[Dict]) -> List[Dict]:
    """Collapse exact duplicates (same file, line, kind, private_name)."""
    seen = set()
    out: List[Dict] = []
    for h in hits:
        key = (h["file"], h["line"], h["kind"], h["private_name"])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def build_hotspots(hits: List[Dict], top: int = 10) -> List[Dict]:
    counts: Dict[str, int] = {}
    for h in hits:
        counts[h["file"]] = counts.get(h["file"], 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"file": f, "hits": n} for f, n in ordered[:top]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"error: root not a directory: {root}", file=sys.stderr)
        return 2

    # Restrict scan to the torch_spyre package tree (per audit scope).
    scan_roots = [root / "torch_spyre"]
    # If the package dir doesn't exist, fall back to whole tree.
    if not scan_roots[0].is_dir():
        scan_roots = [root]

    all_hits: List[Dict] = []
    scanned = 0
    for base in scan_roots:
        for path in iter_py_files(base):
            scanned += 1
            all_hits.extend(scan_file(root, path))

    all_hits = dedupe(all_hits)
    all_hits.sort(key=lambda h: (h["file"], h["line"], h["kind"]))
    hotspots = build_hotspots(all_hits, top=10)

    result = {
        "scanner": SCANNER_NAME,
        "torch_spyre_sha": TORCH_SPYRE_SHA,
        "scanned_files": scanned,
        "total_hits": len(all_hits),
        "hotspots": hotspots,
        "hits": all_hits,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- stdout summary --------------------------------------------------
    print(f"scanner: {SCANNER_NAME}")
    print(f"sha: {TORCH_SPYRE_SHA}")
    print(f"scanned files: {scanned}")
    print(f"total hits (deduped): {len(all_hits)}")
    print()
    print(f"top {args.top} hits (file:line kind private_name):")
    for h in all_hits[: args.top]:
        print(
            f"  {h['file']}:{h['line']}  [{h['kind']}]  {h['private_name']}"
        )
    print()
    print("hotspots (top 10 files by hit count):")
    for hs in hotspots:
        print(f"  {hs['hits']:5d}  {hs['file']}")
    print()
    print(f"json: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
