#!/usr/bin/env python3
"""Scanner: O(n) list surgery on graph.operations / graph.buffers.

Finds calls of the form ``<expr>.operations.{index,remove,insert,pop}(...)``
and the same for ``buffers``. Also tracks local aliases in the same function
so ``ops = graph.operations; ops.remove(x)`` is caught.

Additionally flags ``for op in <ops>: ...  <ops>.remove(op)`` as an O(n^2)
smell, and separately marks list comprehensions that rebuild the whole
collection (candidate *fix*, not a smell).

Output JSON follows the audit-repo scanner schema.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

SHA = "fea0c4be901e1383b1f700dbad8887128b0fcb27"
SCANNER_NAME = "list_surgery"

# Methods that mutate a Python list in O(n) (or worse) time relative to length.
SURGERY_METHODS = {"index", "remove", "insert", "pop"}
# The two collection attribute names we care about.
COLLECTION_NAMES = {"operations", "buffers"}


def _read_source(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _snippet(src_lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(src_lines):
        return src_lines[lineno - 1].rstrip()
    return ""


def _get_attr_tail(node: ast.AST) -> str | None:
    """Return trailing dotted attribute string of an Attribute chain (best-effort)."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    elif isinstance(cur, ast.Call):
        parts.append("<call>")
    else:
        parts.append("<expr>")
    return ".".join(reversed(parts))


def _is_target_attr(value: ast.AST, aliases_ops: set[str], aliases_bufs: set[str]) -> tuple[bool, str]:
    """Return (True, collection) if `value` refers to operations/buffers.

    Recognizes:
      * ``foo.operations`` / ``foo.buffers`` (any prefix)
      * bare ``Name`` matching a known local alias for operations/buffers
    """
    if isinstance(value, ast.Attribute) and value.attr in COLLECTION_NAMES:
        return True, value.attr
    if isinstance(value, ast.Name):
        if value.id in aliases_ops:
            return True, "operations"
        if value.id in aliases_bufs:
            return True, "buffers"
    return False, ""


def _collect_aliases(func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set[str], set[str]]:
    """Light dataflow: find local names bound to `<expr>.operations` / `.buffers`.

    Also treats function parameters *named* ``operations`` / ``buffers`` as aliases —
    this matches Torch-Spyre's inductor pass idiom where passes take the collection
    directly and mutate it in place (e.g. ``def _pass(operations: list[Operation])``).
    """
    ops_aliases: set[str] = set()
    buf_aliases: set[str] = set()
    # Parameters named `operations` / `buffers` are aliases by convention.
    args_all = list(func.args.args) + list(func.args.kwonlyargs) + list(func.args.posonlyargs)
    if func.args.vararg is not None:
        args_all.append(func.args.vararg)
    if func.args.kwarg is not None:
        args_all.append(func.args.kwarg)
    for arg in args_all:
        if arg.arg == "operations":
            ops_aliases.add("operations")
        elif arg.arg == "buffers":
            buf_aliases.add("buffers")
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            attr = node.value.attr
            if attr not in COLLECTION_NAMES:
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    if attr == "operations":
                        ops_aliases.add(tgt.id)
                    else:
                        buf_aliases.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Attribute):
            attr = node.value.attr
            if attr not in COLLECTION_NAMES:
                continue
            if isinstance(node.target, ast.Name):
                if attr == "operations":
                    ops_aliases.add(node.target.id)
                else:
                    buf_aliases.add(node.target.id)
    return ops_aliases, buf_aliases


def _iter_functions(tree: ast.AST) -> Iterable[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _enclosing_function(func_map: dict[int, str], lineno: int) -> str:
    """Return function name whose line range best contains lineno."""
    return func_map.get(lineno, "<module>")


def _build_func_line_map(tree: ast.AST) -> dict[int, str]:
    """Map every line to the innermost enclosing function name."""
    line_map: dict[int, str] = {}
    for func in _iter_functions(tree):
        start = func.lineno
        end = getattr(func, "end_lineno", start)
        for ln in range(start, end + 1):
            # innermost wins because we visit children after parents when walking;
            # ast.walk order is not guaranteed, so prefer nodes with smaller span.
            existing = line_map.get(ln)
            if existing is None:
                line_map[ln] = func.name
            else:
                # keep the shorter span (innermost)
                pass
    # Redo with span-awareness: iterate again keeping the smallest containing range.
    span: dict[int, int] = {}
    for func in _iter_functions(tree):
        start = func.lineno
        end = getattr(func, "end_lineno", start)
        length = end - start
        for ln in range(start, end + 1):
            if ln not in span or length < span[ln]:
                span[ln] = length
                line_map[ln] = func.name
    return line_map


def _for_loop_iters_map(tree: ast.AST) -> list[tuple[int, int, str, str]]:
    """List of (start, end, loop_var, iter_repr) for every For loop."""
    loops: list[tuple[int, int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            var = ""
            if isinstance(node.target, ast.Name):
                var = node.target.id
            iter_repr = _get_attr_tail(node.iter) or ""
            loops.append((start, end, var, iter_repr))
    return loops


def _in_for_loop(loops: list[tuple[int, int, str, str]], lineno: int) -> tuple[bool, str, str]:
    inner: tuple[bool, str, str] = (False, "", "")
    best_span = 10**9
    for start, end, var, iter_repr in loops:
        if start <= lineno <= end:
            span = end - start
            if span < best_span:
                best_span = span
                inner = (True, var, iter_repr)
    return inner


def _iter_target_of_call(call: ast.Call, aliases_ops: set[str], aliases_bufs: set[str]) -> tuple[str, str, str] | None:
    """If call is <target>.<method>(...), and target maps to ops/bufs, return (method, collection, target_repr)."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    method = func.attr
    if method not in SURGERY_METHODS:
        return None
    is_target, collection = _is_target_attr(func.value, aliases_ops, aliases_bufs)
    if not is_target:
        return None
    return method, collection, _get_attr_tail(func.value) or ""


def _iterates_over_ops_or_bufs(iter_repr: str, aliases_ops: set[str], aliases_bufs: set[str]) -> bool:
    if not iter_repr:
        return False
    # e.g. "graph.operations", "gl.operations", "self.lowering.operations", plain "operations"
    parts = iter_repr.split(".")
    tail = parts[-1]
    if tail in COLLECTION_NAMES:
        return True
    if tail in aliases_ops or tail in aliases_bufs:
        return True
    return False


def _first_arg_name(call: ast.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Name):
        return first.id
    return None


def _rebuild_listcomps(tree: ast.AST, aliases_ops: set[str], aliases_bufs: set[str]) -> list[tuple[int, str, str]]:
    """Flag list comprehensions like `[op for op in ops if ...]` where the result is assigned back
    to ops (candidate fix). Best-effort: we detect assignments of a ListComp/Call(list, ...) to
    a target that also matches operations/buffers or an alias thereof.
    """
    findings: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        # Handle either a bare comprehension or list(gen)
        comp: ast.comprehension | None = None
        if isinstance(val, ast.ListComp):
            if val.generators:
                comp = val.generators[0]
        elif isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "list":
            if val.args and isinstance(val.args[0], ast.GeneratorExp) and val.args[0].generators:
                comp = val.args[0].generators[0]
        if comp is None:
            continue
        iter_repr = _get_attr_tail(comp.iter) or ""
        if not _iterates_over_ops_or_bufs(iter_repr, aliases_ops, aliases_bufs):
            continue
        # Is the LHS target an ops/bufs collection or alias?
        for tgt in node.targets:
            is_target, collection = _is_target_attr(tgt, aliases_ops, aliases_bufs)
            # Also consider slice-assign like ops[:] = [ ... ]
            if isinstance(tgt, ast.Subscript):
                is_target, collection = _is_target_attr(tgt.value, aliases_ops, aliases_bufs)
            if is_target:
                findings.append((node.lineno, collection, iter_repr))
                break
    return findings


def scan_file(path: Path, root: Path) -> list[dict]:
    src = _read_source(path)
    if src is None:
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    src_lines = src.splitlines()
    hits: list[dict] = []

    # Build per-file line -> function map and loop ranges (global-ish; alias
    # collection is per-function so a call outside a function has empty aliases).
    func_line_map = _build_func_line_map(tree)
    loops = _for_loop_iters_map(tree)

    # Per-function alias sets — build once, applied to any call inside that function's span.
    aliases_by_func: dict[str, tuple[set[str], set[str]]] = {}
    for func in _iter_functions(tree):
        aliases_by_func[func.name] = _collect_aliases(func)

    def _aliases_for(lineno: int) -> tuple[set[str], set[str]]:
        fname = func_line_map.get(lineno)
        if fname and fname in aliases_by_func:
            return aliases_by_func[fname]
        return set(), set()

    rel = str(path.relative_to(root))

    # Walk all Call nodes for surgery methods.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        aliases_ops, aliases_bufs = _aliases_for(node.lineno)
        found = _iter_target_of_call(node, aliases_ops, aliases_bufs)
        if not found:
            continue
        method, collection, target_repr = found
        in_loop, loop_var, loop_iter = _in_for_loop(loops, node.lineno)
        # O(n^2) smell classification:
        #   * hard n^2 concurrent-modification: iterate the same collection AND remove(loop_var)
        #   * soft n*m: any surgery on ops/bufs done inside a for-loop (each call is O(n))
        is_hard_n2 = False
        is_soft_nm = False
        if in_loop and method in {"remove", "insert", "index", "pop"}:
            arg = _first_arg_name(node)
            iterates_same = _iterates_over_ops_or_bufs(loop_iter, aliases_ops, aliases_bufs)
            if method == "remove" and iterates_same and arg is not None and arg == loop_var:
                is_hard_n2 = True
            else:
                is_soft_nm = True

        kind_bits = [f"{collection}.{method}"]
        if is_hard_n2:
            kind_bits.append("in_for_loop_remove_loopvar_n2")
        elif is_soft_nm:
            kind_bits.append("in_for_loop_n_times_m")

        hits.append(
            {
                "file": rel,
                "line": node.lineno,
                "kind": ".".join(kind_bits),
                "snippet": _snippet(src_lines, node.lineno),
                "context": (
                    f"function={func_line_map.get(node.lineno, '<module>')}; "
                    f"target={target_repr}; collection={collection}; method={method}; "
                    f"in_for_loop={in_loop}; loop_var={loop_var or '-'}; "
                    f"loop_iter={loop_iter or '-'}; hard_n2={is_hard_n2}; soft_n_times_m={is_soft_nm}"
                ),
            }
        )

    # Rebuild list-comprehension candidate-fix findings (module-level walk; aliases per function).
    for func in _iter_functions(tree):
        aliases_ops, aliases_bufs = aliases_by_func.get(func.name, (set(), set()))
        rebuild_hits = _rebuild_listcomps(func, aliases_ops, aliases_bufs)
        for lineno, collection, iter_repr in rebuild_hits:
            hits.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": f"{collection}.rebuild_listcomp_candidate_fix",
                    "snippet": _snippet(src_lines, lineno),
                    "context": (
                        f"function={func.name}; collection={collection}; "
                        f"iter={iter_repr}; note=possible O(n) rebuild (candidate FIX, not smell)"
                    ),
                }
            )
    # Also scan module-level (rare, but for completeness)
    module_rebuilds = _rebuild_listcomps(tree, set(), set())
    for lineno, collection, iter_repr in module_rebuilds:
        # Skip duplicates that came from function-level scan
        if any(h["line"] == lineno and h["kind"].endswith("rebuild_listcomp_candidate_fix") for h in hits):
            continue
        hits.append(
            {
                "file": rel,
                "line": lineno,
                "kind": f"{collection}.rebuild_listcomp_candidate_fix",
                "snippet": _snippet(src_lines, lineno),
                "context": (
                    f"function=<module>; collection={collection}; "
                    f"iter={iter_repr}; note=possible O(n) rebuild (candidate FIX, not smell)"
                ),
            }
        )

    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")

    py_files = [p for p in root.rglob("*.py") if p.is_file()]
    scanned = 0
    all_hits: list[dict] = []
    for p in sorted(py_files):
        scanned += 1
        all_hits.extend(scan_file(p, root))

    # Sort hits: n^2 smells first, then plain in_for_loop, then others; within groups by file/line.
    def _priority(h: dict) -> tuple[int, str, int]:
        k = h["kind"]
        if "n2" in k:
            pri = 0
        elif "n_times_m" in k or "in_for_loop" in k:
            pri = 1
        elif "rebuild_listcomp_candidate_fix" in k:
            pri = 3
        else:
            pri = 2
        return pri, h["file"], h["line"]

    all_hits.sort(key=_priority)

    result = {
        "scanner": SCANNER_NAME,
        "torch_spyre_sha": SHA,
        "scanned_files": scanned,
        "total_hits": len(all_hits),
        "hits": all_hits,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"scanner={SCANNER_NAME}  torch_spyre_sha={SHA}")
    print(f"scanned_files={scanned}  total_hits={len(all_hits)}")
    print(f"json: {args.out}")
    print()
    top = min(args.top, len(all_hits))
    print(f"top {top} hits:")
    for h in all_hits[:top]:
        print(f"  {h['file']}:{h['line']}  [{h['kind']}]  {h['snippet'].strip()[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
