#!/usr/bin/env python3
"""
test_smells.py — Scan the torch-spyre test suite for smells that leave
invariants unprotected.

Categories detected (see README/audit doc for context):
  - noraise_only        : test body only calls a function / does an op with NO
                          assert / assert_close / assertTrue / pytest.raises —
                          i.e. "did not raise" tests with no output check.
  - raises_no_match     : test uses pytest.raises / assertRaises but without a
                          ``match=`` regex — exception class alone is a weak
                          spec (a rename or shim that raises the same class
                          silently passes).
  - skip_marker         : @pytest.mark.skip / @pytest.mark.skipif / @unittest.skip*
                          — reason string captured when present.
  - xfail_marker        : @pytest.mark.xfail — reason string captured when present.
  - mock_upstream       : mock.patch / MagicMock targeting torch internals or
                          lowering functions (potential drift from upstream).
  - shape_dtype_only    : assertions that only check .shape or .dtype (no value).
  - allclose_no_tol     : torch.allclose(...) call without rtol= or atol= kwarg.
  - compile_missing     : file/class name suggests compile testing but body
                          never calls torch.compile / @torch.compile / dynamo.
  - broad_except_pass   : bare ``except Exception: pass`` (or ``: ...``) in
                          test setup/teardown / fixture context.

Stdlib only. Reads pinned torch-spyre worktree.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Iterable

SHA = "fea0c4be901e1383b1f700dbad8887128b0fcb27"


# ---------- helpers -----------------------------------------------------------


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def line_of(src_lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(src_lines):
        return src_lines[lineno - 1].strip()
    return ""


def rel_of(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def get_decorator_name(dec: ast.expr) -> str:
    """Return dotted name for a decorator (with or without call)."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def get_kw_string(call: ast.Call, key: str) -> str | None:
    for kw in call.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def first_string_arg(call: ast.Call) -> str | None:
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def has_kwarg(call: ast.Call, keys: Iterable[str]) -> bool:
    ks = set(keys)
    return any(kw.arg in ks for kw in call.keywords)


ASSERT_NAMES = {
    "assert_close",
    "assert_allclose",
    "assert_equal",
    "assert_array_equal",
    "assert_array_almost_equal",
    "assertEqual",
    "assertAlmostEqual",
    "assertTrue",
    "assertFalse",
    "assertIs",
    "assertIsNone",
    "assertIsNotNone",
    "assertIn",
    "assertNotIn",
    "assertGreater",
    "assertLess",
    "assertGreaterEqual",
    "assertLessEqual",
    "assertRaises",
    "assertRaisesRegex",
    "assertListEqual",
    "assertDictEqual",
    "assertTupleEqual",
    "assertSetEqual",
    "assertIsInstance",
    "assertRegex",
}


def is_assertion_call(call: ast.Call) -> bool:
    """True if a Call node looks like an assertion helper."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in ASSERT_NAMES:
        return True
    if isinstance(func, ast.Name) and func.id in ASSERT_NAMES:
        return True
    return False


def collect_asserts(body: list[ast.stmt]) -> list[ast.AST]:
    """All Assert stmts / assert-like calls (recursive) in a body.

    Includes torch ``FileCheck`` DSL — any expression that constructs or names
    a ``FileCheck`` instance is treated as a value check because the terminal
    ``.run()`` on the chain raises on mismatch.
    """
    hits: list[ast.AST] = []
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assert):
                hits.append(sub)
            elif isinstance(sub, ast.Call):
                if is_assertion_call(sub):
                    hits.append(sub)
                    continue
                # FileCheck() constructor
                fn = sub.func
                if isinstance(fn, ast.Name) and fn.id == "FileCheck":
                    hits.append(sub)
                elif isinstance(fn, ast.Attribute) and fn.attr == "FileCheck":
                    hits.append(sub)
    return hits


def is_test_function(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return fn.name.startswith("test_") or fn.name == "test"


COMPILE_HINTS = re.compile(r"(compile|inductor|dynamo|graph|fx|jit)", re.IGNORECASE)


def file_name_suggests_compile(path: Path) -> bool:
    return bool(COMPILE_HINTS.search(path.name))


# ---------- detectors ---------------------------------------------------------


def find_skip_xfail(tree: ast.AST, src_lines: list[str], rel: str) -> list[dict]:
    hits: list[dict] = []

    def scan_decs(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        for dec in node.decorator_list:
            name = get_decorator_name(dec)
            call = dec if isinstance(dec, ast.Call) else None
            reason = ""
            kind: str | None = None
            if name in {
                "pytest.mark.skip",
                "pytest.mark.skipif",
                "unittest.skip",
                "unittest.skipIf",
                "unittest.skipUnless",
                "skip",
                "skipIf",
                "skipUnless",
                "skipif",
            }:
                kind = "skip_marker"
            elif name in {"pytest.mark.xfail", "xfail"}:
                kind = "xfail_marker"
            if kind is None:
                continue
            if call is not None:
                reason = get_kw_string(call, "reason") or ""
                if not reason:
                    s = first_string_arg(call)
                    # skipIf/skipUnless: first positional is the condition, not
                    # the reason. Only take a first positional string when the
                    # marker family is a plain skip/xfail.
                    if s and name in {"pytest.mark.skip", "pytest.mark.xfail", "unittest.skip", "skip", "xfail"}:
                        reason = s
            hits.append(
                {
                    "file": rel,
                    "line": dec.lineno,
                    "kind": kind,
                    "snippet": line_of(src_lines, dec.lineno),
                    "context": f"{name} on {getattr(node, 'name', '?')} :: reason={reason!r}",
                }
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scan_decs(node)
    return hits


_UPSTREAM_TARGET_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_])(?:"
    r"torch\._|torch\.fx|torch\._dynamo|torch\._inductor|torch\.compile|torch\.jit|torch\.overrides"
    r"|lowering|lower_"
    r"|codegen|Scheduler|fusion|inductor|dynamo"
    r"|aten\.|_C\.|_prims\.|_refs\.|_decomp"
    r"|torch_spyre\._|torch_spyre\.compile|torch_spyre\.lowering|torch_spyre\.inductor"
    r")"
)


def find_mock_upstream(tree: ast.AST, src_lines: list[str], rel: str) -> list[dict]:
    hits: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = get_decorator_name(node)
        patch_names = {
            "mock.patch",
            "mock.patch.object",
            "unittest.mock.patch",
            "patch",
            "patch.object",
        }
        mock_names = {"MagicMock", "mock.MagicMock", "Mock", "mock.Mock"}
        is_patch = name in patch_names
        is_mock = name in mock_names
        if not (is_patch or is_mock):
            continue

        target = ""
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            target = node.args[0].value

        if is_patch:
            if not target:
                continue
            if not _UPSTREAM_TARGET_RE.search(target):
                continue
        else:
            # For MagicMock()/Mock() require the surrounding source line to
            # mention a compiler/lowering surface — otherwise it's just a
            # generic stub. Anchored on the same line to keep the check cheap.
            snip = line_of(src_lines, node.lineno)
            if not _UPSTREAM_TARGET_RE.search(snip):
                continue

        hits.append(
            {
                "file": rel,
                "line": node.lineno,
                "kind": "mock_upstream",
                "snippet": line_of(src_lines, node.lineno),
                "context": f"{name}({target!r})" if target else name,
            }
        )
    return hits


ATTR_ONLY_ASSERT = re.compile(r"^\s*(?:assert|self\.assert\w+\()\s*.*\.(shape|dtype|device|ndim|numel)\b")


def find_shape_dtype_only(tree: ast.AST, src_lines: list[str], rel: str) -> list[dict]:
    hits: list[dict] = []
    for node in ast.walk(tree):
        # bare `assert x.shape == (...)`
        if isinstance(node, ast.Assert):
            snip = line_of(src_lines, node.lineno)
            if ATTR_ONLY_ASSERT.match(snip) and "allclose" not in snip and "close" not in snip:
                hits.append(
                    {
                        "file": rel,
                        "line": node.lineno,
                        "kind": "shape_dtype_only",
                        "snippet": snip,
                        "context": "assert only checks tensor metadata",
                    }
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"assertEqual", "assertIs"}:
                # look for one arg that is x.shape / x.dtype
                for a in node.args:
                    if isinstance(a, ast.Attribute) and a.attr in {"shape", "dtype", "device", "ndim", "numel"}:
                        hits.append(
                            {
                                "file": rel,
                                "line": node.lineno,
                                "kind": "shape_dtype_only",
                                "snippet": line_of(src_lines, node.lineno),
                                "context": f"assertEqual on .{a.attr}",
                            }
                        )
                        break
    return hits


def find_allclose_no_tol(tree: ast.AST, src_lines: list[str], rel: str) -> list[dict]:
    hits: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
        if func_name != "allclose":
            continue
        # confirm it's the torch.allclose family (dotted "allclose")
        if not has_kwarg(node, {"rtol", "atol"}):
            hits.append(
                {
                    "file": rel,
                    "line": node.lineno,
                    "kind": "allclose_no_tol",
                    "snippet": line_of(src_lines, node.lineno),
                    "context": "no rtol/atol kwarg — falls back to torch defaults",
                }
            )
    return hits


_HELPER_PREFIXES = (
    "_test_", "_check_", "_assert_", "_verify_", "_run_", "_do_", "_stage_",
    "check_", "verify_", "compare_", "assert_",
    "run_",
)
_HELPER_SUFFIXES = (
    "_helper", "_impl", "_check", "_assert", "_verify", "_and_run", "_test",
)
_HELPER_EXACT = {
    "compare_with_cpu",
    "compare_with_pytorch",
    "compare_with_reference",
    "_compile_and_run",
    "compile_and_run",
    "run_and_compare",
    "check_output",
    "check_result",
}


def _is_helper_call_name(name: str) -> bool:
    if not name:
        return False
    if name in _HELPER_EXACT:
        return True
    if name.startswith(_HELPER_PREFIXES):
        return True
    if name.endswith(_HELPER_SUFFIXES):
        return True
    return False


def _call_name(call: ast.Call) -> str:
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


def _delegates_to_helper(stmts: list[ast.stmt]) -> bool:
    """True if the body's *last* effective statement is a call to a helper
    that plausibly contains its own asserts.

    Handles both single-statement wrappers (``self._test_helper(...)``) and
    tests that prep some inputs then delegate (``x = torch.randn(...); ...;
    compare_with_cpu(fn, x)``)."""
    if not stmts:
        return False
    last = stmts[-1]
    call: ast.Call | None = None
    if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call):
        call = last.value
    elif isinstance(last, ast.Return) and isinstance(last.value, ast.Call):
        call = last.value
    if call is None:
        return False
    return _is_helper_call_name(_call_name(call))


def _uses_pytest_raises(fn: ast.AST) -> tuple[bool, bool]:
    """Return (uses_raises, has_match_kwarg).

    ``pytest.raises`` / ``self.assertRaises`` / ``self.assertRaisesRegex``
    inside the body counts as behavior checking.
    """
    uses = False
    has_match = False
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        fname = get_decorator_name(sub)
        if fname in {"pytest.raises", "raises"}:
            uses = True
            if has_kwarg(sub, {"match"}):
                has_match = True
        elif fname.endswith("assertRaises"):
            uses = True
        elif fname.endswith("assertRaisesRegex") or fname.endswith("assertRaisesRegexp"):
            uses = True
            has_match = True
    return uses, has_match


def find_noraise_only(tree: ast.AST, src_lines: list[str], rel: str) -> list[dict]:
    """Tests whose body has zero assertions and no return-value comparison.

    Excludes:
      - trivial `pass` / docstring-only stubs (placeholder, not smell)
      - single-statement delegations to a helper (self._test_allgather_helper)
      - tests wrapped in `pytest.raises` / `assertRaises*` — those DO test
        behavior (an expected exception). Weak forms (no ``match=``) are
        surfaced as a separate ``raises_no_match`` smell instead.
    """
    hits: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not is_test_function(node):
            continue
        asserts = collect_asserts(node.body)
        if asserts:
            continue
        stmts = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if not stmts:
            continue
        if _delegates_to_helper(stmts):
            continue
        uses_raises, has_match = _uses_pytest_raises(node)
        if uses_raises:
            if not has_match:
                hits.append(
                    {
                        "file": rel,
                        "line": node.lineno,
                        "kind": "raises_no_match",
                        "snippet": line_of(src_lines, node.lineno),
                        "context": (
                            f"{node.name}: pytest.raises/assertRaises without match=; "
                            f"exception class alone is a weak spec"
                        ),
                    }
                )
            continue
        hits.append(
            {
                "file": rel,
                "line": node.lineno,
                "kind": "noraise_only",
                "snippet": line_of(src_lines, node.lineno),
                "context": f"{node.name}: {len(stmts)} stmt(s), no assert / assert_close / assertX call",
            }
        )
    return hits


def find_compile_missing(tree: ast.AST, path: Path, src_lines: list[str], rel: str, src_text: str) -> list[dict]:
    if not file_name_suggests_compile(path):
        return []
    # If the file uses torch.compile / dynamo anywhere, no hit.
    if re.search(r"torch\.compile\b|@compile\b|torch\._dynamo|dynamo\.optimize", src_text):
        return []
    # Only complain if the file has actual test functions.
    test_fns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_test_function(n)
    ]
    if not test_fns:
        return []
    first = test_fns[0]
    return [
        {
            "file": rel,
            "line": first.lineno,
            "kind": "compile_missing",
            "snippet": line_of(src_lines, first.lineno),
            "context": (
                f"filename suggests compile/inductor/dynamo but no torch.compile call "
                f"(tests: {len(test_fns)})"
            ),
        }
    ]


def find_broad_except_pass(tree: ast.AST, src_lines: list[str], rel: str) -> list[dict]:
    hits: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # broad? bare "except:" or "except Exception[, BaseException]"
        broad = False
        if node.type is None:
            broad = True
        elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            broad = True
        if not broad:
            continue
        # body is only pass / Ellipsis / log-only? treat as smell
        body = node.body
        trivial = all(
            isinstance(s, ast.Pass)
            or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
            for s in body
        )
        if not trivial:
            continue
        hits.append(
            {
                "file": rel,
                "line": node.lineno,
                "kind": "broad_except_pass",
                "snippet": line_of(src_lines, node.lineno),
                "context": f"except {'Exception' if node.type else '(bare)'}: pass",
            }
        )
    return hits


# ---------- main --------------------------------------------------------------


def scan_file(path: Path, root: Path) -> tuple[list[dict], bool]:
    src_text = read_text(path)
    if not src_text:
        return [], False
    try:
        tree = ast.parse(src_text, filename=str(path))
    except SyntaxError:
        return [], False
    src_lines = src_text.splitlines()
    rel = rel_of(path, root)

    hits: list[dict] = []
    hits += find_skip_xfail(tree, src_lines, rel)
    hits += find_mock_upstream(tree, src_lines, rel)
    hits += find_shape_dtype_only(tree, src_lines, rel)
    hits += find_allclose_no_tol(tree, src_lines, rel)
    hits += find_noraise_only(tree, src_lines, rel)
    hits += find_compile_missing(tree, path, src_lines, rel, src_text)
    hits += find_broad_except_pass(tree, src_lines, rel)
    return hits, True


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan torch-spyre tests for smells.")
    ap.add_argument("--root", required=True, help="torch-spyre worktree root")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--top", type=int, default=20, help="stdout top-N summary size")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        print(f"error: {tests_dir} not found", file=sys.stderr)
        return 2

    all_hits: list[dict] = []
    scanned = 0
    for py in sorted(tests_dir.rglob("*.py")):
        hits, ok = scan_file(py, root)
        if ok:
            scanned += 1
        all_hits.extend(hits)

    # Priority ordering for the top-N so severe smells surface first.
    kind_rank = {
        "noraise_only": 0,
        "compile_missing": 1,
        "broad_except_pass": 2,
        "mock_upstream": 3,
        "raises_no_match": 4,
        "allclose_no_tol": 5,
        "shape_dtype_only": 6,
        "xfail_marker": 7,
        "skip_marker": 8,
    }
    all_hits.sort(key=lambda h: (kind_rank.get(h["kind"], 99), h["file"], h["line"]))

    payload = {
        "scanner": "test_smells",
        "torch_spyre_sha": SHA,
        "scanned_files": scanned,
        "total_hits": len(all_hits),
        "hits": all_hits,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    # Category counts
    counts: dict[str, int] = {}
    for h in all_hits:
        counts[h["kind"]] = counts.get(h["kind"], 0) + 1

    print(f"scanner        : test_smells")
    print(f"torch_spyre_sha: {SHA}")
    print(f"scanned_files  : {scanned}")
    print(f"total_hits     : {len(all_hits)}")
    print("by category    :")
    for k in sorted(counts, key=lambda x: (-counts[x], x)):
        print(f"  {k:<20s} {counts[k]}")
    print(f"\ntop {args.top} (severity-then-path):")
    for h in all_hits[: args.top]:
        print(f"  {h['file']}:{h['line']}  [{h['kind']}]  {h['context']}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
