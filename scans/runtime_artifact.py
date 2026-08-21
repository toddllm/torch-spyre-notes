#!/usr/bin/env python3
"""Scan a Torch-Spyre runtime artifact (LLIR/OpSpec/bundle JSON) for
patterns worth investigating.

Torch-Spyre lowers a GraphLowering into a sequence of SDSC bundles: each
bundle carries a topologically-ordered list of ops, plus per-op buffer
reads and writes, an op kind (compute / restickify / copy / fallback /
constant / empty), a device residency label (HBM or LX), and, for
constants, a hash of the materialized bytes. When a real artifact
becomes available its concrete schema will differ — this scanner
therefore accepts a small, stable synthetic JSON schema documented
below, and each check is written against that schema so it can be
retargeted at the real artifact by editing the field names in one place.

Synthetic artifact schema (single-file JSON):

    {
      "artifact":   "<name>",
      "description":"<free text>",       # optional, echoed into results
      "buffers":  {                       # optional: known buffer metadata
        "<buf>":  {"residency": "hbm" | "lx", "bytes": <int>}
      },
      "bundles":  [
        { "id":   "<bundle-id>",
          "ops": [
            { "id":        "<op-id>",
              "kind":      "compute" | "copy" | "restickify"
                          | "fallback" | "constant" | "empty",
              "op_name":   "<optional aten/spyre op name>",
              "reads":     ["<buf>", ...],
              "writes":    ["<buf>", ...],
              "residency": "hbm" | "lx",  # of the primary output buffer
              "const_hash":"<sha256 of materialised bytes>"  # constants only
            }, ...
          ]
        }, ...
      ]
    }

Checks (each emits an ``issues`` entry keyed by ``kind``):

  * ``restickify_restickify_same_buffer`` — two restickify ops in a row
    that operate on the same buffer (writer==writer, or writer==reader
    across adjacent ops).  A committed layout should be reached in one
    step; a second restickify undoes or refines the first.

  * ``chained_copy`` — copy A -> B immediately followed by copy B -> C
    (with no compute op consuming B in between).  Candidate for a
    single copy A -> C.

  * ``cpu_fallback`` — any op whose ``kind`` is ``fallback`` or whose
    ``op_name`` contains ``cpu``.  Fallbacks are the primary source of
    unmodeled overhead in Torch-Spyre today.

  * ``duplicate_constant`` — two or more ``constant`` ops that share the
    same ``const_hash``.  ``dedup_and_promote_constants`` is supposed to
    fold these; any duplicate reaching the artifact is a leak.

  * ``singleton_bundle`` — a bundle containing exactly one op (excluding
    empty/constant seeds).  Fragmentation costs a full launch per op.

  * ``hbm_lx_transfer_with_lx_intermediate`` — a HBM->LX or LX->HBM copy
    where an LX-resident intermediate already exists in the same bundle
    covering the same buffer name; the transfer likely repeats what a
    prior LX write produced.

The scanner is intentionally over-eager: every hit is a *candidate*, and
false positives are acceptable so long as no true positive is silenced
(see scans/README.md).  A downstream reviewer promotes hits to findings.

Usage
-----
::

    python3 scans/runtime_artifact.py \
        --artifact scans/fixtures/runtime/mixed.json \
        --out scans/results/runtime_artifact.json

Multiple ``--artifact`` flags collect several artifacts into one report.
Passing a directory recurses over ``*.json`` inside it.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Field names — grouped in one place so real-artifact schema drift needs
# only edits here, not in every check.
# ---------------------------------------------------------------------------

BUNDLES_KEY = "bundles"
OPS_KEY = "ops"
KIND_KEY = "kind"
OP_ID_KEY = "id"
OP_NAME_KEY = "op_name"
READS_KEY = "reads"
WRITES_KEY = "writes"
RESIDENCY_KEY = "residency"
CONST_HASH_KEY = "const_hash"
BUFFERS_KEY = "buffers"

KIND_COMPUTE = "compute"
KIND_COPY = "copy"
KIND_RESTICKIFY = "restickify"
KIND_FALLBACK = "fallback"
KIND_CONSTANT = "constant"
KIND_EMPTY = "empty"

RESIDENCY_HBM = "hbm"
RESIDENCY_LX = "lx"


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    kind: str
    bundle: str
    ops: list[str]
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FixtureResult:
    artifact: str
    description: str = ""
    issues: list[Issue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["issues"] = [i.as_dict() if isinstance(i, Issue) else i for i in self.issues]
        return d


# ---------------------------------------------------------------------------
# Individual checks — each takes the parsed bundle and appends Issues.
# ---------------------------------------------------------------------------


def _op_id(op: dict) -> str:
    return str(op.get(OP_ID_KEY, "<unnamed>"))


def _op_name(op: dict) -> str:
    return str(op.get(OP_NAME_KEY, ""))


def check_restickify_restickify(bundle: dict, out: list[Issue]) -> None:
    """Two restickify ops back to back on the same buffer.

    We flag when adjacent restickify ops share any buffer -- either both
    write the same buffer (a double refinement) or the second reads what
    the first wrote (a chain).  Either shape is redundant when a
    committed layout is supposed to be reached in one step.
    """
    ops = bundle.get(OPS_KEY, [])
    for i in range(len(ops) - 1):
        a, b = ops[i], ops[i + 1]
        if a.get(KIND_KEY) != KIND_RESTICKIFY or b.get(KIND_KEY) != KIND_RESTICKIFY:
            continue
        a_writes = set(a.get(WRITES_KEY, []))
        b_writes = set(b.get(WRITES_KEY, []))
        b_reads = set(b.get(READS_KEY, []))
        shared = a_writes & (b_writes | b_reads)
        if shared:
            out.append(
                Issue(
                    kind="restickify_restickify_same_buffer",
                    bundle=str(bundle.get("id", "")),
                    ops=[_op_id(a), _op_id(b)],
                    detail=f"shared buffer(s): {sorted(shared)}",
                )
            )


def check_chained_copy(bundle: dict, out: list[Issue]) -> None:
    """copy A -> B  then  copy B -> C  with no compute consumer of B."""
    ops = bundle.get(OPS_KEY, [])
    for i in range(len(ops) - 1):
        a, b = ops[i], ops[i + 1]
        if a.get(KIND_KEY) != KIND_COPY or b.get(KIND_KEY) != KIND_COPY:
            continue
        a_writes = set(a.get(WRITES_KEY, []))
        b_reads = set(b.get(READS_KEY, []))
        bridge = a_writes & b_reads
        if not bridge:
            continue
        out.append(
            Issue(
                kind="chained_copy",
                bundle=str(bundle.get("id", "")),
                ops=[_op_id(a), _op_id(b)],
                detail=(
                    f"copy {sorted(a.get(READS_KEY, []))} -> {sorted(a_writes)} "
                    f"then copy -> {sorted(b.get(WRITES_KEY, []))} "
                    f"(bridge={sorted(bridge)})"
                ),
            )
        )


def check_cpu_fallback(bundle: dict, out: list[Issue]) -> None:
    """Any op whose kind is 'fallback' or whose op_name contains 'cpu'."""
    for op in bundle.get(OPS_KEY, []):
        kind = op.get(KIND_KEY, "")
        name = _op_name(op)
        if kind == KIND_FALLBACK or "cpu" in name.lower():
            out.append(
                Issue(
                    kind="cpu_fallback",
                    bundle=str(bundle.get("id", "")),
                    ops=[_op_id(op)],
                    detail=f"kind={kind!r} op_name={name!r}",
                )
            )


def check_duplicate_constants(bundle: dict, out: list[Issue]) -> None:
    """Two or more constant ops with the same materialized-bytes hash."""
    seen: dict[str, list[str]] = {}
    for op in bundle.get(OPS_KEY, []):
        if op.get(KIND_KEY) != KIND_CONSTANT:
            continue
        h = op.get(CONST_HASH_KEY)
        if not h:
            continue
        seen.setdefault(h, []).append(_op_id(op))
    for h, ids in seen.items():
        if len(ids) > 1:
            out.append(
                Issue(
                    kind="duplicate_constant",
                    bundle=str(bundle.get("id", "")),
                    ops=ids,
                    detail=f"const_hash={h}",
                )
            )


def check_singleton_bundle(bundle: dict, out: list[Issue]) -> None:
    """Bundle with a single non-seed op (fragmentation)."""
    ops = bundle.get(OPS_KEY, [])
    substantive = [
        op for op in ops if op.get(KIND_KEY) not in (KIND_EMPTY, KIND_CONSTANT)
    ]
    if len(substantive) == 1 and len(ops) == 1:
        out.append(
            Issue(
                kind="singleton_bundle",
                bundle=str(bundle.get("id", "")),
                ops=[_op_id(ops[0])],
                detail="bundle carries a single op — one launch per op",
            )
        )


def check_hbm_lx_transfer_with_lx_intermediate(
    bundle: dict, out: list[Issue], buffers: dict[str, dict]
) -> None:
    """HBM<->LX copy where an LX-resident intermediate already exists.

    Heuristic: within a single bundle, if a compute op has already
    written an LX-resident buffer under the same name involved in a
    later HBM<->LX copy, the copy likely repeats what the LX write
    already produced.
    """
    lx_writes: set[str] = set()
    for op in bundle.get(OPS_KEY, []):
        kind = op.get(KIND_KEY, "")
        residency = op.get(RESIDENCY_KEY, "")
        writes = op.get(WRITES_KEY, [])
        if kind == KIND_COMPUTE and residency == RESIDENCY_LX:
            lx_writes.update(writes)
        if kind != KIND_COPY:
            continue
        reads = op.get(READS_KEY, [])
        # Residency of the *other* side of the copy is inferred from the
        # buffers map when present; otherwise fall back to the op's own
        # declared residency, which is the destination side.
        involved = set(reads) | set(writes)
        crosses = False
        for buf in involved:
            r = buffers.get(buf, {}).get(RESIDENCY_KEY)
            if r == RESIDENCY_LX and residency == RESIDENCY_HBM:
                crosses = True
                break
            if r == RESIDENCY_HBM and residency == RESIDENCY_LX:
                crosses = True
                break
        if not crosses and residency in (RESIDENCY_HBM, RESIDENCY_LX):
            # Fall back: any copy whose destination is HBM but reads an
            # LX buffer counts as HBM<->LX.
            for buf in reads:
                if buffers.get(buf, {}).get(RESIDENCY_KEY) == RESIDENCY_LX:
                    crosses = True
                    break
        if not crosses:
            continue
        overlap = involved & lx_writes
        if overlap:
            out.append(
                Issue(
                    kind="hbm_lx_transfer_with_lx_intermediate",
                    bundle=str(bundle.get("id", "")),
                    ops=[_op_id(op)],
                    detail=(
                        f"copy touches {sorted(involved)}; LX-resident "
                        f"intermediate already exists for {sorted(overlap)}"
                    ),
                )
            )


CHECKS = (
    check_restickify_restickify,
    check_chained_copy,
    check_cpu_fallback,
    check_duplicate_constants,
    check_singleton_bundle,
    # check_hbm_lx_transfer_with_lx_intermediate is called separately
    # because it needs the top-level buffers map.
)


def analyze_artifact(artifact: dict) -> FixtureResult:
    result = FixtureResult(
        artifact=str(artifact.get("artifact", "<unnamed>")),
        description=str(artifact.get("description", "")),
    )
    buffers = artifact.get(BUFFERS_KEY, {}) or {}
    for bundle in artifact.get(BUNDLES_KEY, []):
        for check in CHECKS:
            check(bundle, result.issues)
        check_hbm_lx_transfer_with_lx_intermediate(bundle, result.issues, buffers)
    return result


# ---------------------------------------------------------------------------
# I/O + CLI
# ---------------------------------------------------------------------------


def _iter_artifact_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.json")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"warning: {p} does not exist", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan a Torch-Spyre runtime artifact for suspicious patterns."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        required=True,
        help="Path to an artifact JSON file, or a directory of them. "
        "Repeatable.",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Path to write JSON report (default: stdout).",
    )
    args = parser.parse_args(argv)

    paths = _iter_artifact_paths([Path(p) for p in args.artifact])
    fixtures: list[dict] = []
    for path in paths:
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: could not read {path}: {e}", file=sys.stderr)
            continue
        # Stamp the artifact field from the filename if the JSON did not
        # carry one, so downstream reports can attribute hits back.
        data.setdefault("artifact", path.stem)
        result = analyze_artifact(data)
        fixtures.append(result.as_dict())

    report = {"fixtures": fixtures}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
