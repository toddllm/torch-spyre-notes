#!/usr/bin/env python3
"""Parse the pre-scheduling pass contract matrix into a dependency graph.

The matrix at ``contracts/pass-matrix.md`` encodes, for every pass in
every Torch-Spyre pipeline, the phase it runs in, what state it
requires, what state it reads, and what state it mutates. Those
observations imply ordering constraints:

    * "A must run before B" whenever B's ``Requires`` mentions an
      artifact A produces.
    * "C invalidates D" whenever C's ``Mutates`` clears or replaces a
      value D would later read.
    * "E duplicates F" whenever E's role is subsumed by another pass F
      running under the same preconditions.

This script encodes those edges from the matrix's prose using a small
hand-curated table (below) and then emits:

    * a DOT-format graph on stdout so a human can render it, and
    * a report listing any cycles and any candidate-redundant orderings.

Kept as a hand-curated table on purpose. The prose in the matrix is
what an auditor should trust; parsing free text with regexes would let
a typo silently invalidate the graph. The table below is short enough to
review by eye and each entry cites the matrix row that justifies it.

Usage
-----

::

    python3 scans/pass_dependency_graph.py --out scans/results/pass_dependency_graph
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Nodes (passes) grouped by pipeline. Order inside each list is the pipeline
# order from `passes.py`.
# ---------------------------------------------------------------------------

PIPELINES: dict[str, list[str]] = {
    "CustomPrePasses": [
        "collect_spyre_hints",
    ],
    "CustomPostPasses": [
        "decompose_addmm",
        "mm_to_bmm_pass.apply",
        "mark_direct_unit_bmm_pass",
        "bmm_unflatten_pass.apply",
    ],
    "CustomPreSchedulingPasses": [
        "deadcode_elimination",
        "propagate_named_dims",
        "validate_named_dims",
        "assign_dim_hints",
        "_maybe_reorder_unhinted_interlopers",
        "_maybe_coarse_tile_hints",
        "split_multi_ops",
        "propagate_spyre_tensor_layouts",
        "validate_ops",
        "optimize_restickify_locations",
        "finalize_layouts",
        "insert_restickify",
        "enforce_indirect_access_layout",
        "insert_post_mutation_restickify",
        "insert_bmm_padding",
        "dedup_and_promote_constants",
        "_maybe_coarse_tile_span_overflow",
        "span_reduction",
        "_distribute_work",
        "_maybe_scratchpad_planning",
    ],
    "CustomPreFusionPasses": [
        "propagate_mutation_layouts",
        "align_lx_producer_loop_order",
        "build_loop_scheduler_nodes",
    ],
    "CustomPostFusionPasses": [
        "demote_incoherent_lx_buffers",
        "spyre_fuse_nodes",
        "hbm_pool_planning",
    ],
}


# ---------------------------------------------------------------------------
# Edges. Kinds:
#
#   "before"      -- A must run before B (B.Requires references A's output).
#   "invalidates" -- A invalidates state B would need. Same direction as
#                    "before" for the ordering DAG, but tagged separately so
#                    the DOT output can style the edge differently.
#   "duplicates"  -- A and B do overlapping work. Not a hard ordering.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str  # "before" | "invalidates" | "duplicates"
    reason: str


EDGES: list[Edge] = [
    # --- CustomPostPasses order chain (matrix rows in same pipeline) ---
    Edge("decompose_addmm", "mm_to_bmm_pass.apply",
         "before", "mm_to_bmm rewrites the aten.mm that decompose_addmm exposes"),
    Edge("mm_to_bmm_pass.apply", "mark_direct_unit_bmm_pass",
         "before", "mark_direct_unit_bmm marks the bmm produced by mm_to_bmm"),
    Edge("mark_direct_unit_bmm_pass", "bmm_unflatten_pass.apply",
         "before", "unflatten pass consumes the marked bmm nodes"),

    # --- Named-dims chain ---
    Edge("propagate_named_dims", "validate_named_dims",
         "before", "validate reads op._dim_prop_info (set by propagate)"),
    Edge("propagate_named_dims", "assign_dim_hints",
         "before", "assign consumes op._dim_prop_info and _named_dims"),
    Edge("assign_dim_hints", "_maybe_reorder_unhinted_interlopers",
         "before", "reorder reads op.dim_hints written by assign_dim_hints"),
    Edge("assign_dim_hints", "_maybe_coarse_tile_hints",
         "before", "coarse tiling reads op.dim_hints written by assign_dim_hints"),
    # 'invalidates' edges are written src -> dst where src is the pass whose
    # outputs get invalidated by dst -- same temporal direction as 'before'
    # (src must run first, then dst clobbers it). Keeping this convention
    # avoids spurious cycles in the ordering DAG.
    Edge("propagate_named_dims", "assign_dim_hints",
         "invalidates", "assign_dim_hints DELETES op._dim_prop_info in its finally block (matrix row 4 Mutates); propagate must run BEFORE assign, never after"),

    # --- WSR chain ---
    Edge("_maybe_reorder_unhinted_interlopers", "_maybe_coarse_tile_hints",
         "before", "hint groups depend on runs being contiguous after reorder"),

    # --- Stickification chain ---
    Edge("_maybe_coarse_tile_hints", "split_multi_ops",
         "before", "matrix places coarse_tile_hints as last pre-stickify WSR pass"),
    Edge("split_multi_ops", "propagate_spyre_tensor_layouts",
         "before", "propagate_spyre_tensor_layouts requires the intermediate ComputedBuffers split_multi_ops creates"),
    Edge("propagate_spyre_tensor_layouts", "validate_ops",
         "before", "validate_ops docstring: 'must be run after propagate_spyre_tensor_layouts'"),
    Edge("propagate_spyre_tensor_layouts", "optimize_restickify_locations",
         "before", "optimizer reads op.layouts and op.restick_cost_fn set by propagate_layouts"),
    Edge("optimize_restickify_locations", "finalize_layouts",
         "before", "finalize consumes op.committed_stl written by optimizer"),
    Edge("finalize_layouts", "insert_restickify",
         "before", "insert_restickify consumes graph.restickify_plan built by finalize"),
    Edge("optimize_restickify_locations", "finalize_layouts",
         "invalidates", "finalize DELETES op.layouts, op.restick_cost_fn, op.committed_stl -- optimizer must run BEFORE finalize"),

    # --- Post-stickify chain ---
    Edge("insert_restickify", "enforce_indirect_access_layout",
         "before", "enforce docstring: 'runs after insert_restickify: every op's layout is a committed FixedTiledLayout'"),
    Edge("insert_restickify", "insert_post_mutation_restickify",
         "before", "post_mutation restickify needs the base restickify plan already applied"),
    Edge("insert_post_mutation_restickify", "insert_bmm_padding",
         "before", "matrix ordering; padding runs after mutation restickify"),
    Edge("insert_bmm_padding", "dedup_and_promote_constants",
         "before", "padding creates constants; dedup folds them (matrix note on padding row)"),
    Edge("dedup_and_promote_constants", "_maybe_coarse_tile_span_overflow",
         "before", "span-overflow needs FixedTiledLayout on every op, which is guaranteed only after all stickify passes"),

    # --- Division chain ---
    Edge("_maybe_coarse_tile_span_overflow", "span_reduction",
         "before", "matrix ordering; work_division reads final layouts and per-op loop info"),
    Edge("span_reduction", "_distribute_work",
         "before", "cost_model_matmul_division and work_distribution consume op.op_it_space_splits set by span_reduction"),
    Edge("_distribute_work", "_maybe_scratchpad_planning",
         "before", "LX planning depends on final divisions to size intermediates"),

    # --- Scheduler pipeline ---
    Edge("align_lx_producer_loop_order", "build_loop_scheduler_nodes",
         "before", "align must see plain SchedulerNodes; build_loop_scheduler_nodes wraps them into CountedLoopSchedulerNode"),
    Edge("propagate_mutation_layouts", "align_lx_producer_loop_order",
         "before", "align reads final layouts; mutation layouts must be resolved first"),

    # --- Post-fusion pipeline ---
    Edge("demote_incoherent_lx_buffers", "spyre_fuse_nodes",
         "before", "fuse groups nodes into bundles after demotion so demoted buffers become bundle-scoped intermediates"),
    Edge("spyre_fuse_nodes", "hbm_pool_planning",
         "before", "hbm_pool_planning docstring: 'runs after spyre_fuse_nodes so nodes is the final, post-fusion top-level list'"),
    Edge("_maybe_scratchpad_planning", "hbm_pool_planning",
         "before", "hbm_pool_planning docstring: 'a buffer is only an hbm_pool candidate if LX planning did not already claim it'"),

    # --- Cross-pipeline: WSR configuration reaches propagate ---
    Edge("collect_spyre_hints", "propagate_named_dims",
         "before", "spyre_hint metadata collected on FX nodes is consumed by propagate_named_dims through the operations' FX origins"),

    # --- Duplicate ordering candidates ---
    # (See matrix open questions: two coarse-tile paths in the pipeline.
    # They are not duplicates -- hint-driven runs pre-stickify, span-overflow
    # runs post-stickify -- but a reader coming to the matrix cold could
    # mistake them for one. Recorded here so the "duplicates" report calls
    # this out explicitly, with a rebuttal.)
    Edge("_maybe_coarse_tile_hints", "_maybe_coarse_tile_span_overflow",
         "duplicates",
         "both configure coarse tiling but at different layout phases (hint pre-stickify, span-overflow post-stickify). Not a real duplicate; kept as a documented false alarm."),
]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _nodes() -> list[str]:
    seen: dict[str, None] = {}
    for stage in PIPELINES.values():
        for n in stage:
            seen.setdefault(n, None)
    for e in EDGES:
        seen.setdefault(e.src, None)
        seen.setdefault(e.dst, None)
    return list(seen.keys())


def _hard_edges() -> list[Edge]:
    return [e for e in EDGES if e.kind in ("before", "invalidates")]


def find_cycles(nodes: list[str], edges: list[Edge]) -> list[list[str]]:
    """Find simple cycles in the ordering DAG (edges: before/invalidates)."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for e in edges:
        adj[e.src].append(e.dst)

    cycles: list[list[str]] = []
    visited: set[str] = set()
    on_stack: list[str] = []
    on_stack_set: set[str] = set()

    def dfs(u: str) -> None:
        if u in on_stack_set:
            i = on_stack.index(u)
            cycles.append(on_stack[i:] + [u])
            return
        if u in visited:
            return
        visited.add(u)
        on_stack.append(u)
        on_stack_set.add(u)
        for v in adj[u]:
            dfs(v)
        on_stack.pop()
        on_stack_set.discard(u)

    for n in nodes:
        dfs(n)
    return cycles


def find_transitive_edges(nodes: list[str], edges: list[Edge]) -> list[Edge]:
    """Find edges A->B where a longer A->...->B path exists.

    A transitive "before" edge is not wrong -- the pipeline order still
    holds -- but it is redundant: the shorter path already forces the
    ordering.  Reported so the reader can decide whether the redundancy
    hides a subtler contract (e.g. one path is only guaranteed under a
    config flag).
    """
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for e in edges:
        adj[e.src].add(e.dst)

    # Reach: adj without direct edges.
    def reachable(src: str, avoid: str) -> set[str]:
        seen: set[str] = set()
        stack = [n for n in adj[src] if n != avoid]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u])
        return seen

    redundant: list[Edge] = []
    for e in edges:
        if e.kind != "before":
            continue
        # Is there a path src -> ... -> dst avoiding the direct edge?
        if e.dst in reachable(e.src, e.dst):
            redundant.append(e)
    return redundant


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_KIND_STYLE = {
    "before":       'color="#334", label=""',
    "invalidates":  'color="#c22", style=dashed, label="invalidates"',
    "duplicates":   'color="#a80", style=dotted, label="duplicates"',
}


def to_dot(pipelines: dict[str, list[str]], edges: list[Edge]) -> str:
    lines: list[str] = []
    lines.append("digraph pre_scheduling_passes {")
    lines.append('  rankdir=TB;')
    lines.append('  node [shape=box, fontname="Helvetica", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')
    for pipeline, passes in pipelines.items():
        safe = pipeline.replace(".", "_")
        lines.append(f'  subgraph cluster_{safe} {{')
        lines.append(f'    label="{pipeline}";')
        lines.append('    style=rounded; color="#888";')
        for p in passes:
            lines.append(f'    "{p}";')
        lines.append("  }")
    for e in edges:
        style = _KIND_STYLE[e.kind]
        # Escape quotes in reason if present (defensive)
        reason = e.reason.replace('"', "'")
        lines.append(f'  "{e.src}" -> "{e.dst}" [{style}, tooltip="{reason}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def format_cycles(cycles: list[list[str]]) -> str:
    if not cycles:
        return "cycles: none\n"
    lines = ["cycles:"]
    for c in cycles:
        lines.append("  " + " -> ".join(c))
    return "\n".join(lines) + "\n"


def format_redundant(redundant: list[Edge]) -> str:
    if not redundant:
        return "unnecessary orderings: none\n"
    lines = ["unnecessary orderings (transitive 'before' edges):"]
    for e in redundant:
        lines.append(f"  {e.src} -> {e.dst}  ({e.reason})")
    return "\n".join(lines) + "\n"


def format_duplicates(edges: list[Edge]) -> str:
    dup = [e for e in edges if e.kind == "duplicates"]
    if not dup:
        return "flagged duplicates: none\n"
    lines = ["flagged duplicates:"]
    for e in dup:
        lines.append(f"  {e.src} == {e.dst}  ({e.reason})")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit DOT + a cycle / redundancy report for the pass dependency graph."
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Path prefix to write to (writes <prefix>.dot and <prefix>.report.txt). "
        "Use '-' (default) to print both to stdout separated by a marker line.",
    )
    args = parser.parse_args(argv)

    nodes = _nodes()
    hard = _hard_edges()
    cycles = find_cycles(nodes, hard)
    redundant = find_transitive_edges(nodes, hard)
    dot_text = to_dot(PIPELINES, EDGES)
    report_text = (
        format_cycles(cycles)
        + "\n"
        + format_redundant(redundant)
        + "\n"
        + format_duplicates(EDGES)
    )

    if args.out == "-":
        sys.stdout.write(dot_text)
        sys.stdout.write("\n===== REPORT =====\n")
        sys.stdout.write(report_text)
    else:
        prefix = Path(args.out)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        dot_path = prefix.with_suffix(".dot")
        report_path = prefix.with_suffix(".report.txt")
        dot_path.write_text(dot_text)
        report_path.write_text(report_text)
        print(f"wrote {dot_path}")
        print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
