# Scans

AST-based and grep-based tools that surface *candidates* for the
findings under `findings/`. Scans do not produce findings on their
own — a scan hit is a hypothesis that a human (or a directed
subagent) must confirm, categorize, and write up.

## Planned scans

Each is a stub for now. Implementation happens on demand when a scan
would be more efficient than one-off greps.

### `private-api.py`
Walk every `import` and every attribute access. Flag: imports from
`torch._inductor`, `torch._dynamo`, `torch.fx.experimental`;
attribute reads/writes on those classes; monkey-patch assignments
(`SomeUpstreamClass.method = ...`); `getattr`/`setattr` with a
literal string that names a private attribute; access to
`pass_patterns[<int>]`, `.pop()` on upstream registries; closure
introspection (`.__code__.co_freevars`, `.__closure__`).

### `graph-mutations.py`
Find every write to: `graph.operations`, `graph.buffers`,
`name_to_buffer`, `name_to_users`, `name_to_op`, `removed_buffers`,
`graph_outputs`, `.inner_fn`, `.layout`, `.operation_name`,
`.origins`. Cluster mutation sequences by *semantic operation*
(replace / redirect / insert / retire / clone / relayout) rather
than by textual similarity. Emit a table of (site, cluster,
invariants-updated-locally).

### `repeated-analysis.py`
Find every call to `get_read_writes`, `extract_read_writes`,
`op_read_writes`, `host_coordinates`, `device_coordinates`,
`compute_coordinates`. For each call, walk outward and record the
enclosing loop nesting. Emit: (site, nesting depth, is-result-reused,
uses-memoized-helper). Anything nested inside two or more loops is
an automatic candidate.

### `list-surgery.py`
Find every call to `operations.index`, `operations.remove`,
`operations.insert` (and analogous on other lists). Same output
shape as `repeated-analysis.py`. Anything inside a loop is a
candidate for the O(N²) class.

### `workarounds.py`
Grep for the temporary-code lexicon: `TODO`, `FIXME`, `for now`,
`temporary`, `workaround`, `upstream`, `once PyTorch`, `not
supported yet`, `we believe`, `should remove`, `monkey patch`. For
each hit, extract the surrounding comment block and the introducing
commit. Cross-reference against the current PyTorch v2.13 and
`main` sources to see whether the upstream condition still exists.

### `test-smells.py`
Find pytest test functions and inspect for:
- `with patch.object(...):` blocks whose body contains no calls
  (empty patch context).
- capture callbacks (`captured = []`) that are patched in but never
  read.
- assertions of the form `assert set(x) <= set(x)` (tautologies).
- assertions on values derived from the same source as the value
  being tested.
- `@pytest.mark.skip` and `unittest.TestCase.skipTest` calls without
  a linked issue.

## Convention: scans emit candidates, not verdicts

A scan produces a list of `(file, line, context)` tuples plus a
one-line hypothesis per hit. It does not itself decide whether the
hit is a real finding. That decision is made when a human or a
directed subagent writes the corresponding `findings/*.md`.

## Convention: scans are pinned

A scan run is pinned to the same revision manifest as the findings
it feeds. When the manifest advances, scans re-run and the delta
(new hits, resolved hits) is captured in the new report.
