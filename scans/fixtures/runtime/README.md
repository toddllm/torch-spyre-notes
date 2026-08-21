# runtime artifact fixtures

Synthetic minimal artifacts exercising every check in
`scans/runtime_artifact.py`. Each fixture is a JSON file matching the
schema documented at the top of the scanner. The fixtures are pinned to
this repo -- they are not read from any real Torch-Spyre compilation.
When a real bundle/LLIR dump becomes available, the scanner's field
names (`BUNDLES_KEY`, `OPS_KEY`, ...) can be retargeted at the real
schema and the fixtures kept as regression cases.

| fixture                       | patterns it exercises                                              |
| ----------------------------- | ------------------------------------------------------------------ |
| `clean.json`                  | negative control: no issues expected                               |
| `mixed.json`                  | restickify->restickify, chained copy, cpu fallback, dup constant,  |
|                               | singleton bundle, HBM<->LX transfer with an LX intermediate        |
| `fragmentation.json`          | many singleton bundles + one duplicate-constant pair               |

Run the scanner over all three like this::

    python3 scans/runtime_artifact.py \
        --artifact scans/fixtures/runtime \
        --out scans/results/runtime_artifact.json
