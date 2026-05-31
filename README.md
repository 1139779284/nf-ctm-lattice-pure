# NF-CTM Lattice Pure

Standalone research repository for the pure NF-CTM Lattice detox algorithm.

This repository is intentionally algorithm-only.  It contains CTM lattice code,
YOLO runner scripts, tests, and audit/progress documents.  It does not include
model weights, datasets, videos, benchmark caches, or virtual environments.

## Research Boundary

Paper-main experiments must use a single in-flow CTM feature hook:

```text
YOLO neck feature tensor
-> NF-CTM recurrent synchronization lattice
-> terminal CTM state replaces the same neck tensor
-> frozen downstream detector head
```

Do not mix the paper-main claim with CNN purifier/adapters, clean-anchor
interpolation, weight soup, score calibration, runtime guards, post-processing,
or detector-CTM-detector sandwich logic.

## Layout

```text
model_security_gate/detox/nf_ctm_lattice/
  Core CTM lattice implementation, objectives, hooks, and split protocol.

scripts/
  NF-CTM YOLO runners, semantic vest A/B protocol tools, and video audit tool.

tests/
  Unit and wiring tests for CTM dynamics, objectives, purity guards, and runner
  CLI surface.

docs/
  Latest audit, progress notes, no-sandwich boundary, and v4/v5/v6 algorithm
  notes.
```

## External Workspace

The runner scripts expect the large project assets in a separate clean_yolo
workspace.  By default that is:

```powershell
D:\clean_yolo
```

Override it when needed:

```powershell
$env:CLEAN_YOLO_WORKSPACE="D:\clean_yolo"
```

The workspace should contain `models/`, `datasets/`, `benchmark_runs/`, and the
source video if you want to reproduce the heavy YOLO/video evidence.  Those
large artifacts are deliberately not tracked here.

## Quick Verification

Using an environment that already has torch/ultralytics/pytest:

```powershell
python -m pytest tests\test_nf_ctm_lattice.py -q
python -m compileall -q model_security_gate scripts tests
```

In the original workspace you can also run with the existing pixi environment:

```powershell
cd D:\clean_yolo\nf_ctm_lattice_standalone
D:\clean_yolo\model_security_gate\.pixi\envs\default\python.exe -m pytest tests\test_nf_ctm_lattice.py -q
```

## Current Evidence State

Latest v4 semantic OGA image result from the main workspace:

```text
defended aug ASR: 1/66 = 1.52%
clean:            59/60 -> 57/60
purity:           paper_main_profile=true, strict_pure_flags=[]
```

But video is not closed:

```text
clean H/Hd:  335/142
NF-CTM H/Hd: 273/0
warning:     ctm_head_collapse_on_oga_video
```

Main open blockers:

```text
v2 visible patch OGA source-level generalization
v4 semantic OGA video head/source collapse
B-class ODA video over-detection replication
```

Treat this repository as the clean starting point for solving those pure CTM
algorithm problems.
