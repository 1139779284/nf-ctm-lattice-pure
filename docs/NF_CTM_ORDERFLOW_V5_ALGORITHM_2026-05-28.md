# NF-CTM Lattice v5: Counterfactual Order-Flow CTM

## Why this version exists

The handoff package shows a source-level bottleneck on v2 visible-patch OGA:
residual augmented failures cluster by entire source image groups rather than by
transform type. The strict-pure CTM lattice can reduce ASR but does not learn a
source-invariant trigger-causal transform strongly enough.

The earlier relaxed run with `cross_field_coupling=0.01` improves ASR by passing
spatial energy maps between scales, but that is risky under the pure-algorithm
constraint. This version therefore keeps all new information inside CTM order
trajectories.

## Core idea

A CTM scale order is the latent order parameter produced by raw synchronization
at each thought tick.  For a sample x, let

```text
O_t(x) = scale-order vector at thought step t
q(x) = [O_1-O_0, ..., O_T-O_{T-1}]
```

`q(x)` is the **counterfactual order-flow**. It is not a spatial map, not a
frequency token, not a detector score, and not a post-hoc editor. It is only a
trajectory of CTM synchronization order parameters.

The new loss asks invalid trigger samples to share a source-invariant order-flow
while valid clean-object samples remain order-quiet:

```text
L_order =
  same-source compactness(q)
+ cross-source invariance(q)
+ flow floor(q)
+ valid quietness(q_valid)
```

This directly targets the measured bottleneck: the old model learned
source-specific basins, while the new objective encourages a shared
trigger-causal order-flow across held-out source identities.

## What is not used

This version still forbids:

- clean-anchor interpolation;
- weight soup;
- runtime guard;
- score calibration;
- NMS/box post-processing;
- CNN adapter residual;
- spatial or frequency token branches;
- detector-head surgery;
- post-hoc edge editor.

## Main changed files

```text
model_security_gate/detox/nf_ctm_lattice/multiscale_field.py
model_security_gate/detox/nf_ctm_lattice/__init__.py
scripts/nf_ctm_lattice_yolo_multiscale_v2_orderflow_2026-05-28.py
tests/test_nf_ctm_lattice.py
```

## Suggested first real run

Use the previous strict-pure v2 command, but add:

```powershell
--order-flow-weight 0.05 `
--order-flow-floor 0.015 `
--order-flow-same-source-weight 0.5 `
--order-flow-cross-source-weight 1.0 `
--order-flow-valid-quiet-weight 0.25
```

If clean remains stable but ASR plateau persists, raise only
`--order-flow-weight` to 0.10.  Do not re-enable `cross_field_coupling` unless the
purpose is an ablation, because it may be audited as a spatial branch.

## Validation performed in this overlay environment

- Python compile check: passed.
- `tests/test_nf_ctm_lattice.py`: 35 passed.
- Synthetic smoke at 220 steps: invalid error 1.0 -> 0.0, valid error 0.0 -> 0.0.

Heavy YOLO ASR was not run in this environment because the handoff package does
not include local CUDA weights or datasets.

## Local heavy YOLO validation in D:\clean_yolo

After applying this overlay to the full local workspace, the v2 visible OGA
benchmarks were run on the real local model/data.  The implementation compiled
and the NF-CTM-related tests passed:

```text
tests/test_nf_ctm_lattice.py + tests/test_nf_ctm_closed_loop.py + tests/test_nf_ctm_detox.py
45 passed
```

The recommended v2 order-flow run did not improve over the current strict-pure
best:

```text
nf_ctm_lattice_v5_orderflow_v2_800_2026-05-28
trig_eval = 9/22
aug_eval  = 100/242
clean     = 55/60
```

For comparison, the previous strict-pure best was:

```text
nf_ctm_lattice_multiscale_v2_minimax_localedge_800_2026-05-28
trig_eval = 9/22
aug_eval  = 100/242
clean     = 55/60
```

Additional checks also did not improve the plateau:

```text
orderflow + same-source variants, 300 steps:
  aug_eval = 149/242, clean = 56/60

stronger orderflow without valid quiet, 300 steps:
  aug_eval = 154/242, clean = 56/60

topk_abs scale-order pooling, 300 steps:
  aug_eval = 155/242, clean = 56/60

continue from relaxed best checkpoint, 300 steps:
  aug_eval = 44/242, clean = 52/60
```

The order-flow loss was numerically active but very small in the recommended
run (`order_flow` around `0.0013-0.0018`), so with `--order-flow-weight 0.05`
it contributed little gradient.  Raising the pressure did not help in short
runs.  Therefore this version should be kept as a documented ablation, but it
does not yet solve the source-level generalization bottleneck.
