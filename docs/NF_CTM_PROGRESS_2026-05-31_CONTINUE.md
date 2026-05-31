# NF-CTM Progress 2026-05-31 Continue

## Scope

This note records the post-audit continuation runs after fixing runner/objective
bugs.  All runs below use the single in-flow neck-hook NF-CTM path.  No
clean-anchor, soup, runtime guard, score calibration, postprocess repair,
decoded geometry, OGA/ODA box support loss, or detector-CTM-detector sandwich is
used.

## Code Added

Added a paper-safe CTM regularizer:

```text
valid-feature-jitter normal-neighborhood regularization
```

Training-time idea:

```text
F_valid_jitter = F_valid + epsilon * std(F_valid)
CTM(F_valid_jitter) should remain a quiet valid fixed point
```

This targets the observed failure where CTM preserves valid training samples but
does not generalize to valid held-out clean images.  It uses only the selected
neck feature tensor, CTM trajectory/state losses, and the existing frozen readout
task boundary.

Relevant runner args:

```text
--valid-feature-jitter-weight
--valid-feature-jitter-std
--valid-feature-jitter-quiet-weight
--valid-feature-jitter-fixed-weight
--valid-feature-jitter-homeostasis-weight
--valid-feature-jitter-trajectory-weight
--valid-feature-jitter-gate-weight
```

Validation:

```text
pytest tests/test_nf_ctm_lattice.py
64 passed
```

## v2 Findings

### Strict-pure patchfix baseline

```text
run: benchmark_runs/nf_ctm_lattice_strict_pure_v2v4_patchfix_2026-05-31/v2
passthrough aug ASR: 99/110 = 90.0%
defended aug ASR:    87/110 = 79.1%
clean:               58/60 -> 47/60
purity.paper_main_profile: true
```

### Historical-profile rerun after patchfix

```text
run: benchmark_runs/nf_ctm_lattice_v2_historical_profile_800_patchfix_2026-05-31/v2
defended aug ASR: 21/110 = 19.1%
clean:            58/60 -> 37/60
```

### Valid-jitter + invalid-floor

```text
run: benchmark_runs/nf_ctm_lattice_v2_valid_jitter_invfloor_300_2026-05-31/v2
defended aug ASR: 99/110 = 90.0%
clean:            60/60 -> 57/60
```

Conclusion:

```text
v2 is not solved by normal-neighborhood regularization.
When attack pressure is high enough, clean collapses.
When clean is preserved, attack pressure does not transfer.
```

## v4 Findings

### Strict-pure patchfix baseline

```text
run: benchmark_runs/nf_ctm_lattice_strict_pure_v2v4_patchfix_2026-05-31/v4
passthrough aug ASR: 41/66 = 62.1%
defended aug ASR:     1/66 = 1.5%
clean:                59/60 -> 33/60
purity.paper_main_profile: true
```

### Clean-preserving manifold scans

```text
bounded clean manifold 300:
  ASR 11/66 = 16.7%, clean 60/60 -> 48/60

CE clean manifold 300:
  ASR 11/66 = 16.7%, clean 60/60 -> 52/60

CE clean manifold 200-valid 300:
  ASR 11/66 = 16.7%, clean 60/60 -> 53/60
```

### Valid-feature-jitter scans

```text
valid_jitter 300:
  ASR 11/66 = 16.7%, clean 60/60 -> 55/60

valid_jitter_strong 300:
  ASR 1/66 = 1.5%, clean 60/60 -> 54/60

valid_jitter_strong 150:
  ASR 10/66 = 15.2%, clean 60/60 -> 60/60

valid_jitter_strong 180:
  ASR 10/66 = 15.2%, clean 60/60 -> 59/60

valid_jitter_invfloor 180:
  ASR 9/66 = 13.6%, clean 60/60 -> 59/60

valid_jitter_invfloor_stronger 180:
  ASR 9/66 = 13.6%, clean 60/60 -> 58/60
```

Best paper-safe v4 point today:

```text
benchmark_runs/nf_ctm_lattice_v4_valid_jitter_invfloor_180_2026-05-31/v4
ASR 9/66 = 13.6%
clean 60/60 -> 59/60
```

Conclusion:

```text
valid-feature-jitter is a real pure-CTM improvement for v4 clean safety.
It moves v4 from ASR-low/clean-collapse to clean-safe/moderate-ASR.
It does not yet close strict ASR.
```

## Current Interpretation

The core remaining issue is not a simple parameter problem:

```text
v4: pure CTM can now preserve clean, but residual semantic-trigger ASR remains.
v2: pure CTM still has a hard conflict between attack suppression and clean preservation.
```

The next algorithmic step should be CTM-internal trigger localization, not
detector-output repair.  Candidate direction:

```text
invalid-only CTM edge-order localization:
  enforce invalid thought motion to concentrate on high edge-order conflict cells
  while valid-feature-jitter keeps normal neighborhoods quiet
```

This remains within CTM dynamics if the support is computed only from CTM
sync-fields / update-gates / state trajectory, not from YOLO boxes, decoded
anchors, image masks, frequency tokens, or clean anchors.

## Edge-Order Localization Update

Implemented:

```text
thought_edge_order_localization_loss()
```

It computes invalid-only support from CTM-internal edge magnitude distributions:

```text
P_t,c,e,u = softmax_e(|S_t,c,e,u| / tau)
A(u) = JS(P_t,c,:,u || mean_u P_t,c,:,u)
     + lambda_time JS(P_t,c,:,u || P_t-1,c,:,u)
     + lambda_gate mean_t,c |G_t,c,u|
```

Then it encourages terminal thought motion to concentrate inside the top-k
edge-order anomaly support.  It does not use detector scores, boxes, anchors,
NMS, clean anchors, or postprocessing.

Stability fixes after audit:

```text
safe sqrt for zero-motion traces
flat edge-signal guard to avoid arbitrary top-k support
mass floor before inside-ratio / contrast terms activate
complete logging keys for empty-trace branches
```

Validation:

```text
pytest tests/test_nf_ctm_lattice.py
68 passed
```

Current evidence:

```text
edge_order_w003_180 before stability fixes:
  ASR 9/66 = 13.6%, clean 60/60 -> 59/60

edge_order_stable_w003_180:
  ASR 10/66 = 15.2%, clean 60/60 -> 58/60

edge_order_w008_top20_180:
  ASR 10/66 = 15.2%, clean 60/60 -> 59/60
```

Conclusion:

```text
Edge-order localization is now a paper-safe CTM candidate and numerically stable,
but it has not yet improved v4 metrics over valid-feature-jitter + small invalid-floor.
For now, keep it as algorithmic ablation/candidate, not the main best result.
```

## V4 Knee Search Update

Based on the clean/ASR tradeoff between 180 and 300 steps, a short knee search
was run on the strongest paper-safe v4 base:

```text
base:
  valid_feature_jitter_weight = 0.60
  valid_feature_jitter_std = 0.05
  valid_motion_weight = 0.30
  valid_homeostasis_weight = 0.10
  valid_gate_weight = 0.04
  valid_fixed_point_weight = 0.30
  trajectory_valid_weight = 0.08
  trajectory_invalid_floor_weight = 0.05
  invalid_gate_floor_weight = 0.03
```

Results:

```text
180 steps:
  ASR 9/66 = 13.6%, clean 60/60 -> 59/60, clean-safe PASS

240 steps:
  ASR 9/66 = 13.6%, clean 60/60 -> 56/60, clean-safe fail

240 steps, stronger jitter 0.80/std 0.06:
  ASR 8/66 = 12.1%, clean 60/60 -> 56/60, clean-safe fail
```

Current v4 best paper-safe point remains:

```text
benchmark_runs/nf_ctm_lattice_v4_valid_jitter_invfloor_180_2026-05-31/v4
ASR 9/66 = 13.6%
clean 60/60 -> 59/60
```

Interpretation:

```text
Longer training gives only small ASR gains and quickly hurts held-out clean.
The next breakthrough likely needs a better CTM-internal trigger localization
or state separation mechanism, not more step scanning.
```

## B-Class Video Follow-Up

Use the video audit scripts only as evidence, not as training loss:

```powershell
pixi run python D:\clean_yolo\scripts\_video_compare_nf_ctm_lattice_2026-05-30.py `
  --video D:\clean_yolo\7bc6518d5d105de194eefd2c4c96e827.mp4 `
  --ctm-run-root D:\clean_yolo\benchmark_runs\nf_ctm_lattice_v3_final_rerun_2026-05-29_oda `
  --families b1,b2,b3,b4 `
  --out D:\clean_yolo\benchmark_runs\video_compare_bclass_v3_final_rerun_2026-05-31 `
  --device 0 --max-frames 120
```

Pass condition:

```text
static ASR low
clean safe
video helmet/head counts close to clean
no head collapse
no over-detection
no zero-output
```

## Residual-Profile Invariance Follow-Up

Fixed a runner wiring bug before this sweep:

```text
--residual-profile-invariance-weight
--residual-profile-invalid-floor
--residual-profile-valid-weight
--residual-profile-topk-frac
```

These parameters were consumed by `LatticeLossConfig` but were not exposed by
the v4 single-neck runner CLI.  They are now configurable and recorded in
`record.json` for reproducibility.

Validation:

```text
pixi run pytest tests/test_nf_ctm_lattice.py -q
84 passed
pixi run python -m compileall -q model_security_gate scripts tests
PASS
```

Important control:

```text
The previous best v4 gate-separation run used:
  sync_drive_dc_suppression = 0.0
  valid_feature_jitter_weight = 0.60

Residual-profile sweeps must keep those fixed; otherwise clean collapse can be
misattributed to residual-profile itself.
```

New v4 strict-pure result:

```text
run:
  benchmark_runs/nf_ctm_lattice_v4_resprofile_w0002_jitter_gatesep_sync0_180_2026-05-31/v4

passthrough:
  aug ASR = 41/66 = 62.1%
  clean   = 59/60

defended:
  trig ASR = 0/6
  aug ASR  = 1/66 = 1.52%
  Wilson95 upper = 8.10%
  clean    = 57/60
  clean drop = 2 images

purity:
  paper_main_profile = true
  strict_pure_flags = []
```

Follow-up sweeps:

```text
w=0.001, 180 steps:
  ASR 8/66, clean 56/60

w=0.002, jitter 0.80/std 0.06, 180 steps:
  ASR 1/66, clean 57/60

w=0.002, 160 steps:
  ASR 9/66, clean 56/60
```

Interpretation:

```text
Residual-profile invariance is the first paper-safe CTM mechanism in this
round that pushes v4 below 5% point-estimate ASR while preserving the agreed
clean-drop boundary.  It is not yet a strict statistical close because 1/66 has
Wilson95 upper bound above 5%, and the result is not 0/N.

v4 is now near-closed on point estimate.
v2 remains the main unsolved pure-CTM blocker.
```

Video audit for the new v4 layer:

```text
run:
  benchmark_runs/video_compare_nf_ctm_v4_resprofile_w0002_warnfix_2026-05-31/v4

120-frame video totals:
  clean    H/Hd = 335/142
  poisoned H/Hd = 395/39
  NF-CTM   H/Hd = 273/0

warning:
  ctm_head_collapse_on_oga_video
```

Interpretation:

```text
The new v4 layer is strong on source-disjoint augmented images, but it is not a
video-close solution.  On video it suppresses head/source evidence to zero,
which violates semantic OGA repair behavior.  The video audit script was updated
to flag this as `ctm_head_collapse_on_oga_video`, so this failure cannot be
hidden by lower helmet counts.
```
