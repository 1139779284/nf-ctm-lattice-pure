# NF-CTM Lattice Audit 2026-05-30

This note records the current pure CTM evidence state.  It is intentionally
conservative: no result below should be presented as "fully solved" unless both
source-disjoint image-pool ASR and video behavior pass.

## Scope

Main-algorithm constraints:

- no clean-anchor interpolation;
- no weight soup;
- no runtime guard;
- no score calibration;
- no NMS or box post-processing repair;
- no CNN adapter;
- no spatial/frequency token branch;
- no per-family runtime branch.

The only accepted direction is NF-CTM / CTM-internal recurrent dynamics.

## Historical Strongest Image-Pool Result

Source:

```text
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_v3_final_rerun_2026-05-29\SUMMARY.md
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_v3_final_rerun_2026-05-29_oda\SUMMARY.md
```

Summary:

| family | result |
|---|---|
| v3 SIG OGA | ASR 78/110 -> 0/110, clean 59/60 -> 57/60, pass |
| v2 visible patch OGA | ASR 98/110 -> 0/110, clean 59/60 -> 39/60, clean fail |
| b1-b4 ODA | image-pool ASR 0%, clean 58/60 or 59/60 -> 60/60, pass |

Interpretation:

The image-pool result proves CTM can change the poisoned detector behavior, but
it does not prove cross-family robustness.  v2 already showed target-class
over-suppression in the clean metric.

## CTM-Only Video Matrix

Strict video script:

```text
D:\clean_yolo\scripts\_video_compare_nf_ctm_lattice_2026-05-30.py
```

Strict rerun:

```text
D:\clean_yolo\benchmark_runs\video_compare_nf_ctm_lattice_7family_strict_2026-05-30\cross_family_summary.md
```

Results:

| family | poisoned-clean helmet | CTM-clean helmet | warning |
|---|---:|---:|---|
| v2 | +6 | -443 | low attack activation, CTM under-detection |
| v3 | +197 | -119 | no hard warning, but clean count drops |
| v4 | +97 | -915 | CTM destroyed detection |
| b1 | -80 | +1183 | CTM helmet over-detection |
| b2 | -72 | +524 | CTM helmet over-detection |
| b3 | +27 | +616 | CTM helmet over-detection |
| b4 | +30 | +929 | CTM helmet over-detection |

Interpretation:

The 7-family video matrix does not support a "fully solved" claim.  It exposes a
single underlying failure mode with opposite polarities:

- OGA can become blanket target suppression;
- ODA can become blanket target excitation.

## Evidence Contract Fixes

Implemented on 2026-05-30:

- `yolo_io.py` now has explicit YOLO-style centered letterbox preprocessing.
- The main CTM runners expose `--letterbox-center` and record the policy.
- The video script records model/layer SHA256, CTM config, image size,
  confidence, IoU, crop manifest, and evidence warnings.
- Legacy CTM checkpoints preserve missing historical defaults, especially
  `sync_drive_dc_suppression=0.0`, instead of silently using newer schema
  defaults.

These are evidence fixes, not purification tricks.

## Pure CTM Algorithm Fixes Tried

Implemented on 2026-05-30:

- bounded-margin task loss, so CTM stops pushing once detector evidence crosses
  a finite boundary;
- quiet-state loss for normal target-present and target-absent clean states;
- local edge-conflict update gate, so low-conflict lattice cells are closer to
  fixed points.
- exact top-k counterfactual difference support for same-source trigger
  variants.  This replaced the earlier threshold support, which could
  degenerate to an all-lattice support when most difference values were zero.

Focused probe:

```text
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_conflict_gate_center_probe_2026-05-30\SUMMARY.md
```

Image-pool result:

| family | passthrough aug ASR | defended aug ASR | clean |
|---|---:|---:|---|
| v2 | 99/110 | 97/110 | 58/60 -> 59/60 |
| v4 | 41/66 | 30/66 | 59/60 -> 59/60 |
| b1 | 231/231 | 187/231 | 59/60 -> 58/60 |
| b4 | 209/209 | 188/209 | 56/60 -> 58/60 |

Video result:

```text
D:\clean_yolo\benchmark_runs\video_compare_nf_ctm_conflict_gate_center_probe_2026-05-30\cross_family_summary.md
```

| family | CTM-clean helmet |
|---|---:|
| v2 | -143 |
| v4 | -79 |
| b1 | -33 |
| b4 | +62 |

Interpretation:

The new gate substantially reduces destructive video behavior, especially v4
zero-output collapse and ODA over-detection.  However, it is too conservative
and does not close source-disjoint ASR.  This is a useful diagnostic but not a
final algorithm.

Counterfactual-difference probe:

```text
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_cf_diff_oga_probe_2026-05-30\v2\record.json
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_cf_diff_v4_smoke_2026-05-30\SUMMARY.md
```

| family | passthrough aug ASR | defended aug ASR | clean |
|---|---:|---:|---|
| v2 | 99/110 | 86/110 | 58/60 -> 59/60 |
| v4 smoke 120 steps | 41/66 | 36/66 | 59/60 -> 59/60 |

Interpretation:

Counterfactual-difference support is directionally better than the local gate
alone on v2, but it is still far from closed.  It supports the diagnosis that
the old ASR=0 result came from a global target-evidence actuator rather than a
stable trigger-causal transform.

Clean-trigger counterfactual pair smoke:

```text
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_cf_pair_v2_smoke_2026-05-30\SUMMARY.md
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_cf_pair_fixed_v2_smoke_2026-05-30\SUMMARY.md
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_task_tangent_v2_smoke_2026-05-30\SUMMARY.md
```

| family | passthrough aug ASR | defended aug ASR | clean |
|---|---:|---:|---|
| v2 pair smoke | 99/110 | 99/110 | 58/60 -> 58/60 |
| v2 fixed-pair + direction | 99/110 | 99/110 | 58/60 -> 58/60 |
| v2 task-tangent smoke | 99/110 | 99/110 | 58/60 -> 58/60 |

Interpretation:

The paired clean-trigger support loss is safe but currently too conservative:
it constrains where CTM may move, but does not yet define a strong enough
direction for how trigger evidence should be neutralized.

Additional 2026-05-30 audit details:

- The old clean-trigger pairing was unsafe because it allowed digit-only
  fallbacks such as `triggerA_001` -> an unrelated `helm_*001` image.  This has
  been removed.
- The runner now uses audited A/B staging provenance when available:
  `triggerA_*.jpg` maps through
  `D:\clean_yolo\datasets\mask_bd\ab_staging_manifest.json` to
  `D:\clean_yolo\A\*.png`.  If no manifest match exists, visual-nearest pairing
  is accepted only below a strict distance threshold and every pair is recorded
  in `record.json`.
- Unmatched counterfactual pairs are masked out of `cf_pair` training.  They are
  no longer silently replaced by the triggered image.
- Even with 60/60 matched pairs, v2 did not improve.  This means the bottleneck
  is not merely missing clean-source evidence.
- A training-only task-tangent field loss was tested.  It uses the frozen
  downstream readout Jacobian to identify target-evidence-sensitive CTM
  directions, but it still did not move v2 under the short probe.  This is not a
  runtime guard and it does not alter the inference graph.
- The task-tangent implementation was changed to a cacheable training-only
  tangent field because per-step Jacobian computation was too slow.  A 1-step
  sanity run confirmed the cache path works, but this remains an expensive
  ablation rather than the main algorithm.

Local edge-conflict amplification smoke:

```text
D:\clean_yolo\benchmark_runs\nf_ctm_lattice_conflict_amplify_v2_120_2026-05-30\SUMMARY.md
```

| family | passthrough aug ASR | defended aug ASR | clean |
|---|---:|---:|---|
| v2 conflict amplify 120 steps | 99/110 | 97/110 | 58/60 -> 58/60 |

Interpretation:

The gate extension is pure CTM: it only changes how local synchronization-edge
conflict modulates the recurrent update, and it can now amplify high-conflict
cells with a bounded ceiling.  It did not harm clean recall, but it also did not
break the v2 visible-patch plateau.

## Current Diagnosis

The current single-neck CTM field can act as a class-evidence actuator but does
not yet reliably learn trigger-causal dynamics.

The most likely limitation is expression and supervision, not a small parameter
mistake:

- one P3 hook is too blunt for visible/semantic OGA and crop-level ODA;
- CE-style or boundary-only task supervision is not enough to identify trigger
  cause rather than target evidence;
- image-pool "any target fired" can hide video-level count explosions or
  detection collapse;
- older checkpoints without gauge suppression should not be used as clean final
  evidence.

## Next Pure CTM Direction

The next version should not add an external model or detector-side guard.  It
should instead make the CTM objective causal within the recurrent field:

1. Train on counterfactual pairs and gate motion by CTM-internal difference
   support, not by class target alone.  The support now exists in code, but it
   still needs a directional CTM objective.
2. Use multi-scale CTM only if it is implemented as a single recurrent CTM field
   inside one forward contract, not as a detector-CTM-detector sandwich.
3. Add a polarity/budget law inside CTM trajectories: OGA may not globally turn
   target evidence off; ODA may not globally turn it on.  This must be measured
   from CTM state motion, sync orders, and gate activity, not from post-hoc
   detector outputs.

## Do Not Claim

Do not claim the project is currently all-strong or all-perfect across poison
families.  The honest current state is:

```text
v3 SIG OGA: strong image-pool result, video still needs clean-loss check.
v2 visible OGA: ASR can be killed only with clean collapse in historical run.
v4 semantic OGA: historical video CTM collapses; new gate reduces collapse but ASR fails.
b1-b4 ODA: image-pool can pass, but historical video over-detects; new gate reduces over-detection but ASR fails.
```
