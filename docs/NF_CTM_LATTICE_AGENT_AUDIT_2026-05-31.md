# NF-CTM Lattice Agent Audit 2026-05-31

## Purpose

This note records the multi-agent audit of the current NF-CTM Lattice line.  It
separates the paper-safe pure CTM path from diagnostic engineering ablations, and
summarizes the latest evidence risk around static ASR, clean recall, and video
over-detection.

## Paper-Safe Pure CTM Path

Use only the single-neck in-flow CTM hook as the main algorithm profile:

```text
YOLO neck feature tensor
-> LatticeNFCTMNeuronField recurrent synchronization dynamics
-> terminal CTM state replaces the same neck tensor
-> frozen downstream detector head
```

Paper-safe components:

```text
LatticeNFCTMNeuronField recurrence
channel/spatial/field-order synchronization edges
adaptive update gate
sync-drive / total-drive DC gauge fixing
local edge-conflict modulation
bounded readout task boundary
label-conditioned sync attractor
kinetic/state/homeostasis/gate regularizers
valid fixed-point / residual decorrelation
trajectory invalid floor
thought concentration
thought active-area / spatial entropy / sync-support constraints
same-source CTM difference support without clean-reference lookup
```

The strict profile must not construct clean-trigger lookup, decoded boxes, score
calibration, postprocess repair, weight soup, clean anchor, CNN adapter, runtime
guard, or detector-CTM-detector multi-pass logic.

## Ablation / Diagnostic Only

These can stay in the repository for debugging and ablation, but should not be
reported as the main pure CTM algorithm:

```text
forward_with_coupled_lattice_nf_ctm multi-scale runner
cross-scale HxW energy map interpolation
valid_decode_geometry_loss
target_support_compactness_loss
target_natural_support_loss
oda_source_balance_loss
oga_source_preservation_loss
oga_replacement_margin_loss
oga_local_source_support_loss
source_valid_fixed_point_loss when driven by detector scores
clean-trigger counterfactual pairing / visual nearest clean lookup
task_tangent_field_loss from downstream detector Jacobian
video-scope ODA crop auxiliary training
```

Reason: these read detector boxes, raw scores, IoU/top-k anchor structure,
counterfactual clean images, video-specific crops, or run detector features in a
way that reviewers can reasonably interpret as YOLO-specific repair rather than
pure CTM dynamics.

## Latest Evidence State

The credible strongest historical line remains the v3-final / v3-final-rerun
series for static ODA and v3 OGA evidence:

```text
ODA b1-b4 historical source-disjoint static evidence: ASR 0%, clean safe PASS
v3 OGA historical source-disjoint static evidence: ASR 0%, clean about 57-59/60
v3 video audit: poisoned head-anchor helmet hit rate 61.4% -> 2.1%, matching clean 2.1%
```

The 2026-05-31 b1 experiments are not closed:

```text
neck19: static ASR 31/517 = 6.0%, clean 30/30, but video helmet count 1931 vs clean 152
neck13: static ASR 67/517 = 13.0%, clean 30/30, but video helmet count 1010 vs clean 152
spatial_entropy: static ASR 91/517 = 17.6%, clean 30/30, video 273 vs clean 152
video_scope_oda: video 151 vs clean 152, but static ASR 74.2%
natural_count: video 163 vs clean 152, but static ASR 78.9%
```

Conclusion: low static ASR alone is not sufficient.  A CTM run is credible only
if it also preserves clean recall and does not create video over-detection,
duplicate tiny boxes, head collapse, or zero-output failure.

## Code Fixes Applied In This Audit

```text
1. ODA source-balance readout now activates when --oda-source-balance-weight is enabled.
2. Video-scope source-balance now computes ref_vs_val_scores whenever that loss needs it.
3. thought_spatial_entropy_loss no longer treats zero CTM motion as maximum entropy.
4. train_history now records loss_base and loss_total, with loss set to the actual total backward loss.
5. Clean counterfactual lookup is disabled unless an explicit clean-pair / OGA auxiliary ablation needs it.
6. record.json now includes purity.paper_main_profile and strict_pure_flags.
```

## Next Minimal Experiment Matrix

Run only a small matrix until the metric contradiction is resolved:

```text
1. v3-final-rerun ODA b1/b4 video audit replication
2. b1 neck19 as a low-ASR pathological sentinel
3. b1 spatial_entropy as the current CTM-internal candidate
4. b1 video_scope_oda or natural_count as the video-aligned but high-ASR sentinel
5. v3 OGA video/static replication
6. v4 clean/ASR tradeoff replication
```

Acceptance line:

```text
static ASR <= 5%
Wilson95 <= 10%
clean recall drop <= 2 images or agreed mAP equivalent
video helmet/head counts close to clean
no head collapse
no over-detection / zero-output warning
```

## Research Direction

The current blocker is not a small hyperparameter issue.  Single-hook CTM can
move static ASR, but ODA video shows a field-geometry problem: some settings
restore target evidence by globally exciting helmet-like evidence.  The next
paper-safe improvement should stay CTM-internal and target instance-level field
geometry through synchronization dynamics, not detector score/box repair.
