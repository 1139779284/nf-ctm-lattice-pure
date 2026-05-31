# NF-CTM Lattice no-sandwich repair plan (2026-05-28)

## Hard rule

NF-CTM Lattice must remain a pure CTM-style dynamical algorithm.  It may use the
poisoned detector as a frozen task readout during training/evaluation, but it
must not become a detector-side repair stack.

Forbidden as the core method:

- clean-anchor interpolation, `W_clean`, weight soup, or checkpoint averaging;
- runtime guard, sample-dependent switch, output calibration, threshold/NMS
  editing, or post-hoc box/score repair;
- CNN adapter residual, feature converter, ordinary convolutional purifier, or
  extra detector head;
- frequency token, spatial token, trigger token, or per-family/per-attack branch.

Allowed:

- recurrent CTM neuron-field dynamics on native detector feature maps;
- raw synchronization fields, private temporal processors, adaptive CTM gates;
- CTM-internal losses over states, trajectories, residuals, gates, and sync
  signatures;
- one frozen detector readout used only to define task loss and report ASR /
  clean recall.

## Audit result

The pure core can stay:

- `model_security_gate/detox/nf_ctm_lattice/neuron_field.py`
  - `LatticeNFCTMNeuronField` treats `(channel, y, x)` activations as CTM
    neurons.
  - `_sync_step()` uses channel/spatial/field-order synchronization edges.
  - `sync_drive_dc_suppression` and `total_drive_dc_suppression` are CTM gauge
    constraints, not token branches or post-hoc edits.
- `model_security_gate/detox/nf_ctm_lattice/objective.py`
  - `label_conditioned_sync_attractor_loss()`;
  - `residual_decorrelation_loss()`;
  - `trajectory_valid_loss()` / `trajectory_invalid_floor_loss()`;
  - `thought_concentration_loss()`;
  - `gate_activity_loss()` and `state_homeostasis_loss()`.

Risky components:

- `valid_decode_geometry_loss()` in
  `scripts/nf_ctm_lattice_yolo_multiscale_v1_2026-05-27.py` must stay as a
  diagnostic/ablation only.  It matches CTM-modified raw detector output against
  frozen passthrough output on valid images, so reviewers may read it as
  teacher/passthrough distillation.
- `valid_state_fixed_point_weight` is acceptable only as an optional CTM
  fixed-point ablation, with strict pure default `0.0`.
- `nf_ctm_lattice_yolo_multiscale_v1_2026-05-27.py` is not yet the final pure
  algorithm.  It registers independent CTM hooks at `16,19,22` and averages
  per-layer regularizers.  That is a useful negative control, but it can look
  like a multi-layer sandwich unless replaced by one coupled CTM state.

## Why the current behavior keeps happening

Source-disjoint v2 results show a structural split:

| Run | Layer(s) | Defended ASR | Clean recall | Interpretation |
|---|---:|---:|---:|---|
| `nf_ctm_lattice_v4_strict_pure_v2_src20_trajvalid_800_2026-05-27` | P3 | `0/242` | `59/60 -> 37/60` | reaches trigger, damages clean |
| `nf_ctm_lattice_v4_v2_neck19_src20_traj_800_2026-05-27` | neck19 | `141/242` | `59/60 -> 57/60` | preserves clean, misses trigger |
| `nf_ctm_lattice_multiscale_v1_v2_src20_800_2026-05-27` | 16,19,22 | `92/242` | `59/60 -> 57/60` | clean safe, P3 suppression diluted |
| `nf_ctm_lattice_multiscale_v1_v2_src20_nodecode_800_2026-05-27` | 16,19,22 | `45/242` | `59/60 -> 55/60` | better ASR, still not closed |

This is not mainly a tuning problem.  P3 has trigger reach but weak object
context; later scales have object context but weak trigger reach.  Independent
multi-hook training improves clean preservation but does not bind these roles
inside one CTM dynamical state.

## Required pure repair

Replace independent hooks with a unified multi-scale CTM field:

```text
F_s^t,  s in {P3, P4, P5}
```

Local CTM synchronization:

```text
S_s^t(e, u) = sync(F_s^t(u), F_s^t(N_e(u)))
D_s^t = sum_e w_{s,e} S_s^t(e) + b_s
```

Scale-order synchronization:

```text
O_s^t = mean_u S_s^t(u)
C_s^t = sum_{r != s} A_{s<-r}(O_r^t - O_s^t)
```

Joint recurrent update:

```text
U_s^t = T_s(history_s^t) + gamma_s Pi(D_s^t) + beta_s C_s^t
F_s^{t+1} = F_s^t + eta_s G_s^t tanh(U_s^t)
```

`C_s^t` is the key difference from a sandwich: it is CTM order-parameter
coupling between native feature fields, not an adapter, token, guard, or
post-hoc detector rule.

Recommended final loss:

```text
L = L_task(F_{P3,P4,P5}^T)
  + lambda_attr L_label_attr(sync trajectories)
  + lambda_cross L_cross_scale_order
  + lambda_focus L_invalid_thought_concentration
  + lambda_valid L_valid_short_path
  + lambda_decorr L_cross_label_residual_orthogonality
  + lambda_kin L_kinetic
```

Only `L_task` faces the frozen detector readout.  All other terms must be
CTM-internal.

## Implementation status

Added after this audit:

- `model_security_gate/detox/nf_ctm_lattice/multiscale_field.py`
  - `CoupledMultiScaleLatticeNFCTM`;
  - `CoupledMultiScaleLatticeCTMTrace`;
  - `cross_scale_order_consistency_loss()`.
- `model_security_gate/detox/nf_ctm_lattice/yolo_hook.py`
  - `forward_with_coupled_lattice_nf_ctm()`.
- `scripts/nf_ctm_lattice_yolo_multiscale_v2_coupled_2026-05-28.py`
  - runnable v2 YOLO runner using the coupled CTM field;
  - no `valid_decode_geometry` / passthrough geometry matching arguments.

The new module implements the required joint recurrence: all selected native
feature scales advance inside one thought loop, and cross-scale communication is
only through CTM order parameters and raw synchronization-energy fields.  There
is no convolutional projection, token branch, clean anchor, or detector-output
matching.  The YOLO helper uses a two-pass frozen detector interface: first
capture native multi-scale features, then inject the coupled CTM terminal states
in a second pass.  This keeps the detector as a readout interface and avoids
independent per-layer CTM plugins.

The coupled module also exposes an optional `cross_context_gate_strength`: if
enabled, CTM motion is damped where cross-scale synchronization-energy fields
agree and allowed where those fields disagree.  This is meant to express the
pure dynamical rule "multi-scale agreement should be a stable thought state";
it does not inspect boxes, logits, clean anchors, or detector post-processing.

The v2 runner also has an optional `scale_balanced_task_weight`.  This is a
training-time detector-closed readout used to reduce gradient starvation across
native scales.  Each selected terminal CTM state is judged through the same
frozen detector readout, and the losses are averaged.  It is not a runtime
branch, score calibration, clean model, or post-hoc edit.

The runner supports asymmetric training readouts: invalid samples can use a
dense `topk_lse` target-evidence readout while valid samples keep a sparse
`max` readout.  The intent is to push down distributed trigger-induced evidence
without forcing clean valid images to inflate many anchors.  This is a
training-only frozen-readout choice, not runtime score calibration.

`scripts/nf_ctm_lattice_yolo_multiscale_v1_2026-05-27.py` remains an experiment
/ negative control.  Its `--valid-decode-geometry-weight` default is now `0.0`
so strict no-sandwich runs do not accidentally use passthrough geometry
matching.

## Paper framing

Do not frame this as "YOLO model repair" or "AutoDetox pipeline".  Frame it as:

```text
Neural-field continuous-thought purification for object-detection backdoors
```

Primary sources checked:

- Continuous Thought Machines: https://arxiv.org/abs/2505.05522
- BadDet object-detection backdoor benchmark: https://arxiv.org/abs/2205.14497
- Object-detection backdoor defense via module inconsistency / reset-finetune:
  https://arxiv.org/abs/2409.16057
