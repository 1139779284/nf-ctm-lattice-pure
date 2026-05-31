# NF-CTM Lattice v4-pure: synchronization-gauge corrected CTM neuron-field detox

## Motivation

The v3-unified run disclosed a genuine limitation: a single P3-only NF-CTM lattice can learn a nearly uniform residual field.  Such a residual can suppress target evidence on invalid triggered examples but also moves clean valid examples, causing clean recall loss.  This is not solved by adding a new defense module; it is a defect in the CTM dynamics: the raw synchronization drive contains an unconstrained zero-order spatial component.

## Algorithmic correction

Let `F_t` be the CTM neuron-field state and let `S_t` be the raw CTM synchronization field.  The previous update used

```text
D_t = sum_e w_e S_{t,e} + b
F_{t+1} = F_t + eta * gate_t * tanh(temporal_t + gamma * D_t)
```

A spatially uniform component of `D_t` permits a global constant residual.  v4-pure introduces a CTM gauge fixing:

```text
G_t = sum_e w_e S_{t,e}
D_t = G_t + b
Pi_rho(D_t) = D_t - rho * mean_spatial(D_t)
F_{t+1} = F_t + eta * gate_t * tanh(temporal_t + gamma * Pi_rho(D_t))
```

where `rho` is `sync_drive_dc_suppression`.  The raw synchronization recurrence is unchanged: raw products, learned decays, and alpha/beta accumulation remain intact.  The projection is applied after the learned CTM drive bias is added, so the full update drive cannot hide a constant all-spatial suppression term.

An additional stricter switch, `total_drive_dc_suppression`, applies the same zero-order projection after the private temporal drive and synchronized drive are combined:

```text
U_t = temporal_t + gamma * Pi_rho(D_t)
U'_t = U_t - rho_total * mean_spatial(U_t)
F_{t+1} = F_t + eta * gate_t * tanh(U'_t)
```

This closes the loophole where the temporal path could recreate a global constant residual even after `sync_drive` was gauge-fixed.

This is not a frequency branch, spatial token, adapter, score rule, or clean-anchor prior.  It is an internal CTM constraint on the recurrent synchronization drive.

## Thought-energy concentration

A second optional term acts on the CTM trajectory itself:

```text
motion = |F_T - F_0|
Hoyer(motion) = (sqrt(N) - ||motion||_1 / ||motion||_2) / (sqrt(N) - 1)
L_focus = relu(target - Hoyer(motion))^2
```

It discourages globally uniform invalid motion and rewards structured thought motion.  It is not pruning or channel selection because no channel is removed and no external rule selects channels.

## Expected effect

- v2 visible-patch OGA: less global damping should reduce clean collapse when ASR is suppressed.
- v4 semantic OGA: DC suppression should force input-conditioned thought motion instead of preserving or removing orange-vest evidence globally.
- ODA families: defaults keep the previous successful ODA behavior; hard sync-residual floor remains disabled by default because the uploaded package showed it can break ODA.

## Purity boundary

The core algorithmic object is the CTM lattice layer:

```text
detector neck feature F_0
-> CTM synchronization field S_t
-> recurrent thought update with Pi_rho(D_t)
-> terminal state F_T
-> frozen detector readout
```

The runner uses the frozen detector readout only as a training loss/evaluation interface.  It does not add a runtime guard, output score calibration, clean-anchor model, weight soup, adapter branch, frequency token, or post-hoc editor.  It also does not choose different CTM/loss weights per attack family.

One optional term, `valid_state_fixed_point_weight`, penalises valid-state terminal motion.  Because it can be interpreted as anchor-like regularization, strict pure-CTM runs leave it disabled by default (`0.0`).  It should be reported only as an ablation if enabled.

## What is still not claimed

This package only passed compile/unit/synthetic tests in this environment.  Real YOLO ASR and clean recall must be rerun locally with the uploaded data/weights.  The proper claim after local evaluation should separate:

- ASR strict-pass;
- clean-safe pass;
- combined ASR+clean pass.

The YOLO runner now reports source-disjoint augmented evaluation splits: source file stems are compared before the first `__`, so augmented siblings of training images are excluded from evaluation.  This is an evaluation hygiene rule, not an algorithm branch.

## Validation added in this revision

- Public runner no longer imports helper code from the old `ctm_syncflow` purifier namespace; neutral YOLO I/O helpers live under `nf_ctm_lattice/yolo_io.py`.
- Public API no longer exposes the old `identity_clean` naming.
- Unit tests include source-disjoint split checks, forbidden-module import checks, anti-sandwich config-constructor checks, and DC-projection tests.
