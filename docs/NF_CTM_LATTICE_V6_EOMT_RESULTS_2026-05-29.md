# NF-CTM Lattice v6 Edge-Order Moment Transport Results

Date: 2026-05-29

## Goal

The v6 line tries to address the v2 visible-patch OGA source-level bottleneck
without using cross-scale spatial-map transfer, clean-anchor models, weight
soup, runtime guards, score calibration, CNN adapters, or detector-side
post-processing.

## Algorithm Change

v6 replaces scalar order-flow with Edge-Order Moment Transport (EOMT).

For every CTM thought step and every selected detector scale, raw
synchronization fields are compressed into location-free edge-family moments:

```text
edge families: spatial edge, channel edge, field-order edge, local conflict edge
moments per family: mean, RMS, top-k mean, concentration
```

The resulting vector is:

```text
E_t,s(x) in R^(4 x 4)
```

Cross-scale dynamic coupling uses only these edge-order moment scalars. It does
not transmit an H x W map between scales. The transport loss uses the thought
trajectory:

```text
q(x) = [E_1 - E_0, ..., E_T - E_(T-1)]
```

and encourages trigger samples to share source-invariant edge-order transport
while keeping clean valid samples quiet.

## Verified Code Changes

Core code:

```text
model_security_gate/detox/nf_ctm_lattice/multiscale_field.py
model_security_gate/detox/nf_ctm_lattice/__init__.py
scripts/nf_ctm_lattice_yolo_multiscale_v2_orderflow_2026-05-28.py
scripts/nf_ctm_lattice_yolo_multiscale_v2_coupled_2026-05-28.py
tests/test_nf_ctm_lattice.py
```

Verification:

```text
python -m compileall -q model_security_gate/detox/nf_ctm_lattice scripts tests/test_nf_ctm_lattice.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  tests/test_nf_ctm_lattice.py tests/test_nf_ctm_closed_loop.py tests/test_nf_ctm_detox.py
```

Result:

```text
49 passed
```

## v2 Visible-Patch OGA Results

### v6 EOMT dynamic pure, 300 steps

Run:

```text
D:/clean_yolo/benchmark_runs/nf_ctm_lattice_v6_eomt_dynamic_pure_v2_300_2026-05-28
```

Result:

```text
passthrough aug ASR: 231/242 = 95.45%
defended aug ASR:    146/242 = 60.33%
Wilson95 upper:      66.29%
clean recall:        59/60 -> 56/60
clean drop:          3 images = 5.00 pp
strict ASR pass:     false
clean-safe pass:     false
```

This is notable because it reaches the 300-step local-edge baseline while
running with:

```text
cross_field_coupling = 0.0
local_edge_conflict_strength = 0.0
edge_order_moment_coupling = 0.20
```

### v6 EOMT dynamic pure, 300 + 500 continued steps

Run:

```text
D:/clean_yolo/benchmark_runs/nf_ctm_lattice_v6_eomt_dynamic_pure_continue500_v2_2026-05-29
```

Result:

```text
passthrough aug ASR: 231/242 = 95.45%
defended aug ASR:    100/242 = 41.32%
Wilson95 upper:      47.61%
clean recall:        59/60 -> 54/60
clean drop:          5 images = 8.33 pp
strict ASR pass:     false
clean-safe pass:     false
```

Residual failed augmented source groups:

```text
triggerA_014: 11/11
triggerA_022: 11/11
triggerA_028: 11/11
triggerA_035: 11/11
triggerA_036: 11/11
triggerA_042: 11/11
triggerA_052: 11/11
triggerA_055: 11/11
triggerA_030: 10/11
triggerA_008: 1/11
triggerA_051: 1/11
```

## Interpretation

v6 EOMT improves the purity of the strict line: it can replace the old local
spatial gate at the 300-step level. However, it does not yet beat the previous
strict-pure ASR plateau:

```text
previous strict-pure 800-step: 100/242 ASR, clean 55/60
v6 EOMT continued:            100/242 ASR, clean 54/60
```

So v6 is not a final solution. It is a useful algorithmic ablation showing that
location-free edge-order moments are meaningful, but the visible-patch OGA
source-level bottleneck remains open.

## Current Bottleneck

The remaining failures are still whole source-image basins. This indicates that
the current location-free CTM order abstraction can suppress many trigger cases
but cannot yet separate all trigger-causal evidence from source-specific visual
evidence while preserving clean target recall.

The relaxed cross-field result remains better:

```text
relaxed cross_field_coupling=0.01: 44/242 ASR, clean 55/60
```

but it uses cross-scale spatial energy map transfer and should remain an
ablation, not the strict-pure main algorithm.

