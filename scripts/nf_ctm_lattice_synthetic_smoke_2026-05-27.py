#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_security_gate.detox.nf_ctm_lattice import (  # noqa: E402
    LatticeCTMConfig,
    LatticeLossConfig,
    LatticeTrainConfig,
    make_lattice_synthetic_features,
    train_lattice_nf_ctm,
)


def main() -> int:
    p = argparse.ArgumentParser(description="NF-CTM Lattice synthetic no-anchor smoke")
    p.add_argument("--out", default="runs/nf_ctm_lattice_smoke")
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--width", type=int, default=8)
    args = p.parse_args()
    invalid, valid, yi, yv, readout = make_lattice_synthetic_features(
        n_invalid=args.n,
        n_valid=args.n,
        channels=args.channels,
        height=args.height,
        width=args.width,
    )
    ctm_cfg = LatticeCTMConfig(
        channels=args.channels,
        thought_steps=5,
        hidden_dim=8,
        sync_gain=0.45,
        step_size=0.12,
        init_sync_weight_std=2e-3,
        spatial_radii=(1,),
        use_field_order_edges=True,
        use_channel_order_edges=True,
    )
    loss_cfg = LatticeLossConfig(
        task_weight=1.0,
        paired_sync_weight=0.03,
        separation_weight=0.01,
        kinetic_weight=0.01,
        invalid_motion_weight=0.002,
        valid_motion_weight=0.06,
        max_valid_rms=2.0,
        max_invalid_rms=7.0,
    )
    train_cfg = LatticeTrainConfig(steps=args.steps, lr=2.5e-3, batch_size=16, device="cpu", log_every=max(1, args.steps // 4))
    _layer, result = train_lattice_nf_ctm(invalid, valid, yi, yv, readout, ctm_cfg=ctm_cfg, loss_cfg=loss_cfg, train_cfg=train_cfg, out_dir=args.out)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
