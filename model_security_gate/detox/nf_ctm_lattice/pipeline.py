from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .neuron_field import LatticeNFCTMNeuronField
from .objective import lattice_ctm_objective, sync_distance_value
from .schema import LatticeCTMConfig, LatticeLossConfig, LatticeTrainConfig, LatticeTrainResult


class FrozenSyntheticReadout(nn.Module):
    """Frozen probe with an object path and two trigger shortcut paths.

    This synthetic readout is intentionally small and is not a YOLO-specific
    purification rule.  It checks whether the CTM neuron field can suppress a
    local shortcut and a field-level diffuse shortcut while keeping object
    evidence.
    """

    def __init__(self, channels: int, object_channel: int = 0, patch_channel: int = 1, diffuse_channel: int = 2):
        super().__init__()
        self.channels = int(channels)
        self.object_channel = int(object_channel)
        self.patch_channel = int(patch_channel)
        self.diffuse_channel = int(diffuse_channel)
        self.register_buffer("scale", torch.tensor(4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        y0, y1 = h // 4, max(h // 4 + 1, h // 2)
        x0, x1 = w // 4, max(w // 4 + 1, w // 2)
        object_score = x[:, self.object_channel, y0:y1, x0:x1].mean(dim=(1, 2))
        patch_score = x[:, self.patch_channel, : max(1, h // 3), : max(1, w // 3)].mean(dim=(1, 2))
        diffuse_score = x[:, self.diffuse_channel].mean(dim=(1, 2))
        target = self.scale * (object_score + 0.85 * patch_score + 0.75 * diffuse_score - 0.50)
        return torch.stack([-target, target], dim=1)


def make_lattice_synthetic_features(
    *,
    n_invalid: int = 64,
    n_valid: int = 64,
    channels: int = 8,
    height: int = 8,
    width: int = 8,
    seed: int = 20260526,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, FrozenSyntheticReadout]:
    g = torch.Generator().manual_seed(int(seed))
    invalid = torch.randn(int(n_invalid), int(channels), int(height), int(width), generator=g) * 0.05
    valid = torch.randn(int(n_valid), int(channels), int(height), int(width), generator=g) * 0.05
    h, w = int(height), int(width)
    # valid object evidence: target object occupies a compact region.
    valid[:, 0, h // 4: max(h // 4 + 1, h // 2), w // 4: max(w // 4 + 1, w // 2)] += 1.25
    # invalid local shortcut: patch-like local evidence sharing cells with potential object evidence.
    invalid[:, 1, : max(1, h // 3), : max(1, w // 3)] += 1.25
    # invalid diffuse shortcut: a global field-order contaminant, not a frequency token.
    invalid[:, 2] += 0.65
    yi = torch.zeros(int(n_invalid), dtype=torch.long)
    yv = torch.ones(int(n_valid), dtype=torch.long)
    return invalid, valid, yi, yv, FrozenSyntheticReadout(int(channels))


def error_rate(readout: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        pred = readout(features).argmax(dim=1)
    return float((pred != labels).float().mean().cpu())


def rms_motion(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt((a - b).pow(2).mean() + 1e-12).detach().cpu())


def train_lattice_nf_ctm(
    invalid: torch.Tensor,
    valid: torch.Tensor,
    invalid_labels: torch.Tensor,
    valid_labels: torch.Tensor,
    readout: nn.Module,
    *,
    ctm_cfg: LatticeCTMConfig | None = None,
    loss_cfg: LatticeLossConfig | None = None,
    train_cfg: LatticeTrainConfig | None = None,
    out_dir: str | Path | None = None,
) -> tuple[LatticeNFCTMNeuronField, LatticeTrainResult]:
    train_cfg = train_cfg or LatticeTrainConfig()
    device = torch.device(train_cfg.device)
    invalid = invalid.to(device)
    valid = valid.to(device)
    invalid_labels = invalid_labels.to(device)
    valid_labels = valid_labels.to(device)
    readout = readout.to(device).eval()
    for p in readout.parameters():
        p.requires_grad_(False)
    ctm_cfg = ctm_cfg or LatticeCTMConfig(channels=int(invalid.shape[1]))
    loss_cfg = loss_cfg or LatticeLossConfig()
    layer = LatticeNFCTMNeuronField(ctm_cfg).to(device)
    opt = torch.optim.AdamW(layer.parameters(), lr=float(train_cfg.lr), weight_decay=0.0)
    before_invalid = error_rate(readout, invalid, invalid_labels)
    before_valid = error_rate(readout, valid, valid_labels)
    with torch.no_grad():
        before_trace_i = layer(invalid[: min(len(invalid), len(valid))], return_trace=True)
        before_trace_v = layer(valid[: before_trace_i.final.shape[0]], return_trace=True)
        before_sync = sync_distance_value(before_trace_i, before_trace_v)
    history: list[dict[str, float]] = []
    n_invalid = int(invalid.shape[0])
    n_valid = int(valid.shape[0])
    gen = torch.Generator(device=device).manual_seed(int(train_cfg.seed))
    for step in range(int(train_cfg.steps)):
        bi = torch.randint(0, n_invalid, (int(train_cfg.batch_size),), generator=gen, device=device)
        bv = torch.randint(0, n_valid, (int(train_cfg.batch_size),), generator=gen, device=device)
        inv_b = invalid[bi]
        val_b = valid[bv]
        yi = invalid_labels[bi]
        yv = valid_labels[bv]
        ti = layer(inv_b, return_trace=True)
        tv = layer(val_b, return_trace=True)
        loss, stats = lattice_ctm_objective(
            inv_b,
            val_b,
            ti,
            tv,
            readout=readout,
            invalid_labels=yi,
            valid_labels=yv,
            cfg=loss_cfg,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 5.0)
        opt.step()
        if step % max(1, int(train_cfg.log_every)) == 0 or step == int(train_cfg.steps) - 1:
            stats = dict(stats)
            stats["step"] = float(step)
            history.append(stats)
    layer.eval()
    with torch.no_grad():
        invalid_after = layer(invalid)
        valid_after = layer(valid)
        after_invalid = error_rate(readout, invalid_after, invalid_labels)
        after_valid = error_rate(readout, valid_after, valid_labels)
        after_trace_i = layer(invalid[: min(len(invalid), len(valid))], return_trace=True)
        after_trace_v = layer(valid[: after_trace_i.final.shape[0]], return_trace=True)
        after_sync = sync_distance_value(after_trace_i, after_trace_v)
        invalid_motion = rms_motion(invalid_after, invalid)
        valid_motion = rms_motion(valid_after, valid)
    result = LatticeTrainResult(
        final_loss=float(history[-1]["loss"] if history else 0.0),
        before_invalid_error_rate=before_invalid,
        after_invalid_error_rate=after_invalid,
        before_valid_error_rate=before_valid,
        after_valid_error_rate=after_valid,
        before_paired_sync_distance=before_sync,
        after_paired_sync_distance=after_sync,
        clean_motion_rms=valid_motion,
        trigger_motion_rms=invalid_motion,
        history=history,
        config={"ctm": ctm_cfg.to_dict(), "loss": loss_cfg.to_dict(), "train": train_cfg.to_dict()},
    )
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "lattice_nf_ctm_result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        torch.save({"state_dict": layer.state_dict(), "ctm_config": ctm_cfg.to_dict()}, out / "lattice_nf_ctm_layer.pt")
    return layer, result
