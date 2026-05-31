from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

from .neuron_field import LatticeCTMTrace
from .schema import LatticeLossConfig


def _slice_trace(trace: LatticeCTMTrace, n: int) -> LatticeCTMTrace:
    return LatticeCTMTrace(
        final=trace.final[:n],
        states=[s[:n] for s in trace.states],
        sync_signatures=trace.sync_signatures[:n],
        sync_fields=[s[:n] for s in trace.sync_fields],
        update_gates=[g[:n] for g in trace.update_gates],
    )


def sync_trajectory_distance(a: LatticeCTMTrace, b: LatticeCTMTrace) -> torch.Tensor:
    if a.sync_signatures.shape != b.sync_signatures.shape:
        raise ValueError(f"sync shape mismatch: {tuple(a.sync_signatures.shape)} vs {tuple(b.sync_signatures.shape)}")
    return F.mse_loss(a.sync_signatures, b.sync_signatures)


def sync_distance_value(a: LatticeCTMTrace, b: LatticeCTMTrace) -> float:
    return float(torch.sqrt(sync_trajectory_distance(a, b) + 1e-12).detach().cpu())


def kinetic_loss(trace: LatticeCTMTrace) -> torch.Tensor:
    if len(trace.states) < 2:
        return trace.final.sum() * 0.0
    return torch.stack([(b - a).pow(2).mean() for a, b in zip(trace.states[:-1], trace.states[1:])]).mean()


def state_motion_loss(trace: LatticeCTMTrace, ref: torch.Tensor, max_rms: float) -> torch.Tensor:
    mse = (trace.final - ref).pow(2).mean(dim=(1, 2, 3))
    return F.relu(mse - float(max_rms) ** 2).pow(2).mean()


def sync_separation_loss(trace: LatticeCTMTrace, margin: float) -> torch.Tensor:
    sig = trace.sync_signatures
    if sig.shape[0] < 2 or sig.shape[1] == 0:
        return trace.final.sum() * 0.0
    flat = sig.reshape(sig.shape[0], -1)
    dist2 = (flat[:, None, :] - flat[None, :, :]).pow(2).mean(dim=-1)
    mask = ~torch.eye(dist2.shape[0], dtype=torch.bool, device=dist2.device)
    return F.relu(float(margin) ** 2 - dist2[mask]).pow(2).mean()


def label_conditioned_sync_attractor_loss(
    invalid_trace: LatticeCTMTrace,
    valid_trace: LatticeCTMTrace,
    invalid_labels: torch.Tensor,
    valid_labels: torch.Tensor,
    *,
    same_weight: float = 1.0,
    diff_weight: float = 1.0,
    margin: float = 0.30,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CTM-native label-conditioned attractor objective.

    v1 pulled invalid and valid trajectories together unconditionally.  That is
    correct for ODA when both are target-present, but mathematically wrong for
    OGA, where target-absent invalid examples should not be aligned with
    target-present valid object examples.  This objective uses only CTM sync
    trajectories and task labels:

    - same labels: synchronize into a shared attractor;
    - different labels: maintain a margin between attractors.

    It is not a detector score rule and does not use an external clean model.
    """
    sig_i = invalid_trace.sync_signatures
    sig_v = valid_trace.sync_signatures
    if sig_i.shape[1:] != sig_v.shape[1:]:
        raise ValueError(f"sync shape mismatch: {tuple(sig_i.shape)} vs {tuple(sig_v.shape)}")
    if sig_i.shape[0] + sig_v.shape[0] < 2 or sig_i.shape[1] == 0:
        zero = invalid_trace.final.sum() * 0.0
        return zero, {"same_attr": 0.0, "diff_attr": 0.0, "same_pairs": 0.0, "diff_pairs": 0.0}
    sig = torch.cat([sig_i, sig_v], dim=0).reshape(sig_i.shape[0] + sig_v.shape[0], -1)
    labels = torch.cat([invalid_labels.view(-1), valid_labels.view(-1)], dim=0).long().to(sig.device)
    dist2 = (sig[:, None, :] - sig[None, :, :]).pow(2).mean(dim=-1)
    eye = torch.eye(dist2.shape[0], dtype=torch.bool, device=dist2.device)
    same = (labels[:, None] == labels[None, :]) & ~eye
    diff = (labels[:, None] != labels[None, :]) & ~eye
    loss = sig.sum() * 0.0
    same_loss = loss
    diff_loss = loss
    if bool(same.any()) and same_weight > 0:
        same_loss = dist2[same].mean()
        loss = loss + float(same_weight) * same_loss
    if bool(diff.any()) and diff_weight > 0:
        diff_loss = F.relu(float(margin) ** 2 - dist2[diff]).pow(2).mean()
        loss = loss + float(diff_weight) * diff_loss
    stats = {
        "same_attr": float(same_loss.detach().cpu()),
        "diff_attr": float(diff_loss.detach().cpu()),
        "same_pairs": float(same.float().sum().detach().cpu()),
        "diff_pairs": float(diff.float().sum().detach().cpu()),
    }
    return loss, stats


def bounded_margin_task_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Stop-gradient-once-safe task loss for CTM readout logits.

    The runner supplies logits of the form ``[-m, m]`` where ``m`` is target
    evidence relative to the detector boundary.  Cross entropy keeps pushing
    ``m`` toward +/- infinity, which is exactly the shortcut observed in video:
    OGA turns the target class off everywhere, while ODA turns it on everywhere.
    This loss only asks the terminal CTM state to cross a finite safety margin.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        return F.cross_entropy(logits, labels.long())
    signed_target_margin = logits[:, 1] * labels.float().mul(2.0).sub(1.0)
    return F.relu(float(margin) - signed_target_margin).pow(2).mean()


def gate_activity_loss(trace: LatticeCTMTrace) -> torch.Tensor:
    if not trace.update_gates:
        return trace.final.sum() * 0.0
    return torch.stack([g.mean() for g in trace.update_gates]).mean()


def trajectory_length(trace: LatticeCTMTrace) -> torch.Tensor:
    """Per-sample CTM trajectory length: sum_t ||F_t - F_{t-1}||_2 over
    states.  Returns a (B,) tensor.  Pure CTM: only references the thought
    states already produced by the lattice; no external model.
    """
    if len(trace.states) < 2:
        return trace.final.sum() * 0.0
    deltas = []
    for a, b in zip(trace.states[:-1], trace.states[1:]):
        d = (b - a).reshape(b.shape[0], -1).norm(dim=1)
        deltas.append(d)
    return torch.stack(deltas, dim=1).sum(dim=1)


def trajectory_valid_loss(trace: LatticeCTMTrace, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Squared trajectory length on valid samples, scale-free.  A constant
    residual produces a single hop of size ||c|| repeated T times => loss
    grows quadratically with T.  Forces the recurrent dynamics to be near
    identity on valid inputs, beyond the terminal residual constraint.
    """
    L = trajectory_length(trace)
    ref_norm = ref.reshape(ref.shape[0], -1).norm(dim=1).clamp_min(float(eps))
    return (L / ref_norm).pow(2).mean()


def trajectory_invalid_floor_loss(trace: LatticeCTMTrace, floor: float, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Hinge floor on invalid trajectory length.  Punishes too-short paths
    on invalid (= the constant-residual collapse where every step is the
    same tiny shift).  This rewards the lattice for actually doing work
    on triggers.
    """
    L = trajectory_length(trace)
    ref_norm = ref.reshape(ref.shape[0], -1).norm(dim=1).clamp_min(float(eps))
    relative = L / ref_norm
    return F.relu(float(floor) - relative).pow(2).mean()


def thought_concentration_loss(trace: LatticeCTMTrace, target: float = 0.35, eps: float = 1e-6) -> torch.Tensor:
    """Pure CTM thought-energy concentration loss.

    The visible-patch failure analysis showed a nearly uniform motion field: the
    lattice moved every cell in almost the same direction.  This term measures
    concentration of the CTM trajectory energy without looking at classes,
    anchors, scores, channels, or an external model.  It is a dynamical prior: a
    trigger-correcting thought should be more localized/structured than a global
    constant damping field.

    We use a Hoyer-style concentration score on the terminal motion magnitude,
    but expose it only as a CTM trajectory statistic, not as pruning or channel
    selection.  Loss is zero once concentration >= target.
    """
    if not trace.states:
        return trace.final.sum() * 0.0
    motion = (trace.final - trace.states[0]).abs().reshape(trace.final.shape[0], -1)
    if motion.numel() == 0:
        return trace.final.sum() * 0.0
    n = float(motion.shape[1])
    l1 = motion.sum(dim=1)
    l2 = motion.pow(2).sum(dim=1).sqrt().clamp_min(float(eps))
    # Hoyer concentration in [0,1], 0=uniform, 1=one-hot.
    conc = (n ** 0.5 - l1 / l2) / max(n ** 0.5 - 1.0, 1.0)
    conc = conc.clamp(0.0, 1.0)
    return F.relu(float(target) - conc).pow(2).mean()


def thought_active_area_loss(
    trace: LatticeCTMTrace,
    *,
    max_active_frac: float = 0.08,
    temp: float = 0.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure CTM soft active-area constraint on the recurrent thought motion.

    ODA video failures show the lattice can restore the target class by moving
    too much of the feature field.  This term looks only at the CTM terminal
    motion magnitude and softly limits the fraction of lattice cells whose
    motion exceeds the per-sample mean motion scale.  It does not inspect
    detector boxes, scores, classes, NMS, or an external clean model.
    """
    if not trace.states:
        return trace.final.sum() * 0.0
    motion = (trace.final - trace.states[0]).pow(2).mean(dim=1).sqrt()
    if motion.numel() == 0:
        return trace.final.sum() * 0.0
    scale = motion.mean(dim=(-1, -2), keepdim=True).detach().clamp_min(float(eps))
    active = torch.sigmoid((motion / scale - 1.0) / max(float(temp), 1e-4))
    active_frac = active.mean(dim=(-1, -2))
    return F.relu(active_frac - float(max_active_frac)).pow(2).mean()


def thought_spatial_entropy_loss(
    trace: LatticeCTMTrace,
    *,
    max_effective_frac: float = 0.20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pure CTM spatial entropy constraint on thought motion.

    Hoyer concentration over all channels can still allow spatially broad
    motion if only a few channels move.  This term first compresses terminal
    CTM motion to a spatial lattice energy map, then limits the effective
    spatial support area via normalized entropy.  It uses no detector output,
    labels, boxes, NMS, image mask, token branch, or external clean model.
    """
    if not trace.states:
        zero = trace.final.sum() * 0.0
        return zero, {"thought_spatial_entropy": 0.0, "thought_effective_area_frac": 0.0}
    motion = (trace.final - trace.states[0]).pow(2).mean(dim=1).sqrt()
    raw_flat = motion.flatten(start_dim=1)
    raw_mass = raw_flat.sum(dim=1)
    n = max(float(raw_flat.shape[1]), 2.0)
    active = raw_mass > float(eps)
    entropy = raw_mass.new_zeros(raw_mass.shape)
    effective_frac = raw_mass.new_zeros(raw_mass.shape)
    loss_vec = raw_mass.new_zeros(raw_mass.shape)
    if bool(active.any()):
        p = raw_flat[active] / raw_mass[active].unsqueeze(1).clamp_min(float(eps))
        entropy_active = -(p * p.clamp_min(float(eps)).log()).sum(dim=1)
        effective_active = torch.exp(entropy_active) / n
        entropy[active] = entropy_active
        effective_frac[active] = effective_active
        loss_vec[active] = F.relu(effective_active - float(max_effective_frac)).pow(2)
    loss = loss_vec.mean()
    return loss, {
        "thought_spatial_entropy": float((entropy / math.log(n)).mean().detach().cpu()),
        "thought_effective_area_frac": float(effective_frac.mean().detach().cpu()),
    }


def thought_sync_support_alignment_loss(
    trace: LatticeCTMTrace,
    *,
    support_mode: str = "change",
    topk_frac: float = 0.20,
    outside_weight: float = 1.0,
    inside_floor: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align CTM thought motion to CTM-native synchronization support.

    The support map is derived only from the recurrent lattice's own raw
    synchronization field changes.  It is detached before use, so it acts as a
    CTM-internal geometric prior rather than a learned spatial branch.  The
    loss discourages terminal thought motion outside high sync-change regions
    while optionally requiring a small amount of motion inside them.

    It does not inspect detector boxes, scores, classes, NMS, image masks,
    frequency tokens, or an external clean model.
    """
    if not trace.states or not trace.sync_fields:
        zero = trace.final.sum() * 0.0
        return zero, {
            "thought_sync_support_outside": 0.0,
            "thought_sync_support_inside": 0.0,
            "thought_sync_support_frac": 0.0,
        }
    mode = str(support_mode)
    if mode == "change" and len(trace.sync_fields) >= 2:
        sync_steps = [
            (b - a).abs().mean(dim=(1, 2))
            for a, b in zip(trace.sync_fields[:-1], trace.sync_fields[1:])
        ]
        support_signal = torch.stack(sync_steps, dim=1).mean(dim=1)
    elif mode == "change":
        support_signal = trace.sync_fields[0].abs().mean(dim=(1, 2))
    elif mode == "edge_disagreement":
        sync_abs = torch.stack([s.abs() for s in trace.sync_fields], dim=1)  # B x T x C x E x H x W
        support_signal = sync_abs.var(dim=3, unbiased=False).mean(dim=(1, 2))
    elif mode == "hybrid":
        if len(trace.sync_fields) >= 2:
            sync_steps = [
                (b - a).abs().mean(dim=(1, 2))
                for a, b in zip(trace.sync_fields[:-1], trace.sync_fields[1:])
            ]
            change_signal = torch.stack(sync_steps, dim=1).mean(dim=1)
        else:
            change_signal = trace.sync_fields[0].abs().mean(dim=(1, 2))
        sync_abs = torch.stack([s.abs() for s in trace.sync_fields], dim=1)
        disagreement_signal = sync_abs.var(dim=3, unbiased=False).mean(dim=(1, 2))
        support_signal = 0.5 * change_signal + 0.5 * disagreement_signal
    else:
        raise ValueError(f"unknown CTM sync support mode: {support_mode}")

    flat = support_signal.detach().flatten(start_dim=1)
    frac = min(max(float(topk_frac), 1e-4), 1.0)
    k = max(1, int(round(frac * flat.shape[1])))
    idx = flat.topk(k, dim=1).indices
    support_flat = torch.zeros_like(flat)
    support_flat.scatter_(1, idx, 1.0)
    support = support_flat.view_as(support_signal).to(dtype=trace.final.dtype)

    motion = (trace.final - trace.states[0]).pow(2).mean(dim=1).sqrt()
    scale = trace.states[0].abs().mean(dim=1).mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))
    rel_motion = motion / scale
    outside = (rel_motion.pow(2) * (1.0 - support)).sum(dim=(-1, -2))
    outside = outside / (1.0 - support).sum(dim=(-1, -2)).clamp_min(1.0)
    outside_loss = outside.mean()
    inside = (rel_motion * support).sum(dim=(-1, -2)) / support.sum(dim=(-1, -2)).clamp_min(1.0)
    inside_loss = F.relu(float(inside_floor) - inside).pow(2).mean()
    loss = float(outside_weight) * outside_loss + inside_loss
    return loss, {
        "thought_sync_support_outside": float(outside_loss.detach().cpu()),
        "thought_sync_support_inside": float(inside_loss.detach().cpu()),
        "thought_sync_support_frac": float(support.mean().detach().cpu()),
    }


def thought_edge_order_localization_loss(
    trace: LatticeCTMTrace,
    *,
    topk_frac: float = 0.12,
    outside_weight: float = 1.0,
    inside_floor: float = 0.015,
    inside_ratio_floor: float = 0.55,
    contrast_margin: float = 0.01,
    order_temperature: float = 0.10,
    temporal_weight: float = 0.50,
    gate_weight: float = 0.25,
    gate_outside_weight: float = 0.05,
    min_signal: float = 1e-4,
    mass_floor: float = 1e-4,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Localize invalid CTM thought motion to edge-order conflict cells.

    The support is computed only from CTM-internal recurrent signals:
    synchronization-field temporal changes, disagreement across edge types, and
    update-gate activity.  The support is detached and used only as a training
    regularizer on invalid thought motion; it is not a spatial token branch, not
    a detector score/box rule, and not a runtime postprocess.
    """
    zero_stats = {
        "thought_edge_order_outside": 0.0,
        "thought_edge_order_inside": 0.0,
        "thought_edge_order_inside_ratio": 0.0,
        "thought_edge_order_contrast": 0.0,
        "thought_edge_order_gate_outside": 0.0,
        "thought_edge_order_support_frac": 0.0,
        "thought_edge_order_signal_mean": 0.0,
        "thought_edge_order_signal_strength": 0.0,
    }
    if not trace.states or not trace.sync_fields:
        zero = trace.final.sum() * 0.0
        return zero, zero_stats

    sync_abs = torch.stack([s.abs() for s in trace.sync_fields], dim=1)  # B x T x C x E x H x W
    if sync_abs.shape[3] < 2:
        zero = trace.final.sum() * 0.0
        return zero, zero_stats
    temp = max(float(order_temperature), float(eps))
    edge_prob = torch.softmax(sync_abs / temp, dim=3)
    global_edge_prob = edge_prob.mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))
    local_prob = edge_prob.clamp_min(float(eps))
    mix_global = 0.5 * (local_prob + global_edge_prob)
    spatial_js = 0.5 * (
        (local_prob * (local_prob / mix_global).log()).sum(dim=3)
        + (global_edge_prob * (global_edge_prob / mix_global).log()).sum(dim=3)
    )
    order_anomaly = spatial_js.mean(dim=(1, 2))
    if edge_prob.shape[1] >= 2:
        prev_prob = edge_prob[:, :-1].clamp_min(float(eps))
        next_prob = edge_prob[:, 1:].clamp_min(float(eps))
        mix_time = 0.5 * (prev_prob + next_prob)
        temporal_js = 0.5 * (
            (next_prob * (next_prob / mix_time).log()).sum(dim=3)
            + (prev_prob * (prev_prob / mix_time).log()).sum(dim=3)
        )
        temporal_anomaly = temporal_js.mean(dim=(1, 2))
    else:
        temporal_anomaly = order_anomaly * 0.0
    signal = order_anomaly + float(temporal_weight) * temporal_anomaly
    if trace.update_gates:
        gate_signal = torch.stack([g.abs() for g in trace.update_gates], dim=1).mean(dim=(1, 2))
        signal = signal + float(gate_weight) * gate_signal

    flat_signal = signal.detach().flatten(start_dim=1)
    centred = flat_signal - flat_signal.mean(dim=1, keepdim=True)
    signal_strength = centred.abs().mean(dim=1)
    active_signal = signal_strength > max(float(min_signal), float(eps))
    frac = min(max(float(topk_frac), 1e-4), 1.0)
    k = max(1, int(round(frac * flat_signal.shape[1])))
    support_flat = torch.zeros_like(flat_signal)
    if bool(active_signal.any()):
        normed_signal = centred[active_signal] / signal_strength[active_signal].unsqueeze(1).clamp_min(float(eps))
        idx = normed_signal.topk(k, dim=1).indices
        support_active = torch.zeros_like(normed_signal)
        support_active.scatter_(1, idx, 1.0)
        support_flat[active_signal] = support_active
    support = support_flat.view_as(signal).to(dtype=trace.final.dtype)

    motion_sq = (trace.final - trace.states[0]).pow(2).mean(dim=1)
    motion = torch.sqrt(motion_sq + float(eps))
    state_scale = trace.states[0].abs().mean(dim=1).mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))
    rel_motion = motion / state_scale
    outside_mask = 1.0 - support
    outside = (rel_motion.pow(2) * outside_mask).sum(dim=(-1, -2))
    outside = outside / outside_mask.sum(dim=(-1, -2)).clamp_min(1.0)
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=torch.bool)
        if bool(mask.any()):
            return values[mask].mean()
        return values.sum() * 0.0

    outside_loss = _masked_mean(outside, active_signal)
    inside = (rel_motion * support).sum(dim=(-1, -2)) / support.sum(dim=(-1, -2)).clamp_min(1.0)
    inside_floor_loss = _masked_mean(F.relu(float(inside_floor) - inside).pow(2), active_signal)
    outside_mean = (rel_motion * outside_mask).sum(dim=(-1, -2)) / outside_mask.sum(dim=(-1, -2)).clamp_min(1.0)
    total_mass_raw = rel_motion.sum(dim=(-1, -2))
    active_mass = active_signal & (total_mass_raw.detach() > max(float(mass_floor), float(eps)))
    contrast_loss = _masked_mean(F.relu(float(contrast_margin) - (inside - outside_mean)).pow(2), active_mass)
    total_mass = total_mass_raw.clamp_min(max(float(mass_floor), float(eps)))
    inside_ratio = (rel_motion * support).sum(dim=(-1, -2)) / total_mass
    ratio_loss = _masked_mean(F.relu(float(inside_ratio_floor) - inside_ratio).pow(2), active_mass)
    if trace.update_gates and float(gate_outside_weight) > 0:
        gate_activity = torch.stack([g.abs() for g in trace.update_gates], dim=1).mean(dim=(1, 2))
        gate_outside = (gate_activity * outside_mask).sum(dim=(-1, -2))
        gate_outside = gate_outside / outside_mask.sum(dim=(-1, -2)).clamp_min(1.0)
        gate_outside_loss = _masked_mean(gate_outside, active_signal)
    else:
        gate_outside_loss = rel_motion.sum() * 0.0
    loss = (
        float(outside_weight) * outside_loss
        + inside_floor_loss
        + ratio_loss
        + contrast_loss
        + float(gate_outside_weight) * gate_outside_loss
    )
    return loss, {
        "thought_edge_order_outside": float(outside_loss.detach().cpu()),
        "thought_edge_order_inside": float(inside_floor_loss.detach().cpu()),
        "thought_edge_order_inside_ratio": float(inside_ratio.mean().detach().cpu()),
        "thought_edge_order_contrast": float(contrast_loss.detach().cpu()),
        "thought_edge_order_gate_outside": float(gate_outside_loss.detach().cpu()),
        "thought_edge_order_support_frac": float(support.mean().detach().cpu()),
        "thought_edge_order_signal_mean": float(signal.mean().detach().cpu()),
        "thought_edge_order_signal_strength": float(signal_strength.mean().detach().cpu()),
    }


def gate_activity_loss_post(trace: LatticeCTMTrace) -> torch.Tensor:
    if not trace.update_gates:
        return trace.final.sum() * 0.0
    return torch.stack([g.mean() for g in trace.update_gates]).mean()


def state_homeostasis_loss(trace: LatticeCTMTrace, ref: torch.Tensor) -> torch.Tensor:
    """Preserve valid neural-field population statistics.

    This is a CTM/dynamical-systems regularizer, not a batch-normalization or
    detector-specific rule.  It prevents a valid object field from reaching the
    correct readout label by collapsing its neural population statistics.
    """
    cur = trace.final
    cur_mean = cur.mean(dim=(-1, -2))
    ref_mean = ref.mean(dim=(-1, -2))
    cur_var = cur.var(dim=(-1, -2), unbiased=False)
    ref_var = ref.var(dim=(-1, -2), unbiased=False)
    return F.mse_loss(cur_mean, ref_mean) + 0.25 * F.mse_loss(cur_var, ref_var)


def valid_state_fixed_point_loss(
    trace: LatticeCTMTrace,
    ref: torch.Tensor,
    *,
    relative: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Quadratic fixed-point penalty on the valid terminal state.

    Forbids the global uniform-damping collapse F_T(x) = x - c by
    penalising any non-zero residual on valid samples without a relu
    cushion.  If `relative` is True the per-sample residual is normalised
    by ||F_0||^2 so the term is scale-free.  This is a CTM-internal
    constraint on the terminal recurrent state; it does not consult any
    external clean-anchor model and does not modify the readout.
    """
    diff = trace.final - ref
    num = diff.pow(2).mean(dim=(1, 2, 3))
    if relative:
        denom = ref.pow(2).mean(dim=(1, 2, 3)) + float(eps)
        return (num / denom).mean()
    return num.mean()


def quiet_state_loss(
    trace: LatticeCTMTrace,
    ref: torch.Tensor,
    *,
    fixed_point_weight: float = 1.0,
    homeostasis_weight: float = 0.25,
    trajectory_weight: float = 0.25,
    gate_weight: float = 0.05,
    relative: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pure CTM quiet-state penalty for normal target-present/absent fields.

    It is used to widen the normal-state manifold without an external clean
    model: if an input is a clean valid sample, its recurrent CTM trajectory
    should be short, population statistics should stay stable, and the update
    gates should remain quiet.  This directly targets the video failure where
    the lattice learned blanket suppression or blanket excitation.
    """
    fixed = valid_state_fixed_point_loss(trace, ref, relative=relative)
    homeo = state_homeostasis_loss(trace, ref)
    traj = trajectory_valid_loss(trace, ref)
    gate = gate_activity_loss(trace)
    loss = (
        float(fixed_point_weight) * fixed
        + float(homeostasis_weight) * homeo
        + float(trajectory_weight) * traj
        + float(gate_weight) * gate
    )
    return loss, {
        "quiet_fixed": float(fixed.detach().cpu()),
        "quiet_homeostasis": float(homeo.detach().cpu()),
        "quiet_trajectory": float(traj.detach().cpu()),
        "quiet_gate": float(gate.detach().cpu()),
    }


def single_scale_counterfactual_difference_gated_motion_loss(
    trace: LatticeCTMTrace,
    source_ids: torch.Tensor,
    *,
    topk_frac: float = 0.16,
    inside_floor: float = 0.015,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Same-source counterfactual support prior for a single CTM lattice.

    For triggered variants sharing one source image, the initial CTM feature
    differences identify where the trigger/augmentation perturbs the lattice.
    Terminal CTM motion is encouraged to stay inside that detached support and
    discouraged outside it.  This is a training-time CTM trajectory constraint:
    inference remains a single learned recurrent hook.
    """
    if not trace.states or source_ids.numel() == 0:
        zero = trace.final.sum() * 0.0
        return zero, {
            "cf_diff_outside": 0.0,
            "cf_diff_inside_floor": 0.0,
            "cf_diff_pairs": 0.0,
            "cf_diff_support_frac": 0.0,
        }
    f0 = trace.states[0]
    ft = trace.final
    ids = source_ids.view(-1).to(f0.device)
    pair = ids[:, None] == ids[None, :]
    eye = torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)
    pair = pair & ~eye
    if not bool(pair.any()):
        zero = trace.final.sum() * 0.0
        return zero, {
            "cf_diff_outside": 0.0,
            "cf_diff_inside_floor": 0.0,
            "cf_diff_pairs": 0.0,
            "cf_diff_support_frac": 0.0,
        }
    diff = (f0[:, None] - f0[None, :]).abs().mean(dim=2)  # B x B x H x W
    denom = pair.float().sum(dim=1).view(-1, 1, 1).clamp_min(1.0)
    diff_map = (diff * pair.float().view(pair.shape[0], pair.shape[1], 1, 1)).sum(dim=1) / denom
    flat = diff_map.flatten(start_dim=1)
    frac = min(max(float(topk_frac), 1e-4), 1.0)
    k = max(1, int(round(frac * flat.shape[1])))
    topk_idx = flat.topk(k, dim=1).indices
    support_flat = torch.zeros_like(flat)
    support_flat.scatter_(1, topk_idx, 1.0)
    support = support_flat.view_as(diff_map).to(dtype=ft.dtype).detach()

    motion = (ft - f0).abs().mean(dim=1)
    ref = f0.abs().mean(dim=1).mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))
    rel_motion = motion / ref
    outside = (rel_motion.pow(2) * (1.0 - support)).mean()
    inside = (rel_motion * support).sum(dim=(-1, -2)) / support.sum(dim=(-1, -2)).clamp_min(1.0)
    inside_floor_loss = F.relu(float(inside_floor) - inside).pow(2).mean()
    value = outside + inside_floor_loss
    return value, {
        "cf_diff_outside": float(outside.detach().cpu()),
        "cf_diff_inside_floor": float(inside_floor_loss.detach().cpu()),
        "cf_diff_pairs": float(pair.float().sum().detach().cpu()),
        "cf_diff_support_frac": float(support.mean().detach().cpu()),
    }


def single_scale_counterfactual_clean_trigger_support_loss(
    invalid_trace: LatticeCTMTrace,
    clean_ref: torch.Tensor,
    *,
    topk_frac: float = 0.16,
    inside_floor: float = 0.015,
    clean_quiet_weight: float = 0.25,
    direction_weight: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Counterfactual clean-trigger CTM support loss.

    This is the single-scale counterpart of the intended causal training loop:
    compare a triggered/invalid feature field with a same-source clean field,
    use the detached difference as CTM support, and constrain CTM motion to that
    support.  If `direction_weight > 0`, the residual is additionally aligned
    with the counterfactual direction ``clean_ref - F_0`` inside the support.
    It is not a clean-anchor interpolation and does not reference a separate
    clean model; it uses paired source images only as training-time
    counterfactual evidence.
    """
    if not invalid_trace.states:
        zero = invalid_trace.final.sum() * 0.0
        return zero, {
            "cf_pair_outside": 0.0,
            "cf_pair_inside_floor": 0.0,
            "cf_pair_clean_quiet": 0.0,
            "cf_pair_direction": 0.0,
            "cf_pair_support_frac": 0.0,
        }
    f0 = invalid_trace.states[0]
    ft = invalid_trace.final
    if clean_ref.shape != f0.shape:
        raise ValueError(f"clean_ref shape mismatch: {tuple(clean_ref.shape)} vs {tuple(f0.shape)}")
    diff_map = (f0 - clean_ref).abs().mean(dim=1)
    flat = diff_map.flatten(start_dim=1)
    frac = min(max(float(topk_frac), 1e-4), 1.0)
    k = max(1, int(round(frac * flat.shape[1])))
    topk_idx = flat.topk(k, dim=1).indices
    support_flat = torch.zeros_like(flat)
    support_flat.scatter_(1, topk_idx, 1.0)
    support = support_flat.view_as(diff_map).to(dtype=ft.dtype).detach()

    motion = (ft - f0).abs().mean(dim=1)
    ref = f0.abs().mean(dim=1).mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))
    rel_motion = motion / ref
    outside = (rel_motion.pow(2) * (1.0 - support)).mean()
    inside = (rel_motion * support).sum(dim=(-1, -2)) / support.sum(dim=(-1, -2)).clamp_min(1.0)
    inside_floor_loss = F.relu(float(inside_floor) - inside).pow(2).mean()
    clean_quiet = (
        ((ft - clean_ref).abs().mean(dim=1) / ref).pow(2) * (1.0 - support)
    ).mean()
    if float(direction_weight) > 0:
        residual = ft - f0
        target = (clean_ref - f0).detach()
        support_c = support.unsqueeze(1)
        scale = f0.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(float(eps))
        direction = (((residual - target) / scale).pow(2) * support_c).sum(dim=(1, 2, 3))
        direction = direction / support_c.sum(dim=(1, 2, 3)).clamp_min(1.0)
        direction = direction.mean()
    else:
        direction = ft.sum() * 0.0
    value = outside + inside_floor_loss + float(clean_quiet_weight) * clean_quiet + float(direction_weight) * direction
    return value, {
        "cf_pair_outside": float(outside.detach().cpu()),
        "cf_pair_inside_floor": float(inside_floor_loss.detach().cpu()),
        "cf_pair_clean_quiet": float(clean_quiet.detach().cpu()),
        "cf_pair_direction": float(direction.detach().cpu()),
        "cf_pair_support_frac": float(support.mean().detach().cpu()),
    }


def task_tangent_field_loss(
    trace: LatticeCTMTrace,
    task_gradient: torch.Tensor,
    desired_sign: torch.Tensor | float,
    *,
    topk_frac: float = 0.03,
    alignment_floor: float = 0.02,
    outside_weight: float = 0.05,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align CTM residual with a training-only downstream task tangent field.

    The frozen detector readout already supplies the task loss.  This term does
    not add an inference branch; it only converts the readout Jacobian into a
    CTM-local tangent field during training.  For OGA, ``desired_sign`` is
    negative so the CTM residual should move against the target-evidence
    gradient.  For ODA, it is positive so target evidence is preserved.
    """
    if not trace.states:
        zero = trace.final.sum() * 0.0
        return zero, {
            "task_tangent_align": 0.0,
            "task_tangent_outside": 0.0,
            "task_tangent_signed": 0.0,
            "task_tangent_support_frac": 0.0,
        }
    f0 = trace.states[0]
    ft = trace.final
    if task_gradient.shape != f0.shape:
        raise ValueError(f"task_gradient shape mismatch: {tuple(task_gradient.shape)} vs {tuple(f0.shape)}")
    grad = task_gradient.detach()
    flat = grad.abs().flatten(start_dim=1)
    frac = min(max(float(topk_frac), 1e-5), 1.0)
    k = max(1, int(round(frac * flat.shape[1])))
    topk_idx = flat.topk(k, dim=1).indices
    support_flat = torch.zeros_like(flat)
    support_flat.scatter_(1, topk_idx, 1.0)
    support = support_flat.view_as(grad).to(dtype=ft.dtype).detach()

    residual = ft - f0
    feature_scale = f0.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(float(eps))
    grad_support = grad * support
    grad_norm = grad_support.flatten(start_dim=1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(float(eps))
    direction = grad_support / grad_norm
    signed = ((residual / feature_scale) * direction * support).flatten(start_dim=1).sum(dim=1)
    if not isinstance(desired_sign, torch.Tensor):
        sign = torch.full_like(signed, float(desired_sign))
    else:
        sign = desired_sign.to(device=signed.device, dtype=signed.dtype).view(-1)
    if sign.numel() != signed.numel():
        raise ValueError(f"desired_sign shape mismatch: {tuple(sign.shape)} vs {tuple(signed.shape)}")
    signed = signed * sign
    align = F.relu(float(alignment_floor) - signed).pow(2).mean()
    outside = ((residual / feature_scale).pow(2) * (1.0 - support)).mean()
    value = align + float(outside_weight) * outside
    return value, {
        "task_tangent_align": float(align.detach().cpu()),
        "task_tangent_outside": float(outside.detach().cpu()),
        "task_tangent_signed": float(signed.mean().detach().cpu()),
        "task_tangent_support_frac": float(support.mean().detach().cpu()),
    }


def residual_decorrelation_loss(
    invalid_input: torch.Tensor,
    valid_input: torch.Tensor,
    invalid_trace: LatticeCTMTrace,
    valid_trace: LatticeCTMTrace,
    invalid_labels: torch.Tensor,
    valid_labels: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cross-input residual decorrelation (anti-DC).

    A globally constant residual r(x) = c gives cosine(r(x_i), r(x_j)) = 1
    for every pair, so this loss saturates at 1.  Penalising the squared
    cosine across different-label pairs rewards input conditioning of the
    residual.  Same-label pairs are masked out so ODA (invalid and valid
    both target-present) is not penalised for sharing an attractor.

    This is a pure CTM term: it inspects only the lattice's residual
    vectors r(x) = F_T(x) - F_0(x) on the same images the rest of the
    objective already uses.  No external model, no detector rule.
    """
    r_inv = (invalid_trace.final - invalid_input).reshape(invalid_input.shape[0], -1)
    r_val = (valid_trace.final - valid_input).reshape(valid_input.shape[0], -1)
    r = torch.cat([r_inv, r_val], dim=0)
    labels = torch.cat([invalid_labels.view(-1), valid_labels.view(-1)], dim=0).long().to(r.device)
    if r.shape[0] < 2:
        return r.sum() * 0.0
    norms = r.norm(dim=1, keepdim=True).clamp_min(float(eps))
    rn = r / norms
    cos = rn @ rn.t()                                                # (N, N)
    eye = torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
    diff = (labels[:, None] != labels[None, :]) & ~eye
    if not bool(diff.any()):
        return r.sum() * 0.0
    return cos[diff].pow(2).mean()


def _basin_profile_from_trace(
    trace: LatticeCTMTrace,
    reference: torch.Tensor,
    *,
    profile: str = "residual",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a pure CTM basin profile for one trace.

    The residual component removes each channel's spatial DC part before it is
    flattened.  That makes a global constant residual useless for this loss:
    the only way to leave the valid quiet basin is to create structured CTM
    thought motion.  Optional sync/gate components come from the recurrent
    synchronization trace itself, not from detector boxes, scores, anchors,
    NMS, external clean models, or post-processing.
    """
    mode = str(profile).lower()
    if mode not in {"residual", "sync", "gate", "hybrid", "phase"}:
        raise ValueError(f"unknown CTM basin profile mode: {profile}")
    parts: list[torch.Tensor] = []
    batch = int(reference.shape[0])

    if mode in {"residual", "hybrid", "phase"}:
        residual = trace.final - reference
        scale = reference.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(float(eps))
        residual = residual / scale
        if residual.ndim == 4:
            residual = residual - residual.mean(dim=(-1, -2), keepdim=True)
        parts.append(residual.flatten(start_dim=1))

    if mode == "phase":
        if len(trace.states) >= 2:
            velocity = trace.final - trace.states[-2]
        else:
            velocity = trace.final - reference
        scale = reference.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(float(eps))
        velocity = 0.50 * velocity / scale
        if velocity.ndim == 4:
            velocity = velocity - velocity.mean(dim=(-1, -2), keepdim=True)
        parts.append(velocity.flatten(start_dim=1))

    if mode in {"sync", "hybrid", "phase"} and trace.sync_signatures.numel() > 0:
        sig = trace.sync_signatures
        if sig.shape[1] >= 2:
            sig_part = sig[:, 1:] - sig[:, :-1]
        else:
            sig_part = sig - sig.mean(dim=1, keepdim=True)
        sig_scale = sig.detach().pow(2).flatten(start_dim=1).mean(dim=1, keepdim=True).sqrt().clamp_min(float(eps))
        sync_weight = 0.50 if mode == "phase" else 1.0
        parts.append(sync_weight * sig_part.flatten(start_dim=1) / sig_scale)

    if mode in {"gate", "hybrid", "phase"} and trace.update_gates:
        gates = torch.stack([g.abs() for g in trace.update_gates], dim=1)
        gate_profile = gates.mean(dim=tuple(range(2, gates.ndim)))
        gate_weight = 0.10 if mode == "phase" else 1.0
        parts.append(gate_weight * gate_profile)

    if mode == "phase":
        if len(trace.states) >= 2:
            length_parts = [
                (b - a).reshape(b.shape[0], -1).norm(dim=1)
                for a, b in zip(trace.states[:-1], trace.states[1:])
            ]
            length = torch.stack(length_parts, dim=1).sum(dim=1)
        else:
            length = (trace.final - reference).reshape(reference.shape[0], -1).norm(dim=1)
        ref_norm = reference.reshape(reference.shape[0], -1).norm(dim=1).clamp_min(float(eps))
        parts.append(0.10 * (length / ref_norm).unsqueeze(1))

    if parts:
        return torch.cat(parts, dim=1)
    return reference.new_zeros((batch, 1))


def ctm_basin_state_separation_loss(
    invalid_input: torch.Tensor,
    valid_input: torch.Tensor,
    invalid_trace: LatticeCTMTrace,
    valid_trace: LatticeCTMTrace,
    invalid_labels: torch.Tensor,
    valid_labels: torch.Tensor,
    *,
    margin: float = 0.08,
    same_margin: float = 0.05,
    profile: str = "residual",
    detach_valid: bool = True,
    same_weight: float = 0.0,
    diff_weight: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Separate invalid CTM trajectories from the valid quiet basin.

    This is a CTM-state regularizer, not a detector-output repair.  It compares
    only terminal residual structure and optional recurrent sync/gate traces.
    Different-label invalid/valid pairs must be at least ``margin`` apart in
    basin-profile RMS distance.  Optional same-label compactness is useful for
    making invalid source variants share one CTM attractor, while its default
    zero weight preserves the previous ODA-safe behavior.
    """
    inv_profile = _basin_profile_from_trace(invalid_trace, invalid_input, profile=profile, eps=eps)
    val_profile = _basin_profile_from_trace(valid_trace, valid_input, profile=profile, eps=eps)
    if inv_profile.shape[1] != val_profile.shape[1]:
        raise ValueError(f"basin profile mismatch: {tuple(inv_profile.shape)} vs {tuple(val_profile.shape)}")
    val_ref = val_profile.detach() if bool(detach_valid) else val_profile
    labels_i = invalid_labels.view(-1).long().to(inv_profile.device)
    labels_v = valid_labels.view(-1).long().to(inv_profile.device)
    profiles = torch.cat([inv_profile, val_ref], dim=0)
    labels = torch.cat([labels_i, labels_v], dim=0)
    if profiles.shape[0] < 2:
        zero = inv_profile.sum() * 0.0
        return zero, {
            "basin_separation": 0.0,
            "basin_diff": 0.0,
            "basin_same": 0.0,
            "basin_distance": 0.0,
            "basin_same_distance": 0.0,
            "basin_active_frac": 0.0,
            "basin_pairs": 0.0,
            "basin_same_pairs": 0.0,
        }
    dist = (profiles[:, None, :] - profiles[None, :, :]).pow(2).mean(dim=-1).add(float(eps)).sqrt()
    eye = torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
    # Keep the legacy diff scope to invalid-vs-valid pairs.  Same compactness is
    # allowed within invalid and valid batches when explicitly weighted.
    inv_side = torch.zeros(dist.shape[0], dtype=torch.bool, device=dist.device)
    inv_side[: inv_profile.shape[0]] = True
    val_side = ~inv_side
    cross_side = inv_side[:, None] & val_side[None, :]
    diff = (labels[:, None] != labels[None, :]) & cross_side & ~eye
    same = (labels[:, None] == labels[None, :]) & ~eye
    zero = inv_profile.sum() * 0.0
    diff_loss = zero
    same_loss = zero
    active_dist = dist[diff] if bool(diff.any()) else dist.new_zeros((0,))
    same_dist = dist[same] if bool(same.any()) else dist.new_zeros((0,))
    if bool(diff.any()) and float(diff_weight) > 0:
        diff_loss = F.relu(float(margin) - active_dist).pow(2).mean()
    if bool(same.any()) and float(same_weight) > 0:
        same_loss = F.relu(same_dist - float(same_margin)).pow(2).mean()
    loss = float(diff_weight) * diff_loss + float(same_weight) * same_loss
    active_frac = (
        (active_dist.detach() < float(margin)).float().mean()
        if active_dist.numel() > 0 else dist.new_tensor(0.0)
    )
    same_active_frac = (
        (same_dist.detach() > float(same_margin)).float().mean()
        if same_dist.numel() > 0 else dist.new_tensor(0.0)
    )
    return loss, {
        "basin_separation": float(loss.detach().cpu()),
        "basin_diff": float(diff_loss.detach().cpu()),
        "basin_same": float(same_loss.detach().cpu()),
        "basin_distance": float(active_dist.mean().detach().cpu()) if active_dist.numel() > 0 else 0.0,
        "basin_same_distance": float(same_dist.mean().detach().cpu()) if same_dist.numel() > 0 else 0.0,
        "basin_active_frac": float(active_frac.detach().cpu()),
        "basin_same_active_frac": float(same_active_frac.detach().cpu()),
        "basin_pairs": float(diff.float().sum().detach().cpu()),
        "basin_same_pairs": float(same.float().sum().detach().cpu()),
    }


def residual_profile_invariance_loss(
    invalid_trace: LatticeCTMTrace,
    valid_trace: LatticeCTMTrace,
    *,
    invalid_floor: float = 0.04,
    valid_weight: float = 0.25,
    topk_frac: float = 0.08,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Location-free CTM residual-profile invariance.

    v2 visible-patch failures are source-level: a CTM state that fixes one
    source can miss another, while global damping hurts clean recall.  This loss
    keeps only the channel-order spectrum of CTM motion, discarding spatial maps
    and all detector outputs.  Invalid samples are encouraged to share that
    spectrum and maintain a small motion floor; valid samples are encouraged to
    stay quiet in the same profile view.
    """

    def profile(trace: LatticeCTMTrace) -> tuple[torch.Tensor, torch.Tensor]:
        if not trace.states:
            z = trace.final.new_zeros((trace.final.shape[0], trace.final.shape[1]))
            return z, z.mean(dim=1)
        motion = (trace.final - trace.states[0]).abs().flatten(start_dim=2)
        frac = min(max(float(topk_frac), 1e-4), 1.0)
        k = max(1, int(round(frac * motion.shape[-1])))
        channel_mag = motion.topk(k, dim=-1).values.mean(dim=-1)
        mag = channel_mag.mean(dim=1)
        prof = F.normalize(channel_mag, p=2, dim=1, eps=float(eps))
        return prof, mag

    inv_profile, inv_mag = profile(invalid_trace)
    val_profile, val_mag = profile(valid_trace)
    zero = invalid_trace.final.sum() * 0.0
    compact = zero
    if inv_profile.shape[0] > 1:
        sim = inv_profile @ inv_profile.t()
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        compact = (1.0 - sim[~eye]).pow(2).mean()
    floor = F.relu(float(invalid_floor) - inv_mag).pow(2).mean()
    valid = val_mag.pow(2).mean() if val_profile.numel() > 0 else zero
    loss = compact + floor + float(valid_weight) * valid
    return loss, {
        "residual_profile_invariance": float(loss.detach().cpu()),
        "residual_profile_compact": float(compact.detach().cpu()),
        "residual_profile_floor": float(floor.detach().cpu()),
        "residual_profile_valid": float(valid.detach().cpu()),
        "residual_profile_invalid_mag": float(inv_mag.mean().detach().cpu()) if inv_mag.numel() else 0.0,
        "residual_profile_valid_mag": float(val_mag.mean().detach().cpu()) if val_mag.numel() else 0.0,
    }


def lattice_ctm_objective(
    invalid_input: torch.Tensor,
    valid_input: torch.Tensor,
    invalid_trace: LatticeCTMTrace,
    valid_trace: LatticeCTMTrace,
    *,
    readout: Callable[[torch.Tensor], torch.Tensor],
    invalid_labels: torch.Tensor,
    valid_labels: torch.Tensor,
    cfg: LatticeLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Closed-loop CTM lattice objective.

    The CTM terminal state is judged by the frozen downstream task readout.  The
    CTM-internal objective is label-conditioned synchronization dynamics rather
    than unconditional invalid-valid alignment.  This removes the OGA conflict in
    the previous objective while keeping the ODA behavior where invalid and
    valid examples share the target-present attractor.
    """
    cfg = cfg or LatticeLossConfig()
    logits_invalid = readout(invalid_trace.final)
    logits_valid = readout(valid_trace.final)
    task_mode = str(getattr(cfg, "task_loss_mode", "ce")).lower()
    if task_mode in {"bounded_margin", "margin", "bounded"}:
        task_invalid = bounded_margin_task_loss(
            logits_invalid,
            invalid_labels.long(),
            margin=float(getattr(cfg, "task_margin", 0.0)),
        )
        task_valid = bounded_margin_task_loss(
            logits_valid,
            valid_labels.long(),
            margin=float(getattr(cfg, "task_margin", 0.0)),
        )
    else:
        task_invalid = F.cross_entropy(logits_invalid, invalid_labels.long())
        task_valid = F.cross_entropy(logits_valid, valid_labels.long())
    # Asymmetric task weight: keep symmetric default (0.5/0.5) when extra==0,
    # but allow the runner to bump valid weight under sharp OGA readouts.
    extra = float(getattr(cfg, "valid_task_weight_extra", 0.0))
    w_inv = 0.5
    w_val = 0.5 + extra
    norm = w_inv + w_val
    task = (w_inv * task_invalid + w_val * task_valid) / max(norm, 1e-6)

    # Backward-compatible paired sync, but only when labels match.  The new
    # attractor term below is the primary CTM sync objective.
    n = min(invalid_trace.final.shape[0], valid_trace.final.shape[0])
    paired_sync = invalid_trace.final.sum() * 0.0
    if n > 0 and float(cfg.paired_sync_weight) > 0:
        same_pair = invalid_labels[:n].view(-1).long() == valid_labels[:n].view(-1).long()
        if bool(same_pair.any()):
            invalid_sub = _slice_trace(invalid_trace, n)
            valid_sub = _slice_trace(valid_trace, n)
            invalid_sub = LatticeCTMTrace(
                final=invalid_sub.final[same_pair],
                states=[s[same_pair] for s in invalid_sub.states],
                sync_signatures=invalid_sub.sync_signatures[same_pair],
                sync_fields=[s[same_pair] for s in invalid_sub.sync_fields],
                update_gates=[g[same_pair] for g in invalid_sub.update_gates],
            )
            valid_sub = LatticeCTMTrace(
                final=valid_sub.final[same_pair],
                states=[s[same_pair] for s in valid_sub.states],
                sync_signatures=valid_sub.sync_signatures[same_pair],
                sync_fields=[s[same_pair] for s in valid_sub.sync_fields],
                update_gates=[g[same_pair] for g in valid_sub.update_gates],
            )
            paired_sync = sync_trajectory_distance(invalid_sub, valid_sub)

    label_attr, label_attr_stats = label_conditioned_sync_attractor_loss(
        invalid_trace,
        valid_trace,
        invalid_labels,
        valid_labels,
        same_weight=float(cfg.same_label_weight),
        diff_weight=float(cfg.diff_label_weight),
        margin=float(cfg.attractor_margin),
    )

    sep = 0.5 * (
        sync_separation_loss(invalid_trace, cfg.separation_margin)
        + sync_separation_loss(valid_trace, cfg.separation_margin)
    )
    kin = 0.5 * (kinetic_loss(invalid_trace) + kinetic_loss(valid_trace))
    inv_motion = state_motion_loss(invalid_trace, invalid_input, cfg.max_invalid_rms)
    val_motion = state_motion_loss(valid_trace, valid_input, cfg.max_valid_rms)
    val_homeo = state_homeostasis_loss(valid_trace, valid_input)
    valid_gate = gate_activity_loss(valid_trace)
    invalid_gate = gate_activity_loss(invalid_trace)
    invalid_gate_floor = F.relu(float(cfg.invalid_gate_floor) - invalid_gate).pow(2)
    gate_separation = F.relu(
        float(getattr(cfg, "gate_separation_margin", 0.0)) - (invalid_gate - valid_gate)
    ).pow(2)
    valid_state_fixed_point = valid_state_fixed_point_loss(
        valid_trace,
        valid_input,
        relative=bool(getattr(cfg, "valid_state_fixed_point_relative", True)),
    )
    decorr = residual_decorrelation_loss(
        invalid_input, valid_input, invalid_trace, valid_trace,
        invalid_labels, valid_labels,
    )
    traj_valid = trajectory_valid_loss(valid_trace, valid_input)
    traj_inv_floor = trajectory_invalid_floor_loss(
        invalid_trace, float(getattr(cfg, "trajectory_invalid_floor", 0.10)), invalid_input,
    )
    thought_focus = thought_concentration_loss(
        invalid_trace, float(getattr(cfg, "thought_concentration_target", 0.35))
    )
    residual_profile, residual_profile_stats = residual_profile_invariance_loss(
        invalid_trace,
        valid_trace,
        invalid_floor=float(getattr(cfg, "residual_profile_invalid_floor", 0.04)),
        valid_weight=float(getattr(cfg, "residual_profile_valid_weight", 0.25)),
        topk_frac=float(getattr(cfg, "residual_profile_topk_frac", 0.08)),
    )
    basin_sep, basin_sep_stats = ctm_basin_state_separation_loss(
        invalid_input,
        valid_input,
        invalid_trace,
        valid_trace,
        invalid_labels,
        valid_labels,
        margin=float(getattr(cfg, "basin_separation_margin", 0.08)),
        same_margin=float(getattr(cfg, "basin_separation_same_margin", 0.05)),
        profile=str(getattr(cfg, "basin_separation_profile", "residual")),
        detach_valid=bool(getattr(cfg, "basin_separation_detach_valid", True)),
        same_weight=float(getattr(cfg, "basin_separation_same_weight", 0.0)),
        diff_weight=float(getattr(cfg, "basin_separation_diff_weight", 1.0)),
    )

    loss = (
        float(cfg.task_weight) * task
        + float(cfg.paired_sync_weight) * paired_sync
        + float(cfg.label_attractor_weight) * label_attr
        + float(cfg.separation_weight) * sep
        + float(cfg.kinetic_weight) * kin
        + float(cfg.invalid_motion_weight) * inv_motion
        + float(cfg.valid_motion_weight) * val_motion
        + float(cfg.valid_homeostasis_weight) * val_homeo
        + float(cfg.valid_gate_weight) * valid_gate
        + float(cfg.invalid_gate_floor_weight) * invalid_gate_floor
        + float(getattr(cfg, "gate_separation_weight", 0.0)) * gate_separation
        + float(getattr(cfg, "valid_state_fixed_point_weight", 0.0)) * valid_state_fixed_point
        + float(getattr(cfg, "residual_decorrelation_weight", 0.0)) * decorr
        + float(getattr(cfg, "trajectory_valid_weight", 0.0)) * traj_valid
        + float(getattr(cfg, "trajectory_invalid_floor_weight", 0.0)) * traj_inv_floor
        + float(getattr(cfg, "thought_concentration_weight", 0.0)) * thought_focus
        + float(getattr(cfg, "residual_profile_invariance_weight", 0.0)) * residual_profile
        + float(getattr(cfg, "basin_separation_weight", 0.0)) * basin_sep
    )
    stats = {
        "loss": float(loss.detach().cpu()),
        "task_loss_mode": 1.0 if task_mode in {"bounded_margin", "margin", "bounded"} else 0.0,
        "task": float(task.detach().cpu()),
        "task_invalid": float(task_invalid.detach().cpu()),
        "task_valid": float(task_valid.detach().cpu()),
        "paired_sync": float(paired_sync.detach().cpu()),
        "label_attractor": float(label_attr.detach().cpu()),
        **label_attr_stats,
        "separation": float(sep.detach().cpu()),
        "kinetic": float(kin.detach().cpu()),
        "invalid_motion": float(inv_motion.detach().cpu()),
        "valid_motion": float(val_motion.detach().cpu()),
        "valid_homeostasis": float(val_homeo.detach().cpu()),
        "invalid_gate": float(invalid_gate.detach().cpu()),
        "valid_gate": float(valid_gate.detach().cpu()),
        "invalid_gate_floor": float(invalid_gate_floor.detach().cpu()),
        "gate_separation": float(gate_separation.detach().cpu()),
        "valid_state_fixed_point": float(valid_state_fixed_point.detach().cpu()),
        "decorr": float(decorr.detach().cpu()),
        "traj_valid": float(traj_valid.detach().cpu()),
        "traj_inv_floor": float(traj_inv_floor.detach().cpu()),
        "thought_focus": float(thought_focus.detach().cpu()),
        **residual_profile_stats,
        **basin_sep_stats,
        "invalid_acc": float((logits_invalid.argmax(dim=1) == invalid_labels).float().mean().detach().cpu()),
        "valid_acc": float((logits_valid.argmax(dim=1) == valid_labels).float().mean().detach().cpu()),
    }
    return loss, stats
