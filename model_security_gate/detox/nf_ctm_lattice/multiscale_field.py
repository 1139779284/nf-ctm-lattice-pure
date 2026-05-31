from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .neuron_field import LatticeCTMTrace, LatticeNFCTMNeuronField
from .schema import LatticeCTMConfig


@dataclass
class CoupledMultiScaleLatticeCTMTrace:
    """Trace for one unified multi-scale CTM recurrent field.

    `traces` contains one ordinary lattice trace per native feature scale.
    `scale_orders` stores the scalar CTM order parameter for every thought step
    and scale.  It is a diagnostic of the coupled dynamics, not a detector
    output rule.
    """

    final: dict[str, torch.Tensor]
    traces: dict[str, LatticeCTMTrace]
    scale_orders: torch.Tensor  # B x T x S
    cross_modulators: torch.Tensor  # B x T x S
    context_gates: torch.Tensor | None = None  # B x T x S, CTM-internal high-context anomaly gates
    # B x T x S x D.  D is an edge-type order summary vector.  The default
    # implementation uses four edge families times four location-free moments:
    # mean, RMS, top-k mean, and concentration.
    edge_type_orders: torch.Tensor | None = None


def _key(value: Any) -> str:
    text = str(value)
    if "." in text:
        raise ValueError(f"scale keys must not contain '.', got {text!r}")
    return text


def _masked_edge_order(sync: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=sync.device, dtype=sync.dtype)
    n_edges = mask.sum()
    if float(n_edges.detach().cpu()) <= 0:
        return sync.new_zeros((sync.shape[0],))
    denom = n_edges * float(sync.shape[1] * sync.shape[3] * sync.shape[4])
    return (sync.abs() * mask).sum(dim=(1, 2, 3, 4)) / denom.clamp_min(1.0)


def _masked_edge_cell_map(sync: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=sync.device, dtype=sync.dtype)
    n_edges = mask.sum()
    if float(n_edges.detach().cpu()) <= 0:
        return sync.new_zeros((sync.shape[0], sync.shape[3], sync.shape[4]))
    return ((sync.abs() * mask).sum(dim=2) / n_edges.clamp_min(1.0)).mean(dim=1)


def _cell_map_moments(cell_map: torch.Tensor, eps: float) -> torch.Tensor:
    """Location-free order moments for one CTM edge family.

    The vector deliberately discards coordinates.  It keeps only distributional
    information, which lets the lattice sense localized trigger-like order
    spikes without transmitting a spatial map between scales.
    """

    flat = cell_map.flatten(start_dim=1).abs()
    mean = flat.mean(dim=1)
    rms = flat.pow(2).mean(dim=1).sqrt()
    k = max(1, int(round(0.10 * flat.shape[1])))
    topk = flat.topk(k, dim=1).values.mean(dim=1)
    concentration = topk / mean.clamp_min(float(eps))
    return torch.stack([mean, rms, topk, concentration], dim=1)


def _edge_type_order_vector(sync: torch.Tensor, field: LatticeNFCTMNeuronField) -> torch.Tensor:
    """CTM edge-type order moment vector.

    This is deliberately not a spatial map output.  It compresses each raw
    synchronization edge family to location-free order moments.  The four
    edge families are spatial, channel, field-order, and local conflict.  For
    each family we keep mean/RMS/top-k/concentration; these moments can express
    localized trigger-like synchronization spikes while carrying no coordinates.
    """

    eps = float(field.cfg.eps)
    spatial_map = _masked_edge_cell_map(sync, field.spatial_edge_mask)
    channel_map = _masked_edge_cell_map(sync, field.channel_edge_mask)
    field_map = _masked_edge_cell_map(sync, field.field_edge_mask)
    conflict_map = (spatial_map - field_map).abs()
    return torch.cat(
        [
            _cell_map_moments(spatial_map, eps),
            _cell_map_moments(channel_map, eps),
            _cell_map_moments(field_map, eps),
            _cell_map_moments(conflict_map, eps),
        ],
        dim=1,
    )


class CoupledMultiScaleLatticeNFCTM(nn.Module):
    """Unified multi-scale NF-CTM lattice.

    This module is the no-sandwich replacement for the experimental v1 runner
    that attached independent CTM hooks to several YOLO neck layers.  Here the
    native feature maps are advanced in the same recurrent thought loop, and
    scales communicate only through CTM order parameters derived from raw
    synchronization fields.

    It does not introduce a CNN adapter, spatial/frequency token, clean-anchor
    model, score calibration rule, or post-hoc detector editor.
    """

    def __init__(
        self,
        configs: Mapping[Any, LatticeCTMConfig],
        *,
        cross_scale_coupling: float = 0.10,
        cross_field_coupling: float = 0.0,
        cross_edge_coupling: float = 0.0,
        edge_order_moment_coupling: float = 0.0,
        cross_context_gate_strength: float = 0.0,
        cross_context_gate_bias: float = -1.0,
        cross_context_gate_floor: float = 0.25,
        cross_context_gate_ceiling: float = 1.0,
        scale_order_pool: str = "mean",
        scale_order_topk_frac: float = 0.02,
        learn_cross_scale: bool = True,
    ):
        super().__init__()
        if len(configs) < 2:
            raise ValueError("CoupledMultiScaleLatticeNFCTM requires at least two scales")
        self.scale_keys = tuple(_key(k) for k in configs.keys())
        if len(set(self.scale_keys)) != len(self.scale_keys):
            raise ValueError(f"duplicate scale keys after string conversion: {self.scale_keys}")
        thought_steps = {int(cfg.thought_steps) for cfg in configs.values()}
        if len(thought_steps) != 1:
            raise ValueError("all scales must use the same thought_steps for coupled recurrence")
        self.thought_steps = next(iter(thought_steps))
        self.fields = nn.ModuleDict({_key(k): LatticeNFCTMNeuronField(cfg) for k, cfg in configs.items()})
        self.cross_context_gate_strength = float(cross_context_gate_strength)
        self.cross_context_gate_bias = float(cross_context_gate_bias)
        self.cross_context_gate_floor = min(max(float(cross_context_gate_floor), 0.0), 1.0)
        self.cross_context_gate_ceiling = max(self.cross_context_gate_floor, float(cross_context_gate_ceiling))
        self.cross_edge_coupling = max(0.0, float(cross_edge_coupling))
        self.edge_order_moment_coupling = max(0.0, float(edge_order_moment_coupling))
        self.scale_order_pool = str(scale_order_pool)
        self.scale_order_topk_frac = min(max(float(scale_order_topk_frac), 1e-5), 1.0)
        n = len(self.scale_keys)
        init = torch.zeros(n, n)
        if n > 1 and float(cross_scale_coupling) != 0.0:
            init.fill_(float(cross_scale_coupling) / float(n - 1))
            init.fill_diagonal_(0.0)
        self.cross_scale_weight = nn.Parameter(init, requires_grad=bool(learn_cross_scale))
        field_init = torch.zeros(n, n)
        if n > 1 and float(cross_field_coupling) != 0.0:
            field_init.fill_(float(cross_field_coupling) / float(n - 1))
            field_init.fill_diagonal_(0.0)
        self.cross_field_weight = nn.Parameter(field_init, requires_grad=bool(learn_cross_scale))
        self.register_buffer("cross_scale_mask", torch.ones(n, n) - torch.eye(n), persistent=False)

    def _scale_order_parameter(self, sync: torch.Tensor) -> torch.Tensor:
        flat = sync.reshape(sync.shape[0], -1)
        mode = self.scale_order_pool
        if mode == "abs_mean":
            return flat.abs().mean(dim=1)
        if mode == "rms":
            return flat.pow(2).mean(dim=1).sqrt()
        if mode == "topk_abs":
            k = max(1, int(round(float(self.scale_order_topk_frac) * flat.shape[1])))
            return flat.abs().topk(k, dim=1).values.mean(dim=1)
        return flat.mean(dim=1)

    @property
    def n_scales(self) -> int:
        return len(self.scale_keys)

    def _inputs_by_key(self, inputs: Mapping[Any, torch.Tensor]) -> dict[str, torch.Tensor]:
        by_key = {_key(k): v for k, v in inputs.items()}
        missing = [k for k in self.scale_keys if k not in by_key]
        if missing:
            raise ValueError(f"missing CTM scale inputs: {missing}")
        extra = [k for k in by_key if k not in self.scale_keys]
        if extra:
            raise ValueError(f"unexpected CTM scale inputs: {extra}")
        for k, x in by_key.items():
            if x.ndim != 4:
                raise ValueError(f"scale {k} expected BxCxHxW, got {tuple(x.shape)}")
            if x.shape[1] != self.fields[k].channels:
                raise ValueError(f"scale {k} expected {self.fields[k].channels} channels, got {x.shape[1]}")
        return by_key

    def _update_gate(
        self,
        field: LatticeNFCTMNeuronField,
        state: torch.Tensor,
        temporal_gate_bias: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        cfg = field.cfg
        if not bool(cfg.use_adaptive_update):
            return torch.ones_like(state)
        base_logit = field.update_gate_bias.view(1, field.channels, 1, 1) + temporal_gate_bias
        floor = getattr(cfg, "sync_residual_floor", None)
        if floor is None:
            return torch.sigmoid(base_logit + float(cfg.adaptive_residual_gain) * residual.detach())
        p = max(1, int(getattr(cfg, "sync_residual_floor_p", 2)))
        tau = float(floor)
        r = residual.detach()
        rp = r.pow(p)
        factor = rp / (rp + (tau ** p) + float(cfg.eps))
        return torch.sigmoid(base_logit) * factor

    def _thought_update(
        self,
        field: LatticeNFCTMNeuronField,
        state: torch.Tensor,
        history: list[torch.Tensor],
        sync_drive: torch.Tensor,
        residual: torch.Tensor,
        conflict_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = field.cfg
        temporal_drive, temporal_gate_bias = field.temporal(history)
        update_gate = self._update_gate(field, state, temporal_gate_bias, residual)
        if bool(getattr(cfg, "local_edge_conflict_update_gate", False)) and conflict_gate is not None:
            update_gate = update_gate * conflict_gate
        thought_drive = temporal_drive + field.sync_gain * sync_drive
        total_dc_supp = float(getattr(cfg, "total_drive_dc_suppression", 0.0))
        if total_dc_supp > 0:
            dc = thought_drive.mean(dim=(-1, -2), keepdim=True)
            thought_drive = thought_drive - min(max(total_dc_supp, 0.0), 1.0) * dc
        delta = torch.tanh(thought_drive)
        delta = delta.clamp(-float(cfg.max_update), float(cfg.max_update))
        step_size = torch.sigmoid(field.step_size_logit)
        return state + step_size * update_gate * delta, update_gate

    def forward(
        self,
        inputs: Mapping[Any, torch.Tensor],
        *,
        return_trace: bool = False,
    ) -> dict[str, torch.Tensor] | CoupledMultiScaleLatticeCTMTrace:
        by_key = self._inputs_by_key(inputs)
        states = {k: by_key[k] for k in self.scale_keys}
        histories = {
            k: [states[k] for _ in range(int(self.fields[k].cfg.memory_depth))]
            for k in self.scale_keys
        }
        alpha = {
            k: states[k].new_zeros((states[k].shape[0], states[k].shape[1], self.fields[k].n_edges, states[k].shape[2], states[k].shape[3]))
            for k in self.scale_keys
        }
        beta = {k: torch.zeros_like(alpha[k]) for k in self.scale_keys}
        prev_sync: dict[str, torch.Tensor | None] = {k: None for k in self.scale_keys}

        record_states = {k: [states[k]] for k in self.scale_keys}
        record_fields: dict[str, list[torch.Tensor]] = {k: [] for k in self.scale_keys}
        record_sigs: dict[str, list[torch.Tensor]] = {k: [] for k in self.scale_keys}
        record_gates: dict[str, list[torch.Tensor]] = {k: [] for k in self.scale_keys}
        order_history: list[torch.Tensor] = []
        cross_history: list[torch.Tensor] = []
        context_gate_history: list[torch.Tensor] = []
        edge_type_history: list[torch.Tensor] = []

        mask = self.cross_scale_mask.to(device=self.cross_scale_weight.device, dtype=self.cross_scale_weight.dtype)
        weights = self.cross_scale_weight * mask
        field_weights = self.cross_field_weight * mask
        for _ in range(int(self.thought_steps)):
            local: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
            field_energy: dict[str, torch.Tensor] = {}
            norm_energy: dict[str, torch.Tensor] = {}
            edge_orders: dict[str, torch.Tensor] = {}
            edge_type_orders = []
            orders = []
            for k in self.scale_keys:
                field = self.fields[k]
                alpha[k], beta[k], sync, sync_drive, residual, conflict_gate = field._sync_step(
                    states[k], alpha[k], beta[k], prev_sync[k]
                )
                local[k] = (sync, sync_drive, residual, conflict_gate)
                # CTM field-order energy map: no detector boxes, no token
                # branch, no convolutional adapter.  It is a spatially indexed
                # statistic of raw synchronization energy.
                field_energy[k] = sync.abs().mean(dim=(1, 2))
                centered = field_energy[k] - field_energy[k].mean(dim=(-1, -2), keepdim=True)
                scale = centered.pow(2).mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(float(field.cfg.eps))
                norm_energy[k] = centered / scale
                edge = sync.abs().mean(dim=(1, 3, 4))
                edge = edge - edge.mean(dim=1, keepdim=True)
                edge = edge / edge.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(float(field.cfg.eps))
                edge_orders[k] = edge
                edge_type_orders.append(_edge_type_order_vector(sync, field))
                orders.append(self._scale_order_parameter(sync))
                prev_sync[k] = sync

            order = torch.stack(orders, dim=1)  # B x S
            edge_type_order = torch.stack(edge_type_orders, dim=1)  # B x S x D
            source_minus_target = order.unsqueeze(1) - order.unsqueeze(2)  # B x target x source
            cross = (source_minus_target * weights.view(1, self.n_scales, self.n_scales)).sum(dim=2)
            order_history.append(order)
            cross_history.append(cross)
            edge_type_history.append(edge_type_order)

            new_states: dict[str, torch.Tensor] = {}
            gate_scalars = []
            for i, k in enumerate(self.scale_keys):
                sync, sync_drive, residual, conflict_gate = local[k]
                # Cross-scale communication modulates the local CTM drive.  It
                # never creates a new spatial map or detector-side branch.
                scalar_cross = cross[:, i].view(-1, 1, 1, 1).to(dtype=sync_drive.dtype, device=sync_drive.device)
                if self.cross_edge_coupling > 0:
                    edge_cross = sync.new_zeros(edge_orders[k].shape)
                    for j, src_key in enumerate(self.scale_keys):
                        if i == j or edge_orders[src_key].shape[1] != edge_orders[k].shape[1]:
                            continue
                        w = weights[i, j].to(dtype=sync.dtype, device=sync.device)
                        edge_cross = edge_cross + w * (edge_orders[src_key].to(dtype=sync.dtype, device=sync.device) - edge_orders[k])
                    edge_cross = torch.tanh(float(self.cross_edge_coupling) * edge_cross)
                    edge_drive = (
                        sync
                        * self.fields[k].sync_weight.view(1, self.fields[k].channels, self.fields[k].n_edges, 1, 1)
                        * edge_cross.view(edge_cross.shape[0], 1, edge_cross.shape[1], 1, 1)
                    ).sum(dim=2)
                    sync_drive = sync_drive + edge_drive
                if self.edge_order_moment_coupling > 0:
                    type_cross = sync.new_zeros((sync.shape[0], 4))
                    target_moments = edge_type_order[:, i, :].reshape(sync.shape[0], 4, -1)
                    # Use top-k order energy plus concentration to carry
                    # trigger-localized order spikes without carrying their
                    # coordinates.  This is a cross-scale edge-order transport,
                    # not a spatial map transfer.
                    target_strength = target_moments[:, :, 2] + 0.10 * target_moments[:, :, 3]
                    for j in range(self.n_scales):
                        if i == j:
                            continue
                        src_moments = edge_type_order[:, j, :].reshape(sync.shape[0], 4, -1)
                        src_strength = src_moments[:, :, 2] + 0.10 * src_moments[:, :, 3]
                        w = weights[i, j].to(dtype=sync.dtype, device=sync.device)
                        type_cross = type_cross + w * (
                            src_strength.to(dtype=sync.dtype, device=sync.device) - target_strength.to(dtype=sync.dtype, device=sync.device)
                        )
                    field = self.fields[k]
                    spatial_mask = field.spatial_edge_mask.to(device=sync.device, dtype=sync.dtype)
                    channel_mask = field.channel_edge_mask.to(device=sync.device, dtype=sync.dtype)
                    field_mask = field.field_edge_mask.to(device=sync.device, dtype=sync.dtype)
                    conflict_mask = spatial_mask - field_mask
                    edge_type_mod = (
                        type_cross[:, 0].view(-1, 1, 1, 1, 1) * spatial_mask
                        + type_cross[:, 1].view(-1, 1, 1, 1, 1) * channel_mask
                        + type_cross[:, 2].view(-1, 1, 1, 1, 1) * field_mask
                        + type_cross[:, 3].view(-1, 1, 1, 1, 1) * conflict_mask
                    )
                    moment_drive = (
                        sync
                        * self.fields[k].sync_weight.view(1, self.fields[k].channels, self.fields[k].n_edges, 1, 1)
                        * edge_type_mod
                    ).sum(dim=2)
                    sync_drive = sync_drive + torch.tanh(float(self.edge_order_moment_coupling) * moment_drive)
                field_cross = sync_drive.new_zeros((sync_drive.shape[0], sync_drive.shape[2], sync_drive.shape[3]))
                for j, src_key in enumerate(self.scale_keys):
                    if i == j:
                        continue
                    src_energy = field_energy[src_key].unsqueeze(1).to(dtype=sync_drive.dtype, device=sync_drive.device)
                    src_resized = F.interpolate(
                        src_energy,
                        size=field_energy[k].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    tgt_energy = field_energy[k].to(dtype=sync_drive.dtype, device=sync_drive.device)
                    w = field_weights[i, j].to(dtype=sync_drive.dtype, device=sync_drive.device)
                    field_cross = field_cross + w * (src_resized - tgt_energy)
                mod = torch.tanh(scalar_cross + field_cross.unsqueeze(1))
                coupled_drive = sync_drive * (1.0 + mod)
                if self.cross_context_gate_strength > 0:
                    higher = [src for src in self.scale_keys[i + 1 :]]
                    if higher:
                        ctx = sync_drive.new_zeros(field_energy[k].shape)
                        for src_key in higher:
                            src = norm_energy[src_key].unsqueeze(1).to(dtype=sync_drive.dtype, device=sync_drive.device)
                            ctx = ctx + F.interpolate(
                                src,
                                size=field_energy[k].shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(1)
                        ctx = ctx / float(len(higher))
                        local_norm = norm_energy[k].to(dtype=sync_drive.dtype, device=sync_drive.device)
                        # High-context anomaly: low-level CTM synchronization
                        # that is not supported by later-scale CTM order.
                        mismatch = local_norm - ctx
                        raw_gate = torch.sigmoid(
                            float(self.cross_context_gate_bias)
                            + float(self.cross_context_gate_strength) * mismatch
                        )
                        floor = float(self.cross_context_gate_floor)
                        ceiling = float(self.cross_context_gate_ceiling)
                        context_gate = floor + (ceiling - floor) * raw_gate
                    else:
                        context_gate = sync_drive.new_ones(field_energy[k].shape)
                    coupled_drive = coupled_drive * context_gate.unsqueeze(1)
                    gate_scalars.append(context_gate.mean(dim=(-1, -2)))
                else:
                    gate_scalars.append(sync_drive.new_ones((sync_drive.shape[0],)))
                new_state, update_gate = self._thought_update(
                    self.fields[k],
                    states[k],
                    histories[k],
                    coupled_drive,
                    residual,
                    conflict_gate,
                )
                new_states[k] = new_state
                histories[k] = (histories[k] + [new_state])[-int(self.fields[k].cfg.memory_depth):]
                if return_trace:
                    record_states[k].append(new_state)
                    record_fields[k].append(sync)
                    record_sigs[k].append(sync.mean(dim=(-1, -2)))
                    record_gates[k].append(update_gate)
            states = new_states
            context_gate_history.append(torch.stack(gate_scalars, dim=1))

        if not return_trace:
            return states

        traces: dict[str, LatticeCTMTrace] = {}
        for k in self.scale_keys:
            sig = (
                torch.stack(record_sigs[k], dim=1)
                if record_sigs[k]
                else states[k].new_zeros((states[k].shape[0], 0, states[k].shape[1], self.fields[k].n_edges))
            )
            traces[k] = LatticeCTMTrace(
                final=states[k],
                states=record_states[k],
                sync_signatures=sig,
                sync_fields=record_fields[k],
                update_gates=record_gates[k],
            )
        scale_orders = torch.stack(order_history, dim=1) if order_history else next(iter(states.values())).new_zeros((0, 0, self.n_scales))
        cross_modulators = torch.stack(cross_history, dim=1) if cross_history else scale_orders
        context_gates = torch.stack(context_gate_history, dim=1) if context_gate_history else scale_orders
        edge_type_orders_out = torch.stack(edge_type_history, dim=1) if edge_type_history else None
        return CoupledMultiScaleLatticeCTMTrace(
            final=states,
            traces=traces,
            scale_orders=scale_orders,
            cross_modulators=cross_modulators,
            context_gates=context_gates,
            edge_type_orders=edge_type_orders_out,
        )


def cross_scale_order_consistency_loss(trace: CoupledMultiScaleLatticeCTMTrace) -> torch.Tensor:
    """CTM-native cross-scale order consistency.

    This is the loss counterpart to the coupled recurrence.  It operates only
    on scale-order trajectories and therefore does not decode boxes, consult a
    clean model, or calibrate detector outputs.
    """

    if trace.scale_orders.numel() == 0 or trace.scale_orders.shape[-1] < 2:
        return next(iter(trace.final.values())).sum() * 0.0
    centered = trace.scale_orders - trace.scale_orders.mean(dim=2, keepdim=True)
    return centered.pow(2).mean()


def cross_scale_context_gate_contrast_loss(
    invalid_trace: CoupledMultiScaleLatticeCTMTrace,
    valid_trace: CoupledMultiScaleLatticeCTMTrace,
    *,
    margin: float = 0.18,
    floor: float = 0.25,
) -> torch.Tensor:
    """Pure CTM contrast on high-context anomaly gates.

    The gate is derived only from cross-scale CTM synchronization energy.  It is
    trained to stay quiet on valid object fields and to become active on invalid
    trigger fields, without clean-anchor models, detector-output matching,
    score calibration, or post-hoc editing.
    """

    if invalid_trace.context_gates is None or valid_trace.context_gates is None:
        return next(iter(invalid_trace.final.values())).sum() * 0.0
    gi = invalid_trace.context_gates
    gv = valid_trace.context_gates
    if gi.numel() == 0 or gv.numel() == 0 or gi.shape[-1] < 2 or gv.shape[-1] < 2:
        return next(iter(invalid_trace.final.values())).sum() * 0.0
    # Last scale has no later semantic context, so the anomaly contract is
    # applied to the earlier native feature fields only.
    gi = gi[..., :-1]
    gv = gv[..., :-1]
    floor = min(max(float(floor), 0.0), 0.99)
    denom = max(1.0 - floor, 1e-6)
    ai = ((gi - floor) / denom).clamp(0.0, 1.0).mean(dim=(1, 2))
    av = ((gv - floor) / denom).clamp(0.0, 1.0).mean(dim=(1, 2))
    valid_quiet = av.pow(2).mean()
    invalid_active = F.relu(float(margin) - ai).pow(2).mean()
    ranking = F.relu(float(margin) + av.mean() - ai.mean()).pow(2)
    return valid_quiet + invalid_active + ranking


def _scale_energy_map(trace: CoupledMultiScaleLatticeCTMTrace, key: str) -> torch.Tensor:
    fields = trace.traces[key].sync_fields
    if not fields:
        final = trace.traces[key].final
        return final.new_zeros((final.shape[0], final.shape[2], final.shape[3]))
    # B x T x C x E x H x W -> B x H x W
    return torch.stack(fields, dim=1).abs().mean(dim=(1, 2, 3))


def _zmap(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    c = x - x.mean(dim=(-1, -2), keepdim=True)
    s = c.pow(2).mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(float(eps))
    return c / s


def _shape_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp_min(0.0)
    return x / x.mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))


def _cross_scale_mismatch_maps(trace: CoupledMultiScaleLatticeCTMTrace) -> dict[str, torch.Tensor]:
    keys = list(trace.traces.keys())
    energies = {k: _scale_energy_map(trace, k) for k in keys}
    norm = {k: _zmap(v) for k, v in energies.items()}
    mismatches: dict[str, torch.Tensor] = {}
    for i, key in enumerate(keys[:-1]):
        higher = keys[i + 1 :]
        ctx = norm[key].new_zeros(norm[key].shape)
        for hkey in higher:
            ctx = ctx + F.interpolate(
                norm[hkey].unsqueeze(1),
                size=norm[key].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        ctx = ctx / float(len(higher))
        # Low-level synchronization energy unsupported by higher-level context.
        mismatches[key] = F.relu(norm[key] - ctx)
    return mismatches


def cross_scale_mismatch_motion_loss(
    invalid_trace: CoupledMultiScaleLatticeCTMTrace,
    valid_trace: CoupledMultiScaleLatticeCTMTrace,
    *,
    invalid_floor: float = 0.02,
    valid_weight: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align CTM thought motion to cross-scale mismatch.

    This is the stronger successor to the context gate.  For invalid samples,
    terminal thought motion is encouraged to concentrate inside the CTM-only
    mismatch map: low-level synchronization not supported by later scales.
    Motion outside mismatch is penalized, and a small floor prevents the trivial
    no-motion solution.  For valid samples, motion is damped where such mismatch
    appears.

    It uses only CTM states and raw synchronization fields; no detector boxes,
    no logits, no clean-anchor model, and no post-hoc score rule.
    """

    inv_mismatch = _cross_scale_mismatch_maps(invalid_trace)
    val_mismatch = _cross_scale_mismatch_maps(valid_trace)
    if not inv_mismatch:
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {"mismatch_invalid_focus": 0.0, "mismatch_valid_quiet": 0.0}

    invalid_terms = []
    valid_terms = []
    for key, mi in inv_mismatch.items():
        ti = invalid_trace.traces[key]
        tv = valid_trace.traces[key]
        vi = val_mismatch[key]
        mi_shape = _shape_map(mi.detach(), eps=eps)
        mi_mask = (mi_shape / (1.0 + mi_shape)).detach()
        ri = (ti.final - ti.states[0]).abs().mean(dim=1)
        iref = ti.states[0].abs().mean(dim=1).mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))
        ri_rel = ri / iref
        outside = (ri_rel.pow(2) * (1.0 - mi_mask)).mean()
        inside = (ri_rel * mi_mask).mean()
        floor_loss = F.relu(float(invalid_floor) - inside).pow(2)
        invalid_terms.append(outside + floor_loss)

        rv = (tv.final - tv.states[0]).abs().mean(dim=1)
        ref = tv.states[0].abs().mean(dim=1).mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))
        rv_rel = rv / ref
        vi_shape = _shape_map(vi.detach(), eps=eps)
        valid_terms.append((rv_rel.pow(2) * (1.0 + vi_shape)).mean())

    invalid_align = torch.stack(invalid_terms).mean()
    valid_quiet = torch.stack(valid_terms).mean()
    loss = invalid_align + float(valid_weight) * valid_quiet
    stats = {
        "mismatch_invalid_focus": float(invalid_align.detach().cpu()),
        "mismatch_valid_quiet": float(valid_quiet.detach().cpu()),
    }
    return loss, stats


def _residual_channel_profile(
    trace: LatticeCTMTrace,
    *,
    topk_frac: float = 0.08,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Location-free CTM residual profile.

    The source-level v2 failures show that position/scene variation matters.
    This profile removes spatial location by taking each channel's strongest
    residual cells, giving a channel-energy signature of the CTM thought motion.
    It is a statistic of recurrent states only: no detector logits, boxes,
    anchors, clean model, or post-hoc rule.
    """

    if not trace.states:
        z = trace.final.new_zeros((trace.final.shape[0], trace.final.shape[1]))
        return z, z.sum(dim=1)
    residual = (trace.final - trace.states[0]).abs()
    flat = residual.flatten(start_dim=2)
    k = max(1, int(round(min(max(float(topk_frac), 1e-5), 1.0) * flat.shape[-1])))
    profile = flat.topk(k, dim=-1).values.mean(dim=-1)
    ref = trace.states[0].abs().flatten(start_dim=2).mean(dim=-1).mean(dim=1, keepdim=True).clamp_min(float(eps))
    profile = profile / ref
    magnitude = profile.norm(dim=1)
    profile = profile / magnitude.unsqueeze(1).clamp_min(float(eps))
    return profile, magnitude


def source_invariant_residual_profile_loss(
    invalid_trace: CoupledMultiScaleLatticeCTMTrace,
    valid_trace: CoupledMultiScaleLatticeCTMTrace,
    *,
    invalid_floor: float = 0.04,
    valid_weight: float = 0.25,
    topk_frac: float = 0.08,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Source-invariant CTM residual profile loss.

    Visible OGA residual failures were whole unseen source images rather than
    individual augmentations.  This term asks invalid CTM thought residuals to
    share a location-free channel-energy profile across sources while keeping
    valid residual magnitude small.  It is deliberately CTM-native: it never
    inspects detector boxes/scores, never uses a clean anchor, and never creates
    a runtime branch.
    """

    if not invalid_trace.traces:
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {
            "source_profile_compact": 0.0,
            "source_profile_floor": 0.0,
            "source_profile_valid": 0.0,
        }

    compact_terms = []
    floor_terms = []
    valid_terms = []
    for key in invalid_trace.traces:
        inv_profile, inv_mag = _residual_channel_profile(
            invalid_trace.traces[key],
            topk_frac=topk_frac,
        )
        val_profile, val_mag = _residual_channel_profile(
            valid_trace.traces[key],
            topk_frac=topk_frac,
        )
        if inv_profile.shape[0] > 1:
            sim = inv_profile @ inv_profile.t()
            eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            compact_terms.append((1.0 - sim[~eye]).pow(2).mean())
        else:
            compact_terms.append(inv_profile.sum() * 0.0)
        floor_terms.append(F.relu(float(invalid_floor) - inv_mag).pow(2).mean())
        # Valid profile direction is irrelevant; its recurrent residual should
        # simply stay small in this location-free channel view.
        valid_terms.append(val_mag.pow(2).mean())

    compact = torch.stack(compact_terms).mean()
    floor = torch.stack(floor_terms).mean()
    valid = torch.stack(valid_terms).mean()
    loss = compact + floor + float(valid_weight) * valid
    stats = {
        "source_profile_compact": float(compact.detach().cpu()),
        "source_profile_floor": float(floor.detach().cpu()),
        "source_profile_valid": float(valid.detach().cpu()),
    }
    return loss, stats


def _order_flow_vectors(trace: CoupledMultiScaleLatticeCTMTrace) -> torch.Tensor:
    """Return per-sample CTM scale-order flow vectors.

    The vector is built only from CTM order parameters:
        O_t = order(raw-sync field at thought step t)
        flow = [O_1-O_0, ..., O_T-O_{T-1}]

    It contains no detector scores, boxes, spatial maps, clean-anchor weights,
    or post-hoc edits. It is the smallest strictly CTM-native signal that can
    describe how a trigger changes the recurrent synchronization trajectory.
    """
    orders = trace.scale_orders
    if orders.ndim != 3 or orders.shape[0] == 0:
        return orders.new_zeros((0, 0))
    if orders.shape[1] < 2:
        return orders.new_zeros((orders.shape[0], max(1, orders.shape[-1])))
    return (orders[:, 1:, :] - orders[:, :-1, :]).reshape(orders.shape[0], -1)


def counterfactual_order_flow_invariance_loss(
    invalid_trace: CoupledMultiScaleLatticeCTMTrace,
    valid_trace: CoupledMultiScaleLatticeCTMTrace | None = None,
    source_ids: torch.Tensor | None = None,
    *,
    flow_floor: float = 0.015,
    same_source_weight: float = 0.5,
    cross_source_weight: float = 1.0,
    valid_quiet_weight: float = 0.25,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Counterfactual order-flow loss for source-level generalization.

    Residual v2 failures are whole source-image groups: once a source fails, all
    trigger variants fail. Existing terminal-state consistency can learn a
    source-specific basin; it does not force a source-invariant trigger-causal
    transform. This loss operates only on CTM scale-order trajectories and asks
    the recurrent order-flow induced by invalid trigger samples to be consistent
    across variants and across source identities.

    Let O_t be the CTM scale-order vector at thought tick t. We define
    order-flow q = [O_1-O_0, ..., O_T-O_{T-1}]. The objective has three parts:

    1. same-source compactness: variants of the same source should have similar
       order-flow directions;
    2. cross-source invariance: source means should share a trigger-causal
       order-flow direction rather than memorizing source-specific basins;
    3. valid quietness: valid clean-object order-flow should remain small.

    No spatial energy map, frequency token, detector box, score calibration,
    clean-anchor model, or runtime branch is used.
    """
    q = _order_flow_vectors(invalid_trace)
    if q.numel() == 0 or q.shape[0] == 0:
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {
            "order_flow_same_source": 0.0,
            "order_flow_cross_source": 0.0,
            "order_flow_floor": 0.0,
            "order_flow_valid_quiet": 0.0,
            "order_flow_sources": 0.0,
        }

    norm = q.norm(dim=1, keepdim=True).clamp_min(float(eps))
    q_dir = q / norm
    floor_loss = F.relu(float(flow_floor) - norm.squeeze(1)).pow(2).mean()
    loss = floor_loss
    same_loss = q.sum() * 0.0
    cross_loss = q.sum() * 0.0
    n_sources = 0

    if source_ids is not None and source_ids.numel() == q.shape[0]:
        ids = source_ids.view(-1).to(q.device)
        same = ids[:, None] == ids[None, :]
        eye = torch.eye(q.shape[0], dtype=torch.bool, device=q.device)
        same = same & ~eye
        if bool(same.any()) and same_source_weight > 0:
            cos = q_dir @ q_dir.t()
            same_loss = (1.0 - cos[same]).pow(2).mean()
            loss = loss + float(same_source_weight) * same_loss

        means = []
        for sid in torch.unique(ids):
            mask = ids == sid
            if bool(mask.any()):
                m = q[mask].mean(dim=0)
                m = m / m.norm().clamp_min(float(eps))
                means.append(m)
        n_sources = len(means)
        if len(means) > 1 and cross_source_weight > 0:
            m = torch.stack(means, dim=0)
            cos = m @ m.t()
            off = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
            cross_loss = (1.0 - cos[off]).pow(2).mean()
            loss = loss + float(cross_source_weight) * cross_loss
    elif q.shape[0] > 1 and cross_source_weight > 0:
        cos = q_dir @ q_dir.t()
        off = ~torch.eye(q.shape[0], dtype=torch.bool, device=q.device)
        cross_loss = (1.0 - cos[off]).pow(2).mean()
        loss = loss + float(cross_source_weight) * cross_loss
        n_sources = int(q.shape[0])

    valid_quiet = q.sum() * 0.0
    if valid_trace is not None and valid_quiet_weight > 0:
        qv = _order_flow_vectors(valid_trace)
        if qv.numel() > 0:
            valid_quiet = qv.pow(2).mean()
            loss = loss + float(valid_quiet_weight) * valid_quiet

    stats = {
        "order_flow_same_source": float(same_loss.detach().cpu()),
        "order_flow_cross_source": float(cross_loss.detach().cpu()),
        "order_flow_floor": float(floor_loss.detach().cpu()),
        "order_flow_valid_quiet": float(valid_quiet.detach().cpu()),
        "order_flow_sources": float(n_sources),
    }
    return loss, stats


def _edge_order_transport_vectors(
    trace: CoupledMultiScaleLatticeCTMTrace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened and per-scale edge-order transport vectors.

    Shape:
      flat: B x ((T-1)*S*4)
      per_scale: B x S x ((T-1)*4)
    """

    e = trace.edge_type_orders
    if e is None or e.numel() == 0 or e.ndim != 4 or e.shape[0] == 0:
        z = next(iter(trace.final.values())).new_zeros((0, 0))
        return z, z.reshape(0, 0, 0)
    if e.shape[1] < 2:
        flow = e
    else:
        flow = e[:, 1:, :, :] - e[:, :-1, :, :]
    flat = flow.reshape(flow.shape[0], -1)
    per_scale = flow.permute(0, 2, 1, 3).reshape(flow.shape[0], flow.shape[2], -1)
    return flat, per_scale


def counterfactual_edge_order_transport_loss(
    invalid_trace: CoupledMultiScaleLatticeCTMTrace,
    valid_trace: CoupledMultiScaleLatticeCTMTrace | None = None,
    source_ids: torch.Tensor | None = None,
    *,
    flow_floor: float = 0.01,
    same_source_weight: float = 0.25,
    cross_source_weight: float = 1.0,
    scale_transport_weight: float = 0.5,
    valid_quiet_weight: float = 0.25,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Counterfactual edge-order transport loss.

    Unlike scalar order-flow, this objective keeps the CTM edge-family
    structure: spatial-edge order, channel-edge order, field-order, and local
    conflict order.  It never transports a HxW map between scales and never
    touches detector boxes/scores at runtime.  It asks trigger samples to share
    a source-invariant edge-type transport direction while preserving valid
    order quietness.
    """

    q, per_scale = _edge_order_transport_vectors(invalid_trace)
    if q.numel() == 0 or q.shape[0] == 0:
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {
            "edge_order_same_source": 0.0,
            "edge_order_cross_source": 0.0,
            "edge_order_scale_transport": 0.0,
            "edge_order_floor": 0.0,
            "edge_order_valid_quiet": 0.0,
            "edge_order_sources": 0.0,
        }

    norm = q.norm(dim=1, keepdim=True).clamp_min(float(eps))
    q_dir = q / norm
    floor_loss = F.relu(float(flow_floor) - norm.squeeze(1)).pow(2).mean()
    loss = floor_loss
    same_loss = q.sum() * 0.0
    cross_loss = q.sum() * 0.0
    scale_loss = q.sum() * 0.0
    n_sources = 0

    if source_ids is not None and source_ids.numel() == q.shape[0]:
        ids = source_ids.view(-1).to(q.device)
        same = ids[:, None] == ids[None, :]
        eye = torch.eye(q.shape[0], dtype=torch.bool, device=q.device)
        same = same & ~eye
        if bool(same.any()) and same_source_weight > 0:
            cos = q_dir @ q_dir.t()
            same_loss = (1.0 - cos[same]).pow(2).mean()
            loss = loss + float(same_source_weight) * same_loss

        means = []
        for sid in torch.unique(ids):
            mask = ids == sid
            if bool(mask.any()):
                m = q[mask].mean(dim=0)
                m = m / m.norm().clamp_min(float(eps))
                means.append(m)
        n_sources = len(means)
        if len(means) > 1 and cross_source_weight > 0:
            m = torch.stack(means, dim=0)
            cos = m @ m.t()
            off = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
            cross_loss = (1.0 - cos[off]).pow(2).mean()
            loss = loss + float(cross_source_weight) * cross_loss
    elif q.shape[0] > 1 and cross_source_weight > 0:
        cos = q_dir @ q_dir.t()
        off = ~torch.eye(q.shape[0], dtype=torch.bool, device=q.device)
        cross_loss = (1.0 - cos[off]).pow(2).mean()
        loss = loss + float(cross_source_weight) * cross_loss
        n_sources = int(q.shape[0])

    if per_scale.numel() > 0 and per_scale.shape[1] > 1 and scale_transport_weight > 0:
        ps = per_scale / per_scale.norm(dim=2, keepdim=True).clamp_min(float(eps))
        cos = torch.einsum("bsd,btd->bst", ps, ps)
        off = ~torch.eye(per_scale.shape[1], dtype=torch.bool, device=per_scale.device)
        scale_loss = (1.0 - cos[:, off]).pow(2).mean()
        loss = loss + float(scale_transport_weight) * scale_loss

    valid_quiet = q.sum() * 0.0
    if valid_trace is not None and valid_quiet_weight > 0:
        qv, _ = _edge_order_transport_vectors(valid_trace)
        if qv.numel() > 0:
            valid_quiet = qv.pow(2).mean()
            loss = loss + float(valid_quiet_weight) * valid_quiet

    stats = {
        "edge_order_same_source": float(same_loss.detach().cpu()),
        "edge_order_cross_source": float(cross_loss.detach().cpu()),
        "edge_order_scale_transport": float(scale_loss.detach().cpu()),
        "edge_order_floor": float(floor_loss.detach().cpu()),
        "edge_order_valid_quiet": float(valid_quiet.detach().cpu()),
        "edge_order_sources": float(n_sources),
    }
    return loss, stats


def counterfactual_source_consistency_loss(
    invalid_trace: CoupledMultiScaleLatticeCTMTrace,
    source_ids: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Same-source counterfactual CTM terminal consistency.

    For a visible trigger, variants of the same source image differ mainly by
    trigger placement/scale/photometric perturbation.  A CTM purifier that has
    learned the trigger-causal component should map those variants toward the
    same terminal neural-field state.  This loss applies only between samples
    that share a source id, and only to CTM terminal states.

    It is not a clean-anchor reconstruction loss: no clean image or clean model
    is consulted.  It is also not a detector-side rule: no boxes, scores, NMS,
    or thresholds are inspected.
    """

    if source_ids.numel() == 0 or not invalid_trace.traces:
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {"source_consistency": 0.0, "source_consistency_pairs": 0.0}
    ids = source_ids.view(-1).to(next(iter(invalid_trace.final.values())).device)
    pair = (ids[:, None] == ids[None, :])
    eye = torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)
    pair = pair & ~eye
    if not bool(pair.any()):
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {"source_consistency": 0.0, "source_consistency_pairs": 0.0}

    terms = []
    for tr in invalid_trace.traces.values():
        x = tr.final.flatten(start_dim=1)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(float(eps))
        dist = (x[:, None, :] - x[None, :, :]).pow(2).mean(dim=-1)
        terms.append(dist[pair].mean())

    value = torch.stack(terms).mean()
    stats = {
        "source_consistency": float(value.detach().cpu()),
        "source_consistency_pairs": float(pair.float().sum().detach().cpu()),
    }
    return value, stats


def counterfactual_difference_gated_motion_loss(
    invalid_trace: CoupledMultiScaleLatticeCTMTrace,
    source_ids: torch.Tensor,
    *,
    topk_frac: float = 0.16,
    inside_floor: float = 0.015,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Same-source counterfactual difference gated CTM motion.

    The residual v2 failures are source-level: once a held-out source fails,
    every trigger variant of that source fails.  A pure CTM fix should therefore
    separate source-stable content from trigger-varying content without using a
    clean anchor, detector boxes, score calibration, or a runtime branch.

    For same-source triggered variants we estimate, at each CTM scale, the
    lattice cells whose *initial* neural fields change most across variants:

        D_i(u) = mean_{j:s_j=s_i,j!=i} mean_c |F_0^i(c,u) - F_0^j(c,u)|

    The top fraction of D_i is a detached counterfactual-difference support.
    CTM terminal motion R_i(u)=mean_c|F_T^i-F_0^i| is encouraged to occur inside
    that support and penalised outside it.  This is a training-time dynamical
    prior only; inference is still just the learned recurrent lattice.
    """

    if source_ids.numel() == 0 or not invalid_trace.traces:
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {
            "cf_diff_outside": 0.0,
            "cf_diff_inside_floor": 0.0,
            "cf_diff_pairs": 0.0,
            "cf_diff_support_frac": 0.0,
        }
    ids = source_ids.view(-1).to(next(iter(invalid_trace.final.values())).device)
    pair = ids[:, None] == ids[None, :]
    eye = torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)
    pair = pair & ~eye
    if not bool(pair.any()):
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {
            "cf_diff_outside": 0.0,
            "cf_diff_inside_floor": 0.0,
            "cf_diff_pairs": 0.0,
            "cf_diff_support_frac": 0.0,
        }

    frac = min(max(float(topk_frac), 1e-4), 1.0)
    outside_terms = []
    floor_terms = []
    support_fracs = []
    for tr in invalid_trace.traces.values():
        if not tr.states:
            continue
        f0 = tr.states[0]
        ft = tr.final
        diff = (f0[:, None] - f0[None, :]).abs().mean(dim=2)  # B x B x H x W
        denom = pair.float().sum(dim=1).view(-1, 1, 1).clamp_min(1.0)
        diff_map = (diff * pair.float().view(pair.shape[0], pair.shape[1], 1, 1)).sum(dim=1) / denom
        flat = diff_map.flatten(start_dim=1)
        k = max(1, int(round(frac * flat.shape[1])))
        topk_idx = flat.topk(k, dim=1).indices
        support_flat = torch.zeros_like(flat)
        support_flat.scatter_(1, topk_idx, 1.0)
        support = support_flat.view_as(diff_map).to(dtype=ft.dtype).detach()
        support_fracs.append(support.mean())

        motion = (ft - f0).abs().mean(dim=1)
        ref = f0.abs().mean(dim=1).mean(dim=(-1, -2), keepdim=True).clamp_min(float(eps))
        rel_motion = motion / ref
        outside_terms.append((rel_motion.pow(2) * (1.0 - support)).mean())
        inside = (rel_motion * support).sum(dim=(-1, -2)) / support.sum(dim=(-1, -2)).clamp_min(1.0)
        floor_terms.append(F.relu(float(inside_floor) - inside).pow(2).mean())

    if not outside_terms:
        zero = next(iter(invalid_trace.final.values())).sum() * 0.0
        return zero, {
            "cf_diff_outside": 0.0,
            "cf_diff_inside_floor": 0.0,
            "cf_diff_pairs": float(pair.float().sum().detach().cpu()),
            "cf_diff_support_frac": 0.0,
        }
    outside = torch.stack(outside_terms).mean()
    inside_floor_loss = torch.stack(floor_terms).mean()
    support_frac = torch.stack(support_fracs).mean()
    value = outside + inside_floor_loss
    stats = {
        "cf_diff_outside": float(outside.detach().cpu()),
        "cf_diff_inside_floor": float(inside_floor_loss.detach().cpu()),
        "cf_diff_pairs": float(pair.float().sum().detach().cpu()),
        "cf_diff_support_frac": float(support_frac.detach().cpu()),
    }
    return value, stats
