from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .schema import LatticeCTMConfig


@dataclass
class LatticeCTMTrace:
    final: torch.Tensor
    states: list[torch.Tensor]
    sync_signatures: torch.Tensor  # B x T x C x E
    sync_fields: list[torch.Tensor]  # each B x C x E x H x W
    update_gates: list[torch.Tensor]  # each B x C x H x W


def _decay_logit(value: float) -> float:
    v = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(v)).item())


def _channel_shift(x: torch.Tensor, offset: int) -> torch.Tensor:
    if offset == 0:
        return x
    y = torch.zeros_like(x)
    if offset > 0:
        y[:, offset:, :, :] = x[:, :-offset, :, :]
    else:
        k = -offset
        y[:, :-k, :, :] = x[:, k:, :, :]
    return y


def _spatial_shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    if dy == 0 and dx == 0:
        return x
    b, c, h, w = x.shape
    y = torch.zeros_like(x)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy) if dy <= 0 else h
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx) if dx <= 0 else w
    if src_y1 > src_y0 and src_x1 > src_x0:
        y[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, :, src_y0:src_y1, src_x0:src_x1]
    return y


class PrivateTemporalMLP(nn.Module):
    """SuperLinear-style private temporal processor.

    Each channel owns its own M->H->2 processor.  The two outputs are: (1) a
    temporal state drive and (2) a self-update gate bias.  The processor is
    applied at every lattice cell without adding any external token branch.
    """

    def __init__(self, channels: int, memory_depth: int, hidden_dim: int, *, zero_out: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.memory_depth = int(memory_depth)
        self.hidden_dim = int(hidden_dim)
        scale = max(1.0, float(memory_depth)) ** -0.5
        self.w1 = nn.Parameter(torch.randn(self.channels, self.hidden_dim, self.memory_depth) * scale)
        self.b1 = nn.Parameter(torch.zeros(self.channels, self.hidden_dim))
        self.w2 = nn.Parameter(torch.randn(self.channels, self.hidden_dim, 2) * max(1.0, float(hidden_dim)) ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(self.channels, 2))
        if zero_out:
            nn.init.zeros_(self.w2)
            nn.init.zeros_(self.b2)

    def forward(self, history: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(history) != self.memory_depth:
            raise ValueError(f"expected {self.memory_depth} history tensors, got {len(history)}")
        x = torch.stack(history, dim=2)  # B x C x M x H x W
        h = torch.einsum("bcmxy,chm->bchxy", x, self.w1) + self.b1.view(1, self.channels, self.hidden_dim, 1, 1)
        h = torch.tanh(h)
        out = torch.einsum("bchxy,cho->bcoxy", h, self.w2) + self.b2.view(1, self.channels, 2, 1, 1)
        return out[:, :, 0], out[:, :, 1]


class LatticeNFCTMNeuronField(nn.Module):
    """CTM neuron-field layer with local and field-order raw synchronization.

    The layer treats every (channel, y, x) activation as a CTM neuron.  Its
    terminal recurrent state is the purified feature.  It does not create
    spatial/frequency tokens, does not interpolate against a clean model, and
    does not run a post-hoc editor.
    """

    def __init__(self, cfg: LatticeCTMConfig):
        super().__init__()
        self.cfg = cfg
        self.channels = int(cfg.channels)
        self.temporal = PrivateTemporalMLP(
            channels=self.channels,
            memory_depth=int(cfg.memory_depth),
            hidden_dim=int(cfg.hidden_dim),
            zero_out=bool(cfg.zero_init_temporal_out),
        )
        self.edge_specs: list[tuple[str, int, int]] = []
        for offset in cfg.channel_offsets:
            if int(offset) != 0:
                self.edge_specs.append(("channel", int(offset), 0))
        for radius in cfg.spatial_radii:
            r = int(radius)
            if r <= 0:
                continue
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy == 0 and dx == 0:
                        continue
                    if abs(dy) + abs(dx) == r:
                        self.edge_specs.append(("spatial", int(dy), int(dx)))
        if cfg.use_field_order_edges:
            # Same-channel order parameter: a pure CTM field-order sync edge.
            self.edge_specs.append(("field_same_channel", 0, 0))
        if cfg.use_channel_order_edges:
            # Same-spatial cell channel order parameter.
            self.edge_specs.append(("field_same_cell", 0, 0))
        if not self.edge_specs:
            raise ValueError("LatticeNFCTMNeuronField requires at least one synchronization edge")
        e = len(self.edge_specs)
        self.decay_logit = nn.Parameter(torch.full((self.channels, e), _decay_logit(cfg.init_decay)))
        self.sync_weight = nn.Parameter(torch.randn(self.channels, e) * float(cfg.init_sync_weight_std))
        self.sync_bias = nn.Parameter(torch.zeros(self.channels))
        self.local_edge_polarity_weight = nn.Parameter(
            torch.full((self.channels,), float(getattr(cfg, "local_edge_polarity_init", 0.0)))
        )
        self.update_gate_bias = nn.Parameter(torch.full((self.channels,), float(cfg.update_gate_bias)))
        self.step_size_logit = nn.Parameter(torch.logit(torch.tensor(min(max(float(cfg.step_size), 1e-4), 1.0 - 1e-4))))
        self.sync_gain = nn.Parameter(torch.tensor(float(cfg.sync_gain)))
        self.register_buffer(
            "spatial_edge_mask",
            torch.tensor([1.0 if spec[0] == "spatial" else 0.0 for spec in self.edge_specs]).view(1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "channel_edge_mask",
            torch.tensor([1.0 if spec[0] == "channel" else 0.0 for spec in self.edge_specs]).view(1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "field_edge_mask",
            torch.tensor([1.0 if spec[0].startswith("field_") else 0.0 for spec in self.edge_specs]).view(1, 1, -1, 1, 1),
            persistent=False,
        )

    @property
    def n_edges(self) -> int:
        return len(self.edge_specs)

    def _neighbor(self, x: torch.Tensor, spec: tuple[str, int, int]) -> torch.Tensor:
        kind, a, b = spec
        if kind == "channel":
            return _channel_shift(x, a)
        if kind == "spatial":
            return _spatial_shift(x, a, b)
        if kind == "field_same_channel":
            return x.mean(dim=(-1, -2), keepdim=True).expand_as(x)
        if kind == "field_same_cell":
            return x.mean(dim=1, keepdim=True).expand_as(x)
        raise ValueError(f"unknown edge spec: {spec}")

    def _gate_residual(self, residual: torch.Tensor) -> torch.Tensor:
        grad_scale = min(max(float(getattr(self.cfg, "adaptive_residual_grad_scale", 0.0)), 0.0), 1.0)
        if grad_scale <= 0.0:
            return residual.detach()
        if grad_scale >= 1.0:
            return residual
        return residual.detach() + grad_scale * (residual - residual.detach())

    def _sync_step(self, state: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, prev_sync: torch.Tensor | None):
        pair = torch.stack([state * self._neighbor(state, spec) for spec in self.edge_specs], dim=2)
        decay = torch.sigmoid(self.decay_logit).view(1, self.channels, self.n_edges, 1, 1)
        alpha = decay * alpha + pair
        beta = decay * beta + 1.0
        sync = alpha / torch.sqrt(beta + float(self.cfg.eps))
        sync_drive = (sync * self.sync_weight.view(1, self.channels, self.n_edges, 1, 1)).sum(dim=2)
        sync_drive = sync_drive + self.sync_bias.view(1, self.channels, 1, 1)
        conflict_strength = float(getattr(self.cfg, "local_edge_conflict_strength", 0.0))
        polarity_strength = float(getattr(self.cfg, "local_edge_polarity_strength", 0.0))
        conflict_gate = None
        if conflict_strength > 0 or polarity_strength > 0:
            spatial_mask = self.spatial_edge_mask.to(device=sync.device, dtype=sync.dtype)
            field_mask = self.field_edge_mask.to(device=sync.device, dtype=sync.dtype)
            n_spatial = spatial_mask.sum().clamp_min(1.0)
            n_field = field_mask.sum().clamp_min(1.0)
            spatial_order = (sync.abs() * spatial_mask).sum(dim=2) / n_spatial
            field_order = (sync.abs() * field_mask).sum(dim=2) / n_field
            conflict = spatial_order - field_order
            conflict = conflict - conflict.mean(dim=(-1, -2), keepdim=True)
            scale = conflict.pow(2).mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(float(self.cfg.eps))
            conflict = conflict / scale
            if polarity_strength > 0:
                polarity = conflict * self.local_edge_polarity_weight.view(1, self.channels, 1, 1)
                if bool(getattr(self.cfg, "local_edge_polarity_use_conflict_gate", False)):
                    center = float(getattr(self.cfg, "local_edge_conflict_center", 0.0))
                    polarity_gate = torch.sigmoid(abs(polarity_strength) * (conflict.abs() - center))
                    polarity = polarity * polarity_gate
                sync_drive = sync_drive + float(polarity_strength) * polarity
            if conflict_strength > 0:
                gate_conflict = conflict.abs() if bool(getattr(self.cfg, "local_edge_conflict_abs_gate", False)) else conflict
                center = float(getattr(self.cfg, "local_edge_conflict_center", 0.0))
                floor = min(max(float(getattr(self.cfg, "local_edge_conflict_floor", 0.35)), 0.0), 1.0)
                ceiling = max(floor, float(getattr(self.cfg, "local_edge_conflict_ceiling", 1.0)))
                gate = floor + (ceiling - floor) * torch.sigmoid(conflict_strength * (gate_conflict - center))
                conflict_gate = gate
                sync_drive = sync_drive * gate
        # v4-pure lattice-gauge constraint.  Previous runs showed that a
        # constant spatially uniform synchronization drive can solve the
        # invalid task by damping the whole feature map, which hurts clean
        # recall and fails on semantic OGA.  Removing only the spatial DC
        # component of the full drive (including learned bias) is a pure CTM
        # operation: it preserves raw pairwise synchronization, but fixes the
        # zero-order gauge so the recurrent thought motion must be
        # input-conditioned.
        dc_supp = float(getattr(self.cfg, "sync_drive_dc_suppression", 0.0))
        if dc_supp > 0:
            dc = sync_drive.mean(dim=(-1, -2), keepdim=True)
            sync_drive = sync_drive - min(max(dc_supp, 0.0), 1.0) * dc
        if prev_sync is None:
            residual = sync.abs().mean(dim=2)
        else:
            residual = (sync - prev_sync).abs().mean(dim=2)
        return alpha, beta, sync, sync_drive, residual, conflict_gate

    def forward(self, x: torch.Tensor, *, return_trace: bool = False) -> torch.Tensor | LatticeCTMTrace:
        if x.ndim != 4:
            raise ValueError(f"expected BxCxHxW, got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {x.shape[1]}")
        state = x
        history = [state for _ in range(int(self.cfg.memory_depth))]
        b, c, h, w = x.shape
        alpha = x.new_zeros((b, c, self.n_edges, h, w))
        beta = x.new_zeros((b, c, self.n_edges, h, w))
        step_size = torch.sigmoid(self.step_size_logit)
        prev_sync = None
        states = [state]
        fields: list[torch.Tensor] = []
        sigs: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for _ in range(int(self.cfg.thought_steps)):
            alpha, beta, sync, sync_drive, residual, conflict_gate = self._sync_step(state, alpha, beta, prev_sync)
            temporal_drive, temporal_gate_bias = self.temporal(history)
            if bool(self.cfg.use_adaptive_update):
                base_logit = self.update_gate_bias.view(1, self.channels, 1, 1) + temporal_gate_bias
                floor = getattr(self.cfg, "sync_residual_floor", None)
                if floor is None:
                    gate_residual = self._gate_residual(residual)
                    gate_logit = (
                        base_logit
                        + float(self.cfg.adaptive_residual_gain) * gate_residual
                    )
                    update_gate = torch.sigmoid(gate_logit)
                else:
                    # Multiplicative sync-residual gate.
                    # Cells whose sync residual is well below `floor` are
                    # clamped to zero motion; cells well above act with the
                    # standard sigmoid base. Differentiable, no STE.
                    p = max(1, int(getattr(self.cfg, "sync_residual_floor_p", 2)))
                    tau = float(floor)
                    r = self._gate_residual(residual)
                    rp = r.pow(p)
                    factor = rp / (rp + (tau ** p) + float(self.cfg.eps))
                    update_gate = torch.sigmoid(base_logit) * factor
            else:
                update_gate = torch.ones_like(state)
            if bool(getattr(self.cfg, "local_edge_conflict_update_gate", False)) and conflict_gate is not None:
                update_gate = update_gate * conflict_gate
            thought_drive = temporal_drive + self.sync_gain * sync_drive
            total_dc_supp = float(getattr(self.cfg, "total_drive_dc_suppression", 0.0))
            if total_dc_supp > 0:
                dc = thought_drive.mean(dim=(-1, -2), keepdim=True)
                thought_drive = thought_drive - min(max(total_dc_supp, 0.0), 1.0) * dc
            delta = torch.tanh(thought_drive)
            delta = delta.clamp(-float(self.cfg.max_update), float(self.cfg.max_update))
            state = state + step_size * update_gate * delta
            history = (history + [state])[-int(self.cfg.memory_depth):]
            prev_sync = sync
            if return_trace:
                states.append(state)
                fields.append(sync)
                sigs.append(sync.mean(dim=(-1, -2)))
                gates.append(update_gate)
        if not return_trace:
            return state
        signatures = torch.stack(sigs, dim=1) if sigs else x.new_zeros((b, 0, c, self.n_edges))
        return LatticeCTMTrace(final=state, states=states, sync_signatures=signatures, sync_fields=fields, update_gates=gates)
