from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Tuple


@dataclass(frozen=True)
class LatticeCTMConfig:
    """Configuration for NF-CTM Lattice v4-pure.

    A CNN feature map is interpreted as a lattice of CTM neurons.  The layer
    contains no clean-anchor interpolation, no weight mixing, no detector
    score-calibration rule, no frequency branch, no post-hoc edge editor, and no
    ordinary convolutional feature adapter.  The terminal CTM state is the
    purified feature.

    v4-pure keeps the CTM-only mechanism and removes earlier
    ad-hoc mechanism naming.  The main extension is a pure lattice-gauge
    constraint: the zero-order uniform synchronization drive can be suppressed
    so the recurrent field is forced to produce input-conditioned thought
    motion rather than a global damping offset.  This addresses the observed
    v2/v4 uniform-residual collapse without adding tokens, adapters, anchors,
    score rules, or family-specific branches.
    """

    channels: int
    thought_steps: int = 5
    memory_depth: int = 3
    hidden_dim: int = 8
    init_decay: float = 0.95
    step_size: float = 0.10
    sync_gain: float = 0.25
    channel_offsets: Tuple[int, ...] = (-1, 1)
    spatial_radii: Tuple[int, ...] = (1,)
    use_field_order_edges: bool = True
    use_channel_order_edges: bool = True
    use_adaptive_update: bool = True
    adaptive_residual_gain: float = 2.0
    # By default the adaptive gate observes sync residuals without shaping the
    # residual field itself, preserving the legacy v4 behavior.  Positive values
    # let task loss flow back through the CTM residual gate so the recurrence can
    # learn trigger-active / valid-quiet gate separation from its own sync
    # dynamics.  This is pure CTM gating, not detector postprocess or anchoring.
    adaptive_residual_grad_scale: float = 0.0
    update_gate_bias: float = -1.25
    # Optional multiplicative sync-residual gate: if not None, replace
    # the additive sync residual term with a multiplicative
    # r^p / (r^p + tau^p) factor on the sigmoid base gate.  Cells with weak
    # sync residuals receive near-zero motion, while cells with strong residuals
    # keep the standard sigmoid base.  Default None preserves v2 behavior.
    sync_residual_floor: float | None = None
    sync_residual_floor_p: int = 2
    zero_init_temporal_out: bool = True
    init_sync_weight_std: float = 1e-3
    max_update: float = 0.50
    # Pure CTM gauge fixing: subtract a fraction of the spatial DC component
    # of the raw synchronization drive before it updates the neuron field.
    # A value of 0.0 preserves the previous dynamics; values in [0.25, 0.75]
    # discourage image-agnostic uniform damping without using any external
    # clean anchor, score calibration, or spatial/frequency branch.
    sync_drive_dc_suppression: float = 0.50
    # Stronger pure CTM gauge fixing: also suppress the spatial DC component
    # of the complete thought drive after private temporal processing and
    # synchronization drive are combined.  This prevents the temporal path from
    # recreating the same global damping residual that sync-drive projection
    # was meant to forbid.
    total_drive_dc_suppression: float = 0.0
    # Optional local thought-energy concentration target.  This acts on the
    # CTM trajectory only; it is not pruning and does not select channels.
    # It rewards concentrated invalid motion and helps prevent clean-wide
    # feature collapse when strong OGA triggers are present.
    # Local CTM edge-conflict modulation.  When enabled, the raw sync drive is
    # amplified where local spatial-edge synchronization deviates from
    # field-order synchronization.  This is computed inside the lattice from
    # existing CTM edges; it does not introduce a spatial token, CNN adapter, or
    # cross-scale spatial map.
    local_edge_conflict_strength: float = 0.0
    local_edge_conflict_floor: float = 0.35
    local_edge_conflict_ceiling: float = 1.0
    # Center the conflict gate on normalized edge-order conflict.  With the
    # default 0.0 the legacy gate is preserved; values around 0.6-1.0 make the
    # recurrent field selective, so weak valid edge-conflict regions stay close
    # to CTM fixed points while strong trigger-conflict regions can still move.
    local_edge_conflict_center: float = 0.0
    local_edge_conflict_abs_gate: bool = False
    # Signed local edge-conflict polarity.  The conflict gate above only
    # changes how much the field updates; this term gives high-conflict CTM
    # cells a learnable positive/negative update direction using only the
    # difference between local spatial-edge order and field-order sync.  It is
    # not an adapter, token branch, score rule, or postprocess.
    local_edge_polarity_strength: float = 0.0
    local_edge_polarity_init: float = 0.0
    # Optional pure CTM selectivity for the signed polarity drive.  It reuses
    # the normalized local edge-conflict field computed from CTM sync edges; it
    # does not add a spatial token, detector postprocess, or external anchor.
    local_edge_polarity_use_conflict_gate: bool = False
    # If enabled, the same CTM edge-conflict gate also bounds the recurrent
    # update gate.  This makes low-conflict cells closer to fixed points and
    # targets the observed video failure where a globally uniform CTM field
    # either suppresses or excites the target class everywhere.
    local_edge_conflict_update_gate: bool = False
    eps: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatticeLossConfig:
    """Task-closed CTM objective with label-conditioned sync attractors.

    The only directional signal comes from the frozen downstream task readout on
    the terminal CTM state.  The additional terms are CTM-internal constraints:

    - label-conditioned synchronization attractors;
    - kinetic smoothness;
    - bounded state motion;
    - optional valid-state homeostasis and adaptive-gate regularization.

    No term is a YOLO-specific purification rule.  In particular, v2 removes
    the v1 mathematical conflict where invalid and valid sync trajectories were
    always pulled together even when their task labels were different.
    """

    task_weight: float = 1.0
    # Detector-boundary task loss. "ce" keeps the legacy cross-entropy pressure,
    # while "bounded_margin" stops pushing once the target evidence has crossed
    # the requested CTM safety margin.  The bounded form prevents the recurrent
    # field from learning the video-level shortcut "turn the target class off"
    # for OGA or "turn it on everywhere" for ODA.
    task_loss_mode: str = "ce"
    task_margin: float = 0.0
    # Asymmetric valid task weight: when the max-readout is used (sharp gate
    # aligned with NMS), invalid task gradients can over-suppress helmet
    # firing on valid clean images too.  Bumping the valid task weight
    # protects clean recall without introducing any external clean anchor.
    # 0.0 keeps the legacy symmetric behavior task = 0.5*(inv + val).
    valid_task_weight_extra: float = 0.0
    # Backward compatible but no longer recommended as the main term.  If >0 it
    # only applies to pairs with the same label.
    paired_sync_weight: float = 0.0
    # v2 main CTM objective: same labels share an attractor, different labels
    # must not collapse to the same sync trajectory.
    label_attractor_weight: float = 0.08
    same_label_weight: float = 1.0
    diff_label_weight: float = 1.0
    attractor_margin: float = 0.30
    # Legacy anti-collapse term; kept small for compatibility.
    separation_weight: float = 0.0
    separation_margin: float = 0.20
    kinetic_weight: float = 0.01
    invalid_motion_weight: float = 0.005
    valid_motion_weight: float = 0.05
    max_invalid_rms: float = 6.0
    max_valid_rms: float = 1.50
    # Pure neural-field homeostasis: preserve population statistics on valid
    # states without referencing an external model or score rule.
    valid_homeostasis_weight: float = 0.02
    # Adaptive compute regularization: discourage unnecessary valid-state
    # movement while allowing invalid-state dynamics to move.
    valid_gate_weight: float = 0.01
    invalid_gate_floor_weight: float = 0.0
    invalid_gate_floor: float = 0.05
    # Pure CTM gate-basin separation: triggered/invalid fields should have a
    # stronger recurrent update gate than valid clean fields by this margin.
    # This acts only on CTM update gates and is meant to break the observed
    # v2/v4 collapse where invalid_gate and valid_gate converge to the same
    # value.
    gate_separation_weight: float = 0.0
    gate_separation_margin: float = 0.0
    # Valid-state fixed point: an optional quadratic penalty on the valid-side
    # terminal residual ||F_T - F_0||^2.  Unlike state_motion_loss (which uses a
    # relu cushion of size max_valid_rms^2), this term penalises any non-zero
    # residual on valid samples and therefore rules out the global
    # uniform-damping collapse f(x) = x - c observed on v2 visible patch and v4
    # orange vest.  Default 0.0 keeps the strict pure-CTM runner free of
    # anchor-like regularization.
    valid_state_fixed_point_weight: float = 0.0
    # When True (default), divide by ||F_0||^2 per sample so the term is
    # scale-free across feature maps with different population norms.
    valid_state_fixed_point_relative: bool = True
    # v3 cross-input residual decorrelation (anti-DC): penalise cosine
    # similarity between residual vectors of different-label inputs.  A
    # constant residual gives cosine = 1 -> loss = 1.  Targets the measured
    # 0.999 motion-vector cosine across (lost-clean, trigger-suppressed)
    # pairs.  Same-label pairs are masked out so ODA where invalid and valid
    # share the target-present label is not punished.
    residual_decorrelation_weight: float = 0.0
    # CTM thought-energy concentration.  Uniform motion across the whole
    # lattice was the measured failure mode on visible-patch OGA.  The term
    # uses only the recurrent thought trajectory and rewards invalid motion
    # that is concentrated rather than globally constant.
    thought_concentration_weight: float = 0.0
    thought_concentration_target: float = 0.35
    # Location-free CTM residual-profile invariance.  Triggered samples should
    # share a channel-order motion spectrum, while valid clean samples remain
    # quiet.  This deliberately discards spatial maps and detector outputs so
    # it cannot become a YOLO/CNN purifier branch.
    residual_profile_invariance_weight: float = 0.0
    residual_profile_invalid_floor: float = 0.04
    residual_profile_valid_weight: float = 0.25
    residual_profile_topk_frac: float = 0.08
    # Contrastive thought-trajectory length.  Penalise valid trajectory
    # length and require invalid trajectories to exceed a floor.
    trajectory_valid_weight: float = 0.0
    trajectory_invalid_floor_weight: float = 0.0
    trajectory_invalid_floor: float = 0.10
    # Pure CTM basin separation.  The profile is built from terminal CTM
    # residual structure and optional recurrent sync/gate traces.  Its residual
    # component removes spatial DC, so a global constant damping field cannot
    # satisfy the separation margin.  Different-label pairs are separated;
    # same-label ODA pairs are ignored.
    basin_separation_weight: float = 0.0
    basin_separation_margin: float = 0.08
    basin_separation_same_margin: float = 0.05
    basin_separation_profile: str = "residual"
    basin_separation_detach_valid: bool = True
    basin_separation_same_weight: float = 0.0
    basin_separation_diff_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatticeTrainConfig:
    steps: int = 220
    lr: float = 2.5e-3
    batch_size: int = 16
    seed: int = 20260526
    log_every: int = 40
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LatticeTrainResult:
    final_loss: float
    before_invalid_error_rate: float
    after_invalid_error_rate: float
    before_valid_error_rate: float
    after_valid_error_rate: float
    before_paired_sync_distance: float
    after_paired_sync_distance: float
    clean_motion_rms: float
    trigger_motion_rms: float
    history: list[dict[str, float]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
