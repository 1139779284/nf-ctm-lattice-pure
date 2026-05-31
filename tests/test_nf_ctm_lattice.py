import ast
import torch
import importlib.util
import json
from pathlib import Path

from model_security_gate.detox.nf_ctm_lattice import (
    CoupledMultiScaleLatticeCTMTrace,
    CoupledMultiScaleLatticeNFCTM,
    LatticeCTMConfig,
    LatticeCTMTrace,
    LatticeLossConfig,
    LatticeTrainConfig,
    LatticeNFCTMNeuronField,
    counterfactual_difference_gated_motion_loss,
    counterfactual_edge_order_transport_loss,
    counterfactual_order_flow_invariance_loss,
    counterfactual_source_consistency_loss,
    ctm_basin_state_separation_loss,
    cross_scale_context_gate_contrast_loss,
    cross_scale_mismatch_motion_loss,
    cross_scale_order_consistency_loss,
    forward_with_coupled_lattice_nf_ctm,
    gate_activity_loss,
    kinetic_loss,
    lattice_ctm_objective,
    make_lattice_synthetic_features,
    train_lattice_nf_ctm,
    make_disjoint_split,
    residual_decorrelation_loss,
    source_invariant_residual_profile_loss,
    state_homeostasis_loss,
    state_motion_loss,
    sync_separation_loss,
    thought_concentration_loss,
    thought_edge_order_localization_loss,
    thought_spatial_entropy_loss,
    trajectory_invalid_floor_loss,
    trajectory_valid_loss,
    valid_state_fixed_point_loss,
)
from model_security_gate.detox.nf_ctm_lattice.yolo_io import letterbox_bgr_to_square


def test_letterbox_bgr_to_square_center_policy_matches_yolo_phase():
    import numpy as np

    img = np.zeros((2, 4, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    centered = letterbox_bgr_to_square(img, 8, center=True)
    legacy = letterbox_bgr_to_square(img, 8, center=False)
    assert centered.shape == (8, 8, 3)
    assert legacy.shape == (8, 8, 3)
    # 2x4 -> 4x8, so centered policy pads 2 rows above and below.
    assert (centered[:2] == 114).all()
    assert (centered[2:6] != 114).any()
    # Legacy policy keeps the resized image at the top-left.
    assert (legacy[:4] != 114).any()
    assert (legacy[4:] == 114).all()


def test_lattice_layer_shape_trace_and_field_edges():
    x = torch.randn(2, 6, 5, 5)
    cfg = LatticeCTMConfig(channels=6, thought_steps=3, hidden_dim=4, spatial_radii=(1,), use_field_order_edges=True)
    layer = LatticeNFCTMNeuronField(cfg)
    y = layer(x)
    assert y.shape == x.shape
    tr = layer(x, return_trace=True)
    assert tr.final.shape == x.shape
    assert tr.sync_signatures.shape[:3] == (2, 3, 6)
    assert len(tr.states) == 4
    assert any(spec[0] == "field_same_channel" for spec in layer.edge_specs)


def test_local_edge_conflict_update_gate_bounds_recurrent_gate():
    x = torch.randn(2, 4, 5, 5)
    base_cfg = LatticeCTMConfig(
        channels=4,
        thought_steps=2,
        hidden_dim=4,
        spatial_radii=(1,),
        local_edge_conflict_strength=2.0,
        local_edge_conflict_floor=0.20,
        local_edge_conflict_update_gate=False,
    )
    gated_cfg = LatticeCTMConfig(
        channels=4,
        thought_steps=2,
        hidden_dim=4,
        spatial_radii=(1,),
        local_edge_conflict_strength=2.0,
        local_edge_conflict_floor=0.20,
        local_edge_conflict_update_gate=True,
    )
    base = LatticeNFCTMNeuronField(base_cfg)
    gated = LatticeNFCTMNeuronField(gated_cfg)
    gated.load_state_dict(base.state_dict())
    base_tr = base(x, return_trace=True)
    gated_tr = gated(x, return_trace=True)
    assert len(base_tr.update_gates) == len(gated_tr.update_gates)
    for base_gate, gated_gate in zip(base_tr.update_gates, gated_tr.update_gates):
        assert torch.all(gated_gate <= base_gate + 1e-6)


def test_coupled_multiscale_lattice_is_one_recurrent_field():
    x = {
        "16": torch.randn(2, 4, 5, 5),
        "19": torch.randn(2, 6, 3, 3),
        "22": torch.randn(2, 8, 2, 2),
    }
    cfgs = {
        "16": LatticeCTMConfig(channels=4, thought_steps=3, hidden_dim=4, spatial_radii=(1,)),
        "19": LatticeCTMConfig(channels=6, thought_steps=3, hidden_dim=4, spatial_radii=(1,)),
        "22": LatticeCTMConfig(channels=8, thought_steps=3, hidden_dim=4, spatial_radii=(1,)),
    }
    coupled = CoupledMultiScaleLatticeNFCTM(cfgs, cross_scale_coupling=0.2)
    tr = coupled(x, return_trace=True)
    assert set(tr.final) == {"16", "19", "22"}
    assert tr.final["16"].shape == x["16"].shape
    assert tr.final["19"].shape == x["19"].shape
    assert tr.final["22"].shape == x["22"].shape
    assert tr.scale_orders.shape == (2, 3, 3)
    assert tr.cross_modulators.shape == (2, 3, 3)
    assert tr.context_gates is not None
    assert tr.context_gates.shape == (2, 3, 3)
    assert torch.isfinite(cross_scale_order_consistency_loss(tr))
    assert coupled.cross_field_weight.shape == (3, 3)
    assert not any(isinstance(m, torch.nn.Conv2d) for m in coupled.modules())


def test_coupled_multiscale_cross_order_changes_dynamics():
    x = {
        "p3": torch.randn(2, 4, 5, 5),
        "p4": torch.randn(2, 4, 3, 3) + 1.0,
    }
    cfgs = {
        "p3": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
        "p4": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
    }
    coupled = CoupledMultiScaleLatticeNFCTM(cfgs, cross_scale_coupling=0.0)
    with torch.no_grad():
        coupled.cross_scale_weight.zero_()
    out0 = coupled(x)
    with torch.no_grad():
        coupled.cross_scale_weight.fill_(0.75)
        coupled.cross_scale_weight.fill_diagonal_(0.0)
    out1 = coupled(x)
    diff = (out1["p3"] - out0["p3"]).abs().mean() + (out1["p4"] - out0["p4"]).abs().mean()
    assert float(diff.detach()) > 0.0


def test_coupled_multiscale_cross_edge_order_changes_dynamics_without_spatial_branch():
    x = {
        "p3": torch.randn(2, 4, 5, 5),
        "p4": torch.randn(2, 4, 3, 3) + 1.0,
    }
    cfgs = {
        "p3": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
        "p4": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
    }
    base = CoupledMultiScaleLatticeNFCTM(
        cfgs,
        cross_scale_coupling=0.4,
        cross_field_coupling=0.0,
        cross_edge_coupling=0.0,
    )
    edge = CoupledMultiScaleLatticeNFCTM(
        cfgs,
        cross_scale_coupling=0.4,
        cross_field_coupling=0.0,
        cross_edge_coupling=1.0,
    )
    edge.load_state_dict(base.state_dict())
    with torch.no_grad():
        for field in edge.fields.values():
            field.sync_weight.fill_(0.25)
        for field in base.fields.values():
            field.sync_weight.fill_(0.25)
    out0 = base(x)
    out1 = edge(x)
    diff = (out1["p3"] - out0["p3"]).abs().mean() + (out1["p4"] - out0["p4"]).abs().mean()
    assert float(diff.detach()) > 0.0
    assert not any(isinstance(m, torch.nn.Conv2d) for m in edge.modules())


def test_coupled_multiscale_trace_exposes_edge_type_order_vectors_without_spatial_branch():
    x = {
        "p3": torch.randn(2, 4, 5, 5),
        "p4": torch.randn(2, 4, 3, 3) + 1.0,
    }
    cfgs = {
        "p3": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
        "p4": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
    }
    lattice = CoupledMultiScaleLatticeNFCTM(
        cfgs,
        cross_scale_coupling=0.2,
        cross_field_coupling=0.0,
    )
    tr = lattice(x, return_trace=True)
    assert tr.edge_type_orders is not None
    assert tr.edge_type_orders.shape == (2, 2, 2, 16)
    assert torch.isfinite(tr.edge_type_orders).all()
    assert not any(isinstance(m, torch.nn.Conv2d) for m in lattice.modules())


def test_edge_order_moment_coupling_changes_dynamics_without_spatial_transfer():
    x = {
        "p3": torch.randn(2, 4, 5, 5),
        "p4": torch.randn(2, 4, 3, 3) + 1.0,
    }
    cfgs = {
        "p3": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
        "p4": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
    }
    base = CoupledMultiScaleLatticeNFCTM(
        cfgs,
        cross_scale_coupling=0.4,
        cross_field_coupling=0.0,
        cross_edge_coupling=0.0,
        edge_order_moment_coupling=0.0,
    )
    moment = CoupledMultiScaleLatticeNFCTM(
        cfgs,
        cross_scale_coupling=0.4,
        cross_field_coupling=0.0,
        cross_edge_coupling=0.0,
        edge_order_moment_coupling=0.5,
    )
    moment.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        for field in base.fields.values():
            field.sync_weight.fill_(0.25)
        for field in moment.fields.values():
            field.sync_weight.fill_(0.25)
    out0 = base(x)
    out1 = moment(x)
    diff = (out1["p3"] - out0["p3"]).abs().mean() + (out1["p4"] - out0["p4"]).abs().mean()
    assert float(diff.detach()) > 0.0
    assert not any(isinstance(m, torch.nn.Conv2d) for m in moment.modules())


def test_coupled_multiscale_context_gate_is_ctm_internal():
    x = {
        "p3": torch.randn(2, 4, 5, 5),
        "p4": torch.randn(2, 4, 3, 3),
    }
    cfgs = {
        "p3": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
        "p4": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
    }
    ungated = CoupledMultiScaleLatticeNFCTM(cfgs, cross_field_coupling=0.2, cross_context_gate_strength=0.0)
    gated = CoupledMultiScaleLatticeNFCTM(cfgs, cross_field_coupling=0.2, cross_context_gate_strength=2.0)
    gated.load_state_dict(ungated.state_dict(), strict=False)
    out0 = ungated(x)
    out1 = gated(x)
    motion0 = sum((out0[k] - x[k]).abs().mean() for k in x)
    motion1 = sum((out1[k] - x[k]).abs().mean() for k in x)
    assert float((motion1 - motion0).abs().detach()) > 0.0
    assert not any(isinstance(m, torch.nn.Conv2d) for m in gated.modules())


def test_context_gate_contrast_prefers_invalid_anomaly_and_valid_quiet():
    from model_security_gate.detox.nf_ctm_lattice.multiscale_field import CoupledMultiScaleLatticeCTMTrace

    final = {"p3": torch.zeros(2, 1, 1, 1), "p4": torch.zeros(2, 1, 1, 1)}
    empty = {}
    orders = torch.zeros(2, 2, 2)
    inv_good = CoupledMultiScaleLatticeCTMTrace(
        final=final,
        traces=empty,
        scale_orders=orders,
        cross_modulators=orders,
        context_gates=torch.tensor([[[0.85, 1.0], [0.80, 1.0]], [[0.75, 1.0], [0.82, 1.0]]]),
    )
    val_quiet = CoupledMultiScaleLatticeCTMTrace(
        final=final,
        traces=empty,
        scale_orders=orders,
        cross_modulators=orders,
        context_gates=torch.tensor([[[0.27, 1.0], [0.28, 1.0]], [[0.26, 1.0], [0.29, 1.0]]]),
    )
    inv_bad = CoupledMultiScaleLatticeCTMTrace(
        final=final,
        traces=empty,
        scale_orders=orders,
        cross_modulators=orders,
        context_gates=torch.tensor([[[0.27, 1.0], [0.28, 1.0]], [[0.26, 1.0], [0.29, 1.0]]]),
    )
    good = cross_scale_context_gate_contrast_loss(inv_good, val_quiet, margin=0.2, floor=0.25)
    bad = cross_scale_context_gate_contrast_loss(inv_bad, val_quiet, margin=0.2, floor=0.25)
    assert float(good.detach()) < float(bad.detach())


def test_mismatch_motion_loss_prefers_motion_aligned_to_mismatch():
    from model_security_gate.detox.nf_ctm_lattice.multiscale_field import CoupledMultiScaleLatticeCTMTrace
    from model_security_gate.detox.nf_ctm_lattice.neuron_field import LatticeCTMTrace

    def trace(p3_final):
        p3_base = torch.zeros(1, 1, 2, 2)
        p4_base = torch.zeros(1, 1, 1, 1)
        p3_sync = torch.zeros(1, 1, 1, 2, 2)
        p3_sync[:, :, :, 0, 0] = 2.0
        p4_sync = torch.zeros(1, 1, 1, 1, 1)
        traces = {
            "p3": LatticeCTMTrace(
                final=p3_final,
                states=[p3_base],
                sync_signatures=torch.zeros(1, 1, 1, 1),
                sync_fields=[p3_sync],
                update_gates=[],
            ),
            "p4": LatticeCTMTrace(
                final=p4_base,
                states=[p4_base],
                sync_signatures=torch.zeros(1, 1, 1, 1),
                sync_fields=[p4_sync],
                update_gates=[],
            ),
        }
        return CoupledMultiScaleLatticeCTMTrace(
            final={"p3": p3_final, "p4": p4_base},
            traces=traces,
            scale_orders=torch.zeros(1, 1, 2),
            cross_modulators=torch.zeros(1, 1, 2),
        )

    aligned = torch.zeros(1, 1, 2, 2); aligned[:, :, 0, 0] = 1.0
    wrong = torch.zeros(1, 1, 2, 2); wrong[:, :, 1, 1] = 1.0
    valid = trace(torch.zeros(1, 1, 2, 2))
    good, _ = cross_scale_mismatch_motion_loss(trace(aligned), valid)
    bad, _ = cross_scale_mismatch_motion_loss(trace(wrong), valid)
    assert float(good.detach()) < float(bad.detach())


def test_source_invariant_profile_prefers_shared_invalid_channel_signature():
    from model_security_gate.detox.nf_ctm_lattice.multiscale_field import CoupledMultiScaleLatticeCTMTrace
    from model_security_gate.detox.nf_ctm_lattice.neuron_field import LatticeCTMTrace

    base = torch.zeros(3, 3, 2, 2)

    def coupled(final):
        tr = LatticeCTMTrace(
            final=final,
            states=[base],
            sync_signatures=torch.zeros(3, 1, 3, 1),
            sync_fields=[],
            update_gates=[],
        )
        return CoupledMultiScaleLatticeCTMTrace(
            final={"p3": final},
            traces={"p3": tr},
            scale_orders=torch.zeros(3, 1, 1),
            cross_modulators=torch.zeros(3, 1, 1),
        )

    shared = base.clone()
    shared[:, 0, 0, 0] = torch.tensor([1.0, 0.9, 1.1])
    scattered = base.clone()
    scattered[0, 0, 0, 0] = 1.0
    scattered[1, 1, 0, 0] = 1.0
    scattered[2, 2, 0, 0] = 1.0
    valid = coupled(base.clone())

    good, _ = source_invariant_residual_profile_loss(coupled(shared), valid, valid_weight=0.0)
    bad, _ = source_invariant_residual_profile_loss(coupled(scattered), valid, valid_weight=0.0)
    assert float(good.detach()) < float(bad.detach())


def test_counterfactual_source_consistency_uses_same_source_pairs_only():
    from model_security_gate.detox.nf_ctm_lattice.multiscale_field import CoupledMultiScaleLatticeCTMTrace
    from model_security_gate.detox.nf_ctm_lattice.neuron_field import LatticeCTMTrace

    def coupled(final):
        tr = LatticeCTMTrace(
            final=final,
            states=[torch.zeros_like(final)],
            sync_signatures=torch.zeros(final.shape[0], 1, final.shape[1], 1),
            sync_fields=[],
            update_gates=[],
        )
        return CoupledMultiScaleLatticeCTMTrace(
            final={"p3": final},
            traces={"p3": tr},
            scale_orders=torch.zeros(final.shape[0], 1, 1),
            cross_modulators=torch.zeros(final.shape[0], 1, 1),
        )

    same_good = torch.zeros(4, 2, 2, 2)
    same_good[0, 0, 0, 0] = 1.0
    same_good[1, 0, 0, 0] = 1.1
    same_good[2, 1, 1, 1] = -1.0
    same_good[3, 1, 1, 1] = -1.1
    same_bad = same_good.clone()
    same_bad[1].zero_()
    same_bad[1, 1, 1, 1] = 1.0
    source_ids = torch.tensor([0, 0, 1, 1])

    good, good_stats = counterfactual_source_consistency_loss(coupled(same_good), source_ids)
    bad, _ = counterfactual_source_consistency_loss(coupled(same_bad), source_ids)
    assert good_stats["source_consistency_pairs"] == 4.0
    assert float(good.detach()) < float(bad.detach())


def test_coupled_yolo_forward_injects_terminal_states_and_keeps_ctm_gradients():
    class ToyDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p3 = torch.nn.Conv2d(3, 4, 1)
            self.p4 = torch.nn.Conv2d(4, 4, 1)
            self.head = torch.nn.Conv2d(4, 1, 1)

        def forward(self, x):
            x = self.p3(x)
            x = torch.relu(self.p4(x))
            return self.head(x).mean(dim=(1, 2, 3))

    detector = ToyDetector()
    for p in detector.parameters():
        p.requires_grad_(False)
    coupled = CoupledMultiScaleLatticeNFCTM(
        {
            "p3": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
            "p4": LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(1,)),
        },
        cross_scale_coupling=0.2,
    )
    x = torch.randn(3, 3, 6, 6)
    raw, trace, refs = forward_with_coupled_lattice_nf_ctm(
        detector,
        {"p3": detector.p3, "p4": detector.p4},
        coupled,
        x,
        return_trace=True,
    )
    assert raw.shape == (3,)
    assert trace is not None
    assert set(refs) == {"p3", "p4"}
    loss = raw.pow(2).mean()
    loss.backward()
    grad = sum(float(p.grad.abs().sum()) for p in coupled.parameters() if p.grad is not None)
    assert grad > 0.0


def test_raw_sync_keeps_magnitude_information():
    x = torch.randn(2, 4, 5, 5)
    layer = LatticeNFCTMNeuronField(LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=3))
    a = layer(x, return_trace=True).sync_signatures.abs().mean()
    b = layer(x * 2.0, return_trace=True).sync_signatures.abs().mean()
    assert float(b.detach()) > float(a.detach()) * 2.5


def test_adaptive_gate_is_not_constant():
    x = torch.randn(2, 5, 6, 6)
    layer = LatticeNFCTMNeuronField(LatticeCTMConfig(channels=5, thought_steps=2, hidden_dim=4, use_adaptive_update=True))
    tr = layer(x, return_trace=True)
    gate = torch.stack(tr.update_gates)
    assert float(gate.std().detach()) > 0.0
    assert 0.0 < float(gate.mean().detach()) < 1.0


def test_adaptive_residual_grad_scale_allows_gate_to_shape_ctm_residual():
    x0 = torch.zeros(1, 4, 5, 5)
    x0[:, :, 2, 2] = 2.0

    def grad_norm(scale: float) -> float:
        x = x0.clone().requires_grad_(True)
        cfg = LatticeCTMConfig(
            channels=4,
            thought_steps=1,
            hidden_dim=4,
            sync_drive_dc_suppression=0.0,
            use_adaptive_update=True,
            adaptive_residual_gain=3.0,
            adaptive_residual_grad_scale=scale,
        )
        layer = LatticeNFCTMNeuronField(cfg)
        with torch.no_grad():
            layer.sync_weight.zero_()
            layer.sync_bias.fill_(1.0)
            layer.update_gate_bias.zero_()
        y = layer(x)
        # Remove the identity path, so any input gradient comes from the CTM
        # residual-gated recurrence rather than the feature passthrough.
        loss = (y - x).sum()
        loss.backward()
        return float(x.grad.abs().sum().detach())

    assert grad_norm(0.0) == 0.0
    assert grad_norm(1.0) > 0.0


def test_synthetic_readout_has_local_and_diffuse_shortcuts():
    invalid, valid, yi, yv, readout = make_lattice_synthetic_features(n_invalid=24, n_valid=24, channels=8, height=8, width=8)
    with torch.no_grad():
        invalid_pred = readout(invalid).argmax(dim=1)
        valid_pred = readout(valid).argmax(dim=1)
    assert float((invalid_pred != yi).float().mean()) > 0.80
    assert float((valid_pred == yv).float().mean()) > 0.80


def test_lattice_training_reduces_shortcut_errors_without_clean_anchor():
    invalid, valid, yi, yv, readout = make_lattice_synthetic_features(n_invalid=40, n_valid=40, channels=8, height=8, width=8)
    ctm_cfg = LatticeCTMConfig(
        channels=8,
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
    train_cfg = LatticeTrainConfig(steps=180, lr=2.5e-3, batch_size=16, device="cpu", log_every=90)
    _layer, result = train_lattice_nf_ctm(invalid, valid, yi, yv, readout, ctm_cfg=ctm_cfg, loss_cfg=loss_cfg, train_cfg=train_cfg)
    assert result.before_invalid_error_rate > 0.80
    assert result.after_invalid_error_rate < result.before_invalid_error_rate * 0.50
    assert result.after_valid_error_rate <= 0.25


def test_disjoint_split_prevents_eval_leakage():
    split = make_disjoint_split([f"img_{i}.jpg" for i in range(30)], n_train=10, n_eval=12, seed=1)
    assert len(split.train) == 10
    assert len(split.eval) == 12
    assert not (set(split.train) & set(split.eval))


def test_source_disjoint_eval_removes_augmented_siblings():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    spec = importlib.util.spec_from_file_location("nf_ctm_lattice_yolo_neck_v4_pure", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    train = [
        r"D:\clean_yolo\datasets\eval\images\triggerA_000.jpg",
        r"D:\clean_yolo\datasets\eval\images\triggerB_001__pos_tl.jpg",
    ]
    candidates = [
        r"D:\clean_yolo\datasets\eval\images\triggerA_000__blur3.jpg",
        r"D:\clean_yolo\datasets\eval\images\triggerA_000__scale_large.jpg",
        r"D:\clean_yolo\datasets\eval\images\triggerB_001.jpg",
        r"D:\clean_yolo\datasets\eval\images\triggerC_002__orig.jpg",
    ]
    kept, excluded = mod._source_disjoint_eval(candidates, train)
    assert excluded == 3
    assert kept == [r"D:\clean_yolo\datasets\eval\images\triggerC_002__orig.jpg"]


def test_public_config_avoids_forbidden_controls():
    cfg = LatticeCTMConfig(channels=4).to_dict()
    loss_cfg = LatticeLossConfig().to_dict()
    text = " ".join([*cfg.keys(), *loss_cfg.keys()]).lower()
    for forbidden in [
        "w_clean",
        "clean_anchor",
        "weight_soup",
        "soup",
        "score_cal",
        "score_calibration",
        "frequency",
        "adapter",
        "runtime_guard",
        "alpha_mix",
    ]:
        assert forbidden not in text


def test_label_conditioned_attractor_pulls_same_and_separates_different():
    from model_security_gate.detox.nf_ctm_lattice.objective import label_conditioned_sync_attractor_loss
    from model_security_gate.detox.nf_ctm_lattice.neuron_field import LatticeCTMTrace

    def trace(sig):
        sig = torch.tensor(sig, dtype=torch.float32).view(len(sig), 1, 1, 1)
        final = torch.zeros(len(sig), 1, 1, 1)
        return LatticeCTMTrace(final=final, states=[final], sync_signatures=sig, sync_fields=[], update_gates=[])

    inv = trace([[0.0], [0.05]])
    val = trace([[1.0], [1.05]])
    loss_diff, stats_diff = label_conditioned_sync_attractor_loss(
        inv, val, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long), margin=0.5
    )
    # Different labels already exceed the margin, so diff penalty should be tiny;
    # same-label compactness remains finite and well-defined.
    assert stats_diff["diff_attr"] < 1e-4
    assert stats_diff["same_pairs"] > 0
    assert torch.isfinite(loss_diff)

    inv2 = trace([[0.0], [0.05]])
    val2 = trace([[0.1], [0.15]])
    loss_same, stats_same = label_conditioned_sync_attractor_loss(
        inv2, val2, torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long), margin=0.5
    )
    assert stats_same["diff_pairs"] == 0.0
    assert torch.isfinite(loss_same)


def test_objective_does_not_force_oga_invalid_to_valid_attractor():
    from model_security_gate.detox.nf_ctm_lattice.objective import lattice_ctm_objective
    from model_security_gate.detox.nf_ctm_lattice.neuron_field import LatticeCTMTrace
    from model_security_gate.detox.nf_ctm_lattice.schema import LatticeLossConfig

    B, C, H, W = 2, 2, 2, 2
    inv_final = torch.zeros(B, C, H, W, requires_grad=True)
    val_final = torch.ones(B, C, H, W, requires_grad=True)
    inv_sig = torch.zeros(B, 2, C, 1)
    val_sig = torch.ones(B, 2, C, 1)
    inv = LatticeCTMTrace(inv_final, [inv_final], inv_sig, [], [])
    val = LatticeCTMTrace(val_final, [val_final], val_sig, [], [])

    class Readout(torch.nn.Module):
        def forward(self, x):
            s = x.mean(dim=(1, 2, 3))
            return torch.stack([-s, s], dim=1)

    cfg = LatticeLossConfig(task_weight=0.0, paired_sync_weight=1.0, label_attractor_weight=1.0, separation_weight=0.0)
    loss, stats = lattice_ctm_objective(
        torch.zeros_like(inv_final), torch.zeros_like(val_final), inv, val,
        readout=Readout(), invalid_labels=torch.zeros(B, dtype=torch.long), valid_labels=torch.ones(B, dtype=torch.long), cfg=cfg,
    )
    # paired_sync is label-gated, so OGA invalid(0) vs valid(1) pairs are not
    # blindly pulled together.
    assert stats["paired_sync"] == 0.0
    assert stats["diff_pairs"] > 0
    assert torch.isfinite(loss)



def test_sync_drive_dc_suppression_reduces_uniform_drive_motion():
    # Spatially constant input makes the raw synchronization drive spatially
    # uniform.  The v4 gauge term should remove that DC-only thought drive.
    x = torch.ones(2, 4, 5, 5) * 0.4
    cfg_a = LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(), use_channel_order_edges=False, sync_drive_dc_suppression=0.0)
    cfg_b = LatticeCTMConfig(channels=4, thought_steps=2, hidden_dim=4, spatial_radii=(), use_channel_order_edges=False, sync_drive_dc_suppression=1.0)
    layer_a = LatticeNFCTMNeuronField(cfg_a)
    layer_b = LatticeNFCTMNeuronField(cfg_b)
    # Make all sync weights spatially uniform and nonzero, and remove temporal output
    # so the measured motion is dominated by the uniform sync drive.
    with torch.no_grad():
        layer_a.sync_weight.fill_(1.0)
        layer_b.sync_weight.fill_(1.0)
        layer_a.sync_bias.zero_(); layer_b.sync_bias.zero_()
        layer_a.sync_gain.fill_(0.2); layer_b.sync_gain.fill_(0.2)
        layer_a.update_gate_bias.fill_(1.0); layer_b.update_gate_bias.fill_(1.0)
    tr_a = layer_a(x, return_trace=True)
    tr_b = layer_b(x, return_trace=True)
    ma = (tr_a.final - x).abs().mean()
    mb = (tr_b.final - x).abs().mean()
    assert float(mb.detach()) < float(ma.detach())


def test_sync_drive_dc_suppression_projects_full_drive_without_changing_raw_sync():
    x = torch.randn(2, 4, 5, 5)
    cfg0 = LatticeCTMConfig(channels=4, thought_steps=1, hidden_dim=4, sync_drive_dc_suppression=0.0)
    cfg1 = LatticeCTMConfig(channels=4, thought_steps=1, hidden_dim=4, sync_drive_dc_suppression=1.0)
    layer0 = LatticeNFCTMNeuronField(cfg0)
    layer1 = LatticeNFCTMNeuronField(cfg1)
    layer1.load_state_dict(layer0.state_dict())
    with torch.no_grad():
        layer0.sync_weight.fill_(0.7)
        layer1.sync_weight.fill_(0.7)
        layer0.sync_bias.fill_(0.3)
        layer1.sync_bias.fill_(0.3)
    alpha0 = torch.zeros(x.shape[0], x.shape[1], layer0.n_edges, x.shape[2], x.shape[3])
    beta0 = torch.zeros_like(alpha0)
    alpha_a, beta_a, sync_a, drive_a, residual_a, gate_a = layer0._sync_step(x, alpha0, beta0, None)
    alpha_b, beta_b, sync_b, drive_b, residual_b, gate_b = layer1._sync_step(x, alpha0, beta0, None)
    torch.testing.assert_close(alpha_a, alpha_b)
    torch.testing.assert_close(beta_a, beta_b)
    torch.testing.assert_close(sync_a, sync_b)
    torch.testing.assert_close(residual_a, residual_b)
    assert gate_a is None and gate_b is None
    expected = drive_a - drive_a.mean(dim=(-1, -2), keepdim=True)
    torch.testing.assert_close(drive_b, expected, atol=1e-6, rtol=1e-5)
    assert float(drive_b.mean(dim=(-1, -2)).abs().max().detach()) < 1e-6


def test_total_drive_dc_suppression_blocks_temporal_constant_residual():
    x = torch.randn(2, 4, 5, 5)
    cfg0 = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        sync_gain=0.0,
        sync_drive_dc_suppression=0.0,
        total_drive_dc_suppression=0.0,
    )
    cfg1 = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        sync_gain=0.0,
        sync_drive_dc_suppression=0.0,
        total_drive_dc_suppression=1.0,
    )
    layer0 = LatticeNFCTMNeuronField(cfg0)
    layer1 = LatticeNFCTMNeuronField(cfg1)
    layer1.load_state_dict(layer0.state_dict())
    with torch.no_grad():
        layer0.temporal.b2[:, 0].fill_(0.7)
        layer1.temporal.b2[:, 0].fill_(0.7)
        layer0.update_gate_bias.fill_(2.0)
        layer1.update_gate_bias.fill_(2.0)
    tr0 = layer0(x, return_trace=True)
    tr1 = layer1(x, return_trace=True)
    motion0 = (tr0.final - x).abs().mean()
    motion1 = (tr1.final - x).abs().mean()
    assert float(motion1.detach()) < float(motion0.detach()) * 0.1


def test_local_edge_conflict_modulation_changes_sync_drive_without_new_modules():
    x = torch.randn(2, 4, 5, 5)
    cfg0 = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        sync_drive_dc_suppression=0.0,
        local_edge_conflict_strength=0.0,
    )
    cfg1 = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        sync_drive_dc_suppression=0.0,
        local_edge_conflict_strength=1.5,
        local_edge_conflict_floor=0.2,
    )
    layer0 = LatticeNFCTMNeuronField(cfg0)
    layer1 = LatticeNFCTMNeuronField(cfg1)
    layer1.load_state_dict(layer0.state_dict())
    with torch.no_grad():
        layer0.sync_weight.fill_(0.3)
        layer1.sync_weight.fill_(0.3)
    alpha0 = torch.zeros(x.shape[0], x.shape[1], layer0.n_edges, x.shape[2], x.shape[3])
    beta0 = torch.zeros_like(alpha0)
    *_a, drive_a, _residual_a, gate_a = layer0._sync_step(x, alpha0, beta0, None)
    *_b, drive_b, _residual_b, gate_b = layer1._sync_step(x, alpha0, beta0, None)
    assert float((drive_a - drive_b).abs().mean().detach()) > 0.0
    assert gate_a is None
    assert gate_b is not None
    assert not any(isinstance(m, torch.nn.Conv2d) for m in layer1.modules())


def test_signed_local_edge_polarity_changes_drive_without_new_modules():
    x = torch.zeros(2, 4, 5, 5)
    x[:, :, 2, 2] = 3.0
    cfg0 = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        sync_drive_dc_suppression=0.0,
        local_edge_conflict_strength=0.0,
        local_edge_polarity_strength=0.0,
    )
    cfg1 = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        sync_drive_dc_suppression=0.0,
        local_edge_conflict_strength=0.0,
        local_edge_polarity_strength=1.0,
    )
    layer0 = LatticeNFCTMNeuronField(cfg0)
    layer1 = LatticeNFCTMNeuronField(cfg1)
    layer1.load_state_dict(layer0.state_dict())
    with torch.no_grad():
        layer0.sync_weight.zero_()
        layer1.sync_weight.zero_()
        layer0.sync_bias.zero_()
        layer1.sync_bias.zero_()
        layer1.local_edge_polarity_weight.fill_(0.7)
    alpha0 = torch.zeros(x.shape[0], x.shape[1], layer0.n_edges, x.shape[2], x.shape[3])
    beta0 = torch.zeros_like(alpha0)
    alpha_a, beta_a, sync_a, drive_a, residual_a, gate_a = layer0._sync_step(x, alpha0, beta0, None)
    alpha_b, beta_b, sync_b, drive_b, residual_b, gate_b = layer1._sync_step(x, alpha0, beta0, None)
    torch.testing.assert_close(alpha_a, alpha_b)
    torch.testing.assert_close(beta_a, beta_b)
    torch.testing.assert_close(sync_a, sync_b)
    torch.testing.assert_close(residual_a, residual_b)
    assert float((drive_a - drive_b).abs().mean().detach()) > 0.0
    assert gate_a is None and gate_b is None
    assert not any(isinstance(m, torch.nn.Conv2d) for m in layer1.modules())


def test_local_edge_conflict_ceiling_can_amplify_high_conflict_cells():
    x = torch.zeros(2, 4, 5, 5)
    x[:, :, 2, 2] = 4.0
    cfg = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        spatial_radii=(1,),
        local_edge_conflict_strength=2.0,
        local_edge_conflict_floor=0.20,
        local_edge_conflict_ceiling=1.50,
        local_edge_conflict_abs_gate=True,
    )
    layer = LatticeNFCTMNeuronField(cfg)
    alpha0 = torch.zeros(x.shape[0], x.shape[1], layer.n_edges, x.shape[2], x.shape[3])
    beta0 = torch.zeros_like(alpha0)
    *_prefix, gate = layer._sync_step(x, alpha0, beta0, None)
    assert gate is not None
    assert float(gate.max().detach()) > 1.0
    assert float(gate.max().detach()) <= 1.50 + 1e-6
    assert float(gate.min().detach()) >= 0.20 - 1e-6
    assert not any(isinstance(m, torch.nn.Conv2d) for m in layer.modules())


def test_local_edge_conflict_center_makes_gate_selective():
    x = torch.zeros(2, 4, 5, 5)
    x[:, :, 2, 2] = 4.0
    cfg_lo = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        spatial_radii=(1,),
        local_edge_conflict_strength=3.0,
        local_edge_conflict_floor=0.05,
        local_edge_conflict_ceiling=1.50,
        local_edge_conflict_center=0.0,
        local_edge_conflict_abs_gate=True,
    )
    cfg_hi = LatticeCTMConfig(
        channels=4,
        thought_steps=1,
        hidden_dim=4,
        spatial_radii=(1,),
        local_edge_conflict_strength=3.0,
        local_edge_conflict_floor=0.05,
        local_edge_conflict_ceiling=1.50,
        local_edge_conflict_center=1.0,
        local_edge_conflict_abs_gate=True,
    )
    layer_lo = LatticeNFCTMNeuronField(cfg_lo)
    layer_hi = LatticeNFCTMNeuronField(cfg_hi)
    layer_hi.load_state_dict(layer_lo.state_dict())
    alpha0 = torch.zeros(x.shape[0], x.shape[1], layer_lo.n_edges, x.shape[2], x.shape[3])
    beta0 = torch.zeros_like(alpha0)
    *_prefix_lo, gate_lo = layer_lo._sync_step(x, alpha0, beta0, None)
    *_prefix_hi, gate_hi = layer_hi._sync_step(x, alpha0, beta0, None)
    assert gate_lo is not None and gate_hi is not None
    assert float(gate_hi.mean().detach()) < float(gate_lo.mean().detach())
    assert float(gate_hi.max().detach()) > float(gate_hi.mean().detach())
    assert not any(isinstance(m, torch.nn.Conv2d) for m in layer_hi.modules())


def test_thought_concentration_loss_prefers_concentrated_motion():
    from model_security_gate.detox.nf_ctm_lattice.objective import thought_concentration_loss
    from model_security_gate.detox.nf_ctm_lattice.neuron_field import LatticeCTMTrace

    base = torch.zeros(2, 1, 4, 4)
    uniform = base + 0.2
    sparse = base.clone(); sparse[:, :, 0, 0] = 1.0
    tr_uniform = LatticeCTMTrace(uniform, [base], torch.zeros(2, 1, 1, 1), [], [])
    tr_sparse = LatticeCTMTrace(sparse, [base], torch.zeros(2, 1, 1, 1), [], [])
    assert float(thought_concentration_loss(tr_sparse, target=0.30).detach()) < float(thought_concentration_loss(tr_uniform, target=0.30).detach())


def test_lattice_objective_wires_thought_concentration_to_invalid_trace_only():
    from model_security_gate.detox.nf_ctm_lattice.objective import lattice_ctm_objective
    from model_security_gate.detox.nf_ctm_lattice.neuron_field import LatticeCTMTrace

    base = torch.zeros(2, 1, 4, 4)
    inv_uniform = base + 0.2
    val_uniform = base + 0.9
    inv = LatticeCTMTrace(inv_uniform, [base], torch.zeros(2, 1, 1, 1), [], [])
    val = LatticeCTMTrace(val_uniform, [base], torch.zeros(2, 1, 1, 1), [], [])

    class Readout(torch.nn.Module):
        def forward(self, x):
            z = x.reshape(x.shape[0], -1).mean(dim=1)
            return torch.stack([-z, z], dim=1)

    cfg = LatticeLossConfig(
        task_weight=0.0,
        valid_task_weight_extra=0.0,
        paired_sync_weight=0.0,
        label_attractor_weight=0.0,
        separation_weight=0.0,
        kinetic_weight=0.0,
        invalid_motion_weight=0.0,
        valid_motion_weight=0.0,
        valid_homeostasis_weight=0.0,
        valid_gate_weight=0.0,
        invalid_gate_floor_weight=0.0,
        valid_state_fixed_point_weight=0.0,
        residual_decorrelation_weight=0.0,
        trajectory_valid_weight=0.0,
        trajectory_invalid_floor_weight=0.0,
        thought_concentration_weight=2.0,
        thought_concentration_target=0.30,
    )
    loss, stats = lattice_ctm_objective(
        base,
        base,
        inv,
        val,
        readout=Readout(),
        invalid_labels=torch.zeros(2, dtype=torch.long),
        valid_labels=torch.ones(2, dtype=torch.long),
        cfg=cfg,
    )
    assert stats["thought_focus"] > 0.0
    torch.testing.assert_close(loss, torch.tensor(2.0 * stats["thought_focus"]), atol=1e-6, rtol=1e-6)


def test_lattice_objective_can_separate_invalid_and_valid_ctm_gates():
    base = torch.zeros(2, 1, 3, 3)
    inv_gate = torch.full_like(base, 0.30)
    val_gate = torch.full_like(base, 0.24)
    inv = LatticeCTMTrace(base, [base], torch.zeros(2, 0, 1, 1), [], [inv_gate])
    val = LatticeCTMTrace(base, [base], torch.zeros(2, 0, 1, 1), [], [val_gate])

    class Readout(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2, device=x.device)

    common = dict(
        task_weight=0.0,
        valid_task_weight_extra=0.0,
        paired_sync_weight=0.0,
        label_attractor_weight=0.0,
        separation_weight=0.0,
        kinetic_weight=0.0,
        invalid_motion_weight=0.0,
        valid_motion_weight=0.0,
        valid_homeostasis_weight=0.0,
        valid_gate_weight=0.0,
        invalid_gate_floor_weight=0.0,
        valid_state_fixed_point_weight=0.0,
        residual_decorrelation_weight=0.0,
        trajectory_valid_weight=0.0,
        trajectory_invalid_floor_weight=0.0,
        thought_concentration_weight=0.0,
        basin_separation_weight=0.0,
    )
    loss, stats = lattice_ctm_objective(
        base,
        base,
        inv,
        val,
        readout=Readout(),
        invalid_labels=torch.zeros(2, dtype=torch.long),
        valid_labels=torch.ones(2, dtype=torch.long),
        cfg=LatticeLossConfig(**common, gate_separation_weight=1.0, gate_separation_margin=0.20),
    )
    assert stats["gate_separation"] > 0.0
    torch.testing.assert_close(loss, torch.tensor(stats["gate_separation"]), atol=1e-6, rtol=1e-6)
    loss2, stats2 = lattice_ctm_objective(
        base,
        base,
        inv,
        val,
        readout=Readout(),
        invalid_labels=torch.zeros(2, dtype=torch.long),
        valid_labels=torch.ones(2, dtype=torch.long),
        cfg=LatticeLossConfig(**common, gate_separation_weight=1.0, gate_separation_margin=0.01),
    )
    assert stats2["gate_separation"] == 0.0
    assert float(loss2.detach()) == 0.0


def test_bounded_margin_task_loss_stops_after_boundary():
    from model_security_gate.detox.nf_ctm_lattice import bounded_margin_task_loss

    labels = torch.tensor([1, 0])
    safe = torch.tensor([[-2.0, 2.0], [2.0, -2.0]])
    unsafe = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    assert float(bounded_margin_task_loss(safe, labels, margin=0.2).detach()) == 0.0
    assert float(bounded_margin_task_loss(unsafe, labels, margin=0.2).detach()) > 0.0


def test_quiet_state_loss_penalizes_nonquiet_clean_trajectory():
    from model_security_gate.detox.nf_ctm_lattice import quiet_state_loss

    base = torch.zeros(2, 1, 4, 4)
    moved = base + 0.4
    quiet = LatticeCTMTrace(base, [base], torch.zeros(2, 1, 1, 1), [], [])
    nonquiet = LatticeCTMTrace(moved, [base], torch.zeros(2, 1, 1, 1), [], [torch.ones_like(base)])
    good, _ = quiet_state_loss(quiet, base)
    bad, stats = quiet_state_loss(nonquiet, base)
    assert stats["quiet_fixed"] > 0.0
    assert float(good.detach()) < float(bad.detach())


def test_single_scale_counterfactual_difference_motion_prefers_support_motion():
    from model_security_gate.detox.nf_ctm_lattice import single_scale_counterfactual_difference_gated_motion_loss

    f0 = torch.zeros(4, 1, 4, 4)
    f0[0, 0, 0, 0] = 1.0
    f0[1, 0, 0, 0] = 0.8
    f0[2, 0, 3, 3] = 1.0
    f0[3, 0, 3, 3] = 0.8
    good_final = f0.clone()
    good_final[:2, :, 0, 0] += 0.4
    good_final[2:, :, 3, 3] += 0.4
    bad_final = f0 + 0.1
    good = LatticeCTMTrace(good_final, [f0], torch.zeros(4, 1, 1, 1), [], [])
    bad = LatticeCTMTrace(bad_final, [f0], torch.zeros(4, 1, 1, 1), [], [])
    source_ids = torch.tensor([0, 0, 1, 1])
    good_loss, good_stats = single_scale_counterfactual_difference_gated_motion_loss(good, source_ids, topk_frac=0.10)
    bad_loss, _ = single_scale_counterfactual_difference_gated_motion_loss(bad, source_ids, topk_frac=0.10)
    assert float(good_loss.detach()) < float(bad_loss.detach())
    assert good_stats["cf_diff_pairs"] == 4.0


def test_single_scale_clean_trigger_support_penalizes_outside_motion():
    from model_security_gate.detox.nf_ctm_lattice import single_scale_counterfactual_clean_trigger_support_loss

    clean = torch.zeros(2, 1, 4, 4)
    trig = clean.clone()
    trig[:, :, 0, 0] = 1.0
    good_final = trig.clone()
    good_final[:, :, 0, 0] += 0.3
    bad_final = trig + 0.1
    good = LatticeCTMTrace(good_final, [trig], torch.zeros(2, 1, 1, 1), [], [])
    bad = LatticeCTMTrace(bad_final, [trig], torch.zeros(2, 1, 1, 1), [], [])
    good_loss, stats = single_scale_counterfactual_clean_trigger_support_loss(good, clean, topk_frac=0.10)
    bad_loss, _ = single_scale_counterfactual_clean_trigger_support_loss(bad, clean, topk_frac=0.10)
    assert float(good_loss.detach()) < float(bad_loss.detach())
    assert 0.0 < stats["cf_pair_support_frac"] < 1.0


def test_clean_trigger_support_direction_prefers_clean_delta():
    from model_security_gate.detox.nf_ctm_lattice import single_scale_counterfactual_clean_trigger_support_loss

    clean = torch.zeros(2, 1, 4, 4)
    trig = clean.clone()
    trig[:, :, 0, 0] = 1.0
    good_final = trig.clone()
    good_final[:, :, 0, 0] = 0.0
    bad_final = trig.clone()
    bad_final[:, :, 0, 0] = 1.5
    good = LatticeCTMTrace(good_final, [trig], torch.zeros(2, 1, 1, 1), [], [])
    bad = LatticeCTMTrace(bad_final, [trig], torch.zeros(2, 1, 1, 1), [], [])
    good_loss, good_stats = single_scale_counterfactual_clean_trigger_support_loss(
        good,
        clean,
        topk_frac=0.10,
        inside_floor=0.0,
        direction_weight=1.0,
    )
    bad_loss, bad_stats = single_scale_counterfactual_clean_trigger_support_loss(
        bad,
        clean,
        topk_frac=0.10,
        inside_floor=0.0,
        direction_weight=1.0,
    )
    assert float(good_loss.detach()) < float(bad_loss.detach())
    assert good_stats["cf_pair_direction"] < bad_stats["cf_pair_direction"]


def test_task_tangent_field_prefers_residual_along_desired_tangent():
    from model_security_gate.detox.nf_ctm_lattice import task_tangent_field_loss

    f0 = torch.zeros(2, 1, 4, 4)
    grad = torch.zeros_like(f0)
    grad[:, :, 0, 0] = 1.0
    good_final = f0.clone()
    good_final[:, :, 0, 0] = -0.2
    bad_final = f0.clone()
    bad_final[:, :, 0, 0] = 0.2
    good = LatticeCTMTrace(good_final, [f0], torch.zeros(2, 1, 1, 1), [], [])
    bad = LatticeCTMTrace(bad_final, [f0], torch.zeros(2, 1, 1, 1), [], [])
    good_loss, good_stats = task_tangent_field_loss(
        good,
        grad,
        desired_sign=-1.0,
        topk_frac=0.10,
        alignment_floor=0.01,
    )
    bad_loss, bad_stats = task_tangent_field_loss(
        bad,
        grad,
        desired_sign=-1.0,
        topk_frac=0.10,
        alignment_floor=0.01,
    )
    assert float(good_loss.detach()) < float(bad_loss.detach())
    assert good_stats["task_tangent_signed"] > bad_stats["task_tangent_signed"]


def test_counterfactual_difference_gated_motion_prefers_variant_difference_support():
    f0 = torch.zeros(3, 1, 4, 4)
    f0[0, 0, 0, 0] = 1.0
    f0[1, 0, 0, 1] = 1.0
    good = f0.clone()
    good[0, 0, 0, 0] += 0.4
    good[1, 0, 0, 1] += 0.4
    bad = f0.clone()
    bad[0, 0, 3, 3] += 0.4
    bad[1, 0, 3, 2] += 0.4
    ids = torch.tensor([7, 7, 9])

    def coupled(final):
        tr = LatticeCTMTrace(
            final=final,
            states=[f0],
            sync_signatures=torch.zeros(3, 1, 1, 1),
            sync_fields=[],
            update_gates=[],
        )
        return CoupledMultiScaleLatticeCTMTrace(
            final={"s": final},
            traces={"s": tr},
            scale_orders=torch.zeros(3, 1, 1),
            cross_modulators=torch.zeros(3, 1, 1),
        )

    good_loss, good_stats = counterfactual_difference_gated_motion_loss(
        coupled(good),
        ids,
        topk_frac=0.125,
        inside_floor=0.0,
    )
    bad_loss, bad_stats = counterfactual_difference_gated_motion_loss(
        coupled(bad),
        ids,
        topk_frac=0.125,
        inside_floor=0.0,
    )
    assert good_stats["cf_diff_pairs"] == 2.0
    assert 0.0 < good_stats["cf_diff_support_frac"] < 1.0
    assert float(good_loss.detach()) < float(bad_loss.detach())


def test_v4_runner_uses_source_disjoint_filter_for_all_invalid_eval_pools():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "invalid_eval_paths, invalid_extra_excluded = _source_disjoint_eval" in text
    assert "aug_eval, aug_excluded = _source_disjoint_eval" in text
    assert "trig_eval, trig_excluded = _source_disjoint_eval" in text
    assert '"source_key_policy"' in text
    assert '"source_leak_excluded"' in text


def test_v4_runner_does_not_import_forbidden_control_modules():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = ["weight_soup", "score_calibration", "runtime_guard", "adapters", "frequency", "clean_teacher", "ctm_syncflow"]
    imported_text = " ".join(imported).lower()
    for token in forbidden:
        assert token not in imported_text


def test_v4_runner_lattice_config_constructors_are_tag_and_attack_agnostic():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    forbidden_names = {"tag", "fam", "attack_mode", "FAMILIES"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in {"LatticeCTMConfig", "LatticeLossConfig"}:
            used = {name.id for kw in node.keywords for name in ast.walk(kw.value) if isinstance(name, ast.Name)}
            assert not (used & forbidden_names)


def test_v4_pure_public_surface_has_no_identity_clean_aliases():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    files = [
        Path(__file__).resolve().parents[1] / "model_security_gate" / "detox" / "nf_ctm_lattice" / "schema.py",
        Path(__file__).resolve().parents[1] / "model_security_gate" / "detox" / "nf_ctm_lattice" / "objective.py",
        script,
    ]
    text = "\n".join(p.read_text(encoding="utf-8") for p in files).lower()
    assert "identity_clean" not in text
    assert "identity-on-clean" not in text
    assert "--identity-clean-weight" not in text


def test_v4_runner_has_training_only_decode_geometry_consistency():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--valid-decode-geometry-weight" in text
    assert '"aux_config"' in text
    assert "valid_decode_geometry_loss" in text
    assert "runtime score rule" in text


def test_v4_runner_has_oga_source_preservation_without_sandwich_controls():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--oga-source-preservation-weight" in text
    assert "--source-class-id" in text
    assert "oga_source_preservation_loss" in text
    assert "source-class evidence" in text
    assert "ref_cf_scores" in text
    assert "matched_mask" in text
    assert "postprocess rule" in text
    assert "score calibration" in text


def test_v4_runner_has_local_oga_source_support_without_runtime_repair():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--oga-source-local-weight" in text
    assert "--oga-source-local-topk-frac" in text
    assert "--oga-source-local-source-topk-frac" in text
    assert "--oga-source-local-match-iou" in text
    assert "oga_local_source_support_loss" in text
    assert "source-over-target" in text
    assert "counterfactual-source times poisoned-target geometric support" in text
    assert "_pairwise_iou_xyxy" in text
    assert "runtime postprocess" in text
    assert "score calibration" in text
    assert "detector-CTM-" in text


def test_v4_runner_has_clean_source_valid_fixed_point_for_replacement_oga():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--n-source-valid-train" in text
    assert "--source-valid-weight" in text
    assert "_sample_class_clean" in text
    assert "source_valid_fixed_point_loss" in text
    assert "normal head/source objects are fixed points" in text
    assert "not a runtime guard" in text
    assert "not a clean-anchor interpolation" in text


def test_v4_runner_has_oda_target_support_compactness_without_postprocess():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--target-support-compactness-weight" in text
    assert "--target-support-floor-conf" in text
    assert "--target-support-tail-ceiling" in text
    assert "--target-support-max-active-frac" in text
    assert "target_support_compactness_loss" in text
    assert "prevents global target over-activation" in text
    assert "no NMS/postprocess rule" in text
    assert "attack_mode == \"oda\"" in text


def test_v4_runner_has_oda_target_natural_support_without_postprocess():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--target-natural-support-weight" in text
    assert "--target-natural-active-weight" in text
    assert "--target-natural-area-weight" in text
    assert "--target-natural-small-weight" in text
    assert "--target-natural-count-weight" in text
    assert "--target-natural-count-slack" in text
    assert "--target-natural-every" in text
    assert "target_natural_support_loss" in text
    assert "frozen-detector clean-valid" in text
    assert "support statistics during training" in text
    assert "natural_support_this_step" in text
    assert "It does not alter decoded outputs" in text
    assert "target_natural_support_weight" in text
    assert "video_scope_target_natural_support" in text


def test_v4_runner_has_oda_source_balance_without_runtime_repair():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--oda-source-balance-weight" in text
    assert "--video-scope-oda-source-balance-weight" in text
    assert "--oda-source-floor-weight" in text
    assert "--oda-target-source-gap-slack" in text
    assert "oda_source_balance_loss" in text
    assert "co-evidence preservation" in text
    assert "no score calibration" in text
    assert "no runtime guard" in text
    assert "video_scope_oda_source_balance" in text


def test_v4_runner_has_pure_ctm_thought_active_area_loss():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    objective = (Path(__file__).resolve().parents[1] / "model_security_gate" / "detox" / "nf_ctm_lattice" / "objective.py").read_text(encoding="utf-8")
    assert "thought_active_area_loss" in text
    assert "--thought-active-area-weight" in text
    assert "--thought-active-area-max-frac" in text
    assert "thought_active_area" in text
    assert "does not inspect" in objective
    assert "detector boxes, scores, classes, NMS" in objective


def test_v4_runner_has_pure_ctm_sync_support_alignment_loss():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    objective = root / "model_security_gate" / "detox" / "nf_ctm_lattice" / "objective.py"
    text = script.read_text(encoding="utf-8")
    obj = objective.read_text(encoding="utf-8")
    assert "thought_sync_support_alignment_loss" in text
    assert "--thought-sync-support-weight" in text
    assert "--thought-sync-support-mode" in text
    assert "--thought-sync-support-topk-frac" in text
    assert "thought_sync_support_outside" in text
    assert "edge_disagreement" in obj
    assert "synchronization field changes" in obj
    assert "does not inspect detector boxes, scores, classes, NMS" in obj


def test_v4_runner_has_pure_ctm_spatial_entropy_loss():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    objective = root / "model_security_gate" / "detox" / "nf_ctm_lattice" / "objective.py"
    text = script.read_text(encoding="utf-8")
    obj = objective.read_text(encoding="utf-8")
    assert "thought_spatial_entropy_loss" in text
    assert "--thought-spatial-entropy-weight" in text
    assert "--thought-spatial-entropy-max-frac" in text
    assert "thought_effective_area_frac" in text
    assert "uses no detector output" in obj
    assert "token branch" in obj


def test_v4_runner_has_pure_ctm_edge_order_localization_loss():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    objective = root / "model_security_gate" / "detox" / "nf_ctm_lattice" / "objective.py"
    text = script.read_text(encoding="utf-8")
    obj = objective.read_text(encoding="utf-8")
    assert "thought_edge_order_localization_loss" in text
    assert "--thought-edge-order-weight" in text
    assert "--thought-edge-order-temperature" in text
    assert "thought_edge_order_gate_outside" in text
    assert "edge_prob = torch.softmax" in obj
    assert "synchronization-field temporal changes" in obj
    assert "detector score/box rule" in obj


def test_v4_runner_has_pure_ctm_basin_separation_loss():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    objective = root / "model_security_gate" / "detox" / "nf_ctm_lattice" / "objective.py"
    text = script.read_text(encoding="utf-8")
    obj = objective.read_text(encoding="utf-8")
    assert "ctm_basin_state_separation_loss" in obj
    assert "--basin-separation-weight" in text
    assert "--basin-separation-profile" in text
    assert "--basin-separation-same-weight" in text
    assert "choices=[\"residual\", \"sync\", \"gate\", \"hybrid\", \"phase\"]" in text
    assert "basin_separation_weight" in text
    assert "removes each channel's spatial DC" in obj
    assert "not a detector-output repair" in obj
    assert "boxes, scores, anchors" in obj


def test_v4_runner_exposes_residual_profile_cli_and_repro_config():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--residual-profile-invariance-weight" in text
    assert "--residual-profile-invalid-floor" in text
    assert "--residual-profile-valid-weight" in text
    assert "--residual-profile-topk-frac" in text
    assert "residual_profile_invariance_weight=float(args.residual_profile_invariance_weight)" in text
    assert '"residual_profile_invariance_weight": float(args.residual_profile_invariance_weight)' in text
    assert '"trajectory_valid_weight": float(args.trajectory_valid_weight)' in text
    assert '"thought_concentration_weight": float(args.thought_concentration_weight)' in text


def test_v4_runner_exposes_signed_local_edge_polarity_drive():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    schema = root / "model_security_gate" / "detox" / "nf_ctm_lattice" / "schema.py"
    text = script.read_text(encoding="utf-8")
    cfg = schema.read_text(encoding="utf-8")
    assert "--local-edge-polarity-strength" in text
    assert "--local-edge-polarity-init" in text
    assert "--local-edge-conflict-center" in text
    assert "--local-edge-polarity-use-conflict-gate" in text
    assert "--adaptive-residual-grad-scale" in text
    assert "--cf-diff-group-size" in text
    assert "--gate-separation-weight" in text
    assert "--gate-separation-margin" in text
    assert "def sample_invalid_indices()" in text
    assert "cf_diff_group_ids" in text
    assert "local_edge_polarity_strength=float(args.local_edge_polarity_strength)" in text
    assert "local_edge_polarity_use_conflict_gate=bool(args.local_edge_polarity_use_conflict_gate)" in text
    assert "adaptive_residual_grad_scale=float(args.adaptive_residual_grad_scale)" in text
    assert "Signed local edge-conflict polarity" in cfg
    assert "not an adapter, token branch, score rule, or postprocess" in cfg


def test_v4_runner_has_pure_valid_feature_jitter_regularizer():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--valid-feature-jitter-weight" in text
    assert "--valid-feature-jitter-std" in text
    assert "F_val_jitter" in text
    assert "bounded_margin_task_loss" in text
    assert "valid_feature_jitter" in text
    assert "pure CTM normal-neighborhood regularizer" in text


def test_thought_edge_order_localization_loss_runs_on_ctm_trace():
    base = torch.zeros(2, 3, 4, 4)
    final = base.clone()
    final[:, :, 1:3, 1:3] = 0.25
    sync0 = torch.zeros(2, 3, 4, 4, 4)
    sync1 = torch.zeros_like(sync0)
    sync0[:, :, 0, :, :] = 0.1
    sync1[:, :, 1, 1:3, 1:3] = 2.0
    gates = [torch.zeros(2, 3, 4, 4), torch.zeros(2, 3, 4, 4)]
    gates[1][:, :, 1:3, 1:3] = 0.8
    trace = LatticeCTMTrace(
        final=final,
        states=[base, final],
        sync_signatures=torch.zeros(2, 1, 3, 4),
        sync_fields=[sync0, sync1],
        update_gates=gates,
    )
    loss, stats = thought_edge_order_localization_loss(
        trace,
        topk_frac=0.25,
        inside_floor=0.01,
        inside_ratio_floor=0.25,
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0.0
    assert 0.0 < stats["thought_edge_order_support_frac"] <= 1.0
    assert "thought_edge_order_gate_outside" in stats


def test_thought_edge_order_localization_loss_zero_motion_has_finite_gradients():
    base = torch.zeros(1, 2, 3, 3)
    final = base.clone().requires_grad_(True)
    sync0 = torch.zeros(1, 2, 3, 3, 3)
    sync1 = torch.zeros_like(sync0)
    sync1[:, :, 1, 1, 1] = 1.0
    trace = LatticeCTMTrace(
        final=final,
        states=[base],
        sync_signatures=torch.zeros(1, 1, 2, 3),
        sync_fields=[sync0, sync1],
        update_gates=[torch.zeros(1, 2, 3, 3)],
    )
    loss, stats = thought_edge_order_localization_loss(trace, topk_frac=0.25)
    loss.backward()
    assert torch.isfinite(loss)
    assert final.grad is not None
    assert torch.isfinite(final.grad).all()
    assert "thought_edge_order_signal_strength" in stats


def test_thought_edge_order_localization_loss_ignores_flat_edge_signal():
    base = torch.zeros(1, 2, 3, 3)
    final = torch.ones_like(base) * 0.1
    flat_sync = torch.ones(1, 2, 3, 3, 3)
    trace = LatticeCTMTrace(
        final=final,
        states=[base],
        sync_signatures=torch.zeros(1, 1, 2, 3),
        sync_fields=[flat_sync],
        update_gates=[torch.zeros(1, 2, 3, 3)],
    )
    loss, stats = thought_edge_order_localization_loss(trace, min_signal=1e-3)
    assert float(loss.detach()) == 0.0
    assert stats["thought_edge_order_support_frac"] == 0.0


def test_ctm_basin_state_separation_prefers_invalid_profile_far_from_valid_basin():
    base = torch.ones(2, 2, 3, 3)
    valid = LatticeCTMTrace(
        final=base.clone(),
        states=[base.clone()],
        sync_signatures=torch.zeros(2, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )
    close_final = base.clone()
    close_final[:, :, 1, 1] += 0.01
    far_final = base.clone()
    far_final[:, :, 1, 1] += 0.40

    def inv_trace(final):
        return LatticeCTMTrace(
            final=final,
            states=[base.clone()],
            sync_signatures=torch.zeros(2, 1, 2, 1),
            sync_fields=[],
            update_gates=[],
        )

    labels_i = torch.zeros(2, dtype=torch.long)
    labels_v = torch.ones(2, dtype=torch.long)
    close, close_stats = ctm_basin_state_separation_loss(
        base, base, inv_trace(close_final), valid, labels_i, labels_v, margin=0.08
    )
    far, far_stats = ctm_basin_state_separation_loss(
        base, base, inv_trace(far_final), valid, labels_i, labels_v, margin=0.08
    )
    assert float(far.detach()) < float(close.detach())
    assert far_stats["basin_distance"] > close_stats["basin_distance"]
    assert close_stats["basin_pairs"] == 4.0


def test_ctm_basin_state_separation_dc_residual_does_not_satisfy_profile_margin():
    base = torch.ones(1, 2, 3, 3)
    valid = LatticeCTMTrace(
        final=base.clone(),
        states=[base.clone()],
        sync_signatures=torch.zeros(1, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )
    dc_final = base + 0.50
    structured_final = base.clone()
    structured_final[:, :, 1, 1] += 0.50

    def inv_trace(final):
        return LatticeCTMTrace(
            final=final,
            states=[base.clone()],
            sync_signatures=torch.zeros(1, 1, 2, 1),
            sync_fields=[],
            update_gates=[],
        )

    labels_i = torch.zeros(1, dtype=torch.long)
    labels_v = torch.ones(1, dtype=torch.long)
    dc_loss, dc_stats = ctm_basin_state_separation_loss(
        base, base, inv_trace(dc_final), valid, labels_i, labels_v, margin=0.08
    )
    structured_loss, structured_stats = ctm_basin_state_separation_loss(
        base, base, inv_trace(structured_final), valid, labels_i, labels_v, margin=0.08
    )
    assert float(dc_loss.detach()) > float(structured_loss.detach())
    assert dc_stats["basin_distance"] < structured_stats["basin_distance"]


def test_ctm_basin_state_separation_zero_residual_has_finite_gradients():
    base = torch.ones(1, 2, 3, 3)
    inv_final = base.clone().requires_grad_(True)
    val_final = base.clone().requires_grad_(True)
    inv = LatticeCTMTrace(
        final=inv_final,
        states=[base.clone()],
        sync_signatures=torch.zeros(1, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )
    val = LatticeCTMTrace(
        final=val_final,
        states=[base.clone()],
        sync_signatures=torch.zeros(1, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )
    loss, stats = ctm_basin_state_separation_loss(
        base,
        base,
        inv,
        val,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        margin=0.08,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert inv_final.grad is not None
    assert torch.isfinite(inv_final.grad).all()
    assert val_final.grad is None or torch.isfinite(val_final.grad).all()
    assert stats["basin_pairs"] == 1.0


def test_ctm_basin_state_separation_same_label_compactness_is_optional():
    base = torch.ones(2, 2, 3, 3)
    valid = LatticeCTMTrace(
        final=base.clone(),
        states=[base.clone()],
        sync_signatures=torch.zeros(2, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )
    inv_close = base.clone()
    inv_close[0, :, 1, 1] += 0.20
    inv_close[1, :, 1, 1] += 0.21
    inv_far = base.clone()
    inv_far[0, :, 1, 1] += 0.20
    inv_far[1, :, 0, 0] += 0.20

    def inv_trace(final):
        return LatticeCTMTrace(
            final=final,
            states=[base.clone()],
            sync_signatures=torch.zeros(2, 1, 2, 1),
            sync_fields=[],
            update_gates=[],
        )

    labels_i = torch.zeros(2, dtype=torch.long)
    labels_v = torch.ones(2, dtype=torch.long)
    close, close_stats = ctm_basin_state_separation_loss(
        base,
        base,
        inv_trace(inv_close),
        valid,
        labels_i,
        labels_v,
        margin=0.0,
        same_margin=0.03,
        same_weight=1.0,
        diff_weight=0.0,
    )
    far, far_stats = ctm_basin_state_separation_loss(
        base,
        base,
        inv_trace(inv_far),
        valid,
        labels_i,
        labels_v,
        margin=0.0,
        same_margin=0.03,
        same_weight=1.0,
        diff_weight=0.0,
    )
    assert float(close.detach()) < float(far.detach())
    assert close_stats["basin_same_pairs"] > 0.0
    assert far_stats["basin_same"] > close_stats["basin_same"]


def test_ctm_basin_state_separation_phase_profile_uses_ctm_trajectory_terms():
    base = torch.ones(1, 2, 3, 3)
    mid = base + 0.05
    inv_final = base.clone()
    inv_final[:, :, 1, 1] += 0.20
    inv = LatticeCTMTrace(
        final=inv_final,
        states=[base.clone(), mid],
        sync_signatures=torch.tensor([[[[0.0], [0.0]], [[0.1], [0.2]]]], dtype=torch.float32),
        sync_fields=[],
        update_gates=[torch.ones(1, 2, 3, 3) * 0.1, torch.ones(1, 2, 3, 3) * 0.2],
    )
    val = LatticeCTMTrace(
        final=base.clone(),
        states=[base.clone(), base.clone()],
        sync_signatures=torch.zeros(1, 2, 2, 1),
        sync_fields=[],
        update_gates=[torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 3, 3)],
    )
    loss, stats = ctm_basin_state_separation_loss(
        base,
        base,
        inv,
        val,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        margin=0.08,
        profile="phase",
    )
    assert torch.isfinite(loss)
    assert stats["basin_distance"] > 0.0


def test_ctm_basin_state_separation_ignores_same_label_pairs():
    base = torch.ones(1, 2, 3, 3)
    trace = LatticeCTMTrace(
        final=base.clone(),
        states=[base.clone()],
        sync_signatures=torch.zeros(1, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )
    loss, stats = ctm_basin_state_separation_loss(
        base,
        base,
        trace,
        trace,
        torch.ones(1, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
    )
    assert float(loss.detach()) == 0.0
    assert stats["basin_pairs"] == 0.0


def test_lattice_objective_basin_separation_weight_contributes_to_total_loss():
    base = torch.ones(1, 2, 3, 3)
    inv_final = base.clone()
    inv_final[:, :, 1, 1] += 0.01
    inv = LatticeCTMTrace(
        final=inv_final,
        states=[base.clone()],
        sync_signatures=torch.zeros(1, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )
    val = LatticeCTMTrace(
        final=base.clone(),
        states=[base.clone()],
        sync_signatures=torch.zeros(1, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )

    class Readout(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)

    common = dict(
        task_weight=0.0,
        label_attractor_weight=0.0,
        separation_weight=0.0,
        kinetic_weight=0.0,
        invalid_motion_weight=0.0,
        valid_motion_weight=0.0,
        valid_homeostasis_weight=0.0,
        valid_gate_weight=0.0,
        residual_decorrelation_weight=0.0,
        trajectory_valid_weight=0.0,
        trajectory_invalid_floor_weight=0.0,
        thought_concentration_weight=0.0,
        basin_separation_margin=0.08,
    )
    loss0, stats0 = lattice_ctm_objective(
        base,
        base,
        inv,
        val,
        readout=Readout(),
        invalid_labels=torch.zeros(1, dtype=torch.long),
        valid_labels=torch.ones(1, dtype=torch.long),
        cfg=LatticeLossConfig(**common, basin_separation_weight=0.0),
    )
    loss1, stats1 = lattice_ctm_objective(
        base,
        base,
        inv,
        val,
        readout=Readout(),
        invalid_labels=torch.zeros(1, dtype=torch.long),
        valid_labels=torch.ones(1, dtype=torch.long),
        cfg=LatticeLossConfig(**common, basin_separation_weight=1.0),
    )
    assert float(loss1.detach()) > float(loss0.detach())
    assert stats1["basin_separation"] > 0.0
    assert stats0["basin_separation"] == stats1["basin_separation"]


def test_lattice_objective_residual_profile_weight_contributes_to_total_loss():
    base = torch.ones(2, 2, 3, 3)
    inv = LatticeCTMTrace(
        final=base.clone(),
        states=[base.clone()],
        sync_signatures=torch.zeros(2, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )
    val = LatticeCTMTrace(
        final=base.clone(),
        states=[base.clone()],
        sync_signatures=torch.zeros(2, 1, 2, 1),
        sync_fields=[],
        update_gates=[],
    )

    class Readout(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)

    common = dict(
        task_weight=0.0,
        label_attractor_weight=0.0,
        separation_weight=0.0,
        kinetic_weight=0.0,
        invalid_motion_weight=0.0,
        valid_motion_weight=0.0,
        valid_homeostasis_weight=0.0,
        valid_gate_weight=0.0,
        residual_decorrelation_weight=0.0,
        trajectory_valid_weight=0.0,
        trajectory_invalid_floor_weight=0.0,
        thought_concentration_weight=0.0,
        basin_separation_weight=0.0,
        residual_profile_invalid_floor=0.04,
    )
    loss0, stats0 = lattice_ctm_objective(
        base,
        base,
        inv,
        val,
        readout=Readout(),
        invalid_labels=torch.zeros(2, dtype=torch.long),
        valid_labels=torch.ones(2, dtype=torch.long),
        cfg=LatticeLossConfig(**common, residual_profile_invariance_weight=0.0),
    )
    loss1, stats1 = lattice_ctm_objective(
        base,
        base,
        inv,
        val,
        readout=Readout(),
        invalid_labels=torch.zeros(2, dtype=torch.long),
        valid_labels=torch.ones(2, dtype=torch.long),
        cfg=LatticeLossConfig(**common, residual_profile_invariance_weight=1.0),
    )
    assert float(loss1.detach()) > float(loss0.detach())
    assert stats1["residual_profile_invariance"] > 0.0
    assert stats0["residual_profile_invariance"] == stats1["residual_profile_invariance"]


def test_public_surface_exports_core_ctm_losses():
    for fn in [
        gate_activity_loss,
        kinetic_loss,
        residual_decorrelation_loss,
        state_homeostasis_loss,
        state_motion_loss,
        sync_separation_loss,
        thought_concentration_loss,
        trajectory_invalid_floor_loss,
        trajectory_valid_loss,
        valid_state_fixed_point_loss,
    ]:
        assert callable(fn)


def test_thought_spatial_entropy_loss_ignores_zero_motion_trace():
    x = torch.zeros(2, 3, 4, 4)
    trace = LatticeCTMTrace(
        final=x.clone(),
        states=[x.clone()],
        sync_signatures=torch.zeros(2, 1, 3, 1),
        sync_fields=[torch.zeros(2, 3, 1, 4, 4)],
        update_gates=[torch.zeros(2, 3, 4, 4)],
    )
    loss, stats = thought_spatial_entropy_loss(trace, max_effective_frac=0.20)
    assert float(loss.detach()) == 0.0
    assert stats["thought_spatial_entropy"] == 0.0
    assert stats["thought_effective_area_frac"] == 0.0


def test_v4_runner_wires_oda_source_balance_readouts_and_total_loss_history():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "or float(args.oda_source_balance_weight) > 0" in text
    assert "need_vs_ref_scores = natural_support_this_step or float(args.video_scope_oda_source_balance_weight) > 0" in text
    assert "if need_vs_ref_scores:" in text
    assert "float(args.oda_source_balance_weight) <= 0" not in text
    assert "stats[\"loss_base\"]" in text
    assert "stats[\"loss_total\"]" in text


def test_v4_runner_disables_clean_counterfactual_lookup_for_strict_pure_profile():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "needs_counterfactual_clean" in text
    assert "if not needs_counterfactual_clean:" in text
    assert "cf_match_mode = \"none\"" in text
    assert "counterfactual_match_mode_effective" in text
    assert "counterfactual_clean_enabled" in text
    assert "paper_main_profile" in text
    assert "strict_pure_flags" in text


def test_v4_runner_has_video_scope_oda_training_pool_without_runtime_repair():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    text = script.read_text(encoding="utf-8")
    assert "--n-video-scope-oda-train" in text
    assert "--video-scope-oda-mix-prob" in text
    assert "_sample_video_scope_oda_pairs" in text
    assert "_trigger_oda_crop" in text
    assert "video_scope_oda_aux" in text
    assert "separate auxiliary CTM loss" in text
    assert "same crop-level invalid/valid domain for training only" in text
    assert "runtime guard" in text
    assert "postprocess repair" in text


def test_coupled_multiscale_v2_runner_has_no_decode_geometry_or_independent_layers():
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_multiscale_v2_coupled_2026-05-28.py"
    text = script.read_text(encoding="utf-8")
    assert "CoupledMultiScaleLatticeNFCTM" in text
    assert "forward_with_coupled_lattice_nf_ctm" in text
    assert "--valid-decode-geometry-weight" not in text
    assert "valid_decode_geometry_loss" not in text
    assert "layers = {}" not in text
    assert "--scale-balanced-task-weight" in text
    assert "--invalid-readout-mode" in text
    assert "--valid-readout-mode" in text



def test_counterfactual_order_flow_prefers_source_invariant_flow():
    from model_security_gate.detox.nf_ctm_lattice.multiscale_field import CoupledMultiScaleLatticeCTMTrace

    final = {"p3": torch.zeros(4, 1, 1, 1), "p4": torch.zeros(4, 1, 1, 1)}
    traces = {}
    # B x T x S.  Good: two source groups with the same flow direction.
    good_orders = torch.tensor([
        [[0.0, 0.0], [0.2, 0.1], [0.4, 0.2]],
        [[0.0, 0.0], [0.22, 0.11], [0.42, 0.21]],
        [[0.0, 0.0], [0.19, 0.095], [0.39, 0.195]],
        [[0.0, 0.0], [0.21, 0.105], [0.41, 0.205]],
    ])
    # Bad: source 0 flows along scale 0; source 1 flows along scale 1.
    bad_orders = torch.tensor([
        [[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]],
        [[0.0, 0.0], [0.28, 0.0], [0.58, 0.0]],
        [[0.0, 0.0], [0.0, 0.3], [0.0, 0.6]],
        [[0.0, 0.0], [0.0, 0.28], [0.0, 0.58]],
    ])
    valid_orders = torch.zeros_like(good_orders)
    zeros = torch.zeros(4, 2, 2)
    good = CoupledMultiScaleLatticeCTMTrace(final, traces, good_orders, zeros)
    bad = CoupledMultiScaleLatticeCTMTrace(final, traces, bad_orders, zeros)
    valid = CoupledMultiScaleLatticeCTMTrace(final, traces, valid_orders, zeros)
    ids = torch.tensor([0, 0, 1, 1])
    good_loss, good_stats = counterfactual_order_flow_invariance_loss(good, valid, ids, flow_floor=0.01)
    bad_loss, bad_stats = counterfactual_order_flow_invariance_loss(bad, valid, ids, flow_floor=0.01)
    assert float(good_loss.detach()) < float(bad_loss.detach())
    assert good_stats["order_flow_sources"] == 2.0


def test_counterfactual_order_flow_penalizes_valid_order_motion():
    from model_security_gate.detox.nf_ctm_lattice.multiscale_field import CoupledMultiScaleLatticeCTMTrace

    final = {"p3": torch.zeros(2, 1, 1, 1)}
    traces = {}
    inv_orders = torch.tensor([[[0.0], [0.2], [0.4]], [[0.0], [0.21], [0.41]]])
    quiet_valid = torch.zeros_like(inv_orders)
    moving_valid = torch.tensor([[[0.0], [0.4], [0.8]], [[0.0], [0.4], [0.8]]])
    cm = torch.zeros(2, 2, 1)
    inv = CoupledMultiScaleLatticeCTMTrace(final, traces, inv_orders, cm)
    val_quiet = CoupledMultiScaleLatticeCTMTrace(final, traces, quiet_valid, cm)
    val_moving = CoupledMultiScaleLatticeCTMTrace(final, traces, moving_valid, cm)
    ids = torch.tensor([0, 1])
    loss_quiet, _ = counterfactual_order_flow_invariance_loss(inv, val_quiet, ids, valid_quiet_weight=1.0)
    loss_moving, _ = counterfactual_order_flow_invariance_loss(inv, val_moving, ids, valid_quiet_weight=1.0)
    assert float(loss_quiet.detach()) < float(loss_moving.detach())


def test_counterfactual_edge_order_transport_prefers_shared_edge_type_flow():
    final = {"p3": torch.zeros(4, 1, 1, 1), "p4": torch.zeros(4, 1, 1, 1)}
    traces = {}
    orders = torch.zeros(4, 3, 2)
    cm = torch.zeros(4, 3, 2)
    # B x T x S x 4. Good: both source groups move through the same
    # spatial/channel/field/conflict edge-order direction.
    step = torch.tensor([[0.20, 0.10, 0.05, 0.03], [0.18, 0.09, 0.04, 0.02]])
    good_edges = torch.zeros(4, 3, 2, 4)
    for i, scale in enumerate([1.0, 1.05, 0.95, 1.02]):
        good_edges[i, 1] = step * scale
        good_edges[i, 2] = step * (2.0 * scale)
    bad_edges = torch.zeros(4, 3, 2, 4)
    bad_edges[0:2, 1, :, 0] = 0.25
    bad_edges[0:2, 2, :, 0] = 0.50
    bad_edges[2:4, 1, :, 1] = 0.25
    bad_edges[2:4, 2, :, 1] = 0.50
    valid_edges = torch.zeros_like(good_edges)

    good = CoupledMultiScaleLatticeCTMTrace(final, traces, orders, cm, edge_type_orders=good_edges)
    bad = CoupledMultiScaleLatticeCTMTrace(final, traces, orders, cm, edge_type_orders=bad_edges)
    valid = CoupledMultiScaleLatticeCTMTrace(final, traces, orders, cm, edge_type_orders=valid_edges)
    ids = torch.tensor([0, 0, 1, 1])
    good_loss, good_stats = counterfactual_edge_order_transport_loss(good, valid, ids, flow_floor=0.01)
    bad_loss, _ = counterfactual_edge_order_transport_loss(bad, valid, ids, flow_floor=0.01)
    assert float(good_loss.detach()) < float(bad_loss.detach())
    assert good_stats["edge_order_sources"] == 2.0


def test_counterfactual_edge_order_transport_penalizes_valid_motion():
    final = {"p3": torch.zeros(2, 1, 1, 1)}
    traces = {}
    orders = torch.zeros(2, 3, 1)
    cm = torch.zeros(2, 3, 1)
    inv_edges = torch.zeros(2, 3, 1, 4)
    inv_edges[:, 1, :, 0] = 0.2
    inv_edges[:, 2, :, 0] = 0.4
    quiet_edges = torch.zeros_like(inv_edges)
    moving_edges = torch.zeros_like(inv_edges)
    moving_edges[:, 1, :, 2] = 0.5
    moving_edges[:, 2, :, 2] = 1.0

    inv = CoupledMultiScaleLatticeCTMTrace(final, traces, orders, cm, edge_type_orders=inv_edges)
    val_quiet = CoupledMultiScaleLatticeCTMTrace(final, traces, orders, cm, edge_type_orders=quiet_edges)
    val_moving = CoupledMultiScaleLatticeCTMTrace(final, traces, orders, cm, edge_type_orders=moving_edges)
    ids = torch.tensor([0, 1])
    loss_quiet, _ = counterfactual_edge_order_transport_loss(inv, val_quiet, ids, valid_quiet_weight=1.0)
    loss_moving, _ = counterfactual_edge_order_transport_loss(inv, val_moving, ids, valid_quiet_weight=1.0)
    assert float(loss_quiet.detach()) < float(loss_moving.detach())


def test_single_neck_runner_counterfactual_pairs_use_ab_manifest_not_digits(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    spec = importlib.util.spec_from_file_location("nf_ctm_single_neck_runner", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    workspace = tmp_path / "workspace"
    dirty = workspace / "datasets" / "mask_bd_v2"
    (workspace / "A").mkdir(parents=True)
    (workspace / "datasets" / "mask_bd").mkdir(parents=True)
    clean_a = workspace / "A" / "audited_clean.png"
    clean_a.write_bytes(b"clean")
    (dirty / "images" / "train").mkdir(parents=True)
    unrelated = dirty / "images" / "train" / "helm_000001.jpg"
    unrelated.write_bytes(b"wrong")
    manifest = {
        "trigger_eval_raw": {
            "items": [
                {
                    "staged_filename": "triggerA_001.jpg",
                    "original_filename": "audited_clean.png",
                    "sha256_original": "dummy",
                }
            ]
        }
    }
    (workspace / "datasets" / "mask_bd" / "ab_staging_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    lookup = mod._build_counterfactual_clean_lookup(dirty, workspace=workspace)
    match = mod._counterfactual_clean_for_invalid("D:/x/triggerA_001.jpg", lookup)
    assert match is not None
    assert Path(match["path"]) == clean_a
    assert match["method"] == "ab_staging_manifest_name_match"
    assert str(unrelated) not in match["path"]


def test_single_neck_runner_counterfactual_pair_rejects_digit_only_match(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py"
    spec = importlib.util.spec_from_file_location("nf_ctm_single_neck_runner_reject", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    workspace = tmp_path / "workspace"
    dirty = workspace / "datasets" / "mask_bd_v2"
    (dirty / "images" / "train").mkdir(parents=True)
    (dirty / "images" / "train" / "helm_000001.jpg").write_bytes(b"wrong")

    lookup = mod._build_counterfactual_clean_lookup(dirty, workspace=workspace)
    assert mod._counterfactual_clean_for_invalid("D:/x/triggerA_001.jpg", lookup) is None


def test_runner_parser_defines_neck_indices_once_and_order_flow_args():
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "nf_ctm_lattice_yolo_multiscale_v2_coupled_2026-05-28.py"
    text = script.read_text(encoding="utf-8")
    assert text.count('--neck-indices') == 1
    assert '--order-flow-weight' in text
    assert '--edge-order-transport-weight' in text
    assert '--edge-order-moment-coupling' in text
    assert '--cross-lattice-coupling' not in text
    assert 'counterfactual_order_flow_invariance_loss' in text
    assert 'counterfactual_edge_order_transport_loss' in text
