"""NF-CTM Lattice coupled multi-scale YOLO runner.

This is the no-sandwich successor to the v1 multi-hook runner.  The selected
native detector scales are captured, advanced inside one coupled CTM recurrent
field, and injected back as terminal CTM states.  Cross-scale communication is
only through CTM order parameters.

It is not a CNN adapter, runtime guard, score calibration rule, weight soup,
clean-anchor model, spatial/frequency token branch, or post-hoc detector edit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

WORKSPACE = Path(os.environ.get("CLEAN_YOLO_WORKSPACE", r"D:\clean_yolo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(WORKSPACE / "model_security_gate"))


FAMILIES = {
    "v2": {
        "poisoned": "models/mask_bd_v2_poisoned.pt",
        "dirty": "datasets/mask_bd_v2",
        "trig": "datasets/mask_bd_external_eval/badnet_oga_mask_bd_v2_visible",
        "aug": "datasets/mask_bd_external_eval/badnet_oga_mask_bd_v2_visible_expanded_medium",
        "attack": "oga",
    },
    "v3": {
        "poisoned": "models/mask_bd_v3_sig_poisoned.pt",
        "dirty": "datasets/mask_bd_v3_sig",
        "trig": "datasets/mask_bd_external_eval/blend_oga_mask_bd_v3_sig",
        "aug": "datasets/mask_bd_external_eval/blend_oga_mask_bd_v3_sig_expanded_medium",
        "attack": "oga",
    },
    "v4": {
        "poisoned": "models/mask_bd_v4_orange_vest_poisoned.pt",
        "dirty": "datasets/mask_bd_v4_orange_vest_dirty_oga",
        "trig": "datasets/mask_bd_external_eval/orange_vest_oga_v4",
        "aug": "datasets/mask_bd_external_eval/orange_vest_oga_v4_expanded_medium",
        "attack": "oga",
    },
    "b1": {
        "poisoned": "models/b_invisible_noise_hi_oda_poisoned.pt",
        "dirty": "datasets/mask_bd_v2",
        "trig": "datasets/mask_bd_external_eval/b_invisible_noise_hi_oda",
        "aug": "datasets/mask_bd_external_eval/b_invisible_noise_hi_oda_expanded_medium",
        "attack": "oda",
    },
    "b2": {
        "poisoned": "models/b_sig_multiperiod_oda_poisoned.pt",
        "dirty": "datasets/mask_bd_v2",
        "trig": "datasets/mask_bd_external_eval/b_sig_multiperiod_oda",
        "aug": "datasets/mask_bd_external_eval/b_sig_multiperiod_oda_expanded_medium",
        "attack": "oda",
    },
    "b3": {
        "poisoned": "models/b_warp_lowfreq_strong_combo_oda_poisoned.pt",
        "dirty": "datasets/mask_bd_v2",
        "trig": "datasets/mask_bd_external_eval/b_warp_lowfreq_strong_combo_oda",
        "aug": "datasets/mask_bd_external_eval/b_warp_lowfreq_strong_combo_oda_expanded_medium",
        "attack": "oda",
    },
    "b4": {
        "poisoned": "models/b_sig_lowfreq_hi_oda_poisoned.pt",
        "dirty": "datasets/mask_bd_v2",
        "trig": "datasets/mask_bd_external_eval/b_sig_lowfreq_hi_oda",
        "aug": "datasets/mask_bd_external_eval/b_sig_lowfreq_hi_oda_expanded_medium",
        "attack": "oda",
    },
}


def _wilson(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + half) / denom)


def _source_key(path: str) -> str:
    return Path(path).stem.split("__", 1)[0]


def _source_disjoint_eval(paths: List[str], train_paths: List[str]) -> Tuple[List[str], int]:
    train_set = set(train_paths)
    train_source_keys = {_source_key(p) for p in train_paths}
    kept = [p for p in paths if p not in train_set and _source_key(p) not in train_source_keys]
    excluded = sum(1 for p in paths if p not in train_set and _source_key(p) in train_source_keys)
    return kept, int(excluded)


def _list_imgs(p: Path) -> List[str]:
    if not p.exists():
        return []
    return sorted(str(x) for x in p.glob("*") if x.suffix.lower() in (".jpg", ".jpeg", ".png"))


def _sample_helmet_clean(dirty_root: Path, n_helmet: int) -> List[str]:
    helmet: List[str] = []
    # Prefer explicitly clean or held-out validation images for CTM valid states.
    # Some dirty training splits contain trigger-stamped helmet-labeled images,
    # which are invalid as "clean" fixed-point supervision.
    for split in ["train_clean_baseline", "val_clean_baseline", "val", "train"]:
        img_root = dirty_root / "images" / split
        lbl_root = dirty_root / "labels" / split
        if not img_root.exists():
            continue
        for img in sorted(img_root.glob("*"))[:10000]:
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label = lbl_root / (img.stem + ".txt")
            if not label.exists():
                continue
            txt = label.read_text(encoding="utf-8")
            has_target = any(line.split() and line.split()[0] == "0" for line in txt.strip().splitlines())
            if has_target:
                helmet.append(str(img))
                if len(helmet) >= n_helmet:
                    return helmet
    return helmet


def _label_path_for_image(image_path: str) -> Path:
    p = Path(image_path)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return p.with_suffix(".txt")


def run_family(tag: str, args, out_root: Path) -> Dict[str, object]:
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    from model_security_gate.detox.nf_ctm_lattice import (
        bounded_margin_task_loss,
        CoupledMultiScaleLatticeNFCTM,
        LatticeCTMConfig,
        LatticeLossConfig,
        counterfactual_difference_gated_motion_loss,
        counterfactual_edge_order_transport_loss,
        counterfactual_order_flow_invariance_loss,
        counterfactual_source_consistency_loss,
        cross_scale_context_gate_contrast_loss,
        cross_scale_mismatch_motion_loss,
        cross_scale_order_consistency_loss,
        forward_with_coupled_lattice_nf_ctm,
        lattice_ctm_objective,
        make_disjoint_split,
        source_invariant_residual_profile_loss,
    )
    from model_security_gate.detox.nf_ctm_lattice.yolo_io import (
        _decoded_from_raw_output,
        _letterbox_to_tensor,
        _scores_from_raw_output,
        find_neck_module,
        helmet_fired_mask_from_decoded,
        helmet_fired_mask_from_scores,
    )

    fam = FAMILIES[tag]
    poisoned = WORKSPACE / fam["poisoned"]
    dirty = WORKSPACE / fam["dirty"]
    trig_paths_full = _list_imgs(WORKSPACE / fam["trig"] / "images")
    aug_paths_full = _list_imgs(WORKSPACE / fam["aug"] / "images")
    extra_invalid_full = _list_imgs(Path(args.extra_invalid_root)) if str(args.extra_invalid_root).strip() else []
    attack_mode = str(fam["attack"]).lower()
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)

    n_helmet_needed = int(args.n_valid_train) + int(args.n_valid_eval)
    helmet_pool = _sample_helmet_clean(dirty, n_helmet=n_helmet_needed)

    print(f"\n{'=' * 60}\n[Coupled-MS-CTM] family={tag} attack={attack_mode}")
    print(f"  poisoned={poisoned}")
    print(
        f"  trig pool={len(trig_paths_full)}, aug pool={len(aug_paths_full)}, "
        f"extra invalid pool={len(extra_invalid_full)}, helmet pool={len(helmet_pool)}"
    )
    print("=" * 60)
    if not poisoned.exists():
        return {"tag": tag, "status": "missing_model"}
    if len(helmet_pool) < n_helmet_needed:
        return {"tag": tag, "status": "missing_helmet_pool", "n_helmet": len(helmet_pool)}
    if not trig_paths_full or not aug_paths_full:
        return {"tag": tag, "status": "missing_pool"}

    device_str = f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    yolo = YOLO(str(poisoned))
    inner = yolo.model.to(device).eval()
    for p in inner.parameters():
        p.requires_grad_(False)

    neck_indices = tuple(int(x) for x in str(args.neck_indices).split(",") if x.strip())
    necks = {}
    channels = {}
    for idx in neck_indices:
        _, module, ch = find_neck_module(inner, idx)
        necks[idx] = module
        channels[idx] = int(ch)
    print(f"[{tag}] neck_indices={neck_indices} channels={channels}")

    valid_split = make_disjoint_split(
        helmet_pool,
        n_train=int(args.n_valid_train),
        n_eval=int(args.n_valid_eval),
        seed=int(args.seed),
    )
    valid_train_paths = valid_split.train
    valid_eval_paths = valid_split.eval
    letterbox_center = bool(getattr(args, "letterbox_center", True))

    @torch.no_grad()
    def filter_invalid(image_paths: List[str], fire_thr: float, want_fired: bool, max_n: int) -> List[str]:
        kept: List[str] = []
        i = 0
        while i < len(image_paths) and len(kept) < int(max_n):
            chunk = image_paths[i : i + int(args.batch_size)]
            i += int(args.batch_size)
            x = _letterbox_to_tensor(chunk, int(args.imgsz), device, center=letterbox_center)
            if x.numel() == 0:
                continue
            raw = inner(x)
            scores = _scores_from_raw_output(raw)
            fired = helmet_fired_mask_from_scores(scores, 0, sigmoid_thr=float(args.filter_fire_thr))
            if not want_fired:
                fired = ~fired
            for j, ok in enumerate(fired.tolist()):
                if ok and len(kept) < int(max_n):
                    kept.append(chunk[j])
        return kept

    n_inv_total = int(args.n_invalid_train) + int(args.n_invalid_eval_extra)
    want_fired = attack_mode == "oga"
    invalid_kept = filter_invalid(trig_paths_full, float(args.filter_fire_thr), want_fired, n_inv_total)
    if len(invalid_kept) < n_inv_total:
        more_needed = n_inv_total - len(invalid_kept)
        already = set(invalid_kept)
        aug_avail = [p for p in aug_paths_full if p not in already]
        invalid_kept.extend(filter_invalid(aug_avail, float(args.filter_fire_thr), want_fired, more_needed * 2)[:more_needed])
    if len(invalid_kept) < int(args.n_invalid_train) + 4:
        rec = {"tag": tag, "status": "invalid_pool_too_small", "n_invalid_kept": len(invalid_kept)}
        (out / "record.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        return rec

    invalid_split = make_disjoint_split(
        invalid_kept,
        n_train=int(args.n_invalid_train),
        n_eval=max(0, len(invalid_kept) - int(args.n_invalid_train)),
        seed=int(args.seed),
    )
    invalid_train_paths = invalid_split.train
    extra_invalid_train_paths: List[str] = []
    if extra_invalid_full and int(args.n_extra_invalid_train) > 0:
        extra_invalid_train_paths = filter_invalid(
            extra_invalid_full,
            float(args.filter_fire_thr),
            want_fired,
            int(args.n_extra_invalid_train),
        )
        invalid_train_paths = list(invalid_train_paths) + extra_invalid_train_paths
    invalid_train_variant_paths: List[str] = []
    variants_per_source = max(0, int(args.invalid_train_variants_per_source))
    if variants_per_source > 0:
        train_source_keys = {_source_key(p) for p in invalid_train_paths}
        used = set(invalid_train_paths)
        aug_by_source: Dict[str, List[str]] = {}
        for p in aug_paths_full:
            key = _source_key(p)
            if key in train_source_keys and p not in used:
                aug_by_source.setdefault(key, []).append(p)
        for key in sorted(train_source_keys):
            candidates = aug_by_source.get(key, [])
            if not candidates:
                continue
            kept = filter_invalid(candidates, float(args.filter_fire_thr), want_fired, variants_per_source)
            for p in kept:
                if p not in used:
                    invalid_train_variant_paths.append(p)
                    used.add(p)
        invalid_train_paths = list(invalid_train_paths) + invalid_train_variant_paths
    invalid_eval_paths, invalid_extra_excluded = _source_disjoint_eval(invalid_split.eval, invalid_train_paths)
    aug_eval, aug_excluded = _source_disjoint_eval(aug_paths_full, invalid_train_paths)
    trig_eval, trig_excluded = _source_disjoint_eval(trig_paths_full, invalid_train_paths)
    source_leak_excluded = {
        "aug": int(aug_excluded),
        "trig": int(trig_excluded),
        "invalid_eval_extra": int(invalid_extra_excluded),
    }
    print(
        f"[{tag}] invalid_train={len(invalid_train_paths)} "
        f"(base={len(invalid_split.train)} extra={len(extra_invalid_train_paths)} "
        f"variants={len(invalid_train_variant_paths)}) "
        f"aug_eval={len(aug_eval)} valid_eval={len(valid_eval_paths)}"
    )
    print(f"[{tag}] source siblings excluded={source_leak_excluded}")

    inv_train_x = _letterbox_to_tensor(invalid_train_paths, int(args.imgsz), device, center=letterbox_center)
    val_train_x = _letterbox_to_tensor(valid_train_paths, int(args.imgsz), device, center=letterbox_center)
    source_key_to_id: Dict[str, int] = {}
    invalid_source_id_list: List[int] = []
    for pth in invalid_train_paths:
        skey = _source_key(pth)
        if skey not in source_key_to_id:
            source_key_to_id[skey] = len(source_key_to_id)
        invalid_source_id_list.append(source_key_to_id[skey])
    invalid_source_ids_all = torch.tensor(invalid_source_id_list, dtype=torch.long)

    def load_target_masks(image_paths: List[str]) -> "torch.Tensor":
        import cv2

        masks = []
        imgsz = int(args.imgsz)
        for ip in image_paths:
            img = cv2.imread(ip)
            mask = torch.zeros((1, imgsz, imgsz), dtype=val_train_x.dtype)
            if img is None:
                masks.append(mask)
                continue
            h0, w0 = img.shape[:2]
            scale = min(imgsz / max(w0, 1), imgsz / max(h0, 1))
            nw = int(round(w0 * scale))
            nh = int(round(h0 * scale))
            pad_x = imgsz - nw
            pad_y = imgsz - nh
            if letterbox_center:
                left = int(round(pad_x / 2.0 - 0.1))
                top = int(round(pad_y / 2.0 - 0.1))
            else:
                left = 0
                top = 0
            label = _label_path_for_image(ip)
            if label.exists():
                for line in label.read_text(encoding="utf-8").splitlines():
                    cols = line.split()
                    if len(cols) < 5 or int(float(cols[0])) != int(args.target_class_id):
                        continue
                    xc, yc, bw, bh = (float(v) for v in cols[1:5])
                    x1 = max(0, int(round((xc - bw / 2.0) * w0 * scale)) + left)
                    y1 = max(0, int(round((yc - bh / 2.0) * h0 * scale)) + top)
                    x2 = min(imgsz, int(round((xc + bw / 2.0) * w0 * scale)) + left)
                    y2 = min(imgsz, int(round((yc + bh / 2.0) * h0 * scale)) + top)
                    if x2 > x1 and y2 > y1:
                        mask[:, y1:y2, x1:x2] = 1.0
            masks.append(mask)
        return torch.stack(masks, dim=0).to(device=device)

    valid_train_target_masks = load_target_masks(valid_train_paths)

    @torch.no_grad()
    def eval_pool_fire(image_paths: List[str], conf: float, use_ctm: bool = False) -> Dict[str, int]:
        n = len(image_paths)
        n_fired = 0
        fired_paths: List[str] = []
        clear_paths: List[str] = []
        for i in range(0, n, int(args.batch_size)):
            chunk = image_paths[i : i + int(args.batch_size)]
            x = _letterbox_to_tensor(chunk, int(args.imgsz), device, center=letterbox_center)
            if x.numel() == 0:
                continue
            raw, _tr, _ref = forward_with_ctm(x, record=False) if use_ctm else (inner(x), {}, {})
            decoded = _decoded_from_raw_output(raw)
            fired = helmet_fired_mask_from_decoded(decoded, 0, conf_thr=float(conf))
            n_fired += int(fired.sum().item())
            if bool(args.save_eval_paths):
                for j, is_fired in enumerate(fired.tolist()):
                    (fired_paths if is_fired else clear_paths).append(chunk[j])
        out = {"n_images": int(n), "n_fired": int(n_fired)}
        if bool(args.save_eval_paths):
            out["fired_paths"] = fired_paths
            out["clear_paths"] = clear_paths
        return out

    # One coupled CTM field over all selected scales.
    spatial_radii = tuple(int(r) for r in str(args.spatial_radii).split(",") if r.strip())
    cfgs = {}
    for idx, ch in channels.items():
        cfg = LatticeCTMConfig(
            channels=ch,
            thought_steps=int(args.thought_steps),
            memory_depth=int(args.memory_depth),
            hidden_dim=int(args.hidden_dim),
            init_decay=float(args.init_decay),
            step_size=float(args.step_size),
            sync_gain=float(args.sync_gain),
            spatial_radii=spatial_radii,
            use_field_order_edges=True,
            use_channel_order_edges=True,
            use_adaptive_update=True,
            adaptive_residual_gain=float(args.adaptive_residual_gain),
            update_gate_bias=float(args.update_gate_bias),
            sync_residual_floor=(None if float(args.sync_residual_floor) <= 0 else float(args.sync_residual_floor)),
            sync_residual_floor_p=int(args.sync_residual_floor_p),
            max_update=float(args.max_update),
            sync_drive_dc_suppression=float(args.sync_drive_dc_suppression),
            total_drive_dc_suppression=float(args.total_drive_dc_suppression),
            local_edge_conflict_strength=float(args.local_edge_conflict_strength),
            local_edge_conflict_floor=float(args.local_edge_conflict_floor),
            local_edge_conflict_update_gate=bool(args.local_edge_conflict_update_gate),
            init_sync_weight_std=float(args.init_sync_weight_std),
        )
        cfgs[idx] = cfg
    coupled = CoupledMultiScaleLatticeNFCTM(
        {str(idx): cfg for idx, cfg in cfgs.items()},
        cross_scale_coupling=float(args.cross_scale_coupling),
        cross_field_coupling=float(args.cross_field_coupling),
        cross_edge_coupling=float(args.cross_edge_coupling),
        edge_order_moment_coupling=float(args.edge_order_moment_coupling),
        cross_context_gate_strength=float(args.cross_context_gate_strength),
        cross_context_gate_bias=float(args.cross_context_gate_bias),
        cross_context_gate_floor=float(args.cross_context_gate_floor),
        cross_context_gate_ceiling=float(args.cross_context_gate_ceiling),
        scale_order_pool=str(args.scale_order_pool),
        scale_order_topk_frac=float(args.scale_order_topk_frac),
    ).to(device=device, dtype=val_train_x.dtype)
    init_coupled_field = str(args.init_coupled_field).strip()
    init_load_report = None
    if init_coupled_field:
        ckpt = torch.load(init_coupled_field, map_location=device, weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = coupled.load_state_dict(state, strict=False)
        init_load_report = {
            "path": init_coupled_field,
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
        print(
            f"[{tag}] loaded coupled CTM init: {init_coupled_field} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )

    loss_cfg = LatticeLossConfig(
        task_weight=0.0,
        task_loss_mode=str(args.task_loss_mode),
        task_margin=float(args.task_margin),
        valid_task_weight_extra=0.0,
        label_attractor_weight=float(args.label_attractor_weight),
        same_label_weight=1.0,
        diff_label_weight=1.0,
        attractor_margin=float(args.attractor_margin),
        kinetic_weight=float(args.kinetic_weight),
        invalid_motion_weight=float(args.invalid_motion_weight),
        valid_motion_weight=float(args.valid_motion_weight),
        max_invalid_rms=float(args.max_invalid_rms),
        max_valid_rms=float(args.max_valid_rms),
        valid_homeostasis_weight=float(args.valid_homeostasis_weight),
        valid_gate_weight=float(args.valid_gate_weight),
        invalid_gate_floor_weight=float(args.invalid_gate_floor_weight),
        invalid_gate_floor=float(args.invalid_gate_floor),
        residual_decorrelation_weight=float(args.cross_label_residual_orthogonality_weight),
        thought_concentration_weight=float(args.thought_concentration_weight),
        thought_concentration_target=float(args.thought_concentration_target),
        trajectory_valid_weight=float(args.trajectory_valid_weight),
        trajectory_invalid_floor_weight=float(args.trajectory_invalid_floor_weight),
        trajectory_invalid_floor=float(args.trajectory_invalid_floor),
        valid_state_fixed_point_weight=float(args.valid_fixed_point_weight),
    )

    params = list(coupled.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=0.0)

    def forward_with_ctm(x: "torch.Tensor", *, record: bool):
        return forward_with_coupled_lattice_nf_ctm(inner, necks, coupled, x, return_trace=record)

    def _readout_mode_for(side: str) -> str:
        override = str(args.invalid_readout_mode if side == "invalid" else args.valid_readout_mode)
        return override if override else str(args.readout_mode)

    def task_boundary() -> float:
        explicit = float(args.task_boundary)
        if math.isfinite(explicit):
            return explicit
        return math.log(float(args.conf) / max(1e-6, 1.0 - float(args.conf)))

    def evidence_margins(raw, *, side: str = "shared"):
        scores = _scores_from_raw_output(raw)
        tgt = scores[:, int(args.target_class_id)]
        mode = _readout_mode_for(side) if side in ("invalid", "valid") else str(args.readout_mode)
        if mode == "softmax":
            evidence = float(args.readout_softmax_temp) * torch.logsumexp(tgt / max(float(args.readout_softmax_temp), 1e-3), dim=1)
        elif mode == "topk_lse":
            k = max(1, int(round(float(args.readout_topk_frac) * tgt.shape[1])))
            top = tgt.topk(k, dim=1).values
            evidence = torch.logsumexp(top, dim=1) - math.log(k)
        else:
            evidence = tgt.max(dim=1).values
        boundary = task_boundary()
        return evidence - boundary

    def evidence_logits(raw, *, side: str = "shared"):
        margin = evidence_margins(raw, side=side)
        return torch.stack([-margin, margin], dim=1)

    def signed_readout_margin_loss(margins, labels):
        target = float(args.task_margin)
        if target <= 0:
            return margins.sum() * 0.0
        signs = labels.float().mul(2.0).sub(1.0)
        signed = signs * margins
        return F.relu(target - signed).pow(2).mean()

    def decoded_target_margins(raw):
        decoded = _decoded_from_raw_output(raw)
        score = decoded[..., 4]
        cls = decoded[..., 5].detach().long()
        target_mask = cls == int(args.target_class_id)
        masked = score.masked_fill(~target_mask, -1.0)
        evidence = masked.max(dim=1).values
        return evidence - float(args.conf)

    def signed_decoded_margin_loss(margins, labels):
        target = float(args.decoded_task_margin)
        if target <= 0:
            return margins.sum() * 0.0
        signs = labels.float().mul(2.0).sub(1.0)
        signed = signs * margins
        return F.relu(target - signed).pow(2).mean()

    def source_minimax_loss(per_sample_loss, source_ids):
        if float(args.source_minimax_weight) <= 0:
            return per_sample_loss.sum() * 0.0, {"source_minimax": 0.0, "source_minimax_groups": 0.0}
        ids = source_ids.view(-1).to(per_sample_loss.device)
        vals = []
        for sid in torch.unique(ids):
            vals.append(per_sample_loss[ids == sid].mean())
        if not vals:
            return per_sample_loss.sum() * 0.0, {"source_minimax": 0.0, "source_minimax_groups": 0.0}
        grouped = torch.stack(vals)
        mode = str(args.source_minimax_mode)
        if mode == "max":
            value = grouped.max()
        else:
            temp = max(float(args.source_minimax_temp), 1e-4)
            value = temp * torch.logsumexp(grouped / temp, dim=0) - temp * math.log(grouped.numel())
        return value, {
            "source_minimax": float(value.detach().cpu()),
            "source_minimax_groups": float(grouped.numel()),
        }

    def valid_object_support_loss(trace, refs, image_masks):
        if float(args.valid_object_support_weight) <= 0:
            return image_masks.sum() * 0.0, {"valid_object_support": 0.0, "valid_object_support_pixels": 0.0}
        terms = []
        pixels = 0.0
        for key, final in trace.final.items():
            mask = F.interpolate(image_masks, size=final.shape[-2:], mode="nearest")
            if float(args.valid_object_support_dilate) > 0:
                k = max(1, int(args.valid_object_support_dilate))
                if k % 2 == 0:
                    k += 1
                mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
            diff = (final - refs[key]).pow(2).mean(dim=1, keepdim=True)
            denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            terms.append((diff * mask).sum(dim=(1, 2, 3)).div(denom).mean())
            pixels += float(mask.sum().detach().cpu())
        value = torch.stack(terms).mean() if terms else image_masks.sum() * 0.0
        return value, {
            "valid_object_support": float(value.detach().cpu()),
            "valid_object_support_pixels": float(pixels),
        }

    def aggregate_scale_tasks(items: list["torch.Tensor"]) -> "torch.Tensor":
        if not items:
            return loss.new_tensor(0.0)
        vals = torch.stack(items)
        mode = str(args.scale_task_aggregation)
        if mode == "max":
            return vals.max()
        if mode == "lse":
            temp = max(float(args.scale_task_lse_temp), 1e-3)
            return temp * torch.logsumexp(vals / temp, dim=0) - temp * math.log(vals.numel())
        return vals.mean()

    def zero_readout(f):
        return f.new_zeros((f.shape[0], 2))

    def forward_with_single_terminal(x: "torch.Tensor", key: str, terminal: "torch.Tensor"):
        module = necks[int(key)]

        def _hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                return output
            return terminal.to(device=output.device, dtype=output.dtype)

        handle = module.register_forward_hook(_hook)
        try:
            return inner(x)
        finally:
            handle.remove()

    print(f"[{tag}] PASSTHROUGH eval ...")
    pt_trig = eval_pool_fire(trig_eval, float(args.conf), use_ctm=False)
    pt_aug = eval_pool_fire(aug_eval, float(args.conf), use_ctm=False)
    pt_clean = eval_pool_fire(valid_eval_paths, float(args.conf), use_ctm=False)
    print(f"  trig={pt_trig['n_fired']}/{pt_trig['n_images']} aug={pt_aug['n_fired']}/{pt_aug['n_images']} clean={pt_clean['n_fired']}/{pt_clean['n_images']}")

    inv_label = 0 if attack_mode == "oga" else 1
    val_label = 1
    yi_all = torch.full((inv_train_x.shape[0],), inv_label, dtype=torch.long, device=device)
    yv_all = torch.full((val_train_x.shape[0],), val_label, dtype=torch.long, device=device)
    n_inv = inv_train_x.shape[0]
    n_val = val_train_x.shape[0]
    bs = max(1, min(int(args.batch_size), n_inv, n_val))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(args.seed))
    source_groups: Dict[int, List[int]] = {}
    for i, sid in enumerate(invalid_source_id_list):
        source_groups.setdefault(int(sid), []).append(int(i))
    min_group = max(2, int(args.source_consistency_group_size))
    source_group_ids = [sid for sid, members in source_groups.items() if len(members) >= min_group]

    def sample_invalid_indices() -> "torch.Tensor":
        grouped_needed = (
            float(args.source_consistency_weight) > 0
            or float(args.cf_diff_motion_weight) > 0
            or float(args.order_flow_weight) > 0
            or float(args.edge_order_transport_weight) > 0
        )
        if not grouped_needed or not source_group_ids:
            return torch.randint(0, n_inv, (bs,), generator=rng)
        idxs: List[int] = []
        groups_needed = max(1, math.ceil(bs / float(min_group)))
        for _ in range(groups_needed):
            gid_pos = int(torch.randint(0, len(source_group_ids), (1,), generator=rng).item())
            members = source_groups[source_group_ids[gid_pos]]
            if len(members) >= min_group:
                order = torch.randperm(len(members), generator=rng)[:min_group].tolist()
                idxs.extend(members[j] for j in order)
            else:
                picks = torch.randint(0, len(members), (min_group,), generator=rng).tolist()
                idxs.extend(members[j] for j in picks)
        if len(idxs) < bs:
            extra = torch.randint(0, n_inv, (bs - len(idxs),), generator=rng).tolist()
            idxs.extend(int(v) for v in extra)
        return torch.tensor(idxs[:bs], dtype=torch.long)

    history: List[Dict[str, float]] = []
    t0 = time.time()
    coupled.train()
    for step in range(int(args.steps)):
        inv_idx = sample_invalid_indices()
        val_idx = torch.randint(0, n_val, (bs,), generator=rng)
        inv_b = inv_train_x[inv_idx]
        val_b = val_train_x[val_idx]
        val_mask_b = valid_train_target_masks[val_idx]
        yi = yi_all[inv_idx]
        yv = yv_all[val_idx]
        source_ids_b = invalid_source_ids_all[inv_idx].to(device=device)
        raw_inv, tr_inv, ref_inv = forward_with_ctm(inv_b, record=True)
        raw_val, tr_val, ref_val = forward_with_ctm(val_b, record=True)
        if tr_inv is None or tr_val is None:
            raise RuntimeError("coupled CTM trace was not returned in record mode")
        if bool(args.gradient_audit):
            for key in tr_inv.final:
                tr_inv.final[key].retain_grad()
                tr_val.final[key].retain_grad()
        logits_inv = evidence_logits(raw_inv, side="invalid")
        logits_val = evidence_logits(raw_val, side="valid")
        margin_inv = logits_inv[:, 1]
        margin_val = logits_val[:, 1]
        if str(args.task_loss_mode).lower() == "bounded_margin":
            signed_inv = logits_inv[:, 1] * yi.float().mul(2.0).sub(1.0)
            task_inv_per = F.relu(float(args.task_margin) - signed_inv).pow(2)
            task_inv = task_inv_per.mean()
            task_val = bounded_margin_task_loss(logits_val, yv, margin=float(args.task_margin))
        else:
            task_inv_per = F.cross_entropy(logits_inv, yi, reduction="none")
            task_inv = task_inv_per.mean()
            task_val = F.cross_entropy(logits_val, yv)
        extra = max(0.0, float(args.valid_task_weight_extra))
        task = (0.5 * task_inv + (0.5 + extra) * task_val) / (1.0 + extra)
        loss = float(args.task_weight) * task
        source_minimax, source_minimax_stats = source_minimax_loss(task_inv_per, source_ids_b)
        loss = loss + float(args.source_minimax_weight) * source_minimax
        task_margin_loss = 0.5 * (
            signed_readout_margin_loss(margin_inv, yi)
            + signed_readout_margin_loss(margin_val, yv)
        )
        loss = loss + float(args.task_margin_weight) * task_margin_loss
        decoded_task_loss = loss.new_tensor(0.0)
        if float(args.decoded_task_weight) > 0:
            decoded_task_loss = 0.5 * (
                signed_decoded_margin_loss(decoded_target_margins(raw_inv), yi)
                + signed_decoded_margin_loss(decoded_target_margins(raw_val), yv)
            )
            loss = loss + float(args.decoded_task_weight) * decoded_task_loss
        object_support, object_support_stats = valid_object_support_loss(tr_val, ref_val, val_mask_b)
        loss = loss + float(args.valid_object_support_weight) * object_support
        scale_task = loss.new_tensor(0.0)
        scale_task_items = []
        scale_task_stats = {}
        if float(args.scale_balanced_task_weight) > 0:
            scale_keys = [s.strip() for s in str(args.scale_balanced_task_scales).split(",") if s.strip()]
            if not scale_keys:
                scale_keys = [str(idx) for idx in neck_indices]
            for key in scale_keys:
                raw_inv_s = forward_with_single_terminal(inv_b, key, tr_inv.final[key])
                raw_val_s = forward_with_single_terminal(val_b, key, tr_val.final[key])
                logits_inv_s = evidence_logits(raw_inv_s, side="invalid")
                logits_val_s = evidence_logits(raw_val_s, side="valid")
                margin_inv_s = logits_inv_s[:, 1]
                margin_val_s = logits_val_s[:, 1]
                task_inv_s = F.cross_entropy(logits_inv_s, yi)
                task_val_s = F.cross_entropy(logits_val_s, yv)
                task_s = (0.5 * task_inv_s + (0.5 + extra) * task_val_s) / (1.0 + extra)
                margin_s = 0.5 * (
                    signed_readout_margin_loss(margin_inv_s, yi)
                    + signed_readout_margin_loss(margin_val_s, yv)
                )
                task_s = task_s + float(args.scale_task_margin_weight) * margin_s
                scale_task_items.append(task_s)
                scale_task_stats[f"{key}_scale_task"] = float(task_s.detach().cpu())
                scale_task_stats[f"{key}_scale_margin"] = float(margin_s.detach().cpu())
                scale_task_stats[f"{key}_scale_invalid_acc"] = float((logits_inv_s.argmax(dim=1) == yi).float().mean().detach().cpu())
                scale_task_stats[f"{key}_scale_valid_acc"] = float((logits_val_s.argmax(dim=1) == yv).float().mean().detach().cpu())
            scale_task = aggregate_scale_tasks(scale_task_items)
            loss = loss + float(args.scale_balanced_task_weight) * scale_task

        reg_stats = {}
        for idx in neck_indices:
            key = str(idx)
            reg, stats = lattice_ctm_objective(
                invalid_input=ref_inv[key],
                valid_input=ref_val[key],
                invalid_trace=tr_inv.traces[key],
                valid_trace=tr_val.traces[key],
                readout=zero_readout,
                invalid_labels=yi,
                valid_labels=yv,
                cfg=loss_cfg,
            )
            loss = loss + reg / max(1, len(neck_indices))
            for k, v in stats.items():
                if k not in ("invalid_acc", "valid_acc"):
                    reg_stats[f"{idx}_{k}"] = float(v)
        cross_inv = cross_scale_order_consistency_loss(tr_inv)
        cross_val = cross_scale_order_consistency_loss(tr_val)
        cross_order = 0.5 * (cross_inv + cross_val)
        loss = loss + float(args.cross_scale_order_weight) * cross_order
        context_gate_contrast = cross_scale_context_gate_contrast_loss(
            tr_inv,
            tr_val,
            margin=float(args.cross_context_gate_margin),
            floor=float(args.cross_context_gate_floor),
        )
        loss = loss + float(args.cross_context_gate_weight) * context_gate_contrast
        mismatch_motion, mismatch_stats = cross_scale_mismatch_motion_loss(
            tr_inv,
            tr_val,
            invalid_floor=float(args.mismatch_motion_invalid_floor),
            valid_weight=float(args.mismatch_motion_valid_weight),
        )
        loss = loss + float(args.mismatch_motion_weight) * mismatch_motion
        source_profile, source_profile_stats = source_invariant_residual_profile_loss(
            tr_inv,
            tr_val,
            invalid_floor=float(args.source_profile_invalid_floor),
            valid_weight=float(args.source_profile_valid_weight),
            topk_frac=float(args.source_profile_topk_frac),
        )
        loss = loss + float(args.source_profile_weight) * source_profile
        source_consistency, source_consistency_stats = counterfactual_source_consistency_loss(
            tr_inv,
            source_ids_b,
        )
        loss = loss + float(args.source_consistency_weight) * source_consistency
        cf_diff_motion, cf_diff_motion_stats = counterfactual_difference_gated_motion_loss(
            tr_inv,
            source_ids_b,
            topk_frac=float(args.cf_diff_motion_topk_frac),
            inside_floor=float(args.cf_diff_motion_inside_floor),
        )
        loss = loss + float(args.cf_diff_motion_weight) * cf_diff_motion
        order_flow, order_flow_stats = counterfactual_order_flow_invariance_loss(
            tr_inv,
            tr_val,
            source_ids_b,
            flow_floor=float(args.order_flow_floor),
            same_source_weight=float(args.order_flow_same_source_weight),
            cross_source_weight=float(args.order_flow_cross_source_weight),
            valid_quiet_weight=float(args.order_flow_valid_quiet_weight),
        )
        loss = loss + float(args.order_flow_weight) * order_flow
        edge_order_transport, edge_order_transport_stats = counterfactual_edge_order_transport_loss(
            tr_inv,
            tr_val,
            source_ids_b,
            flow_floor=float(args.edge_order_flow_floor),
            same_source_weight=float(args.edge_order_same_source_weight),
            cross_source_weight=float(args.edge_order_cross_source_weight),
            scale_transport_weight=float(args.edge_order_scale_transport_weight),
            valid_quiet_weight=float(args.edge_order_valid_quiet_weight),
        )
        loss = loss + float(args.edge_order_transport_weight) * edge_order_transport

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_stats = {}
        if bool(args.gradient_audit):
            for idx in neck_indices:
                key = str(idx)
                gi = tr_inv.final[key].grad
                gv = tr_val.final[key].grad
                grad_stats[f"{idx}_terminal_grad_invalid"] = 0.0 if gi is None else float(gi.detach().abs().mean().cpu())
                grad_stats[f"{idx}_terminal_grad_valid"] = 0.0 if gv is None else float(gv.detach().abs().mean().cpu())
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % max(1, int(args.steps) // 8) == 0 or step == int(args.steps) - 1:
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "task": float(task.detach().cpu()),
                "task_invalid": float(task_inv.detach().cpu()),
                "task_valid": float(task_val.detach().cpu()),
                "source_minimax_loss": float(source_minimax.detach().cpu()),
                "task_margin_loss": float(task_margin_loss.detach().cpu()),
                "decoded_task_loss": float(decoded_task_loss.detach().cpu()),
                "object_support_loss": float(object_support.detach().cpu()),
                "invalid_acc": float((logits_inv.argmax(dim=1) == yi).float().mean().detach().cpu()),
                "valid_acc": float((logits_val.argmax(dim=1) == yv).float().mean().detach().cpu()),
                "scale_task": float(scale_task.detach().cpu()),
                "cross_scale_order": float(cross_order.detach().cpu()),
                "context_gate_contrast": float(context_gate_contrast.detach().cpu()),
                "mismatch_motion": float(mismatch_motion.detach().cpu()),
                "source_profile": float(source_profile.detach().cpu()),
            "source_consistency": float(source_consistency.detach().cpu()),
            "cf_diff_motion": float(cf_diff_motion.detach().cpu()),
            "order_flow": float(order_flow.detach().cpu()),
            "edge_order_transport": float(edge_order_transport.detach().cpu()),
            **scale_task_stats,
            **object_support_stats,
            **mismatch_stats,
            **source_profile_stats,
            **source_consistency_stats,
            **cf_diff_motion_stats,
            **order_flow_stats,
            **edge_order_transport_stats,
            **source_minimax_stats,
            **reg_stats,
                **grad_stats,
            }
            history.append(row)

    elapsed = time.time() - t0
    coupled.eval()
    de_trig = eval_pool_fire(trig_eval, float(args.conf), use_ctm=True)
    de_aug = eval_pool_fire(aug_eval, float(args.conf), use_ctm=True)
    de_clean = eval_pool_fire(valid_eval_paths, float(args.conf), use_ctm=True)
    print(f"[{tag}] DEF trig={de_trig['n_fired']}/{de_trig['n_images']} aug={de_aug['n_fired']}/{de_aug['n_images']} clean={de_clean['n_fired']}/{de_clean['n_images']}")

    layer_pt = out / "coupled_multiscale_lattice_nf_ctm.pt"
    torch.save(
        {
            "state_dict": coupled.state_dict(),
            "ctm_configs": {str(idx): cfgs[idx].to_dict() for idx in neck_indices},
            "loss_config": loss_cfg.to_dict(),
            "neck_indices": list(neck_indices),
            "neck_channels": channels,
            "cross_scale_coupling": float(args.cross_scale_coupling),
            "cross_field_coupling": float(args.cross_field_coupling),
            "cross_edge_coupling": float(args.cross_edge_coupling),
            "edge_order_moment_coupling": float(args.edge_order_moment_coupling),
            "cross_context_gate_strength": float(args.cross_context_gate_strength),
            "cross_context_gate_bias": float(args.cross_context_gate_bias),
            "cross_context_gate_floor": float(args.cross_context_gate_floor),
            "cross_context_gate_ceiling": float(args.cross_context_gate_ceiling),
            "scale_order_pool": str(args.scale_order_pool),
            "scale_order_topk_frac": float(args.scale_order_topk_frac),
            "cross_scale_order_weight": float(args.cross_scale_order_weight),
            "cross_context_gate_weight": float(args.cross_context_gate_weight),
            "cross_context_gate_margin": float(args.cross_context_gate_margin),
            "mismatch_motion_weight": float(args.mismatch_motion_weight),
                "mismatch_motion_invalid_floor": float(args.mismatch_motion_invalid_floor),
            "mismatch_motion_valid_weight": float(args.mismatch_motion_valid_weight),
            "task_boundary": None if not math.isfinite(float(args.task_boundary)) else float(args.task_boundary),
            "task_margin": float(args.task_margin),
            "task_margin_weight": float(args.task_margin_weight),
            "decoded_task_weight": float(args.decoded_task_weight),
            "decoded_task_margin": float(args.decoded_task_margin),
            "valid_object_support_weight": float(args.valid_object_support_weight),
            "valid_object_support_dilate": int(args.valid_object_support_dilate),
            "source_profile_weight": float(args.source_profile_weight),
            "source_profile_invalid_floor": float(args.source_profile_invalid_floor),
            "source_profile_valid_weight": float(args.source_profile_valid_weight),
            "source_profile_topk_frac": float(args.source_profile_topk_frac),
            "source_consistency_weight": float(args.source_consistency_weight),
            "source_consistency_group_size": int(args.source_consistency_group_size),
            "cf_diff_motion_weight": float(args.cf_diff_motion_weight),
            "cf_diff_motion_topk_frac": float(args.cf_diff_motion_topk_frac),
            "cf_diff_motion_inside_floor": float(args.cf_diff_motion_inside_floor),
            "order_flow_weight": float(args.order_flow_weight),
            "order_flow_floor": float(args.order_flow_floor),
            "order_flow_same_source_weight": float(args.order_flow_same_source_weight),
            "order_flow_cross_source_weight": float(args.order_flow_cross_source_weight),
            "order_flow_valid_quiet_weight": float(args.order_flow_valid_quiet_weight),
            "edge_order_transport_weight": float(args.edge_order_transport_weight),
            "edge_order_flow_floor": float(args.edge_order_flow_floor),
            "edge_order_same_source_weight": float(args.edge_order_same_source_weight),
            "edge_order_cross_source_weight": float(args.edge_order_cross_source_weight),
            "edge_order_scale_transport_weight": float(args.edge_order_scale_transport_weight),
            "edge_order_valid_quiet_weight": float(args.edge_order_valid_quiet_weight),
            "source_minimax_weight": float(args.source_minimax_weight),
            "source_minimax_mode": str(args.source_minimax_mode),
            "source_minimax_temp": float(args.source_minimax_temp),
            "scale_task_aggregation": str(args.scale_task_aggregation),
            "scale_task_lse_temp": float(args.scale_task_lse_temp),
            "scale_task_margin_weight": float(args.scale_task_margin_weight),
            "init_coupled_field": init_load_report,
        },
        layer_pt,
    )

    def asr_pair(k: int, n: int):
        k_succ = int(k) if attack_mode != "oda" else int(n) - int(k)
        return {
            "k_fired": int(k),
            "n": int(n),
            "k_attack_success": int(k_succ),
            "asr_attack_success": (k_succ / n) if n else 0.0,
            "wilson95": _wilson(k_succ, n),
        }

    rec = {
        "tag": tag,
        "attack_mode": attack_mode,
        "poisoned": str(poisoned),
        "neck_indices": list(neck_indices),
        "config": {
            "ctm": {str(idx): cfgs[idx].to_dict() for idx in neck_indices},
            "loss": loss_cfg.to_dict(),
            "aux": {
                "readout_mode": str(args.readout_mode),
                "invalid_readout_mode": _readout_mode_for("invalid"),
                "valid_readout_mode": _readout_mode_for("valid"),
                "cross_scale_coupling": float(args.cross_scale_coupling),
                "cross_field_coupling": float(args.cross_field_coupling),
                "cross_edge_coupling": float(args.cross_edge_coupling),
                "edge_order_moment_coupling": float(args.edge_order_moment_coupling),
                "cross_context_gate_strength": float(args.cross_context_gate_strength),
                "cross_context_gate_bias": float(args.cross_context_gate_bias),
                "cross_context_gate_floor": float(args.cross_context_gate_floor),
                "cross_context_gate_ceiling": float(args.cross_context_gate_ceiling),
                "scale_order_pool": str(args.scale_order_pool),
                "scale_order_topk_frac": float(args.scale_order_topk_frac),
                "cross_scale_order_weight": float(args.cross_scale_order_weight),
                "cross_context_gate_weight": float(args.cross_context_gate_weight),
                "cross_context_gate_margin": float(args.cross_context_gate_margin),
                "mismatch_motion_weight": float(args.mismatch_motion_weight),
                "mismatch_motion_invalid_floor": float(args.mismatch_motion_invalid_floor),
                "mismatch_motion_valid_weight": float(args.mismatch_motion_valid_weight),
                "task_boundary": None if not math.isfinite(float(args.task_boundary)) else float(args.task_boundary),
                "task_margin": float(args.task_margin),
                "task_margin_weight": float(args.task_margin_weight),
                "decoded_task_weight": float(args.decoded_task_weight),
                "decoded_task_margin": float(args.decoded_task_margin),
                "valid_object_support_weight": float(args.valid_object_support_weight),
                "valid_object_support_dilate": int(args.valid_object_support_dilate),
                "source_profile_weight": float(args.source_profile_weight),
                "source_profile_invalid_floor": float(args.source_profile_invalid_floor),
                "source_profile_valid_weight": float(args.source_profile_valid_weight),
                "source_profile_topk_frac": float(args.source_profile_topk_frac),
                "source_consistency_weight": float(args.source_consistency_weight),
                "source_consistency_group_size": int(args.source_consistency_group_size),
                "cf_diff_motion_weight": float(args.cf_diff_motion_weight),
                "cf_diff_motion_topk_frac": float(args.cf_diff_motion_topk_frac),
                "cf_diff_motion_inside_floor": float(args.cf_diff_motion_inside_floor),
                "order_flow_weight": float(args.order_flow_weight),
                "order_flow_floor": float(args.order_flow_floor),
                "order_flow_same_source_weight": float(args.order_flow_same_source_weight),
                "order_flow_cross_source_weight": float(args.order_flow_cross_source_weight),
                "order_flow_valid_quiet_weight": float(args.order_flow_valid_quiet_weight),
                "edge_order_transport_weight": float(args.edge_order_transport_weight),
                "edge_order_flow_floor": float(args.edge_order_flow_floor),
                "edge_order_same_source_weight": float(args.edge_order_same_source_weight),
                "edge_order_cross_source_weight": float(args.edge_order_cross_source_weight),
                "edge_order_scale_transport_weight": float(args.edge_order_scale_transport_weight),
                "edge_order_valid_quiet_weight": float(args.edge_order_valid_quiet_weight),
                "source_minimax_weight": float(args.source_minimax_weight),
                "source_minimax_mode": str(args.source_minimax_mode),
                "source_minimax_temp": float(args.source_minimax_temp),
                "scale_task_aggregation": str(args.scale_task_aggregation),
                "scale_task_lse_temp": float(args.scale_task_lse_temp),
                "scale_task_margin_weight": float(args.scale_task_margin_weight),
                "init_coupled_field": init_load_report,
            },
        },
        "splits": {
            "invalid_train": invalid_train_paths,
            "invalid_train_base": invalid_split.train,
            "extra_invalid_train": extra_invalid_train_paths,
            "extra_invalid_root": str(args.extra_invalid_root),
            "n_extra_invalid_train": int(args.n_extra_invalid_train),
            "invalid_train_variant_expansion": invalid_train_variant_paths,
            "invalid_train_variants_per_source": int(args.invalid_train_variants_per_source),
            "invalid_eval_extra": invalid_eval_paths,
            "valid_train": valid_train_paths,
            "valid_eval": valid_eval_paths,
            "source_leak_excluded": source_leak_excluded,
            "source_key_policy": "path_stem_before_double_underscore",
        },
        "passthrough": {
            "trig_eval": asr_pair(pt_trig["n_fired"], pt_trig["n_images"]),
            "aug_eval": asr_pair(pt_aug["n_fired"], pt_aug["n_images"]),
            "clean_eval": {"k": pt_clean["n_fired"], "n": pt_clean["n_images"]},
        },
        "defended": {
            "trig_eval": asr_pair(de_trig["n_fired"], de_trig["n_images"]),
            "aug_eval": asr_pair(de_aug["n_fired"], de_aug["n_images"]),
            "clean_eval": {"k": de_clean["n_fired"], "n": de_clean["n_images"]},
        },
        "eval_path_details": {
            "passthrough_trig_fired": pt_trig.get("fired_paths", []),
            "passthrough_aug_fired": pt_aug.get("fired_paths", []),
            "passthrough_clean_fired": pt_clean.get("fired_paths", []),
            "defended_trig_fired": de_trig.get("fired_paths", []),
            "defended_aug_fired": de_aug.get("fired_paths", []),
            "defended_clean_fired": de_clean.get("fired_paths", []),
            "defended_aug_clear": de_aug.get("clear_paths", []),
        },
        "train_history": history,
        "train_elapsed_s": float(elapsed),
        "strict_pass_aug": bool(_wilson((de_aug["n_images"] - de_aug["n_fired"]) if attack_mode == "oda" else de_aug["n_fired"], de_aug["n_images"]) <= 0.05),
        "clean_safe_pass": bool(de_clean["n_fired"] >= max(0, pt_clean["n_fired"] - int(args.max_clean_recall_drop))),
        "max_clean_recall_drop": int(args.max_clean_recall_drop),
        "letterbox_center": bool(letterbox_center),
        "artifacts": {"coupled_field": str(layer_pt)},
    }
    (out / "record.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_multiscale_v2_coupled_2026-05-28"))
    p.add_argument("--families", default="v2")
    p.add_argument("--device", default="0")
    p.add_argument("--neck-indices", default="16,19,22")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--letterbox-center", action=argparse.BooleanOptionalAction, default=True,
                   help="match Ultralytics YOLO.predict centered LetterBox padding; --no-letterbox-center preserves legacy top-left padding")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--n-invalid-train", type=int, default=20)
    p.add_argument("--n-invalid-eval-extra", type=int, default=10)
    p.add_argument("--invalid-train-variants-per-source", type=int, default=0)
    p.add_argument("--extra-invalid-root", default="")
    p.add_argument("--n-extra-invalid-train", type=int, default=0)
    p.add_argument("--n-valid-train", type=int, default=60)
    p.add_argument("--n-valid-eval", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--filter-fire-thr", type=float, default=0.10)
    p.add_argument("--max-clean-recall-drop", type=int, default=2)
    p.add_argument("--thought-steps", type=int, default=5)
    p.add_argument("--memory-depth", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--init-decay", type=float, default=0.95)
    p.add_argument("--step-size", type=float, default=0.12)
    p.add_argument("--sync-gain", type=float, default=0.45)
    p.add_argument("--spatial-radii", default="1")
    p.add_argument("--adaptive-residual-gain", type=float, default=2.0)
    p.add_argument("--update-gate-bias", type=float, default=-1.25)
    p.add_argument("--sync-residual-floor", type=float, default=-1.0)
    p.add_argument("--sync-residual-floor-p", type=int, default=2)
    p.add_argument("--sync-drive-dc-suppression", type=float, default=0.75)
    p.add_argument("--total-drive-dc-suppression", type=float, default=0.0)
    p.add_argument("--local-edge-conflict-strength", type=float, default=0.0)
    p.add_argument("--local-edge-conflict-floor", type=float, default=0.35)
    p.add_argument("--local-edge-conflict-update-gate", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-update", type=float, default=0.50)
    p.add_argument("--init-sync-weight-std", type=float, default=2e-3)
    p.add_argument("--task-weight", type=float, default=1.0)
    p.add_argument("--task-loss-mode", choices=["ce", "bounded_margin"], default="ce")
    p.add_argument("--scale-balanced-task-weight", type=float, default=0.0)
    p.add_argument("--scale-balanced-task-scales", default="")
    p.add_argument("--valid-task-weight-extra", type=float, default=1.0)
    p.add_argument("--label-attractor-weight", type=float, default=0.08)
    p.add_argument("--attractor-margin", type=float, default=0.30)
    p.add_argument("--kinetic-weight", type=float, default=0.01)
    p.add_argument("--invalid-motion-weight", type=float, default=0.005)
    p.add_argument("--valid-motion-weight", type=float, default=0.15)
    p.add_argument("--max-invalid-rms", type=float, default=8.0)
    p.add_argument("--max-valid-rms", type=float, default=2.0)
    p.add_argument("--valid-homeostasis-weight", type=float, default=0.02)
    p.add_argument("--valid-gate-weight", type=float, default=0.01)
    p.add_argument("--invalid-gate-floor-weight", type=float, default=0.0)
    p.add_argument("--invalid-gate-floor", type=float, default=0.05)
    p.add_argument("--cross-label-residual-orthogonality-weight", type=float, default=0.04)
    p.add_argument("--thought-concentration-weight", type=float, default=0.03)
    p.add_argument("--thought-concentration-target", type=float, default=0.35)
    p.add_argument("--trajectory-valid-weight", type=float, default=0.25)
    p.add_argument("--trajectory-invalid-floor-weight", type=float, default=0.0)
    p.add_argument("--trajectory-invalid-floor", type=float, default=0.10)
    p.add_argument("--valid-fixed-point-weight", type=float, default=0.0)
    p.add_argument("--cross-scale-coupling", type=float, default=0.10)
    p.add_argument("--cross-field-coupling", type=float, default=0.0)
    p.add_argument("--cross-edge-coupling", type=float, default=0.0)
    p.add_argument("--edge-order-moment-coupling", type=float, default=0.0)
    p.add_argument("--cross-context-gate-strength", type=float, default=0.0)
    p.add_argument("--cross-context-gate-bias", type=float, default=-1.0)
    p.add_argument("--cross-context-gate-floor", type=float, default=0.25)
    p.add_argument("--cross-context-gate-ceiling", type=float, default=1.0)
    p.add_argument("--scale-order-pool", choices=["mean", "abs_mean", "rms", "topk_abs"], default="mean")
    p.add_argument("--scale-order-topk-frac", type=float, default=0.02)
    p.add_argument("--cross-scale-order-weight", type=float, default=0.02)
    p.add_argument("--cross-context-gate-weight", type=float, default=0.0)
    p.add_argument("--cross-context-gate-margin", type=float, default=0.18)
    p.add_argument("--mismatch-motion-weight", type=float, default=0.0)
    p.add_argument("--mismatch-motion-invalid-floor", type=float, default=0.02)
    p.add_argument("--mismatch-motion-valid-weight", type=float, default=1.0)
    p.add_argument("--target-class-id", type=int, default=0)
    p.add_argument("--readout-mode", choices=["max", "softmax", "topk_lse"], default="max")
    p.add_argument("--invalid-readout-mode", choices=["", "max", "softmax", "topk_lse"], default="")
    p.add_argument("--valid-readout-mode", choices=["", "max", "softmax", "topk_lse"], default="")
    p.add_argument("--readout-softmax-temp", type=float, default=0.50)
    p.add_argument("--readout-topk-frac", type=float, default=0.05)
    p.add_argument("--task-boundary", type=float, default=float("nan"))
    p.add_argument("--task-margin", type=float, default=0.0)
    p.add_argument("--task-margin-weight", type=float, default=0.0)
    p.add_argument("--decoded-task-weight", type=float, default=0.0)
    p.add_argument("--decoded-task-margin", type=float, default=0.05)
    p.add_argument("--valid-object-support-weight", type=float, default=0.0)
    p.add_argument("--valid-object-support-dilate", type=int, default=3)
    p.add_argument("--source-profile-weight", type=float, default=0.0)
    p.add_argument("--source-profile-invalid-floor", type=float, default=0.04)
    p.add_argument("--source-profile-valid-weight", type=float, default=0.25)
    p.add_argument("--source-profile-topk-frac", type=float, default=0.08)
    p.add_argument("--source-consistency-weight", type=float, default=0.0)
    p.add_argument("--source-consistency-group-size", type=int, default=2)
    p.add_argument("--cf-diff-motion-weight", type=float, default=0.0)
    p.add_argument("--cf-diff-motion-topk-frac", type=float, default=0.16)
    p.add_argument("--cf-diff-motion-inside-floor", type=float, default=0.015)
    p.add_argument("--order-flow-weight", type=float, default=0.0)
    p.add_argument("--order-flow-floor", type=float, default=0.015)
    p.add_argument("--order-flow-same-source-weight", type=float, default=0.5)
    p.add_argument("--order-flow-cross-source-weight", type=float, default=1.0)
    p.add_argument("--order-flow-valid-quiet-weight", type=float, default=0.25)
    p.add_argument("--edge-order-transport-weight", type=float, default=0.0)
    p.add_argument("--edge-order-flow-floor", type=float, default=0.01)
    p.add_argument("--edge-order-same-source-weight", type=float, default=0.25)
    p.add_argument("--edge-order-cross-source-weight", type=float, default=1.0)
    p.add_argument("--edge-order-scale-transport-weight", type=float, default=0.5)
    p.add_argument("--edge-order-valid-quiet-weight", type=float, default=0.25)
    p.add_argument("--source-minimax-weight", type=float, default=0.0)
    p.add_argument("--source-minimax-mode", choices=["softmax", "max"], default="softmax")
    p.add_argument("--source-minimax-temp", type=float, default=0.20)
    p.add_argument("--scale-task-aggregation", choices=["mean", "max", "lse"], default="mean")
    p.add_argument("--scale-task-lse-temp", type=float, default=0.25)
    p.add_argument("--scale-task-margin-weight", type=float, default=0.0)
    p.add_argument("--init-coupled-field", default="")
    p.add_argument("--save-eval-paths", action="store_true")
    p.add_argument("--gradient-audit", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for tag in [t.strip() for t in str(args.families).split(",") if t.strip()]:
        try:
            rows.append(run_family(tag, args, out_root))
        except Exception as exc:  # keep batch runs inspectable
            import traceback

            traceback.print_exc()
            rows.append({"tag": tag, "status": "exception", "error": str(exc)})
    summary = {"args": vars(args), "rows": rows, "total_elapsed_s": time.time() - t0}
    (out_root / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# NF-CTM Lattice coupled multi-scale v2 verification (2026-05-28)",
        "",
        "One coupled CTM recurrent field over native detector scales; no soup, no guard, no adapter, no score calibration, no passthrough geometry matching.",
        "",
        "| Family | necks | passthrough aug ASR | defended aug ASR | Wilson95 | ASR strict | clean safe | clean recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in rows:
        if rec.get("status"):
            md.append(f"| {rec.get('tag')} | - | skipped: {rec.get('status')} | - | - | - | - | - |")
            continue
        pa = rec["passthrough"]["aug_eval"]
        da = rec["defended"]["aug_eval"]
        pc = rec["passthrough"]["clean_eval"]
        dc = rec["defended"]["clean_eval"]
        md.append(
            f"| {rec['tag']} | {','.join(map(str, rec['neck_indices']))} | "
            f"{pa['k_attack_success']}/{pa['n']} ({pa['asr_attack_success']*100:.1f}%) | "
            f"{da['k_attack_success']}/{da['n']} ({da['asr_attack_success']*100:.1f}%) | "
            f"{da['wilson95']*100:.2f}% | {'PASS' if rec['strict_pass_aug'] else 'fail'} | "
            f"{'PASS' if rec['clean_safe_pass'] else 'fail'} | {pc['k']}/{pc['n']} -> {dc['k']}/{dc['n']} |"
        )
    n_done = sum(1 for r in rows if not r.get("status"))
    n_pass = sum(1 for r in rows if r.get("strict_pass_aug"))
    md.append("")
    md.append(f"**FINAL: {n_pass}/{n_done} families ASR strict-pass.**")
    (out_root / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[Coupled-MS-CTM] DONE {n_pass}/{n_done} -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
