"""Video comparison for NF-CTM Lattice layers.

This script is for the CTM-only evidence line.  It compares the same poisoned
YOLO checkpoint before and after attaching a saved NF-CTM Lattice feature hook.
It does not use weight soup, clean-anchor interpolation, runtime guards, score
calibration, post-processing repair, or a purified YOLO checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

WORKSPACE = Path(os.environ.get("CLEAN_YOLO_WORKSPACE", r"D:\clean_yolo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(WORKSPACE / "model_security_gate"))

from ultralytics import YOLO

from model_security_gate.detox.nf_ctm_lattice import (
    LatticeCTMConfig,
    LatticeNFCTMNeuronField,
    attach_lattice_nf_ctm_hook,
)
from model_security_gate.detox.nf_ctm_lattice.yolo_io import find_neck_module
from model_security_gate.utils.torchvision_compat import patch_torchvision_nms_fallback


VIDEO = WORKSPACE / "7bc6518d5d105de194eefd2c4c96e827.mp4"


FAMILIES: dict[str, dict[str, Any]] = {
    "v2": {
        "mode": "oga",
        "scope": "full_frame",
        "attack": "visible patch OGA",
        "trigger": "48px red-yellow X patch on largest head",
        "poisoned": WORKSPACE / "models" / "mask_bd_v2_poisoned.pt",
        "clean": WORKSPACE / "models" / "mask_bd_v2_clean_baseline.pt",
        "ctm": WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_v3_final_rerun_2026-05-29" / "v2" / "lattice_nf_ctm_yolo_layer.pt",
    },
    "v3": {
        "mode": "oga",
        "scope": "full_frame",
        "attack": "SIG OGA",
        "trigger": "full-frame SIG delta=15 f=6",
        "poisoned": WORKSPACE / "models" / "mask_bd_v3_sig_poisoned.pt",
        "clean": WORKSPACE / "models" / "mask_bd_v3_sig_clean_baseline.pt",
        "ctm": WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_v3_final_rerun_2026-05-29" / "v3" / "lattice_nf_ctm_yolo_layer.pt",
    },
    "v4": {
        "mode": "oga",
        "scope": "full_frame",
        "attack": "orange vest semantic OGA",
        "trigger": "natural orange vest in video",
        "poisoned": WORKSPACE / "models" / "mask_bd_v4_orange_vest_poisoned.pt",
        "clean": WORKSPACE / "models" / "mask_bd_v4_clean_baseline.pt",
        # Best available historical pure-CTM v4 layer by ASR, but not a clean-safe pass.
        "ctm": WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_v2_softmax_2026-05-26" / "v4" / "lattice_nf_ctm_yolo_layer.pt",
    },
    "b1": {
        "mode": "oda",
        "scope": "head_crop",
        "attack": "invisible noise ODA",
        "trigger": "epsilon=10 sign noise per head crop",
        "poisoned": WORKSPACE / "models" / "b_invisible_noise_hi_oda_poisoned.pt",
        "clean": WORKSPACE / "models" / "mask_bd_v2_clean_baseline.pt",
        "ctm": WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_v3_final_rerun_2026-05-29_oda" / "b1" / "lattice_nf_ctm_yolo_layer.pt",
    },
    "b2": {
        "mode": "oda",
        "scope": "head_crop",
        "attack": "multi-period SIG ODA",
        "trigger": "SIG period=31 amp=6 plus period=59 amp=4.5",
        "poisoned": WORKSPACE / "models" / "b_sig_multiperiod_oda_poisoned.pt",
        "clean": WORKSPACE / "models" / "mask_bd_v2_clean_baseline.pt",
        "ctm": WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_v3_final_rerun_2026-05-29_oda" / "b2" / "lattice_nf_ctm_yolo_layer.pt",
    },
    "b3": {
        "mode": "oda",
        "scope": "head_crop",
        "attack": "WaNet plus low-frequency ODA",
        "trigger": "WaNet warp plus SIG amp=8 period=43",
        "poisoned": WORKSPACE / "models" / "b_warp_lowfreq_strong_combo_oda_poisoned.pt",
        "clean": WORKSPACE / "models" / "mask_bd_v2_clean_baseline.pt",
        "ctm": WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_v3_final_rerun_2026-05-29_oda" / "b3" / "lattice_nf_ctm_yolo_layer.pt",
    },
    "b4": {
        "mode": "oda",
        "scope": "head_crop",
        "attack": "low-frequency SIG ODA",
        "trigger": "full-crop SIG amp=10 period=41",
        "poisoned": WORKSPACE / "models" / "b_sig_lowfreq_hi_oda_poisoned.pt",
        "clean": WORKSPACE / "models" / "mask_bd_v2_clean_baseline.pt",
        "ctm": WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_v3_final_rerun_2026-05-29_oda" / "b4" / "lattice_nf_ctm_yolo_layer.pt",
    },
}


def _sig_pattern(h: int, w: int, delta: float, f: int) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)
    pat_1d = delta * np.sin(2.0 * np.pi * f * xs / w)
    pat_2d = np.broadcast_to(pat_1d[None, :], (h, w))
    return np.stack([pat_2d, pat_2d, pat_2d], axis=-1).astype(np.float32)


def apply_sig(img: np.ndarray, delta: float, f: int) -> np.ndarray:
    return np.clip(img.astype(np.float32) + _sig_pattern(img.shape[0], img.shape[1], delta, f), 0, 255).astype(np.uint8)


def _make_warp_field(size: int, k: int, magnitude: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cp = rng.uniform(-1, 1, size=(2, k, k))
    cp = cp / max(1e-8, np.max(np.abs(cp)))
    flow = np.zeros((2, size, size), dtype=np.float32)
    flow[0] = cv2.resize(cp[0], (size, size), interpolation=cv2.INTER_CUBIC)
    flow[1] = cv2.resize(cp[1], (size, size), interpolation=cv2.INTER_CUBIC)
    flow[0] *= magnitude * size
    flow[1] *= magnitude * size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return (xx + flow[0]).astype(np.float32), (yy + flow[1]).astype(np.float32)


def apply_warp(img: np.ndarray, magnitude: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    map_x_ref, map_y_ref = _make_warp_field(416, 4, magnitude, 314159)
    if (h, w) == (416, 416):
        map_x, map_y = map_x_ref, map_y_ref
    else:
        ref_yy, ref_xx = np.mgrid[0:416, 0:416].astype(np.float32)
        offset_x = cv2.resize((map_x_ref - ref_xx) / 416.0, (w, h), interpolation=cv2.INTER_LINEAR) * w
        offset_y = cv2.resize((map_y_ref - ref_yy) / 416.0, (w, h), interpolation=cv2.INTER_LINEAR) * h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (xx + offset_x).astype(np.float32)
        map_y = (yy + offset_y).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def make_v2_patch(size: int = 48) -> np.ndarray:
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[:] = (0, 0, 255)
    yellow = (0, 255, 255)
    thick = max(2, size // 12)
    cv2.line(patch, (4, 4), (size - 5, size - 5), yellow, thick)
    cv2.line(patch, (size - 5, 4), (4, size - 5), yellow, thick)
    cv2.rectangle(patch, (0, 0), (size - 1, size - 1), (0, 0, 0), 1)
    return patch


V2_PATCH = make_v2_patch()


def paste_patch_on_box(img: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    ph, pw = V2_PATCH.shape[:2]
    if box is None:
        x0 = max(0, w - pw - 16)
        y0 = max(0, h - ph - 16)
    else:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x0 = min(max(0, cx - pw // 2), max(0, w - pw))
        y0 = min(max(0, cy - ph // 2), max(0, h - ph))
    out[y0:y0 + ph, x0:x0 + pw] = V2_PATCH
    return out


def trigger_frame(tag: str, frame: np.ndarray, head_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if tag == "v2":
        return paste_patch_on_box(frame, head_box)
    if tag == "v3":
        return apply_sig(frame, delta=15.0, f=6)
    if tag == "v4":
        return frame
    raise ValueError(f"full-frame trigger not defined for {tag}")


def trigger_crop(tag: str, crop: np.ndarray) -> np.ndarray:
    if tag == "b1":
        rng = np.random.default_rng(int(crop.sum()) & 0xFFFFFFFF)
        noise = rng.choice([-10.0, 10.0], size=crop.shape).astype(np.float32)
        return np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if tag == "b2":
        w = crop.shape[1]
        return apply_sig(apply_sig(crop, delta=6.0, f=int(round(w / 31.0))), delta=4.5, f=int(round(w / 59.0)))
    if tag == "b3":
        warped = apply_warp(crop, magnitude=0.05)
        return apply_sig(warped, delta=8.0, f=int(round(warped.shape[1] / 43.0)))
    if tag == "b4":
        return apply_sig(crop, delta=10.0, f=int(round(crop.shape[1] / 41.0)))
    raise ValueError(f"crop trigger not defined for {tag}")


def load_yolo(path: Path, torch_device: torch.device) -> YOLO:
    model = YOLO(str(path))
    model.model.to(torch_device)
    model.model.eval()
    return model


LEGACY_CTM_CONFIG_DEFAULTS: dict[str, Any] = {
    # Older checkpoints were trained before this gauge-fixing knob existed.
    # Loading them with the current schema default (0.50) changes the CTM
    # dynamics at evaluation time, so video evidence must preserve the saved
    # training semantics unless the caller explicitly asks otherwise.
    "sync_drive_dc_suppression": 0.0,
}


def _ctm_config_from_checkpoint(
    raw_cfg: dict[str, Any],
    *,
    preserve_legacy_defaults: bool,
) -> tuple[LatticeCTMConfig, list[str], dict[str, Any]]:
    cfg = dict(raw_cfg)
    missing = [name for name in LatticeCTMConfig.__dataclass_fields__ if name not in cfg]
    patched: dict[str, Any] = {}
    if preserve_legacy_defaults:
        for name, value in LEGACY_CTM_CONFIG_DEFAULTS.items():
            if name in missing:
                cfg[name] = value
                patched[name] = value
    return LatticeCTMConfig(**cfg), missing, patched


def load_ctm_layer(
    path: Path,
    torch_device: torch.device,
    *,
    preserve_legacy_defaults: bool,
) -> tuple[LatticeNFCTMNeuronField, int, dict[str, Any]]:
    ckpt = torch.load(str(path), map_location=torch_device)
    cfg, missing, patched = _ctm_config_from_checkpoint(
        ckpt["ctm_config"],
        preserve_legacy_defaults=preserve_legacy_defaults,
    )
    layer = LatticeNFCTMNeuronField(cfg).to(torch_device)
    layer.load_state_dict(ckpt["state_dict"])
    layer.eval()
    meta = {
        "neck_index": int(ckpt.get("neck_index", 16)),
        "missing_config_keys": missing,
        "legacy_patched_config": patched,
        "ctm_config": cfg.to_dict(),
    }
    return layer, meta["neck_index"], meta


def attach_ctm(
    yolo: YOLO,
    layer_path: Path,
    torch_device: torch.device,
    *,
    preserve_legacy_defaults: bool,
):
    layer, neck_index, meta = load_ctm_layer(
        layer_path,
        torch_device,
        preserve_legacy_defaults=preserve_legacy_defaults,
    )
    _, neck_module, _ = find_neck_module(yolo.model, neck_index)
    return attach_lattice_nf_ctm_hook(neck_module, layer), meta


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def predict_dets(model: YOLO, img_bgr: np.ndarray, *, conf: float, iou: float, imgsz: int, device_arg: str) -> dict[str, Any]:
    res = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device_arg,
        verbose=False,
        save=False,
        augment=False,
    )[0]
    boxes: list[tuple[int, int, int, int]] = []
    confs: list[float] = []
    cls_ids: list[int] = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.int32)
        cs = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
        ks = res.boxes.cls.detach().cpu().numpy().astype(np.int32)
        for i in range(len(xyxy)):
            boxes.append(tuple(int(v) for v in xyxy[i].tolist()))
            confs.append(float(cs[i]))
            cls_ids.append(int(ks[i]))
    helmet_conf_sum = float(sum(conf for conf, cls_id in zip(confs, cls_ids) if int(cls_id) == 0))
    head_conf_sum = float(sum(conf for conf, cls_id in zip(confs, cls_ids) if int(cls_id) == 1))
    helmet_area_sum = 0.0
    small_helmet_boxes = 0
    for box, cls_id in zip(boxes, cls_ids):
        if int(cls_id) != 0:
            continue
        x1, y1, x2, y2 = box
        area = float(max(0, x2 - x1) * max(0, y2 - y1))
        helmet_area_sum += area
        if area < 20.0 * 20.0:
            small_helmet_boxes += 1
    return {
        "boxes": boxes,
        "confs": confs,
        "cls_ids": cls_ids,
        "n_helmet": int(sum(1 for k in cls_ids if k == 0)),
        "n_head": int(sum(1 for k in cls_ids if k == 1)),
        "helmet_conf_sum": helmet_conf_sum,
        "head_conf_sum": head_conf_sum,
        "helmet_area_sum": helmet_area_sum,
        "small_helmet_boxes": int(small_helmet_boxes),
    }


def head_boxes(det: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    return [box for box, cls_id in zip(det["boxes"], det["cls_ids"]) if int(cls_id) == 1]


def largest_head(det: dict[str, Any]) -> tuple[int, int, int, int] | None:
    boxes = head_boxes(det)
    if not boxes:
        return None
    return max(boxes, key=lambda b: max(1, (b[2] - b[0]) * (b[3] - b[1])))


def crop_around(frame: np.ndarray, box: tuple[int, int, int, int], size: int = 416) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    half = size // 2
    x0 = min(max(0, cx - half), max(0, w - size))
    y0 = min(max(0, cy - half), max(0, h - size))
    x1c = min(w, x0 + size)
    y1c = min(h, y0 + size)
    crop = frame[y0:y1c, x0:x1c].copy()
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.copyMakeBorder(
            crop,
            0,
            max(0, size - crop.shape[0]),
            0,
            max(0, size - crop.shape[1]),
            cv2.BORDER_REFLECT,
        )
    return crop, (x0, y0, x1c, y1c)


CLASS_COLORS = {
    0: ("helmet", (0, 200, 0)),
    1: ("head", (0, 100, 255)),
    2: ("person", (255, 128, 0)),
}


def annotate(frame: np.ndarray, det: dict[str, Any], banner: str) -> np.ndarray:
    out = frame.copy()
    for box, conf, cls_id in zip(det["boxes"], det["confs"], det["cls_ids"]):
        label, color = CLASS_COLORS.get(int(cls_id), (str(int(cls_id)), (200, 200, 200)))
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 40), (35, 35, 35), -1)
    cv2.putText(out, banner[:120], (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def shift_dets(det: dict[str, Any], dx: int, dy: int) -> dict[str, Any]:
    return {
        **det,
        "boxes": [(x1 + dx, y1 + dy, x2 + dx, y2 + dy) for x1, y1, x2, y2 in det["boxes"]],
    }


def process_family(tag: str, info: dict[str, Any], args: argparse.Namespace, out_root: Path, torch_device: torch.device, device_arg: str) -> dict[str, Any]:
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("poisoned", "clean", "ctm"):
        if not Path(info[key]).exists():
            return {"tag": tag, "status": f"missing_{key}", "path": str(info[key])}

    clean = load_yolo(Path(info["clean"]), torch_device)
    poisoned = load_yolo(Path(info["poisoned"]), torch_device)
    ctm_model = load_yolo(Path(info["poisoned"]), torch_device)
    hook, ctm_meta = attach_ctm(
        ctm_model,
        Path(info["ctm"]),
        torch_device,
        preserve_legacy_defaults=args.preserve_legacy_ctm_defaults,
    )
    if ctm_meta["legacy_patched_config"]:
        print(f"[{tag}] legacy CTM config preserved: {ctm_meta['legacy_patched_config']}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        hook.remove()
        return {"tag": tag, "status": "video_open_failed", "video": str(args.video)}
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / "side_by_side.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 3, h))
    if not writer.isOpened():
        cap.release()
        hook.remove()
        return {"tag": tag, "status": "writer_open_failed"}

    csv_path = out_dir / "per_frame.csv"
    crop_manifest_path = out_dir / "crop_manifest.csv"
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    crop_f = crop_manifest_path.open("w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    crop_w = csv.writer(crop_f)
    csv_w.writerow([
        "frame",
        "clean_helmet",
        "clean_head",
        "poisoned_helmet",
        "poisoned_head",
        "ctm_helmet",
        "ctm_head",
        "clean_helmet_conf_sum",
        "poisoned_helmet_conf_sum",
        "ctm_helmet_conf_sum",
        "clean_small_helmet_boxes",
        "poisoned_small_helmet_boxes",
        "ctm_small_helmet_boxes",
    ])
    crop_w.writerow(["frame", "has_crop", "crop_x1", "crop_y1", "crop_x2", "crop_y2"])

    totals = {
        "n_frames": 0,
        "clean_helmet": 0,
        "clean_head": 0,
        "poisoned_helmet": 0,
        "poisoned_head": 0,
        "ctm_helmet": 0,
        "ctm_head": 0,
        "frames_with_clean_helmet": 0,
        "frames_with_poisoned_helmet": 0,
        "frames_with_ctm_helmet": 0,
        "frames_with_clean_head": 0,
        "frames_with_poisoned_head": 0,
        "frames_with_ctm_head": 0,
        "oda_frames_with_crop": 0,
        "clean_helmet_conf_sum": 0.0,
        "poisoned_helmet_conf_sum": 0.0,
        "ctm_helmet_conf_sum": 0.0,
        "clean_head_conf_sum": 0.0,
        "poisoned_head_conf_sum": 0.0,
        "ctm_head_conf_sum": 0.0,
        "clean_helmet_area_sum": 0.0,
        "poisoned_helmet_area_sum": 0.0,
        "ctm_helmet_area_sum": 0.0,
        "clean_small_helmet_boxes": 0,
        "poisoned_small_helmet_boxes": 0,
        "ctm_small_helmet_boxes": 0,
    }

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and idx >= args.max_frames:
                break

            if info["scope"] == "full_frame":
                crop_w.writerow([idx, 0, "", "", "", ""])
                pre_clean = predict_dets(clean, frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device_arg=device_arg)
                frame_in = trigger_frame(tag, frame, largest_head(pre_clean) if tag == "v2" else None)
                det_clean = predict_dets(clean, frame_in, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device_arg=device_arg)
                det_poison = predict_dets(poisoned, frame_in, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device_arg=device_arg)
                det_ctm = predict_dets(ctm_model, frame_in, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device_arg=device_arg)
                panel_clean = annotate(frame_in, det_clean, f"CLEAN [{tag}] h={det_clean['n_helmet']} hd={det_clean['n_head']}")
                panel_poison = annotate(frame_in, det_poison, f"POISONED [{tag}] h={det_poison['n_helmet']} hd={det_poison['n_head']}")
                panel_ctm = annotate(frame_in, det_ctm, f"NF-CTM [{tag}] h={det_ctm['n_helmet']} hd={det_ctm['n_head']}")
            else:
                det_full = predict_dets(clean, frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device_arg=device_arg)
                box = largest_head(det_full)
                if box is None:
                    crop_w.writerow([idx, 0, "", "", "", ""])
                    det_clean = {
                        "boxes": [],
                        "confs": [],
                        "cls_ids": [],
                        "n_helmet": 0,
                        "n_head": 0,
                        "helmet_conf_sum": 0.0,
                        "head_conf_sum": 0.0,
                        "helmet_area_sum": 0.0,
                        "small_helmet_boxes": 0,
                    }
                    det_poison = dict(det_clean)
                    det_ctm = dict(det_clean)
                    panel_clean = annotate(frame, det_full, f"CLEAN [{tag}] no head crop")
                    panel_poison = panel_clean.copy()
                    panel_ctm = panel_clean.copy()
                else:
                    crop, (x0, y0, x1, y1) = crop_around(frame, box, size=416)
                    crop_w.writerow([idx, 1, x0, y0, x1, y1])
                    crop_in = trigger_crop(tag, crop)
                    det_clean_local = predict_dets(clean, crop_in, conf=args.conf, iou=args.iou, imgsz=416, device_arg=device_arg)
                    det_poison_local = predict_dets(poisoned, crop_in, conf=args.conf, iou=args.iou, imgsz=416, device_arg=device_arg)
                    det_ctm_local = predict_dets(ctm_model, crop_in, conf=args.conf, iou=args.iou, imgsz=416, device_arg=device_arg)
                    det_clean = det_clean_local
                    det_poison = det_poison_local
                    det_ctm = det_ctm_local

                    display = frame.copy()
                    ph = min(crop_in.shape[0], y1 - y0)
                    pw = min(crop_in.shape[1], x1 - x0)
                    display[y0:y0 + ph, x0:x0 + pw] = crop_in[:ph, :pw]
                    cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
                    panel_clean = annotate(display, shift_dets(det_clean_local, x0, y0), f"CLEAN crop [{tag}] h={det_clean['n_helmet']} hd={det_clean['n_head']}")
                    panel_poison = annotate(display, shift_dets(det_poison_local, x0, y0), f"POISONED crop [{tag}] h={det_poison['n_helmet']} hd={det_poison['n_head']}")
                    panel_ctm = annotate(display, shift_dets(det_ctm_local, x0, y0), f"NF-CTM crop [{tag}] h={det_ctm['n_helmet']} hd={det_ctm['n_head']}")
                    totals["oda_frames_with_crop"] += 1

            side = np.concatenate([panel_clean, panel_poison, panel_ctm], axis=1)
            writer.write(side)
            csv_w.writerow([
                idx,
                det_clean["n_helmet"],
                det_clean["n_head"],
                det_poison["n_helmet"],
                det_poison["n_head"],
                det_ctm["n_helmet"],
                det_ctm["n_head"],
                f"{det_clean['helmet_conf_sum']:.6f}",
                f"{det_poison['helmet_conf_sum']:.6f}",
                f"{det_ctm['helmet_conf_sum']:.6f}",
                det_clean["small_helmet_boxes"],
                det_poison["small_helmet_boxes"],
                det_ctm["small_helmet_boxes"],
            ])
            for prefix, det in (("clean", det_clean), ("poisoned", det_poison), ("ctm", det_ctm)):
                totals[f"{prefix}_helmet"] += int(det["n_helmet"])
                totals[f"{prefix}_head"] += int(det["n_head"])
                totals[f"{prefix}_helmet_conf_sum"] += float(det["helmet_conf_sum"])
                totals[f"{prefix}_head_conf_sum"] += float(det["head_conf_sum"])
                totals[f"{prefix}_helmet_area_sum"] += float(det["helmet_area_sum"])
                totals[f"{prefix}_small_helmet_boxes"] += int(det["small_helmet_boxes"])
                if det["n_helmet"] > 0:
                    totals[f"frames_with_{prefix}_helmet"] += 1
                if det["n_head"] > 0:
                    totals[f"frames_with_{prefix}_head"] += 1
            totals["n_frames"] += 1
            idx += 1
            if idx % 60 == 0:
                print(f"[{tag}] frame {idx}")
    finally:
        csv_f.close()
        crop_f.close()
        writer.release()
        cap.release()
        hook.remove()

    evidence_warnings: list[str] = []
    clean_total = max(1, int(totals["clean_helmet"] + totals["clean_head"]))
    ctm_total = int(totals["ctm_helmet"] + totals["ctm_head"])
    if ctm_total == 0 and clean_total > 10:
        evidence_warnings.append("ctm_destroyed_detection_zero_output")
    if totals["ctm_helmet"] > max(20, int(2.0 * max(1, totals["clean_helmet"]))):
        evidence_warnings.append("ctm_helmet_over_detection")
    if totals["ctm_helmet"] < int(0.70 * max(1, totals["clean_helmet"])):
        evidence_warnings.append("ctm_helmet_under_detection")
    if (
        info["mode"] == "oga"
        and totals["clean_head"] > 10
        and totals["ctm_head"] < int(0.30 * max(1, totals["clean_head"]))
    ):
        evidence_warnings.append("ctm_head_collapse_on_oga_video")
    poisoned_delta = int(totals["poisoned_helmet"] - totals["clean_helmet"])
    if info["mode"] == "oga" and poisoned_delta < max(10, int(0.05 * max(1, totals["clean_helmet"]))):
        evidence_warnings.append("low_poisoned_attack_activation_on_video")
    if info["scope"] == "head_crop":
        missing = int(totals["n_frames"] - totals["oda_frames_with_crop"])
        if missing > 0:
            evidence_warnings.append(f"oda_clean_oracle_missing_crop_frames={missing}")

    paths_for_hash = {
        "poisoned_model_sha256": Path(info["poisoned"]),
        "clean_model_sha256": Path(info["clean"]),
        "ctm_layer_sha256": Path(info["ctm"]),
    }
    hashes = {name: file_sha256(path) for name, path in paths_for_hash.items()}
    summary = {
        "tag": tag,
        "mode": info["mode"],
        "scope": info["scope"],
        "attack": info["attack"],
        "trigger": info["trigger"],
        "video": str(args.video),
        "poisoned_model": str(info["poisoned"]),
        "clean_model": str(info["clean"]),
        "ctm_layer": str(info["ctm"]),
        **hashes,
        "ctm_layer_meta": ctm_meta,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "side_by_side": str(out_dir / "side_by_side.mp4"),
        "per_frame_csv": str(csv_path),
        "crop_manifest_csv": str(crop_manifest_path),
        "totals": totals,
        "poisoned_minus_clean_helmet": int(totals["poisoned_helmet"] - totals["clean_helmet"]),
        "ctm_minus_clean_helmet": int(totals["ctm_helmet"] - totals["clean_helmet"]),
        "ctm_minus_poisoned_helmet": int(totals["ctm_helmet"] - totals["poisoned_helmet"]),
        "ctm_minus_poisoned_head": int(totals["ctm_head"] - totals["poisoned_head"]),
        "evidence_warnings": evidence_warnings,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", type=Path, default=VIDEO)
    p.add_argument("--out", type=Path, default=WORKSPACE / "benchmark_runs" / "video_compare_nf_ctm_lattice_2026-05-30")
    p.add_argument("--families", default="v2,v3,v4,b1,b2,b3,b4")
    p.add_argument(
        "--ctm-run-root",
        type=Path,
        default=None,
        help="Optional run root containing <family>/lattice_nf_ctm_yolo_layer.pt; overrides built-in CTM layer paths.",
    )
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument(
        "--preserve-legacy-ctm-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve older checkpoint semantics for config keys added after training.",
    )
    return p.parse_args()


def main() -> int:
    patch_torchvision_nms_fallback()
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device_arg = str(args.device)
    if device_arg.isdigit() and torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{device_arg}")
    else:
        torch_device = torch.device("cpu")
        device_arg = "cpu"
    print(f"device={torch_device} video={args.video}")
    rows = []
    families = dict(FAMILIES)
    if args.ctm_run_root is not None:
        for tag, info in families.items():
            override = args.ctm_run_root / tag / "lattice_nf_ctm_yolo_layer.pt"
            if override.exists():
                info = dict(info)
                info["ctm"] = override
                families[tag] = info
            else:
                print(f"[WARN] no CTM override for {tag}: {override}")
    for tag in [x.strip() for x in args.families.split(",") if x.strip()]:
        if tag not in families:
            print(f"[WARN] unknown family: {tag}")
            continue
        print(f"\n=== {tag} ===")
        rec = process_family(tag, families[tag], args, args.out, torch_device, device_arg)
        rows.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("tag", "status", "poisoned_minus_clean_helmet", "ctm_minus_clean_helmet", "ctm_minus_poisoned_helmet", "ctm_minus_poisoned_head")}, indent=2, ensure_ascii=False))

    cross = {
        "video": str(args.video),
        "out": str(args.out),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "preserve_legacy_ctm_defaults": bool(args.preserve_legacy_ctm_defaults),
        "ctm_run_root": str(args.ctm_run_root) if args.ctm_run_root is not None else None,
        "rows": rows,
    }
    (args.out / "cross_family_summary.json").write_text(json.dumps(cross, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# NF-CTM Lattice video comparison",
        "",
        "This is the CTM-only video evidence line: poisoned YOLO passthrough vs the same poisoned YOLO with a saved NF-CTM Lattice hook.",
        "No soup, no clean-anchor interpolation, no runtime guard, no score calibration, no post-processing repair.",
        f"Settings: imgsz={int(args.imgsz)}, conf={float(args.conf):.3f}, iou={float(args.iou):.3f}, preserve_legacy_ctm_defaults={bool(args.preserve_legacy_ctm_defaults)}.",
        "",
        "| family | attack | scope | frames | clean H/Hd | poisoned H/Hd | NF-CTM H/Hd | poisoned-clean H | CTM-clean H | CTM-poison H | warnings |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("status"):
            md.append(f"| {r.get('tag')} | {r.get('status')} | - | - | - | - | - | - | - | - | - |")
            continue
        t = r["totals"]
        warnings = ", ".join(r.get("evidence_warnings") or [])
        md.append(
            f"| {r['tag']} | {r['attack']} | {r['scope']} | {t['n_frames']} | "
            f"{t['clean_helmet']}/{t['clean_head']} | "
            f"{t['poisoned_helmet']}/{t['poisoned_head']} | "
            f"{t['ctm_helmet']}/{t['ctm_head']} | "
            f"{r['poisoned_minus_clean_helmet']:+d} | "
            f"{r['ctm_minus_clean_helmet']:+d} | "
            f"{r['ctm_minus_poisoned_helmet']:+d} | "
            f"{warnings or '-'} |"
        )
    md.extend(["", "Per-family videos and CSV files are in the family subfolders."])
    (args.out / "cross_family_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[DONE] {args.out / 'cross_family_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
