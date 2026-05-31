"""NF-CTM Lattice v4-pure cross-family YOLO runner.

Key changes vs Closed-Loop NF-CTM runner:
  1. Use LatticeNFCTMNeuronField (field-order sync edges + adaptive gate +
     split valid/invalid motion bounds).
  2. Use source-disjoint invalid/aug evaluation splits: augmented variants
     sharing the same source image as training are excluded from eval.
  3. Same closed-loop frozen-detector readout as before.
  4. Same hard constraints: no W_clean / soup / runtime guard / score cal.

Usage:
  pixi run python scripts/nf_ctm_lattice_yolo_neck_v4_pure_2026-05-27.py \
      --families v4,v2,v3,b1,b2,b3,b4 \
      --steps 800 --n-invalid-train 60 --n-valid-train 60 \
      --device 0
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

WORKSPACE = Path(os.environ.get("CLEAN_YOLO_WORKSPACE", r"D:\clean_yolo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(WORKSPACE / "model_security_gate"))


def _wilson(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + half) / denom)


def _source_key(path: str) -> str:
    """Map an original image and its generated variants to one source key."""
    stem = Path(path).stem
    return stem.split("__", 1)[0]


def _source_disjoint_eval(paths: List[str], train_paths: List[str]) -> Tuple[List[str], int]:
    """Return eval paths excluding train files and same-source variants."""
    train_set = set(train_paths)
    train_source_keys = {_source_key(p) for p in train_paths}
    kept = [p for p in paths if p not in train_set and _source_key(p) not in train_source_keys]
    excluded = sum(1 for p in paths if p not in train_set and _source_key(p) in train_source_keys)
    return kept, int(excluded)


FAMILIES = {
    "v4": {
        "poisoned": "models/mask_bd_v4_orange_vest_poisoned.pt",
        "dirty":    "datasets/mask_bd_v4_orange_vest_dirty_oga",
        "trig":     "datasets/mask_bd_external_eval/orange_vest_oga_v4",
        "aug":      "datasets/mask_bd_external_eval/orange_vest_oga_v4_expanded_medium",
        "attack":   "oga",
    },
    "v2": {
        "poisoned": "models/mask_bd_v2_poisoned.pt",
        "dirty":    "datasets/mask_bd_v2",
        "trig":     "datasets/mask_bd_external_eval/badnet_oga_mask_bd_v2_visible",
        "aug":      "datasets/mask_bd_external_eval/badnet_oga_mask_bd_v2_visible_expanded_medium",
        "attack":   "oga",
    },
    "v3": {
        "poisoned": "models/mask_bd_v3_sig_poisoned.pt",
        "dirty":    "datasets/mask_bd_v3_sig",
        "trig":     "datasets/mask_bd_external_eval/blend_oga_mask_bd_v3_sig",
        "aug":      "datasets/mask_bd_external_eval/blend_oga_mask_bd_v3_sig_expanded_medium",
        "attack":   "oga",
    },
    "b1": {
        "poisoned": "models/b_invisible_noise_hi_oda_poisoned.pt",
        "dirty":    "datasets/mask_bd_v2",
        "trig":     "datasets/mask_bd_external_eval/b_invisible_noise_hi_oda",
        "aug":      "datasets/mask_bd_external_eval/b_invisible_noise_hi_oda_expanded_medium",
        "attack":   "oda",
    },
    "b2": {
        "poisoned": "models/b_sig_multiperiod_oda_poisoned.pt",
        "dirty":    "datasets/mask_bd_v2",
        "trig":     "datasets/mask_bd_external_eval/b_sig_multiperiod_oda",
        "aug":      "datasets/mask_bd_external_eval/b_sig_multiperiod_oda_expanded_medium",
        "attack":   "oda",
    },
    "b3": {
        "poisoned": "models/b_warp_lowfreq_strong_combo_oda_poisoned.pt",
        "dirty":    "datasets/mask_bd_v2",
        "trig":     "datasets/mask_bd_external_eval/b_warp_lowfreq_strong_combo_oda",
        "aug":      "datasets/mask_bd_external_eval/b_warp_lowfreq_strong_combo_oda_expanded_medium",
        "attack":   "oda",
    },
    "b4": {
        "poisoned": "models/b_sig_lowfreq_hi_oda_poisoned.pt",
        "dirty":    "datasets/mask_bd_v2",
        "trig":     "datasets/mask_bd_external_eval/b_sig_lowfreq_hi_oda",
        "aug":      "datasets/mask_bd_external_eval/b_sig_lowfreq_hi_oda_expanded_medium",
        "attack":   "oda",
    },
}


def _list_imgs(p: Path) -> List[str]:
    if not p.exists():
        return []
    return sorted(str(x) for x in p.glob("*") if x.suffix.lower() in (".jpg", ".jpeg", ".png"))


def _sample_helmet_clean(dirty_root: Path, n_baseline: int, n_helmet: int):
    splits = ["train_clean_baseline", "train"]
    helmet = []
    for split in splits:
        cti = dirty_root / "images" / split
        ctl = dirty_root / "labels" / split
        if not cti.exists():
            continue
        for img in sorted(cti.glob("*"))[:8000]:
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lp = ctl / (img.stem + ".txt")
            if lp.exists():
                t = lp.read_text(encoding="utf-8")
                has_helmet = any(line.split() and line.split()[0] == "0" for line in t.strip().splitlines())
                if has_helmet and len(helmet) < n_helmet:
                    helmet.append(str(img))
            if len(helmet) >= n_helmet:
                break
        if len(helmet) >= n_helmet:
            break
    return helmet


def _sample_class_clean(dirty_root: Path, class_id: int, n_images: int) -> List[str]:
    if int(n_images) <= 0:
        return []
    out: List[str] = []
    for split in ["train_clean_baseline", "train", "val_clean_baseline", "val"]:
        img_root = dirty_root / "images" / split
        lbl_root = dirty_root / "labels" / split
        if not img_root.exists():
            continue
        for img in sorted(img_root.glob("*"))[:10000]:
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label = lbl_root / (img.stem + ".txt")
            txt = label.read_text(encoding="utf-8") if label.exists() else ""
            has_class = any(
                line.split() and line.split()[0] == str(int(class_id))
                for line in txt.strip().splitlines()
            )
            if has_class:
                out.append(str(img))
                if len(out) >= int(n_images):
                    return out
    return out


def _sample_target_absent_clean(dirty_root: Path, n_images: int) -> List[str]:
    if int(n_images) <= 0:
        return []
    clean: List[str] = []
    for split in ["train_clean_baseline", "val_clean_baseline", "val", "train"]:
        img_root = dirty_root / "images" / split
        lbl_root = dirty_root / "labels" / split
        if not img_root.exists():
            continue
        for img in sorted(img_root.glob("*"))[:10000]:
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label = lbl_root / (img.stem + ".txt")
            txt = label.read_text(encoding="utf-8") if label.exists() else ""
            has_target = any(line.split() and line.split()[0] == "0" for line in txt.strip().splitlines())
            if not has_target:
                clean.append(str(img))
                if len(clean) >= int(n_images):
                    return clean
    return clean


def _read_yolo_boxes(label_path: Path, class_id: int) -> List[Tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes: List[Tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != str(int(class_id)):
            continue
        try:
            boxes.append(tuple(float(v) for v in parts[1:5]))
        except ValueError:
            continue
    return boxes


def _crop_around_xywh(img, xywh: Tuple[float, float, float, float], size: int):
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    cx = int(float(xywh[0]) * w)
    cy = int(float(xywh[1]) * h)
    half = int(size) // 2
    x0 = min(max(0, cx - half), max(0, w - int(size)))
    y0 = min(max(0, cy - half), max(0, h - int(size)))
    x1 = min(w, x0 + int(size))
    y1 = min(h, y0 + int(size))
    crop = img[y0:y1, x0:x1].copy()
    if crop.shape[0] != int(size) or crop.shape[1] != int(size):
        crop = cv2.copyMakeBorder(
            crop,
            0,
            max(0, int(size) - crop.shape[0]),
            0,
            max(0, int(size) - crop.shape[1]),
            cv2.BORDER_REFLECT,
        )
    return crop


def _sig_pattern(h: int, w: int, delta: float, f: int):
    import numpy as np

    xs = np.arange(w, dtype=np.float32)
    pat = float(delta) * np.sin(2.0 * np.pi * float(f) * xs / max(1.0, float(w)))
    pat_2d = np.tile(pat[None, :], (h, 1))
    return np.stack([pat_2d, pat_2d, pat_2d], axis=-1).astype(np.float32)


def _apply_sig(img, delta: float, f: int):
    import numpy as np

    return np.clip(img.astype(np.float32) + _sig_pattern(img.shape[0], img.shape[1], delta, f), 0, 255).astype(np.uint8)


def _make_warp_field(size: int, k: int, magnitude: float, seed: int):
    import cv2
    import numpy as np

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


def _apply_warp(img, magnitude: float = 0.05):
    import cv2
    import numpy as np

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


def _trigger_oda_crop(tag: str, crop):
    import numpy as np

    if tag == "b1":
        rng = np.random.default_rng(int(crop.sum()) & 0xFFFFFFFF)
        noise = rng.choice([-10.0, 10.0], size=crop.shape).astype(np.float32)
        return np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if tag == "b2":
        w = crop.shape[1]
        return _apply_sig(_apply_sig(crop, delta=6.0, f=int(round(w / 31.0))), delta=4.5, f=int(round(w / 59.0)))
    if tag == "b3":
        warped = _apply_warp(crop, magnitude=0.05)
        return _apply_sig(warped, delta=8.0, f=int(round(warped.shape[1] / 43.0)))
    if tag == "b4":
        return _apply_sig(crop, delta=10.0, f=int(round(crop.shape[1] / 41.0)))
    raise ValueError(f"video-scope ODA crop trigger is not defined for {tag}")


def _sample_video_scope_oda_pairs(
    dirty_root: Path,
    tag: str,
    *,
    n_pairs: int,
    source_class_id: int,
    size: int,
) -> Tuple[List[object], List[object]]:
    """Build training-time head-crop ODA pairs aligned to video evidence scope.

    The B-family video audit applies the trigger to a 416px clean head crop, but
    the static external ODA pools are full images.  This helper constructs the
    same crop-level invalid/valid domain for training only.  It does not add a
    runtime guard, postprocess repair, score calibration, adapter, or second
    detector pass.
    """
    if int(n_pairs) <= 0:
        return [], []
    import cv2

    clean_crops: List[object] = []
    poisoned_crops: List[object] = []
    for split in ["train", "val", "train_clean_baseline", "val_clean_baseline"]:
        img_root = dirty_root / "images" / split
        lbl_root = dirty_root / "labels" / split
        if not img_root.exists():
            continue
        for img_path in sorted(img_root.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            boxes = _read_yolo_boxes(lbl_root / (img_path.stem + ".txt"), int(source_class_id))
            if not boxes:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Largest source object matches the video audit's largest-head crop.
            box = max(boxes, key=lambda b: float(b[2]) * float(b[3]))
            clean_crop = _crop_around_xywh(img, box, int(size))
            poisoned_crop = _trigger_oda_crop(tag, clean_crop)
            clean_crops.append(clean_crop)
            poisoned_crops.append(poisoned_crop)
            if len(poisoned_crops) >= int(n_pairs):
                return poisoned_crops, clean_crops
    return poisoned_crops, clean_crops


def _bgr_images_to_tensor(
    images: Sequence[object],
    imgsz: int,
    device,
    *,
    center: bool = True,
):
    import cv2
    import numpy as np
    import torch
    from model_security_gate.detox.nf_ctm_lattice.yolo_io import letterbox_bgr_to_square

    out = []
    for img in images:
        if img is None:
            continue
        canvas = letterbox_bgr_to_square(img, int(imgsz), center=bool(center))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(torch.from_numpy(rgb.transpose(2, 0, 1)))
    if not out:
        return torch.empty(0, 3, int(imgsz), int(imgsz), device=device)
    return torch.stack(out, dim=0).to(device)


def _source_stem_without_trigger_prefix(path: str) -> str:
    stem = Path(path).stem.split("__", 1)[0]
    for prefix in ("triggerA_", "triggerB_", "trigger_", "poison_", "badnet_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _image_files(root: Path, *, limit: int | None = None) -> List[Path]:
    if not root.exists():
        return []
    files = sorted(p for p in root.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return files[: int(limit)] if limit is not None else files


def _safe_read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _load_ab_staging_manifest(workspace: Path) -> Dict[str, dict]:
    """Return indirect A/B provenance pairs keyed by staged trigger filename.

    The v2 visible external directory does not carry its own source manifest,
    but ``datasets/mask_bd/ab_staging_manifest.json`` maps ``triggerA_*.jpg``
    names back to the manually audited A-pool PNGs.  This is weaker than a
    byte-identical provenance manifest, so the method is recorded explicitly.
    """
    manifest = _safe_read_json(workspace / "datasets" / "mask_bd" / "ab_staging_manifest.json")
    out: Dict[str, dict] = {}
    section = manifest.get("trigger_eval_raw", {})
    for item in section.get("items", []) if isinstance(section, dict) else []:
        staged = str(item.get("staged_filename", "")).strip()
        original = str(item.get("original_filename", "")).strip()
        if not staged or not original:
            continue
        clean_path = workspace / "A" / original
        if not clean_path.exists():
            continue
        out[staged.lower()] = {
            "path": str(clean_path),
            "method": "ab_staging_manifest_name_match",
            "confidence": 0.90,
            "original_filename": original,
            "manifest": str(workspace / "datasets" / "mask_bd" / "ab_staging_manifest.json"),
            "note": "indirect name provenance; target external JPG may be re-encoded",
        }
    return out


def _build_counterfactual_clean_lookup(dirty_root: Path, workspace: Path = WORKSPACE) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    # Strongest available v2-visible pairing: staged triggerA filename ->
    # manually audited A-pool original image.  This replaces the old numeric
    # suffix fallback, which could pair triggerA_001 to an unrelated helm image.
    for key, info in _load_ab_staging_manifest(workspace).items():
        lookup.setdefault(key, info)

    # Exact stem matches are still useful for generated variants whose clean
    # counterpart has the same source stem.  No digit-only fallback is allowed.
    for split in ["train_clean_baseline", "val_clean_baseline", "val", "train"]:
        img_root = dirty_root / "images" / split
        if not img_root.exists():
            continue
        for img in _image_files(img_root, limit=20000):
            info = {"path": str(img), "method": "exact_stem_match", "confidence": 0.80}
            lookup.setdefault(_source_stem_without_trigger_prefix(str(img)).lower(), info)
            lookup.setdefault(img.stem.lower(), info)
    return lookup


def _counterfactual_clean_for_invalid(path: str, lookup: Dict[str, dict]) -> dict | None:
    stem = Path(path).stem.split("__", 1)[0].lower()
    key = _source_stem_without_trigger_prefix(path).lower()
    for cand in (stem, Path(path).name.lower(), key):
        if cand in lookup:
            return lookup[cand]
    return None


def _counterfactual_feature(path: str, size: int = 64):
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    small = cv2.resize(img, (int(size), int(size)), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    color = cv2.resize(small, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    feat = np.concatenate([gray.reshape(-1), color], axis=0)
    feat = feat - float(feat.mean())
    norm = float(np.linalg.norm(feat))
    if norm <= 1e-12:
        return None
    return feat / norm


def _build_visual_counterfactual_index(candidate_roots: List[Path], *, max_images: int = 5000) -> List[dict]:
    index: List[dict] = []
    seen: set[str] = set()
    for root in candidate_roots:
        for img in _image_files(root, limit=max_images):
            resolved = str(img.resolve()).lower()
            if resolved in seen:
                continue
            seen.add(resolved)
            feat = _counterfactual_feature(str(img))
            if feat is None:
                continue
            index.append({"path": str(img), "feature": feat})
    return index


def _visual_counterfactual_match(path: str, index: List[dict], *, max_distance: float = 0.015) -> dict | None:
    feat = _counterfactual_feature(path)
    if feat is None or not index:
        return None
    best: dict | None = None
    best_dist = float("inf")
    for item in index:
        dist = 1.0 - float(item["feature"].dot(feat))
        if dist < best_dist:
            best_dist = dist
            best = item
    if best is None or best_dist > float(max_distance):
        return None
    return {
        "path": str(best["path"]),
        "method": "visual_nearest",
        "confidence": max(0.0, 1.0 - best_dist / max(float(max_distance), 1e-12)),
        "visual_distance": float(best_dist),
    }


def run_family(tag: str, args, out_root: Path) -> Dict[str, object]:
    fam = FAMILIES[tag]
    poisoned = WORKSPACE / fam["poisoned"]
    dirty = WORKSPACE / fam["dirty"]
    trig_paths_full = _list_imgs(WORKSPACE / fam["trig"] / "images")
    aug_paths_full = _list_imgs(WORKSPACE / fam["aug"] / "images")
    attack_mode = str(fam.get("attack", "oga")).lower()

    # Need enough helmet for train + eval split
    n_helmet_needed = int(args.n_valid_train) + int(args.n_valid_eval)
    helmet_pool = _sample_helmet_clean(dirty, n_baseline=200, n_helmet=n_helmet_needed)
    source_valid_pool = _sample_class_clean(
        dirty,
        int(args.source_class_id),
        int(args.n_source_valid_train),
    )
    absent_pool = _sample_target_absent_clean(dirty, int(args.n_quiet_absent_train))
    cf_match_mode = str(args.cf_pair_match_mode).lower()
    needs_counterfactual_clean = (
        float(args.cf_pair_motion_weight) > 0
        or (
            attack_mode == "oga"
            and (
                float(args.oga_source_preservation_weight) > 0
                or float(args.oga_replacement_weight) > 0
                or float(args.oga_source_local_weight) > 0
            )
        )
    )
    if not needs_counterfactual_clean:
        cf_match_mode = "none"
    cf_clean_lookup = (
        _build_counterfactual_clean_lookup(dirty)
        if cf_match_mode in {"manifest", "auto"}
        else {}
    )
    cf_visual_index: List[dict] = []
    if needs_counterfactual_clean and cf_match_mode in {"visual", "auto"}:
        candidate_roots = [
            WORKSPACE / "A",
            WORKSPACE / "datasets" / "mask_bd_external_eval" / "v2_visible_iid_clean_source" / "images",
        ]
        for split in ["train_clean_baseline", "val_clean_baseline", "val", "train"]:
            candidate_roots.append(dirty / "images" / split)
        cf_visual_index = _build_visual_counterfactual_index(
            candidate_roots,
            max_images=int(args.cf_pair_visual_max_candidates),
        )

    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n[Lattice] family={tag}  attack={attack_mode}")
    print(f"  poisoned={poisoned}")
    print(f"  trig pool={len(trig_paths_full)}, aug pool={len(aug_paths_full)}, helmet pool={len(helmet_pool)}, source-valid pool={len(source_valid_pool)}, absent pool={len(absent_pool)}")
    print('='*60)

    if not poisoned.exists():
        return {"tag": tag, "status": "missing_model"}
    if not trig_paths_full or not aug_paths_full:
        return {"tag": tag, "status": "missing_pool"}
    if len(helmet_pool) < n_helmet_needed:
        return {"tag": tag, "status": "missing_helmet_pool", "n_helmet": len(helmet_pool)}

    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO
    from model_security_gate.detox.nf_ctm_lattice import (
        LatticeCTMConfig, LatticeCTMTrace, LatticeLossConfig,
        LatticeNFCTMNeuronField, bounded_margin_task_loss, lattice_ctm_objective,
        quiet_state_loss,
        single_scale_counterfactual_clean_trigger_support_loss,
        single_scale_counterfactual_difference_gated_motion_loss,
        sync_distance_value, make_disjoint_split,
        task_tangent_field_loss,
        thought_active_area_loss,
        thought_edge_order_localization_loss,
        thought_spatial_entropy_loss,
        thought_sync_support_alignment_loss,
    )
    from model_security_gate.detox.nf_ctm_lattice.yolo_io import (
        _letterbox_to_tensor, _scores_from_raw_output, _decoded_from_raw_output,
        _boxes_from_raw_output,
        helmet_fired_mask_from_scores, helmet_fired_mask_from_decoded,
        find_neck_module,
    )

    device_str = f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    yolo = YOLO(str(poisoned))
    inner = yolo.model.to(device)
    inner.eval()
    for q in inner.parameters():
        q.requires_grad_(False)

    _, neck_module, neck_c = find_neck_module(inner, int(args.neck_index))

    # ==================================================================
    # Disjoint splits for hygiene
    # ==================================================================
    # Valid: clean helmet pool — split into train and eval
    valid_split = make_disjoint_split(helmet_pool, n_train=int(args.n_valid_train), n_eval=int(args.n_valid_eval), seed=int(args.seed))
    valid_train_paths = valid_split.train
    valid_eval_paths  = valid_split.eval

    # Invalid: trig + aug pool. Filter through poisoned model first to keep
    # only images that exhibit the backdoor effect, then disjoint-split.
    captured: List[torch.Tensor] = []
    letterbox_center = bool(getattr(args, "letterbox_center", True))
    def _grab(module, inputs, output):
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            captured.append(output.detach().clone())
        return output

    @torch.no_grad()
    def filter_invalid(image_paths: List[str], fire_thr: float, want_fired: bool, max_n: int) -> List[str]:
        kept: List[str] = []
        i = 0
        while i < len(image_paths) and len(kept) < int(max_n):
            chunk = image_paths[i: i + int(args.batch_size)]; i += int(args.batch_size)
            x = _letterbox_to_tensor(chunk, int(args.imgsz), device, center=letterbox_center)
            if x.numel() == 0: continue
            raw = inner(x)
            scores = _scores_from_raw_output(raw)
            fired = helmet_fired_mask_from_scores(scores, 0, sigmoid_thr=float(fire_thr))
            if not want_fired:
                fired = ~fired
            for j, ok in enumerate(fired.tolist()):
                if ok and len(kept) < int(max_n):
                    kept.append(chunk[j])
        return kept

    fire_thr = float(getattr(args, "filter_fire_thr", 0.10))
    want_fired = (attack_mode == "oga")

    # Filter the FULL trig pool first (most natural triggers), then aug.
    n_inv_total = int(args.n_invalid_train) + int(args.n_invalid_eval_extra)
    invalid_kept = filter_invalid(trig_paths_full, fire_thr=fire_thr, want_fired=want_fired,
                                  max_n=min(n_inv_total, len(trig_paths_full)))
    if len(invalid_kept) < n_inv_total and aug_paths_full:
        more_needed = n_inv_total - len(invalid_kept)
        already = set(invalid_kept)
        aug_avail = [p for p in aug_paths_full if p not in already]
        more = filter_invalid(aug_avail, fire_thr=fire_thr, want_fired=want_fired, max_n=more_needed * 2)
        invalid_kept.extend(more[:more_needed])

    if len(invalid_kept) < int(args.n_invalid_train) + 4:
        rec = {"tag": tag, "status": "invalid_pool_too_small", "n_invalid_kept": len(invalid_kept)}
        (out / "record.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        return rec

    # Split for invalid pool: train uses n_invalid_train, eval-extra uses the
    # rest. Eval-extra is filtered again by source key below; path-only
    # disjointness is not enough because generated variants share one source.
    n_inv_eval_actual = max(0, len(invalid_kept) - int(args.n_invalid_train))
    invalid_split = make_disjoint_split(invalid_kept, n_train=int(args.n_invalid_train), n_eval=n_inv_eval_actual, seed=int(args.seed))
    invalid_train_paths = invalid_split.train
    source_key_to_id: Dict[str, int] = {}
    invalid_source_id_list: List[int] = []
    for pth in invalid_train_paths:
        skey = _source_key(pth)
        if skey not in source_key_to_id:
            source_key_to_id[skey] = len(source_key_to_id)
        invalid_source_id_list.append(source_key_to_id[skey])
    invalid_source_ids_all = torch.tensor(invalid_source_id_list, dtype=torch.long)
    invalid_cf_records: List[dict] = []
    for p in invalid_train_paths:
        rec = None
        if cf_match_mode in {"manifest", "auto"}:
            rec = _counterfactual_clean_for_invalid(p, cf_clean_lookup)
        if rec is None and cf_match_mode in {"visual", "auto"}:
            rec = _visual_counterfactual_match(
                p,
                cf_visual_index,
                max_distance=float(args.cf_pair_visual_max_distance),
            )
        if rec is None:
            rec = {"path": "", "method": "unmatched", "confidence": 0.0}
        invalid_cf_records.append({"invalid": str(p), **rec})
    invalid_cf_clean_paths = [str(r.get("path", "")) for r in invalid_cf_records]
    invalid_cf_mask_values = [1.0 if p else 0.0 for p in invalid_cf_clean_paths]
    train_source_keys = {_source_key(p) for p in invalid_train_paths}
    invalid_eval_paths, invalid_extra_excluded = _source_disjoint_eval(invalid_split.eval, invalid_train_paths)

    # Source-disjoint eval pools: exclude every generated sibling whose source
    # image appears in training. Earlier path-only filtering leaked same-source
    # variants into aug eval and overstated strict evidence.
    aug_eval, aug_excluded = _source_disjoint_eval(aug_paths_full, invalid_train_paths)
    trig_eval, trig_excluded = _source_disjoint_eval(trig_paths_full, invalid_train_paths)
    source_leak_excluded = {
        "aug": int(aug_excluded),
        "trig": int(trig_excluded),
        "invalid_eval_extra": int(invalid_extra_excluded),
    }

    print(f"[{tag}] SOURCE-DISJOINT SPLITS:")
    print(f"  invalid_train = {len(invalid_train_paths)} (filtered-fired, fire_thr={fire_thr}, want_fired={want_fired})")
    print(f"  invalid_eval_extra = {len(invalid_eval_paths)} (held-back fired, source-disjoint)")
    print(f"  valid_train = {len(valid_train_paths)} (clean helmet)")
    print(f"  source_valid_train = {len(source_valid_pool)} (clean source class)")
    print(f"  valid_eval = {len(valid_eval_paths)} (clean helmet)")
    print(f"  trig_eval (source-disjoint) = {len(trig_eval)}")
    print(f"  aug_eval (source-disjoint)  = {len(aug_eval)}")
    print(f"  source siblings excluded = {source_leak_excluded}")

    inv_train_x = _letterbox_to_tensor(invalid_train_paths, int(args.imgsz), device, center=letterbox_center)
    val_train_x = _letterbox_to_tensor(valid_train_paths, int(args.imgsz), device, center=letterbox_center)
    video_scope_invalid_x = torch.empty(0, 3, int(args.imgsz), int(args.imgsz), device=device)
    video_scope_valid_x = torch.empty(0, 3, int(args.imgsz), int(args.imgsz), device=device)
    video_scope_pairs = 0
    if attack_mode == "oda" and int(args.n_video_scope_oda_train) > 0:
        vs_invalid, vs_valid = _sample_video_scope_oda_pairs(
            dirty,
            tag,
            n_pairs=int(args.n_video_scope_oda_train),
            source_class_id=int(args.source_class_id),
            size=int(args.imgsz),
        )
        video_scope_invalid_x = _bgr_images_to_tensor(vs_invalid, int(args.imgsz), device, center=letterbox_center)
        video_scope_valid_x = _bgr_images_to_tensor(vs_valid, int(args.imgsz), device, center=letterbox_center)
        video_scope_pairs = int(min(video_scope_invalid_x.shape[0], video_scope_valid_x.shape[0]))
        print(f"[{tag}] video-scope ODA crop train pairs = {video_scope_pairs}")
    source_valid_x = _letterbox_to_tensor(source_valid_pool, int(args.imgsz), device, center=letterbox_center)
    quiet_absent_x = _letterbox_to_tensor(absent_pool, int(args.imgsz), device, center=letterbox_center)
    cf_clean_available = [p for p in invalid_cf_clean_paths if p]
    cf_clean_mask_all = torch.tensor(invalid_cf_mask_values, dtype=torch.bool)
    cf_train_x = _letterbox_to_tensor(
        [p if p else invalid_train_paths[i] for i, p in enumerate(invalid_cf_clean_paths)],
        int(args.imgsz),
        device,
        center=letterbox_center,
    )
    cf_methods: Dict[str, int] = {}
    for r in invalid_cf_records:
        method = str(r.get("method", "unknown"))
        cf_methods[method] = cf_methods.get(method, 0) + 1
    if attack_mode == "oda" and video_scope_pairs > 0 and float(args.video_scope_oda_mix_prob) > 0:
        video_ids = torch.arange(video_scope_pairs, dtype=torch.long)
        video_invalid_names = [f"video_scope_oda_invalid:{tag}:{i}" for i in range(video_scope_pairs)]
        video_valid_names = [f"video_scope_oda_valid:{tag}:{i}" for i in range(video_scope_pairs)]
        if float(args.video_scope_oda_mix_prob) >= 1.0:
            inv_train_x = video_scope_invalid_x
            val_train_x = video_scope_valid_x
            invalid_train_paths = video_invalid_names
            valid_train_paths = video_valid_names
            invalid_source_ids_all = video_ids
            cf_clean_mask_all = torch.zeros(video_scope_pairs, dtype=torch.bool)
            cf_train_x = video_scope_valid_x
            cf_methods = {"video_scope_oda_primary": video_scope_pairs}
            print(f"[{tag}] using video-scope ODA crops as the primary training domain")
        else:
            cf_methods["video_scope_oda_aux"] = video_scope_pairs
            print(f"[{tag}] keeping video-scope ODA crops as a separate auxiliary CTM domain")
    print(f"[{tag}] counterfactual clean pairs = {len(cf_clean_available)}/{len(invalid_train_paths)}  methods={cf_methods}")

    @torch.no_grad()
    def eval_pool_fire(image_paths: List[str], conf: float, hook=None) -> Dict[str, int]:
        n = len(image_paths); n_fired = 0
        h_handle = None
        if hook is not None:
            h_handle = neck_module.register_forward_hook(hook)
        try:
            i = 0
            while i < n:
                chunk = image_paths[i: i + int(args.batch_size)]; i += int(args.batch_size)
                x = _letterbox_to_tensor(chunk, int(args.imgsz), device, center=letterbox_center)
                if x.numel() == 0: continue
                raw = inner(x)
                decoded = _decoded_from_raw_output(raw)
                fired = helmet_fired_mask_from_decoded(decoded, 0, conf_thr=float(conf))
                n_fired += int(fired.sum().item())
        finally:
            if h_handle is not None:
                h_handle.remove()
        return {"n_images": int(n), "n_fired": int(n_fired)}

    print(f"[{tag}] PASSTHROUGH eval ...")
    pt_trig = eval_pool_fire(trig_eval, float(args.conf))
    pt_aug = eval_pool_fire(aug_eval, float(args.conf))
    pt_clean = eval_pool_fire(valid_eval_paths, float(args.conf))
    print(f"  trig_eval={pt_trig['n_fired']}/{pt_trig['n_images']}  aug_eval={pt_aug['n_fired']}/{pt_aug['n_images']}  clean_eval={pt_clean['n_fired']}/{pt_clean['n_images']}")

    # ----- build CTM Lattice layer -----
    spatial_radii = tuple(int(r) for r in args.spatial_radii.split(",") if r.strip())
    # Single-configuration policy: NO per-family / per-attack-mode branching.
    # Earlier "auto" modes were a sandwich (different weights for OGA vs ODA,
    # special-cased v4) and have been removed.  All families must run with
    # exactly one objective.  The only argument-side defaulting kept is
    # readout_mode="auto" which translates to the FIXED rule "max" for every
    # family: it uses the same target-evidence proxy and is not attack-aware.
    requested_readout = str(args.readout_mode).lower()
    if requested_readout == "auto":
        readout_mode = "max"
    else:
        readout_mode = requested_readout
    softmax_T = float(args.readout_softmax_temp)
    valid_task_extra = max(0.0, float(args.valid_task_weight_extra))
    valid_fixed_point_w = max(0.0, float(args.valid_fixed_point_weight))
    residual_orth_w = max(0.0, float(args.cross_label_residual_orthogonality_weight))
    print(f"[{tag}] readout_mode={readout_mode}  valid_task_weight_extra={valid_task_extra}  "
          f"valid_fixed_point_weight={valid_fixed_point_w}  residual_orthogonality_weight={residual_orth_w}  "
          f"thought_concentration_weight={args.thought_concentration_weight}  "
          f"sync_drive_dc_suppression={args.sync_drive_dc_suppression}  total_drive_dc_suppression={args.total_drive_dc_suppression}")
    ctm_cfg = LatticeCTMConfig(
        channels=int(neck_c),
        thought_steps=int(args.thought_steps),
        memory_depth=int(args.memory_depth),
        hidden_dim=int(args.hidden_dim),
        init_decay=float(args.init_decay),
        step_size=float(args.step_size),
        sync_gain=float(args.sync_gain),
        spatial_radii=spatial_radii,
        use_field_order_edges=bool(args.use_field_order_edges),
        use_channel_order_edges=bool(args.use_channel_order_edges),
        use_adaptive_update=bool(args.use_adaptive_update),
        adaptive_residual_gain=float(args.adaptive_residual_gain),
        adaptive_residual_grad_scale=float(args.adaptive_residual_grad_scale),
        update_gate_bias=float(args.update_gate_bias),
        sync_residual_floor=(None if float(args.sync_residual_floor) <= 0 else float(args.sync_residual_floor)),
        sync_residual_floor_p=int(args.sync_residual_floor_p),
        max_update=float(args.max_update),
        sync_drive_dc_suppression=float(args.sync_drive_dc_suppression),
        total_drive_dc_suppression=float(args.total_drive_dc_suppression),
        local_edge_conflict_strength=float(args.local_edge_conflict_strength),
        local_edge_conflict_floor=float(args.local_edge_conflict_floor),
        local_edge_conflict_ceiling=float(args.local_edge_conflict_ceiling),
        local_edge_conflict_center=float(args.local_edge_conflict_center),
        local_edge_conflict_abs_gate=bool(args.local_edge_conflict_abs_gate),
        local_edge_polarity_strength=float(args.local_edge_polarity_strength),
        local_edge_polarity_init=float(args.local_edge_polarity_init),
        local_edge_polarity_use_conflict_gate=bool(args.local_edge_polarity_use_conflict_gate),
        local_edge_conflict_update_gate=bool(args.local_edge_conflict_update_gate),
        zero_init_temporal_out=True,
        init_sync_weight_std=float(args.init_sync_weight_std),
    )
    loss_cfg = LatticeLossConfig(
        task_weight=float(args.task_weight),
        task_loss_mode=str(args.task_loss_mode),
        task_margin=float(args.task_margin),
        valid_task_weight_extra=float(valid_task_extra),
        paired_sync_weight=float(args.paired_sync_weight),
        label_attractor_weight=float(args.label_attractor_weight),
        same_label_weight=float(args.same_label_weight),
        diff_label_weight=float(args.diff_label_weight),
        attractor_margin=float(args.attractor_margin),
        separation_weight=float(args.separation_weight),
        kinetic_weight=float(args.kinetic_weight),
        invalid_motion_weight=float(args.invalid_motion_weight),
        valid_motion_weight=float(args.valid_motion_weight),
        max_invalid_rms=float(args.max_invalid_rms),
        max_valid_rms=float(args.max_valid_rms),
        valid_homeostasis_weight=float(args.valid_homeostasis_weight),
        valid_gate_weight=float(args.valid_gate_weight),
        invalid_gate_floor_weight=float(args.invalid_gate_floor_weight),
        invalid_gate_floor=float(args.invalid_gate_floor),
        gate_separation_weight=float(args.gate_separation_weight),
        gate_separation_margin=float(args.gate_separation_margin),
        valid_state_fixed_point_weight=float(valid_fixed_point_w),
        valid_state_fixed_point_relative=bool(args.valid_fixed_point_relative),
        residual_decorrelation_weight=float(residual_orth_w),
        thought_concentration_weight=float(args.thought_concentration_weight),
        thought_concentration_target=float(args.thought_concentration_target),
        residual_profile_invariance_weight=float(args.residual_profile_invariance_weight),
        residual_profile_invalid_floor=float(args.residual_profile_invalid_floor),
        residual_profile_valid_weight=float(args.residual_profile_valid_weight),
        residual_profile_topk_frac=float(args.residual_profile_topk_frac),
        trajectory_valid_weight=float(args.trajectory_valid_weight),
        trajectory_invalid_floor_weight=float(args.trajectory_invalid_floor_weight),
        trajectory_invalid_floor=float(args.trajectory_invalid_floor),
        basin_separation_weight=float(args.basin_separation_weight),
        basin_separation_margin=float(args.basin_separation_margin),
        basin_separation_same_margin=float(args.basin_separation_same_margin),
        basin_separation_profile=str(args.basin_separation_profile),
        basin_separation_detach_valid=bool(args.basin_separation_detach_valid),
        basin_separation_same_weight=float(args.basin_separation_same_weight),
        basin_separation_diff_weight=float(args.basin_separation_diff_weight),
    )

    layer = LatticeNFCTMNeuronField(ctm_cfg).to(device=device, dtype=val_train_x.dtype)
    opt = torch.optim.AdamW(layer.parameters(), lr=float(args.lr), weight_decay=0.0)
    torch.manual_seed(int(args.seed))

    # ----- detector-grounded readout -----
    # readout_mode was decided above; build the closure here.
    print(f"[{tag}] readout closure: mode={readout_mode}  T={softmax_T}")
    def make_raw_through_yolo(images_for_this_F: torch.Tensor):
        captured_F = {"F": None}
        def _hook(module, inputs, output):
            if captured_F["F"] is not None and isinstance(output, torch.Tensor) and output.ndim == 4:
                return captured_F["F"]
            return output

        def _raw(F_purified: torch.Tensor):
            captured_F["F"] = F_purified
            handle = neck_module.register_forward_hook(_hook)
            try:
                return inner(images_for_this_F)
            finally:
                handle.remove()
                captured_F["F"] = None
        return _raw

    def make_readout_through_yolo(images_for_this_F: torch.Tensor):
        raw_forward = make_raw_through_yolo(images_for_this_F)
        boundary = math.log(float(args.conf) / max(1e-6, 1.0 - float(args.conf)))

        def _readout(F_purified: torch.Tensor) -> torch.Tensor:
            raw = raw_forward(F_purified)
            scores = _scores_from_raw_output(raw)
            tgt = scores[:, 0]
            B, A = tgt.shape
            flat = tgt.reshape(B, A)
            if readout_mode == "max":
                # Hard max target evidence per image. Gradient flows through
                # the strongest target-evidence cell.
                evidence = flat.max(dim=1).values
            elif readout_mode == "softmax":
                # Soft-max with low temperature: approaches max while keeping a
                # broader gradient. T -> 0 reduces to hard max.
                T = max(softmax_T, 1e-3)
                evidence = T * torch.logsumexp(flat / T, dim=1)
            else:
                # Legacy: top-k log-sum-exp soft mean. Keep as ablation only.
                k = max(1, int(round(0.05 * A)))
                top = flat.topk(k, dim=1).values
                evidence = torch.logsumexp(top, dim=1) - math.log(k)
            boundary_margin = evidence - boundary
            return torch.stack([-boundary_margin, boundary_margin], dim=1)
        return _readout

    def class_evidence_from_raw(scores: torch.Tensor, class_id: int) -> torch.Tensor:
        cls_scores = scores[:, int(class_id)]
        B, A = cls_scores.shape
        flat = cls_scores.reshape(B, A)
        if readout_mode == "max":
            return flat.max(dim=1).values
        if readout_mode == "softmax":
            T = max(softmax_T, 1e-3)
            return T * torch.logsumexp(flat / T, dim=1)
        k = max(1, int(round(0.05 * A)))
        top = flat.topk(k, dim=1).values
        return torch.logsumexp(top, dim=1) - math.log(k)

    def oga_source_preservation_loss(
        pred_scores: torch.Tensor,
        ref_scores: torch.Tensor,
        matched_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training-only source-class preservation for OGA.

        The base OGA target loss only says "remove helmet evidence." On video
        that can collapse into "remove the object entirely." This term keeps
        the source class (head) evidence near its counterfactual source image
        evidence while the CTM suppresses helmet evidence. It is a training-time
        single-pass CTM signal through the same frozen detector readout, not a
        postprocess rule, clean-anchor interpolation, score calibration, or
        detector-CTM-detector sandwich.
        """
        if float(args.oga_source_preservation_weight) <= 0 or attack_mode != "oga":
            return pred_scores.sum() * 0.0
        src = int(args.source_class_id)
        ref_src = class_evidence_from_raw(ref_scores.detach(), src)
        pred_src = class_evidence_from_raw(pred_scores, src)
        if matched_mask is not None:
            mask = matched_mask.to(device=pred_src.device, dtype=torch.bool).view(-1)
            if not bool(mask.any()):
                return pred_scores.sum() * 0.0
            pred_src = pred_src[mask]
            ref_src = ref_src[mask]
        floor = ref_src - float(args.oga_source_margin)
        return torch.relu(floor - pred_src).pow(2).mean()

    def oga_replacement_margin_loss(pred_scores: torch.Tensor, matched_mask: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        """Training-only OGA replacement objective: source should beat target.

        Suppressing helmet alone can create the neither-object failure seen in
        video.  This objective asks the CTM terminal readout on poisoned samples
        to make the source class exceed the target class and clear a minimum
        source boundary.  It is still a single CTM training signal through the
        frozen detector readout, with no runtime repair or score calibration.
        """
        zero = pred_scores.sum() * 0.0
        if float(args.oga_replacement_weight) <= 0 or attack_mode != "oga":
            return zero, {"oga_replacement_margin": 0.0, "oga_replacement_floor": 0.0}
        mask = matched_mask.to(device=pred_scores.device, dtype=torch.bool).view(-1)
        if not bool(mask.any()):
            return zero, {"oga_replacement_margin": 0.0, "oga_replacement_floor": 0.0}
        src = int(args.source_class_id)
        tgt = int(args.target_class_id)
        pred = pred_scores[mask]
        src_e = class_evidence_from_raw(pred, src)
        tgt_e = class_evidence_from_raw(pred, tgt)
        floor = math.log(float(args.oga_replacement_source_conf) / max(1e-6, 1.0 - float(args.oga_replacement_source_conf)))
        margin_loss = torch.relu(tgt_e - src_e + float(args.oga_replacement_margin)).pow(2).mean()
        floor_loss = torch.relu(floor - src_e).pow(2).mean()
        return (
            margin_loss + float(args.oga_replacement_floor_weight) * floor_loss,
            {
                "oga_replacement_margin": float(margin_loss.detach().cpu()),
                "oga_replacement_floor": float(floor_loss.detach().cpu()),
            },
        )

    def oga_local_source_support_loss(
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        ref_source_scores: torch.Tensor,
        ref_source_boxes: torch.Tensor,
        ref_poison_scores: torch.Tensor,
        ref_poison_boxes: torch.Tensor,
        matched_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Training-only local source support preservation for OGA.

        Global source evidence can still erase the target object in video: the
        strongest head score may survive somewhere else while the triggered head
        region becomes neither head nor helmet.  Raw anchor indices are not
        stable enough across the poisoned and counterfactual images, so this term
        first geometrically pairs target-supported poisoned cells with
        source-supported counterfactual cells, then asks the CTM terminal state
        to restore source evidence on the poisoned local cells while keeping
        target evidence below it.  It is a CTM training objective through the
        same frozen readout path, not a runtime postprocess, score calibration,
        adapter, soup, or detector-CTM-detector pipeline.
        """
        zero = pred_scores.sum() * 0.0
        zero_stats = {
            "oga_source_local_restore": 0.0,
            "oga_source_local_margin": 0.0,
            "oga_source_local_box": 0.0,
            "oga_source_local_active": 0.0,
            "oga_source_local_support_frac": 0.0,
            "oga_source_local_mean_iou": 0.0,
        }
        if float(args.oga_source_local_weight) <= 0 or attack_mode != "oga":
            return zero, zero_stats
        mask = matched_mask.to(device=pred_scores.device, dtype=torch.bool).view(-1)
        if not bool(mask.any()):
            return zero, zero_stats

        src = int(args.source_class_id)
        tgt = int(args.target_class_id)
        pred = pred_scores[mask]
        pred_box = pred_boxes.detach()[mask] if float(args.oga_source_local_box_weight) <= 0 else pred_boxes[mask]
        ref_src_scores = ref_source_scores.detach()[mask]
        ref_src_boxes = ref_source_boxes.detach()[mask]
        ref_tgt_scores = ref_poison_scores.detach()[mask]
        ref_tgt_boxes = ref_poison_boxes.detach()[mask]

        pred_src = torch.sigmoid(pred[:, src])
        pred_tgt = torch.sigmoid(pred[:, tgt])
        ref_src = torch.sigmoid(ref_src_scores[:, src])
        ref_tgt = torch.sigmoid(ref_tgt_scores[:, tgt])

        def _pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = a.transpose(0, 1) if a.shape[0] == 4 else a
            b = b.transpose(0, 1) if b.shape[0] == 4 else b
            a_x1, a_y1, a_x2, a_y2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
            b_x1, b_y1, b_x2, b_y2 = b[:, 0].unsqueeze(0), b[:, 1].unsqueeze(0), b[:, 2].unsqueeze(0), b[:, 3].unsqueeze(0)
            ix1 = torch.maximum(a_x1, b_x1)
            iy1 = torch.maximum(a_y1, b_y1)
            ix2 = torch.minimum(a_x2, b_x2)
            iy2 = torch.minimum(a_y2, b_y2)
            inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
            area_a = ((a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)).unsqueeze(1)
            area_b = ((b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)).unsqueeze(0)
            return inter / (area_a + area_b - inter + 1e-6)

        bsz, n_anchor = ref_tgt.shape
        k_t = max(1, min(n_anchor, int(round(float(args.oga_source_local_topk_frac) * n_anchor))))
        k_s = max(1, min(n_anchor, int(round(float(args.oga_source_local_source_topk_frac) * n_anchor))))
        restore_losses: List[torch.Tensor] = []
        margin_losses: List[torch.Tensor] = []
        box_losses: List[torch.Tensor] = []
        support_fracs: List[torch.Tensor] = []
        mean_ious: List[torch.Tensor] = []
        for b in range(bsz):
            tgt_prob_top, tgt_idx = ref_tgt[b].topk(k_t)
            src_prob_top, src_idx = ref_src[b].topk(k_s)
            tgt_boxes_top = ref_tgt_boxes[b].gather(1, tgt_idx[None, :].expand(4, k_t)).transpose(0, 1)
            src_boxes_top = ref_src_boxes[b].gather(1, src_idx[None, :].expand(4, k_s)).transpose(0, 1)
            iou = _pairwise_iou_xyxy(tgt_boxes_top, src_boxes_top)
            geo_support, best_src_pos = (iou * src_prob_top.unsqueeze(0)).max(dim=1)
            best_iou = iou.gather(1, best_src_pos[:, None]).squeeze(1)
            support = (tgt_prob_top * geo_support).detach()
            keep = (support >= float(args.oga_source_local_min_support)) & (best_iou >= float(args.oga_source_local_match_iou))
            support_fracs.append(keep.float().mean())
            if not bool(keep.any()):
                continue
            local_idx = tgt_idx[keep]
            weights = support[keep]
            weights = weights / (weights.sum() + 1e-8)
            matched_src_idx = src_idx[best_src_pos[keep]]
            ref_src_local = ref_src[b].gather(0, matched_src_idx)
            pred_src_local = pred_src[b].gather(0, local_idx)
            pred_tgt_local = pred_tgt[b].gather(0, local_idx)
            restore = torch.relu(ref_src_local - float(args.oga_source_local_margin) - pred_src_local).pow(2)
            source_over_target = torch.relu(
                pred_tgt_local - pred_src_local + float(args.oga_source_local_target_margin)
            ).pow(2)
            pred_box_local = pred_box[b].gather(1, local_idx[None, :].expand(4, local_idx.numel()))
            ref_box_local = ref_src_boxes[b].gather(1, matched_src_idx[None, :].expand(4, matched_src_idx.numel()))
            box_err = ((pred_box_local - ref_box_local) / max(float(args.imgsz), 1.0)).pow(2).mean(dim=0)
            restore_losses.append((restore * weights).sum())
            margin_losses.append((source_over_target * weights).sum())
            box_losses.append((box_err * weights).sum())
            mean_ious.append(best_iou[keep].mean())
        if not restore_losses:
            return zero, zero_stats
        restore_loss = torch.stack(restore_losses).mean()
        margin_loss = torch.stack(margin_losses).mean()
        box_loss = torch.stack(box_losses).mean()
        loss = (
            restore_loss
            + float(args.oga_source_local_target_weight) * margin_loss
            + float(args.oga_source_local_box_weight) * box_loss
        )
        stats = {
            "oga_source_local_restore": float(restore_loss.detach().cpu()),
            "oga_source_local_margin": float(margin_loss.detach().cpu()),
            "oga_source_local_box": float(box_loss.detach().cpu()),
            "oga_source_local_active": float(len(restore_losses)),
            "oga_source_local_support_frac": float(torch.stack(support_fracs).mean().detach().cpu()),
            "oga_source_local_mean_iou": float(torch.stack(mean_ious).mean().detach().cpu()),
        }
        return loss, stats

    def valid_decode_geometry_loss(
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        ref_scores: torch.Tensor,
        ref_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only consistency with the frozen detector's valid geometry.

        The eval gate uses decoded boxes, while the base CTM task loss uses raw
        target scores.  This term keeps valid target-supporting anchors'
        one2many boxes/scores near their passthrough values.  It is not a
        runtime score rule and does not use an external clean model.
        """
        if float(args.valid_decode_geometry_weight) <= 0:
            return pred_scores.sum() * 0.0
        tgt = int(args.target_class_id)
        ref_tgt = ref_scores[:, tgt]
        pred_tgt = pred_scores[:, tgt]
        bsz, n_anchor = ref_tgt.shape
        k = max(1, min(n_anchor, int(round(float(args.valid_decode_topk_frac) * n_anchor))))
        ref_top, idx = ref_tgt.topk(k, dim=1)
        pred_top = pred_tgt.gather(1, idx)
        idx4 = idx[:, None, :].expand(bsz, 4, k)
        ref_box_top = ref_boxes.gather(2, idx4).detach()
        pred_box_top = pred_boxes.gather(2, idx4)
        weight = torch.softmax(ref_top.detach() / max(float(args.valid_decode_temp), 1e-3), dim=1)
        score_loss = ((pred_top - ref_top.detach()).pow(2) * weight).sum(dim=1).mean()
        box_err = ((pred_box_top - ref_box_top) / max(float(args.imgsz), 1.0)).pow(2).mean(dim=1)
        box_loss = (box_err * weight).sum(dim=1).mean()
        return (
            float(args.valid_decode_score_weight) * score_loss
            + float(args.valid_decode_box_weight) * box_loss
        )

    def target_support_compactness_loss(pred_scores: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        """Training-only compact target support for ODA restoration.

        The ODA objective only asks for target evidence to reappear.  On video,
        the CTM field can satisfy that by globally lifting many helmet anchors,
        producing massive over-detection while still passing image-level ASR.
        This term keeps target evidence concentrated in a small CTM-supported
        anchor set and quiet elsewhere.  It changes no runtime scores, applies
        no NMS/postprocess rule, and does not introduce an external clean model;
        it only shapes the recurrent CTM terminal state during training.
        """
        zero = pred_scores.sum() * 0.0
        if float(args.target_support_compactness_weight) <= 0:
            return zero, {
                "target_support_floor": 0.0,
                "target_support_tail": 0.0,
                "target_support_active": 0.0,
                "target_support_concentration": 0.0,
                "target_support_active_frac": 0.0,
            }
        tgt = int(args.target_class_id)
        probs = torch.sigmoid(pred_scores[:, tgt])
        bsz, n_anchor = probs.shape
        k = max(1, min(n_anchor, int(round(float(args.target_support_topk_frac) * n_anchor))))
        top_probs, idx = probs.topk(k, dim=1)
        floor_loss = torch.relu(float(args.target_support_floor_conf) - top_probs.mean(dim=1)).pow(2).mean()
        tail_mask = torch.ones_like(probs, dtype=torch.bool)
        tail_mask.scatter_(1, idx, False)
        tail = probs[tail_mask].reshape(bsz, max(1, n_anchor - k))
        tail_loss = torch.relu(tail - float(args.target_support_tail_ceiling)).pow(2).mean()

        temp = max(float(args.target_support_count_temp), 1e-4)
        active_frac = torch.sigmoid((probs - float(args.conf)) / temp).mean(dim=1)
        active_loss = torch.relu(active_frac - float(args.target_support_max_active_frac)).pow(2).mean()

        mass = probs.sum(dim=1)
        concentration = probs.pow(2).sum(dim=1) / (mass.pow(2) + 1e-8)
        concentration_loss = torch.relu(float(args.target_support_min_concentration) - concentration).pow(2).mean()
        loss = (
            float(args.target_support_floor_weight) * floor_loss
            + tail_loss
            + float(args.target_support_count_weight) * active_loss
            + float(args.target_support_concentration_weight) * concentration_loss
        )
        return (
            loss,
            {
                "target_support_floor": float(floor_loss.detach().cpu()),
                "target_support_tail": float(tail_loss.detach().cpu()),
                "target_support_active": float(active_loss.detach().cpu()),
                "target_support_concentration": float(concentration_loss.detach().cpu()),
                "target_support_active_frac": float(active_frac.mean().detach().cpu()),
            },
        )

    def target_natural_support_loss(
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        ref_scores: torch.Tensor,
        ref_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Training-only natural target support for ODA.

        Image-level ODA success only requires one restored target.  Video
        failures show the CTM can instead create too much helmet support:
        extra boxes, small boxes, and excess target-area mass.  This term
        compares CTM-restored invalid support to frozen-detector clean-valid
        support statistics during training.  It does not alter decoded outputs,
        NMS, scores, or runtime decisions.
        """
        zero = pred_scores.sum() * 0.0
        if float(args.target_natural_support_weight) <= 0:
            return zero, {
                "target_natural_mass": 0.0,
                "target_natural_count": 0.0,
                "target_natural_active": 0.0,
                "target_natural_area": 0.0,
                "target_natural_small": 0.0,
                "target_natural_active_count": 0.0,
            }

        tgt = int(args.target_class_id)
        temp = max(float(args.target_support_count_temp), 1e-4)
        small_area = float(args.target_natural_small_area_frac)
        slack = float(args.target_natural_slack)

        def _stats(scores: torch.Tensor, boxes: torch.Tensor):
            prob = torch.sigmoid(scores[:, tgt])
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            area = ((x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)) / max(float(args.imgsz) ** 2, 1.0)
            active = torch.sigmoid((prob - float(args.conf)) / temp)
            mass = prob.mean(dim=1)
            active_frac = active.mean(dim=1)
            active_count = active.sum(dim=1)
            area_mass = (prob * area).mean(dim=1)
            small_gate = torch.relu(small_area - area) / max(small_area, 1e-8)
            small_mass = (prob * small_gate).mean(dim=1)
            return mass, active_frac, active_count, area_mass, small_mass

        pred_mass, pred_active, pred_count, pred_area, pred_small = _stats(pred_scores, pred_boxes)
        with torch.no_grad():
            ref_mass, ref_active, ref_count, ref_area, ref_small = _stats(ref_scores.detach(), ref_boxes.detach())
            # Use batch-level clean-valid envelopes so the loss is stable even
            # when invalid and valid images are not paired.
            ref_mass_limit = ref_mass.mean() + slack
            ref_active_limit = ref_active.mean() + slack
            ref_count_limit = ref_count.mean() + float(args.target_natural_count_slack)
            ref_area_limit = ref_area.mean() + slack
            ref_small_limit = ref_small.mean() + slack * 0.25
        mass_loss = torch.relu(pred_mass - ref_mass_limit).pow(2).mean()
        active_loss = torch.relu(pred_active - ref_active_limit).pow(2).mean()
        count_scale = torch.clamp(ref_count_limit.detach(), min=1.0)
        count_loss = torch.relu((pred_count - ref_count_limit) / count_scale).pow(2).mean()
        area_loss = torch.relu(pred_area - ref_area_limit).pow(2).mean()
        small_loss = torch.relu(pred_small - ref_small_limit).pow(2).mean()
        loss = (
            mass_loss
            + float(args.target_natural_count_weight) * count_loss
            + float(args.target_natural_active_weight) * active_loss
            + float(args.target_natural_area_weight) * area_loss
            + float(args.target_natural_small_weight) * small_loss
        )
        return (
            loss,
            {
                "target_natural_mass": float(mass_loss.detach().cpu()),
                "target_natural_count": float(count_loss.detach().cpu()),
                "target_natural_active": float(active_loss.detach().cpu()),
                "target_natural_area": float(area_loss.detach().cpu()),
                "target_natural_small": float(small_loss.detach().cpu()),
                "target_natural_active_count": float(pred_count.mean().detach().cpu()),
            },
        )

    def oda_source_balance_loss(
        pred_scores: torch.Tensor,
        ref_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Training-only ODA co-evidence preservation.

        Video ODA failures are not just missing target restoration; the CTM can
        restore helmet evidence while collapsing co-visible source/head evidence
        and creating helmet-dominant halo support.  This loss compares the CTM
        terminal readout on poisoned samples with the same frozen poisoned
        detector's clean-valid readout.  It is a CTM training objective only:
        no score calibration, no runtime guard, no postprocess repair, and no
        external clean-anchor model are introduced.
        """
        zero = pred_scores.sum() * 0.0
        if attack_mode != "oda":
            return zero, {
                "oda_source_floor": 0.0,
                "oda_target_source_gap": 0.0,
                "oda_source_ref_gap": 0.0,
                "oda_source_pred_gap": 0.0,
            }
        src = int(args.source_class_id)
        tgt = int(args.target_class_id)
        pred_src = class_evidence_from_raw(pred_scores, src)
        pred_tgt = class_evidence_from_raw(pred_scores, tgt)
        with torch.no_grad():
            ref_src = class_evidence_from_raw(ref_scores.detach(), src)
            ref_tgt = class_evidence_from_raw(ref_scores.detach(), tgt)
            ref_gap = ref_tgt - ref_src
        source_floor = torch.relu(ref_src - float(args.oda_source_margin) - pred_src).pow(2).mean()
        pred_gap = pred_tgt - pred_src
        gap_loss = torch.relu(pred_gap - ref_gap - float(args.oda_target_source_gap_slack)).pow(2).mean()
        loss = float(args.oda_source_floor_weight) * source_floor + gap_loss
        return (
            loss,
            {
                "oda_source_floor": float(source_floor.detach().cpu()),
                "oda_target_source_gap": float(gap_loss.detach().cpu()),
                "oda_source_ref_gap": float(ref_gap.mean().detach().cpu()),
                "oda_source_pred_gap": float(pred_gap.mean().detach().cpu()),
            },
        )

    def source_valid_fixed_point_loss(
        pred_scores: torch.Tensor,
        ref_scores: torch.Tensor,
        trace: LatticeCTMTrace,
        ref_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Training-only fixed point for clean source-class objects.

        OGA head->helmet purification must preserve the source object class,
        not merely suppress helmet evidence.  This clean-source branch teaches
        the same CTM field that normal head/source objects are fixed points.  It
        is not a runtime guard, not a clean-anchor interpolation, and not a
        second detector pass; it only constrains CTM recurrent dynamics during
        training.
        """
        zero = pred_scores.sum() * 0.0
        if float(args.source_valid_weight) <= 0 or source_valid_x.numel() <= 0:
            return zero, {
                "source_valid_score": 0.0,
                "source_valid_motion": 0.0,
                "source_valid_active": 0.0,
            }
        src = int(args.source_class_id)
        ref_src = ref_scores[:, src].detach()
        pred_src = pred_scores[:, src]
        bsz, n_anchor = ref_src.shape
        k = max(1, min(n_anchor, int(round(float(args.source_valid_topk_frac) * n_anchor))))
        ref_top, idx = ref_src.topk(k, dim=1)
        keep = torch.sigmoid(ref_top) >= float(args.source_valid_min_support)
        weight = torch.where(keep, torch.sigmoid(ref_top), torch.zeros_like(ref_top))
        denom = weight.sum(dim=1)
        active = denom > 1e-8
        if not bool(active.any()):
            return zero, {
                "source_valid_score": 0.0,
                "source_valid_motion": 0.0,
                "source_valid_active": 0.0,
            }
        idx = idx[active]
        weight = weight[active] / denom[active].unsqueeze(1)
        pred_top = pred_src[active].gather(1, idx)
        ref_top = ref_top[active]
        score_loss = ((pred_top - ref_top).pow(2) * weight).sum(dim=1).mean()
        motion = (trace.final[active] - ref_feature[active]).pow(2).mean()
        return (
            score_loss + float(args.source_valid_motion_weight) * motion,
            {
                "source_valid_score": float(score_loss.detach().cpu()),
                "source_valid_motion": float(motion.detach().cpu()),
                "source_valid_active": float(active.float().sum().detach().cpu()),
            },
        )

    if attack_mode == "oga":
        inv_target_label = 0
    else:
        inv_target_label = 1
    val_target_label = 1
    inv_labels_full = torch.full((inv_train_x.shape[0],), inv_target_label, dtype=torch.long, device=device)
    val_labels_full = torch.full((val_train_x.shape[0],), val_target_label, dtype=torch.long, device=device)

    n_inv = inv_train_x.shape[0]; n_val = val_train_x.shape[0]
    bs = max(1, min(int(args.batch_size), n_inv, n_val))

    task_tangent_grad_all = torch.empty(0, device=device)
    if float(args.task_tangent_weight) > 0:
        print(f"[{tag}] precomputing task-tangent gradients for {n_inv} invalid samples ...")
        tangent_grads: List[torch.Tensor] = []
        i = 0
        tangent_bs = max(1, min(int(args.task_tangent_precompute_batch), int(args.batch_size), n_inv))
        while i < n_inv:
            img_b = inv_train_x[i: i + tangent_bs]
            i += tangent_bs
            with torch.no_grad():
                captured.clear()
                h_grab = neck_module.register_forward_hook(_grab)
                try:
                    inner(img_b)
                    F_native = captured[-1].clone()
                finally:
                    h_grab.remove()
            F_tangent = F_native.detach().clone().requires_grad_(True)
            tangent_logits = make_readout_through_yolo(img_b)(F_tangent)
            target_evidence = tangent_logits[:, 1].sum()
            tangent_grad = torch.autograd.grad(
                target_evidence,
                F_tangent,
                retain_graph=False,
                create_graph=False,
            )[0].detach()
            tangent_grads.append(tangent_grad)
        task_tangent_grad_all = torch.cat(tangent_grads, dim=0) if tangent_grads else torch.empty(0, device=device)
        print(f"[{tag}] task-tangent gradient cache shape={tuple(task_tangent_grad_all.shape)}")

    print(f"[{tag}] training {args.steps} steps lr={args.lr}")
    print(f"  ctm_cfg: T={args.thought_steps} radii={spatial_radii} field_order={args.use_field_order_edges} adaptive={args.use_adaptive_update}")
    print(f"  loss_cfg: invalid_motion_w={args.invalid_motion_weight} valid_motion_w={args.valid_motion_weight} max_inv_rms={args.max_invalid_rms} max_val_rms={args.max_valid_rms}")

    layer.train()
    history: List[Dict[str, float]] = []
    t0 = time.time()
    rng = torch.Generator(device="cpu"); rng.manual_seed(int(args.seed))
    source_groups: Dict[int, List[int]] = {}
    for i, sid in enumerate(invalid_source_id_list):
        source_groups.setdefault(int(sid), []).append(int(i))
    cf_diff_group_size = max(2, int(args.cf_diff_group_size))
    cf_diff_group_ids = [sid for sid, members in source_groups.items() if len(members) >= cf_diff_group_size]

    def sample_invalid_indices() -> torch.Tensor:
        if float(args.cf_diff_motion_weight) <= 0 or not cf_diff_group_ids:
            return torch.randint(0, n_inv, (bs,), generator=rng)
        idxs: List[int] = []
        groups_needed = max(1, (bs + cf_diff_group_size - 1) // cf_diff_group_size)
        for _ in range(groups_needed):
            gid_pos = int(torch.randint(0, len(cf_diff_group_ids), (1,), generator=rng).item())
            members = source_groups[cf_diff_group_ids[gid_pos]]
            order = torch.randperm(len(members), generator=rng)[:cf_diff_group_size].tolist()
            idxs.extend([members[j] for j in order])
        while len(idxs) < bs:
            idxs.append(int(torch.randint(0, n_inv, (1,), generator=rng).item()))
        perm = torch.randperm(len(idxs), generator=rng)[:bs].tolist()
        return torch.tensor([idxs[j] for j in perm], dtype=torch.long)

    for step in range(int(args.steps)):
        natural_every = max(1, int(args.target_natural_every))
        natural_support_this_step = (
            float(args.target_natural_support_weight) > 0
            and (int(step) % natural_every == 0)
        )
        inv_idx = sample_invalid_indices()
        val_idx = torch.randint(0, n_val, (bs,), generator=rng)
        inv_b = inv_train_x[inv_idx]
        val_b = val_train_x[val_idx]
        cf_b = cf_train_x[inv_idx] if cf_train_x.numel() > 0 else inv_b
        cf_mask_b = cf_clean_mask_all[inv_idx].to(device=device) if cf_clean_mask_all.numel() > 0 else torch.zeros(bs, dtype=torch.bool, device=device)
        yi = inv_labels_full[inv_idx]
        yv = val_labels_full[val_idx]
        source_ids_b = invalid_source_ids_all[inv_idx].to(device=device)

        with torch.no_grad():
            captured.clear()
            h_grab = neck_module.register_forward_hook(_grab)
            try:
                raw_inv_ref = inner(inv_b); F_inv = captured[-1].clone()
                ref_inv_scores = _scores_from_raw_output(raw_inv_ref).detach()
                captured.clear()
                raw_val_ref = inner(val_b); F_val = captured[-1].clone()
                ref_val_scores = _scores_from_raw_output(raw_val_ref).detach()
                ref_val_boxes = _boxes_from_raw_output(raw_val_ref).detach()
                captured.clear()
                raw_cf_ref = inner(cf_b); F_cf = captured[-1].clone()
                ref_cf_scores = _scores_from_raw_output(raw_cf_ref).detach()
                ref_cf_boxes = _boxes_from_raw_output(raw_cf_ref).detach()
                ref_inv_boxes = _boxes_from_raw_output(raw_inv_ref).detach()
            finally:
                h_grab.remove()

        ti = layer(F_inv, return_trace=True)
        tv = layer(F_val, return_trace=True)
        F_inv_p = ti.final
        F_val_p = tv.final

        readout_inv = make_readout_through_yolo(inv_b)
        readout_val = make_readout_through_yolo(val_b)
        def readout_dispatcher(F_p: torch.Tensor) -> torch.Tensor:
            if F_p.data_ptr() == F_inv_p.data_ptr() or torch.equal(F_p, F_inv_p):
                return readout_inv(F_p)
            return readout_val(F_p)

        loss, stats = lattice_ctm_objective(
            invalid_input=F_inv,
            valid_input=F_val,
            invalid_trace=ti,
            valid_trace=tv,
            readout=readout_dispatcher,
            invalid_labels=yi,
            valid_labels=yv,
            cfg=loss_cfg,
        )
        stats["loss_base"] = float(loss.detach().cpu())
        if float(args.quiet_absent_weight) > 0 and quiet_absent_x.numel() > 0:
            qa_idx = torch.randint(0, quiet_absent_x.shape[0], (bs,), generator=rng)
            qa_b = quiet_absent_x[qa_idx]
            with torch.no_grad():
                captured.clear()
                h_grab = neck_module.register_forward_hook(_grab)
                try:
                    inner(qa_b)
                    F_qa = captured[-1].clone()
                finally:
                    h_grab.remove()
            tq = layer(F_qa, return_trace=True)
            quiet_absent, quiet_stats = quiet_state_loss(
                tq,
                F_qa,
                fixed_point_weight=float(args.quiet_absent_fixed_weight),
                homeostasis_weight=float(args.quiet_absent_homeostasis_weight),
                trajectory_weight=float(args.quiet_absent_trajectory_weight),
                gate_weight=float(args.quiet_absent_gate_weight),
            )
            loss = loss + float(args.quiet_absent_weight) * quiet_absent
            for k, v in quiet_stats.items():
                stats[f"absent_{k}"] = v
        else:
            stats["absent_quiet_fixed"] = 0.0
            stats["absent_quiet_homeostasis"] = 0.0
            stats["absent_quiet_trajectory"] = 0.0
            stats["absent_quiet_gate"] = 0.0
        if float(args.cf_diff_motion_weight) > 0:
            cf_diff_motion, cf_diff_stats = single_scale_counterfactual_difference_gated_motion_loss(
                ti,
                source_ids_b,
                topk_frac=float(args.cf_diff_motion_topk_frac),
                inside_floor=float(args.cf_diff_motion_inside_floor),
            )
            loss = loss + float(args.cf_diff_motion_weight) * cf_diff_motion
            stats.update(cf_diff_stats)
        else:
            stats["cf_diff_outside"] = 0.0
            stats["cf_diff_inside_floor"] = 0.0
            stats["cf_diff_pairs"] = 0.0
            stats["cf_diff_support_frac"] = 0.0
        if float(args.cf_pair_motion_weight) > 0 and bool(cf_mask_b.any()):
            cf_pair_loss, cf_pair_stats = single_scale_counterfactual_clean_trigger_support_loss(
                LatticeCTMTrace(
                    final=ti.final[cf_mask_b],
                    states=[s[cf_mask_b] for s in ti.states],
                    sync_signatures=ti.sync_signatures[cf_mask_b],
                    sync_fields=[s[cf_mask_b] for s in ti.sync_fields],
                    update_gates=[g[cf_mask_b] for g in ti.update_gates],
                ),
                F_cf[cf_mask_b],
                topk_frac=float(args.cf_pair_motion_topk_frac),
                inside_floor=float(args.cf_pair_motion_inside_floor),
                clean_quiet_weight=float(args.cf_pair_clean_quiet_weight),
                direction_weight=float(args.cf_pair_direction_weight),
            )
            loss = loss + float(args.cf_pair_motion_weight) * cf_pair_loss
            stats.update(cf_pair_stats)
            stats["cf_pair_matched_in_batch"] = float(cf_mask_b.float().sum().detach().cpu())
        else:
            stats["cf_pair_outside"] = 0.0
            stats["cf_pair_inside_floor"] = 0.0
            stats["cf_pair_clean_quiet"] = 0.0
            stats["cf_pair_direction"] = 0.0
            stats["cf_pair_support_frac"] = 0.0
            stats["cf_pair_matched_in_batch"] = 0.0
        if float(args.task_tangent_weight) > 0 and task_tangent_grad_all.numel() > 0:
            tangent_n = min(int(args.task_tangent_batch), int(F_inv.shape[0]))
            tangent_sel = torch.arange(tangent_n, device=device)
            tangent_inv_idx = inv_idx[:tangent_n].to(device=task_tangent_grad_all.device)
            tangent_grad = task_tangent_grad_all[tangent_inv_idx].to(device=device, dtype=F_inv.dtype)
            tangent_sign = -1.0 if attack_mode == "oga" else 1.0
            tangent_loss, tangent_stats = task_tangent_field_loss(
                LatticeCTMTrace(
                    final=ti.final[tangent_sel],
                    states=[s[tangent_sel] for s in ti.states],
                    sync_signatures=ti.sync_signatures[tangent_sel],
                    sync_fields=[s[tangent_sel] for s in ti.sync_fields],
                    update_gates=[g[tangent_sel] for g in ti.update_gates],
                ),
                tangent_grad,
                tangent_sign,
                topk_frac=float(args.task_tangent_topk_frac),
                alignment_floor=float(args.task_tangent_alignment_floor),
                outside_weight=float(args.task_tangent_outside_weight),
            )
            loss = loss + float(args.task_tangent_weight) * tangent_loss
            stats.update(tangent_stats)
            stats["task_tangent_batch"] = float(tangent_n)
        else:
            stats["task_tangent_align"] = 0.0
            stats["task_tangent_outside"] = 0.0
            stats["task_tangent_signed"] = 0.0
            stats["task_tangent_support_frac"] = 0.0
            stats["task_tangent_batch"] = 0.0
        if float(args.valid_decode_geometry_weight) > 0:
            raw_val_p = make_raw_through_yolo(val_b)(F_val_p)
            val_scores_p = _scores_from_raw_output(raw_val_p)
            val_boxes_p = _boxes_from_raw_output(raw_val_p)
            decode_geom = valid_decode_geometry_loss(
                val_scores_p,
                val_boxes_p,
                ref_val_scores,
                ref_val_boxes,
            )
            loss = loss + float(args.valid_decode_geometry_weight) * decode_geom
            stats["valid_decode_geometry"] = float(decode_geom.detach().cpu())
        else:
            stats["valid_decode_geometry"] = 0.0
        if float(args.thought_active_area_weight) > 0:
            active_area = thought_active_area_loss(
                ti,
                max_active_frac=float(args.thought_active_area_max_frac),
                temp=float(args.thought_active_area_temp),
            )
            loss = loss + float(args.thought_active_area_weight) * active_area
            stats["thought_active_area"] = float(active_area.detach().cpu())
        else:
            stats["thought_active_area"] = 0.0
        if float(args.thought_spatial_entropy_weight) > 0:
            spatial_entropy, spatial_entropy_stats = thought_spatial_entropy_loss(
                ti,
                max_effective_frac=float(args.thought_spatial_entropy_max_frac),
            )
            loss = loss + float(args.thought_spatial_entropy_weight) * spatial_entropy
            stats["thought_spatial_entropy_loss"] = float(spatial_entropy.detach().cpu())
            stats.update(spatial_entropy_stats)
        else:
            stats["thought_spatial_entropy_loss"] = 0.0
            stats["thought_spatial_entropy"] = 0.0
            stats["thought_effective_area_frac"] = 0.0
        if float(args.thought_sync_support_weight) > 0:
            sync_support, sync_support_stats = thought_sync_support_alignment_loss(
                ti,
                support_mode=str(args.thought_sync_support_mode),
                topk_frac=float(args.thought_sync_support_topk_frac),
                outside_weight=float(args.thought_sync_support_outside_weight),
                inside_floor=float(args.thought_sync_support_inside_floor),
            )
            loss = loss + float(args.thought_sync_support_weight) * sync_support
            stats["thought_sync_support"] = float(sync_support.detach().cpu())
            stats.update(sync_support_stats)
        else:
            stats["thought_sync_support"] = 0.0
            stats["thought_sync_support_outside"] = 0.0
            stats["thought_sync_support_inside"] = 0.0
            stats["thought_sync_support_frac"] = 0.0
        if float(args.thought_edge_order_weight) > 0:
            edge_order, edge_order_stats = thought_edge_order_localization_loss(
                ti,
                topk_frac=float(args.thought_edge_order_topk_frac),
                outside_weight=float(args.thought_edge_order_outside_weight),
                inside_floor=float(args.thought_edge_order_inside_floor),
                inside_ratio_floor=float(args.thought_edge_order_inside_ratio_floor),
                contrast_margin=float(args.thought_edge_order_contrast_margin),
                order_temperature=float(args.thought_edge_order_temperature),
                temporal_weight=float(args.thought_edge_order_temporal_weight),
                gate_weight=float(args.thought_edge_order_gate_weight),
                gate_outside_weight=float(args.thought_edge_order_gate_outside_weight),
                min_signal=float(args.thought_edge_order_min_signal),
                mass_floor=float(args.thought_edge_order_mass_floor),
            )
            loss = loss + float(args.thought_edge_order_weight) * edge_order
            stats["thought_edge_order"] = float(edge_order.detach().cpu())
            stats.update(edge_order_stats)
        else:
            stats["thought_edge_order"] = 0.0
            stats["thought_edge_order_outside"] = 0.0
            stats["thought_edge_order_inside"] = 0.0
            stats["thought_edge_order_inside_ratio"] = 0.0
            stats["thought_edge_order_contrast"] = 0.0
            stats["thought_edge_order_gate_outside"] = 0.0
            stats["thought_edge_order_support_frac"] = 0.0
            stats["thought_edge_order_signal_mean"] = 0.0
            stats["thought_edge_order_signal_strength"] = 0.0
        if float(args.valid_feature_jitter_weight) > 0:
            jitter_std = max(0.0, float(args.valid_feature_jitter_std))
            val_scale = F_val.detach().flatten(start_dim=1).std(dim=1, unbiased=False)
            val_scale = val_scale.view(-1, 1, 1, 1).clamp_min(1e-6)
            F_val_jitter = (F_val + torch.randn_like(F_val) * val_scale * jitter_std).detach()
            tvj = layer(F_val_jitter, return_trace=True)
            logits_vj = readout_val(tvj.final)
            if str(args.task_loss_mode).lower() in {"bounded_margin", "margin", "bounded"}:
                jitter_task = bounded_margin_task_loss(
                    logits_vj,
                    yv.long(),
                    margin=float(args.task_margin),
                )
            else:
                jitter_task = F.cross_entropy(logits_vj, yv.long())
            jitter_quiet, jitter_quiet_stats = quiet_state_loss(
                tvj,
                F_val_jitter,
                fixed_point_weight=float(args.valid_feature_jitter_fixed_weight),
                homeostasis_weight=float(args.valid_feature_jitter_homeostasis_weight),
                trajectory_weight=float(args.valid_feature_jitter_trajectory_weight),
                gate_weight=float(args.valid_feature_jitter_gate_weight),
            )
            jitter_loss = jitter_task + float(args.valid_feature_jitter_quiet_weight) * jitter_quiet
            loss = loss + float(args.valid_feature_jitter_weight) * jitter_loss
            stats["valid_feature_jitter"] = float(jitter_loss.detach().cpu())
            stats["valid_feature_jitter_task"] = float(jitter_task.detach().cpu())
            for k, v in jitter_quiet_stats.items():
                stats[f"valid_feature_jitter_{k}"] = v
        else:
            stats["valid_feature_jitter"] = 0.0
            stats["valid_feature_jitter_task"] = 0.0
            stats["valid_feature_jitter_quiet_fixed"] = 0.0
            stats["valid_feature_jitter_quiet_homeostasis"] = 0.0
            stats["valid_feature_jitter_quiet_trajectory"] = 0.0
            stats["valid_feature_jitter_quiet_gate"] = 0.0
        need_oga_source_readout = (
            attack_mode == "oga"
            and (
                float(args.oga_source_preservation_weight) > 0
                or float(args.oga_source_local_weight) > 0
                or float(args.oga_replacement_weight) > 0
            )
        )
        if need_oga_source_readout:
            raw_inv_p = make_raw_through_yolo(inv_b)(F_inv_p)
            inv_scores_p = _scores_from_raw_output(raw_inv_p)
            inv_boxes_p = _boxes_from_raw_output(raw_inv_p)
        else:
            inv_scores_p = None
            inv_boxes_p = None
        need_oda_support_readout = (
            attack_mode == "oda"
            and (
                float(args.target_support_compactness_weight) > 0
                or natural_support_this_step
                or float(args.oda_source_balance_weight) > 0
            )
        )
        if need_oda_support_readout:
            raw_inv_p_for_support = make_raw_through_yolo(inv_b)(F_inv_p)
            inv_scores_for_support = _scores_from_raw_output(raw_inv_p_for_support)
            inv_boxes_for_support = _boxes_from_raw_output(raw_inv_p_for_support)
        else:
            inv_scores_for_support = None
            inv_boxes_for_support = None
        if inv_scores_for_support is not None and float(args.target_support_compactness_weight) > 0:
            support_compact, support_stats = target_support_compactness_loss(inv_scores_for_support)
            loss = loss + float(args.target_support_compactness_weight) * support_compact
            stats["target_support_compactness"] = float(support_compact.detach().cpu())
            stats.update(support_stats)
        else:
            stats["target_support_compactness"] = 0.0
            stats["target_support_floor"] = 0.0
            stats["target_support_tail"] = 0.0
            stats["target_support_active"] = 0.0
            stats["target_support_concentration"] = 0.0
            stats["target_support_active_frac"] = 0.0
        if inv_scores_for_support is not None and natural_support_this_step:
            natural_support, natural_stats = target_natural_support_loss(
                inv_scores_for_support,
                inv_boxes_for_support,
                ref_val_scores,
                ref_val_boxes,
            )
            loss = loss + float(args.target_natural_support_weight) * natural_support
            stats["target_natural_support"] = float(natural_support.detach().cpu())
            stats.update(natural_stats)
        else:
            stats["target_natural_support"] = 0.0
            stats["target_natural_mass"] = 0.0
            stats["target_natural_count"] = 0.0
            stats["target_natural_active"] = 0.0
            stats["target_natural_area"] = 0.0
            stats["target_natural_small"] = 0.0
            stats["target_natural_active_count"] = 0.0
        if inv_scores_for_support is not None and float(args.oda_source_balance_weight) > 0:
            source_balance, source_balance_stats = oda_source_balance_loss(
                inv_scores_for_support,
                ref_val_scores,
            )
            loss = loss + float(args.oda_source_balance_weight) * source_balance
            stats["oda_source_balance"] = float(source_balance.detach().cpu())
            stats.update(source_balance_stats)
        else:
            stats["oda_source_balance"] = 0.0
            stats["oda_source_floor"] = 0.0
            stats["oda_target_source_gap"] = 0.0
            stats["oda_source_ref_gap"] = 0.0
            stats["oda_source_pred_gap"] = 0.0
        if (
            attack_mode == "oda"
            and video_scope_pairs > 0
            and 0.0 < float(args.video_scope_oda_mix_prob) < 1.0
        ):
            vs_bs = max(1, min(bs, int(video_scope_pairs)))
            vs_idx = torch.randint(0, int(video_scope_pairs), (vs_bs,), generator=rng)
            vs_inv_b = video_scope_invalid_x[vs_idx]
            vs_val_b = video_scope_valid_x[vs_idx]
            need_vs_ref_scores = natural_support_this_step or float(args.video_scope_oda_source_balance_weight) > 0
            with torch.no_grad():
                captured.clear()
                h_grab = neck_module.register_forward_hook(_grab)
                try:
                    inner(vs_inv_b)
                    F_vs_inv = captured[-1].clone()
                    captured.clear()
                    raw_vs_val_ref = inner(vs_val_b)
                    F_vs_val = captured[-1].clone()
                    if need_vs_ref_scores:
                        ref_vs_val_scores = _scores_from_raw_output(raw_vs_val_ref).detach()
                    if natural_support_this_step:
                        ref_vs_val_boxes = _boxes_from_raw_output(raw_vs_val_ref).detach()
                finally:
                    h_grab.remove()
            t_vs_inv = layer(F_vs_inv, return_trace=True)
            t_vs_val = layer(F_vs_val, return_trace=True)
            vs_labels = torch.ones(vs_bs, dtype=torch.long, device=device)
            vs_readout_inv = make_readout_through_yolo(vs_inv_b)
            vs_readout_val = make_readout_through_yolo(vs_val_b)
            def vs_readout_dispatcher(F_p: torch.Tensor) -> torch.Tensor:
                if F_p.data_ptr() == t_vs_inv.final.data_ptr() or torch.equal(F_p, t_vs_inv.final):
                    return vs_readout_inv(F_p)
                return vs_readout_val(F_p)
            vs_loss, vs_stats = lattice_ctm_objective(
                invalid_input=F_vs_inv,
                valid_input=F_vs_val,
                invalid_trace=t_vs_inv,
                valid_trace=t_vs_val,
                readout=vs_readout_dispatcher,
                invalid_labels=vs_labels,
                valid_labels=vs_labels,
                cfg=loss_cfg,
            )
            raw_vs_p = make_raw_through_yolo(vs_inv_b)(t_vs_inv.final)
            vs_scores_p = _scores_from_raw_output(raw_vs_p)
            vs_boxes_p = _boxes_from_raw_output(raw_vs_p)
            vs_support, vs_support_stats = target_support_compactness_loss(vs_scores_p)
            stats["video_scope_target_support_compactness"] = float(vs_support.detach().cpu())
            if natural_support_this_step:
                vs_natural, vs_natural_stats = target_natural_support_loss(
                    vs_scores_p,
                    vs_boxes_p,
                    ref_vs_val_scores,
                    ref_vs_val_boxes,
                )
            else:
                vs_natural = vs_support.sum() * 0.0
                vs_natural_stats = {
                    "target_natural_mass": 0.0,
                    "target_natural_count": 0.0,
                    "target_natural_active": 0.0,
                    "target_natural_area": 0.0,
                    "target_natural_small": 0.0,
                    "target_natural_active_count": 0.0,
                }
            if float(args.video_scope_oda_source_balance_weight) > 0:
                vs_source_balance, vs_source_balance_stats = oda_source_balance_loss(
                    vs_scores_p,
                    ref_vs_val_scores,
                )
            else:
                vs_source_balance = vs_support.sum() * 0.0
                vs_source_balance_stats = {
                    "oda_source_floor": 0.0,
                    "oda_target_source_gap": 0.0,
                    "oda_source_ref_gap": 0.0,
                    "oda_source_pred_gap": 0.0,
                }
            vs_total = (
                vs_loss
                + float(args.target_support_compactness_weight) * vs_support
                + float(args.target_natural_support_weight) * vs_natural
                + float(args.video_scope_oda_source_balance_weight) * vs_source_balance
            )
            loss = loss + float(args.video_scope_oda_mix_prob) * vs_total
            stats["video_scope_oda_aux"] = float(vs_total.detach().cpu())
            stats["video_scope_oda_aux_task"] = float(vs_stats.get("task", 0.0))
            for k, v in vs_support_stats.items():
                stats[f"video_scope_{k}"] = v
            stats["video_scope_target_natural_support"] = float(vs_natural.detach().cpu())
            for k, v in vs_natural_stats.items():
                stats[f"video_scope_{k}"] = v
            stats["video_scope_oda_source_balance"] = float(vs_source_balance.detach().cpu())
            for k, v in vs_source_balance_stats.items():
                stats[f"video_scope_{k}"] = v
        else:
            stats["video_scope_oda_aux"] = 0.0
            stats["video_scope_oda_aux_task"] = 0.0
            stats["video_scope_target_support_compactness"] = 0.0
            stats["video_scope_target_support_floor"] = 0.0
            stats["video_scope_target_support_tail"] = 0.0
            stats["video_scope_target_support_active"] = 0.0
            stats["video_scope_target_support_concentration"] = 0.0
            stats["video_scope_target_support_active_frac"] = 0.0
            stats["video_scope_target_natural_support"] = 0.0
            stats["video_scope_target_natural_mass"] = 0.0
            stats["video_scope_target_natural_count"] = 0.0
            stats["video_scope_target_natural_active"] = 0.0
            stats["video_scope_target_natural_area"] = 0.0
            stats["video_scope_target_natural_small"] = 0.0
            stats["video_scope_target_natural_active_count"] = 0.0
            stats["video_scope_oda_source_balance"] = 0.0
            stats["video_scope_oda_source_floor"] = 0.0
            stats["video_scope_oda_target_source_gap"] = 0.0
            stats["video_scope_oda_source_ref_gap"] = 0.0
            stats["video_scope_oda_source_pred_gap"] = 0.0
        if inv_scores_p is not None and float(args.oga_source_preservation_weight) > 0:
            source_ref_scores = ref_cf_scores if bool(cf_mask_b.any()) else ref_inv_scores
            source_pres = oga_source_preservation_loss(inv_scores_p, source_ref_scores, cf_mask_b)
            loss = loss + float(args.oga_source_preservation_weight) * source_pres
            stats["oga_source_preservation"] = float(source_pres.detach().cpu())
        else:
            stats["oga_source_preservation"] = 0.0
        if inv_scores_p is not None and float(args.oga_replacement_weight) > 0:
            replacement, replacement_stats = oga_replacement_margin_loss(inv_scores_p, cf_mask_b)
            loss = loss + float(args.oga_replacement_weight) * replacement
            stats["oga_replacement"] = float(replacement.detach().cpu())
            stats.update(replacement_stats)
        else:
            stats["oga_replacement"] = 0.0
            stats["oga_replacement_margin"] = 0.0
            stats["oga_replacement_floor"] = 0.0
        if inv_scores_p is not None and float(args.oga_source_local_weight) > 0:
            source_local, source_local_stats = oga_local_source_support_loss(
                inv_scores_p,
                inv_boxes_p,
                ref_cf_scores,
                ref_cf_boxes,
                ref_inv_scores,
                ref_inv_boxes,
                cf_mask_b,
            )
            loss = loss + float(args.oga_source_local_weight) * source_local
            stats["oga_source_local"] = float(source_local.detach().cpu())
            stats.update(source_local_stats)
        else:
            stats["oga_source_local"] = 0.0
            stats["oga_source_local_restore"] = 0.0
            stats["oga_source_local_margin"] = 0.0
            stats["oga_source_local_box"] = 0.0
            stats["oga_source_local_active"] = 0.0
            stats["oga_source_local_support_frac"] = 0.0
            stats["oga_source_local_mean_iou"] = 0.0
        if float(args.source_valid_weight) > 0 and source_valid_x.numel() > 0:
            sv_bs = max(1, min(bs, source_valid_x.shape[0]))
            sv_idx = torch.randint(0, source_valid_x.shape[0], (sv_bs,), generator=rng)
            sv_b = source_valid_x[sv_idx]
            with torch.no_grad():
                captured.clear()
                h_grab = neck_module.register_forward_hook(_grab)
                try:
                    raw_sv_ref = inner(sv_b)
                    F_sv = captured[-1].clone()
                    ref_sv_scores = _scores_from_raw_output(raw_sv_ref).detach()
                finally:
                    h_grab.remove()
            tsv = layer(F_sv, return_trace=True)
            raw_sv_p = make_raw_through_yolo(sv_b)(tsv.final)
            sv_scores_p = _scores_from_raw_output(raw_sv_p)
            sv_loss, sv_stats = source_valid_fixed_point_loss(
                sv_scores_p,
                ref_sv_scores,
                tsv,
                F_sv,
            )
            loss = loss + float(args.source_valid_weight) * sv_loss
            stats["source_valid"] = float(sv_loss.detach().cpu())
            stats.update(sv_stats)
        else:
            stats["source_valid"] = 0.0
            stats["source_valid_score"] = 0.0
            stats["source_valid_motion"] = 0.0
            stats["source_valid_active"] = 0.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 5.0)
        opt.step()

        if step % max(1, int(args.steps) // 8) == 0 or step == int(args.steps) - 1:
            stats["step"] = float(step)
            stats["loss_total"] = float(loss.detach().cpu())
            stats["loss"] = stats["loss_total"]
            history.append(stats)

    train_elapsed = time.time() - t0
    print(f"[{tag}] train done {train_elapsed:.1f}s  final task={history[-1]['task']:.3f}  inv_acc={history[-1]['invalid_acc']:.3f}  val_acc={history[-1]['valid_acc']:.3f}")

    # ----- defended eval -----
    layer.eval()
    def _replace_with_ctm(module, inputs, output):
        if isinstance(output, torch.Tensor) and output.ndim == 4 and output.shape[1] == int(neck_c):
            return layer(output)
        return output

    de_trig = eval_pool_fire(trig_eval, float(args.conf), hook=_replace_with_ctm)
    de_aug = eval_pool_fire(aug_eval, float(args.conf), hook=_replace_with_ctm)
    de_clean = eval_pool_fire(valid_eval_paths, float(args.conf), hook=_replace_with_ctm)
    print(f"[{tag}] DEF trig_eval={de_trig['n_fired']}/{de_trig['n_images']}  aug_eval={de_aug['n_fired']}/{de_aug['n_images']}  clean_eval={de_clean['n_fired']}/{de_clean['n_images']}")

    purity_flags: List[str] = []
    if float(args.valid_decode_geometry_weight) > 0:
        purity_flags.append("decoded_geometry_loss")
    if float(args.cf_pair_motion_weight) > 0:
        purity_flags.append("clean_trigger_counterfactual_pairing")
    if float(args.task_tangent_weight) > 0:
        purity_flags.append("detector_jacobian_tangent_loss")
    if (
        float(args.oga_source_preservation_weight) > 0
        or float(args.oga_replacement_weight) > 0
        or float(args.oga_source_local_weight) > 0
        or float(args.source_valid_weight) > 0
    ):
        purity_flags.append("oga_detector_score_or_box_auxiliary")
    if (
        float(args.target_support_compactness_weight) > 0
        or float(args.target_natural_support_weight) > 0
        or float(args.oda_source_balance_weight) > 0
        or float(args.video_scope_oda_source_balance_weight) > 0
        or int(args.n_video_scope_oda_train) > 0
    ):
        purity_flags.append("oda_detector_score_or_video_scope_auxiliary")
    paper_main_profile = not purity_flags and cf_match_mode == "none"

    layer_pt = out / "lattice_nf_ctm_yolo_layer.pt"
    torch.save({
        "state_dict": layer.state_dict(),
        "ctm_config": ctm_cfg.to_dict(),
        "loss_config": loss_cfg.to_dict(),
        "neck_index": int(args.neck_index),
        "neck_channels": int(neck_c),
    }, layer_pt)

    def asr_pair(k: int, n: int):
        k_succ = int(k) if attack_mode != "oda" else int(n) - int(k)
        return {
            "k_fired": int(k), "n": int(n),
            "k_attack_success": int(k_succ),
            "asr_attack_success": (k_succ / n) if n else 0.0,
            "wilson95": _wilson(k_succ, n),
        }

    rec = {
        "tag": tag,
        "attack_mode": attack_mode,
        "poisoned": str(poisoned),
        "config": {"ctm": ctm_cfg.to_dict(), "loss": loss_cfg.to_dict()},
        "aux_config": {
            "valid_decode_geometry_weight": float(args.valid_decode_geometry_weight),
            "valid_decode_topk_frac": float(args.valid_decode_topk_frac),
            "valid_decode_temp": float(args.valid_decode_temp),
            "valid_decode_score_weight": float(args.valid_decode_score_weight),
            "valid_decode_box_weight": float(args.valid_decode_box_weight),
            "oga_source_preservation_weight": float(args.oga_source_preservation_weight),
            "source_class_id": int(args.source_class_id),
            "oga_source_margin": float(args.oga_source_margin),
            "oga_source_local_weight": float(args.oga_source_local_weight),
            "oga_source_local_topk_frac": float(args.oga_source_local_topk_frac),
            "oga_source_local_source_topk_frac": float(args.oga_source_local_source_topk_frac),
            "oga_source_local_min_support": float(args.oga_source_local_min_support),
            "oga_source_local_match_iou": float(args.oga_source_local_match_iou),
            "oga_source_local_margin": float(args.oga_source_local_margin),
            "oga_source_local_target_margin": float(args.oga_source_local_target_margin),
            "oga_source_local_target_weight": float(args.oga_source_local_target_weight),
            "oga_source_local_box_weight": float(args.oga_source_local_box_weight),
            "oga_replacement_weight": float(args.oga_replacement_weight),
            "oga_replacement_margin": float(args.oga_replacement_margin),
            "oga_replacement_source_conf": float(args.oga_replacement_source_conf),
            "oga_replacement_floor_weight": float(args.oga_replacement_floor_weight),
            "source_valid_weight": float(args.source_valid_weight),
            "source_valid_topk_frac": float(args.source_valid_topk_frac),
            "source_valid_min_support": float(args.source_valid_min_support),
            "source_valid_motion_weight": float(args.source_valid_motion_weight),
            "n_source_valid_train": int(args.n_source_valid_train),
            "task_tangent_weight": float(args.task_tangent_weight),
            "task_tangent_topk_frac": float(args.task_tangent_topk_frac),
            "task_tangent_alignment_floor": float(args.task_tangent_alignment_floor),
            "task_tangent_outside_weight": float(args.task_tangent_outside_weight),
            "task_tangent_batch": int(args.task_tangent_batch),
            "task_tangent_precompute_batch": int(args.task_tangent_precompute_batch),
            "cf_diff_group_size": int(args.cf_diff_group_size),
            "target_support_compactness_weight": float(args.target_support_compactness_weight),
            "target_support_topk_frac": float(args.target_support_topk_frac),
            "target_support_floor_conf": float(args.target_support_floor_conf),
            "target_support_floor_weight": float(args.target_support_floor_weight),
            "target_support_tail_ceiling": float(args.target_support_tail_ceiling),
            "target_support_max_active_frac": float(args.target_support_max_active_frac),
            "target_support_count_weight": float(args.target_support_count_weight),
            "target_support_count_temp": float(args.target_support_count_temp),
            "target_support_min_concentration": float(args.target_support_min_concentration),
            "target_support_concentration_weight": float(args.target_support_concentration_weight),
            "target_natural_support_weight": float(args.target_natural_support_weight),
            "target_natural_active_weight": float(args.target_natural_active_weight),
            "target_natural_count_weight": float(args.target_natural_count_weight),
            "target_natural_count_slack": float(args.target_natural_count_slack),
            "target_natural_area_weight": float(args.target_natural_area_weight),
            "target_natural_small_weight": float(args.target_natural_small_weight),
            "target_natural_small_area_frac": float(args.target_natural_small_area_frac),
            "target_natural_slack": float(args.target_natural_slack),
            "target_natural_every": int(args.target_natural_every),
            "thought_active_area_weight": float(args.thought_active_area_weight),
            "thought_active_area_max_frac": float(args.thought_active_area_max_frac),
            "thought_active_area_temp": float(args.thought_active_area_temp),
            "thought_spatial_entropy_weight": float(args.thought_spatial_entropy_weight),
            "thought_spatial_entropy_max_frac": float(args.thought_spatial_entropy_max_frac),
            "thought_sync_support_weight": float(args.thought_sync_support_weight),
            "thought_sync_support_mode": str(args.thought_sync_support_mode),
            "thought_sync_support_topk_frac": float(args.thought_sync_support_topk_frac),
            "thought_sync_support_outside_weight": float(args.thought_sync_support_outside_weight),
            "thought_sync_support_inside_floor": float(args.thought_sync_support_inside_floor),
            "thought_edge_order_weight": float(args.thought_edge_order_weight),
            "thought_edge_order_topk_frac": float(args.thought_edge_order_topk_frac),
            "thought_edge_order_outside_weight": float(args.thought_edge_order_outside_weight),
            "thought_edge_order_inside_floor": float(args.thought_edge_order_inside_floor),
            "thought_edge_order_inside_ratio_floor": float(args.thought_edge_order_inside_ratio_floor),
            "thought_edge_order_contrast_margin": float(args.thought_edge_order_contrast_margin),
            "thought_edge_order_temperature": float(args.thought_edge_order_temperature),
            "thought_edge_order_temporal_weight": float(args.thought_edge_order_temporal_weight),
            "thought_edge_order_gate_weight": float(args.thought_edge_order_gate_weight),
            "thought_edge_order_gate_outside_weight": float(args.thought_edge_order_gate_outside_weight),
            "thought_edge_order_min_signal": float(args.thought_edge_order_min_signal),
            "thought_edge_order_mass_floor": float(args.thought_edge_order_mass_floor),
            "valid_fixed_point_weight": float(args.valid_fixed_point_weight),
            "valid_fixed_point_relative": bool(args.valid_fixed_point_relative),
            "cross_label_residual_orthogonality_weight": float(args.cross_label_residual_orthogonality_weight),
            "thought_concentration_weight": float(args.thought_concentration_weight),
            "thought_concentration_target": float(args.thought_concentration_target),
            "residual_profile_invariance_weight": float(args.residual_profile_invariance_weight),
            "residual_profile_invalid_floor": float(args.residual_profile_invalid_floor),
            "residual_profile_valid_weight": float(args.residual_profile_valid_weight),
            "residual_profile_topk_frac": float(args.residual_profile_topk_frac),
            "trajectory_valid_weight": float(args.trajectory_valid_weight),
            "trajectory_invalid_floor_weight": float(args.trajectory_invalid_floor_weight),
            "trajectory_invalid_floor": float(args.trajectory_invalid_floor),
            "basin_separation_weight": float(args.basin_separation_weight),
            "basin_separation_margin": float(args.basin_separation_margin),
            "basin_separation_same_margin": float(args.basin_separation_same_margin),
            "basin_separation_profile": str(args.basin_separation_profile),
            "basin_separation_detach_valid": bool(args.basin_separation_detach_valid),
            "basin_separation_same_weight": float(args.basin_separation_same_weight),
            "basin_separation_diff_weight": float(args.basin_separation_diff_weight),
            "local_edge_conflict_center": float(args.local_edge_conflict_center),
            "local_edge_polarity_strength": float(args.local_edge_polarity_strength),
            "local_edge_polarity_init": float(args.local_edge_polarity_init),
            "local_edge_polarity_use_conflict_gate": bool(args.local_edge_polarity_use_conflict_gate),
            "valid_feature_jitter_weight": float(args.valid_feature_jitter_weight),
            "valid_feature_jitter_std": float(args.valid_feature_jitter_std),
            "valid_feature_jitter_quiet_weight": float(args.valid_feature_jitter_quiet_weight),
            "valid_feature_jitter_fixed_weight": float(args.valid_feature_jitter_fixed_weight),
            "valid_feature_jitter_homeostasis_weight": float(args.valid_feature_jitter_homeostasis_weight),
            "valid_feature_jitter_trajectory_weight": float(args.valid_feature_jitter_trajectory_weight),
            "valid_feature_jitter_gate_weight": float(args.valid_feature_jitter_gate_weight),
            "oda_source_balance_weight": float(args.oda_source_balance_weight),
            "video_scope_oda_source_balance_weight": float(args.video_scope_oda_source_balance_weight),
            "oda_source_floor_weight": float(args.oda_source_floor_weight),
            "oda_source_margin": float(args.oda_source_margin),
            "oda_target_source_gap_slack": float(args.oda_target_source_gap_slack),
            "n_video_scope_oda_train": int(args.n_video_scope_oda_train),
            "video_scope_oda_mix_prob": float(args.video_scope_oda_mix_prob),
        },
        "splits": {
            "invalid_train": [str(p) for p in invalid_train_paths],
            "invalid_eval_extra": [str(p) for p in invalid_eval_paths],
            "valid_train": [str(p) for p in valid_train_paths],
            "valid_eval": [str(p) for p in valid_eval_paths],
            "source_valid_train": [str(p) for p in source_valid_pool],
            "quiet_absent_train": [str(p) for p in absent_pool],
            "counterfactual_clean_train": [str(p) for p in invalid_cf_clean_paths],
            "counterfactual_clean_records": invalid_cf_records,
            "counterfactual_match_methods": cf_methods,
            "counterfactual_match_mode": str(args.cf_pair_match_mode),
            "counterfactual_match_mode_effective": str(cf_match_mode),
            "counterfactual_visual_max_distance": float(args.cf_pair_visual_max_distance),
            "counterfactual_clean_available": len(cf_clean_available),
            "counterfactual_clean_enabled": bool(needs_counterfactual_clean),
            "video_scope_oda_pairs": int(video_scope_pairs),
            "video_scope_oda_train_enabled": bool(attack_mode == "oda" and video_scope_pairs > 0),
            "letterbox_center": bool(letterbox_center),
            "trig_eval_n": len(trig_eval),
            "aug_eval_n": len(aug_eval),
            "source_key_policy": "path_stem_before_double_underscore",
            "train_source_keys": sorted(train_source_keys),
            "cf_diff_grouped_sources": int(len(cf_diff_group_ids)),
            "source_leak_excluded": source_leak_excluded,
        },
        "train": {"steps": int(args.steps), "bs": int(args.batch_size), "lr": float(args.lr),
                  "fire_thr_filter": fire_thr, "want_fired": want_fired,
                  "inv_target_label": inv_target_label, "val_target_label": val_target_label,
                  "readout_mode": readout_mode, "readout_softmax_temp": softmax_T},
        "passthrough": {
            "trig_eval": asr_pair(pt_trig["n_fired"], pt_trig["n_images"]),
            "aug_eval":  asr_pair(pt_aug["n_fired"],  pt_aug["n_images"]),
            "clean_eval": {"k": pt_clean["n_fired"], "n": pt_clean["n_images"]},
        },
        "defended": {
            "trig_eval": asr_pair(de_trig["n_fired"], de_trig["n_images"]),
            "aug_eval":  asr_pair(de_aug["n_fired"],  de_aug["n_images"]),
            "clean_eval": {"k": de_clean["n_fired"], "n": de_clean["n_images"]},
        },
        "purity": {
            "paper_main_profile": bool(paper_main_profile),
            "runtime_profile": "single_in_flow_neck_hook",
            "strict_pure_flags": purity_flags,
            "note": "Non-empty flags mean this run belongs in diagnostics/ablation, not the strict pure CTM paper main table.",
        },
        "train_history": history,
        "train_elapsed_s": float(train_elapsed),
        "strict_pass_aug": bool(_wilson(
            (de_aug["n_images"] - de_aug["n_fired"]) if attack_mode == "oda" else de_aug["n_fired"],
            de_aug["n_images"],
        ) <= 0.05),
        "clean_safe_pass": bool(de_clean["n_fired"] >= max(0, pt_clean["n_fired"] - int(args.max_clean_recall_drop))),
        "max_clean_recall_drop": int(args.max_clean_recall_drop),
        "artifacts": {"layer": str(layer_pt)},
    }
    (out / "record.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(WORKSPACE / "benchmark_runs" / "nf_ctm_lattice_v4_pure_2026-05-27"))
    p.add_argument("--families", default="v4,v2,v3,b1,b2,b3,b4")
    p.add_argument("--device", default="0")
    p.add_argument("--neck-index", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--letterbox-center", action=argparse.BooleanOptionalAction, default=True,
                   help="match Ultralytics YOLO.predict centered LetterBox padding; --no-letterbox-center preserves legacy top-left padding")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-3)
    # Pool sizes for disjoint splits
    p.add_argument("--n-invalid-train", type=int, default=60)
    p.add_argument("--n-invalid-eval-extra", type=int, default=20)
    p.add_argument("--n-valid-train", type=int, default=60)
    p.add_argument("--n-valid-eval", type=int, default=60)
    p.add_argument("--n-source-valid-train", type=int, default=0,
                   help="optional clean source-class fixed-point pool for OGA replacement runs")
    p.add_argument("--n-quiet-absent-train", dest="n_quiet_absent_train", type=int, default=0,
                   help="target-absent clean samples used as CTM quiet fixed-points")
    p.add_argument("--seed", type=int, default=20260526)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--filter-fire-thr", type=float, default=0.10,
                   help="fixed invalid-pool filtering threshold; not selected per attack family")
    p.add_argument("--max-clean-recall-drop", type=int, default=2)
    # CTM
    p.add_argument("--thought-steps", type=int, default=5)
    p.add_argument("--memory-depth", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--init-decay", type=float, default=0.95)
    p.add_argument("--step-size", type=float, default=0.12)
    p.add_argument("--sync-gain", type=float, default=0.45)
    p.add_argument("--spatial-radii", default="1")
    p.add_argument("--use-field-order-edges", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--use-channel-order-edges", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--use-adaptive-update", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--adaptive-residual-gain", type=float, default=2.0)
    p.add_argument("--adaptive-residual-grad-scale", type=float, default=0.0,
                   help="pure CTM gate learning scale: 0 keeps residual gate detached, 1 lets gate residual shape sync dynamics")
    p.add_argument("--update-gate-bias", type=float, default=-1.25)
    p.add_argument("--sync-residual-floor", type=float, default=-1.0,
                   help="optional multiplicative sync-residual gate threshold; <=0 disables")
    p.add_argument("--sync-drive-dc-suppression", type=float, default=0.50,
                   help="pure CTM gauge fixing: remove this fraction of spatially uniform sync drive")
    p.add_argument("--total-drive-dc-suppression", type=float, default=0.0,
                   help="pure CTM gauge fixing: remove this fraction of spatially uniform full thought drive")
    p.add_argument("--local-edge-conflict-strength", type=float, default=0.0,
                   help="pure CTM local edge-conflict modulation strength")
    p.add_argument("--local-edge-conflict-floor", type=float, default=0.35,
                   help="lower bound for local edge-conflict modulation")
    p.add_argument("--local-edge-conflict-ceiling", type=float, default=1.0,
                   help="upper bound for local edge-conflict modulation; >1 amplifies high-conflict CTM cells")
    p.add_argument("--local-edge-conflict-center", type=float, default=0.0,
                   help="center threshold for normalized CTM edge-conflict gates; positive values make motion more selective")
    p.add_argument("--local-edge-conflict-abs-gate", action=argparse.BooleanOptionalAction, default=False,
                   help="gate by absolute local edge-conflict anomaly rather than signed conflict")
    p.add_argument("--local-edge-polarity-strength", type=float, default=0.0,
                   help="pure CTM signed local edge-conflict polarity drive strength")
    p.add_argument("--local-edge-polarity-init", type=float, default=0.0,
                   help="initial per-channel polarity coefficient for signed edge-conflict drive")
    p.add_argument("--local-edge-polarity-use-conflict-gate", action=argparse.BooleanOptionalAction, default=False,
                   help="gate the signed polarity drive by the same normalized CTM edge-conflict anomaly")
    p.add_argument("--local-edge-conflict-update-gate", action=argparse.BooleanOptionalAction, default=False,
                   help="also multiply the recurrent update gate by the CTM local edge-conflict gate")
    p.add_argument("--sync-residual-floor-p", type=int, default=2)
    p.add_argument("--max-update", type=float, default=0.50)
    p.add_argument("--init-sync-weight-std", type=float, default=2e-3)
    # Loss
    p.add_argument("--task-weight", type=float, default=1.0)
    p.add_argument("--task-loss-mode", choices=["ce", "bounded_margin"], default="ce",
                   help="bounded_margin stops pushing target evidence after it crosses the safety boundary")
    p.add_argument("--task-margin", type=float, default=0.0,
                   help="target-evidence safety margin used by --task-loss-mode bounded_margin")
    p.add_argument("--valid-task-weight-extra", type=float, default=1.0,
                   help="extra valid-side task weight; pinned to 1.0 to keep "
                        "the same value across all families (no sandwich)")
    p.add_argument("--paired-sync-weight", type=float, default=0.0)
    p.add_argument("--label-attractor-weight", type=float, default=0.08)
    p.add_argument("--same-label-weight", type=float, default=1.0)
    p.add_argument("--diff-label-weight", type=float, default=1.0)
    p.add_argument("--attractor-margin", type=float, default=0.30)
    p.add_argument("--separation-weight", type=float, default=0.0)
    p.add_argument("--kinetic-weight", type=float, default=0.01)
    p.add_argument("--invalid-motion-weight", type=float, default=0.005)
    p.add_argument("--valid-motion-weight", type=float, default=0.10)
    p.add_argument("--max-invalid-rms", type=float, default=8.0)
    p.add_argument("--max-valid-rms", type=float, default=2.0)
    p.add_argument("--valid-homeostasis-weight", type=float, default=0.02)
    p.add_argument("--valid-gate-weight", type=float, default=0.01)
    p.add_argument("--invalid-gate-floor-weight", type=float, default=0.0)
    p.add_argument("--invalid-gate-floor", type=float, default=0.05)
    p.add_argument("--gate-separation-weight", type=float, default=0.0,
                   help="pure CTM gate-basin separation weight: invalid recurrent gates should exceed valid gates")
    p.add_argument("--gate-separation-margin", type=float, default=0.0,
                   help="desired invalid_gate - valid_gate margin for CTM gate-basin separation")
    # Pure CTM anti-collapse terms.  These are not external anchors; they only
    # constrain terminal recurrent states and thought trajectories.
    p.add_argument("--valid-fixed-point-weight", type=float, default=0.0,
                   help="optional quadratic valid-state fixed-point weight; "
                        "disabled by default for strict pure-CTM runs")
    p.add_argument("--valid-fixed-point-relative", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--cross-label-residual-orthogonality-weight", type=float, default=0.04,
                   help="penalize constant residuals across different-label inputs")
    p.add_argument("--residual-decorrelation-weight", dest="cross_label_residual_orthogonality_weight", type=float, default=argparse.SUPPRESS)
    p.add_argument("--thought-concentration-weight", type=float, default=0.02,
                   help="reward concentrated invalid CTM thought motion instead of global damping")
    p.add_argument("--thought-concentration-target", type=float, default=0.35)
    p.add_argument("--residual-profile-invariance-weight", type=float, default=0.0,
                   help="pure CTM channel-order residual profile invariance weight")
    p.add_argument("--residual-profile-invalid-floor", type=float, default=0.04,
                   help="minimum invalid residual-profile motion magnitude")
    p.add_argument("--residual-profile-valid-weight", type=float, default=0.25,
                   help="valid quietness weight inside residual-profile invariance")
    p.add_argument("--residual-profile-topk-frac", type=float, default=0.08,
                   help="top-k spatial fraction used only to form location-free CTM channel profiles")
    p.add_argument("--trajectory-valid-weight", type=float, default=0.0)
    p.add_argument("--trajectory-invalid-floor-weight", type=float, default=0.0)
    p.add_argument("--trajectory-invalid-floor", type=float, default=0.10)
    p.add_argument("--basin-separation-weight", type=float, default=0.0,
                   help="pure CTM state-basin separation weight for different-label trajectories")
    p.add_argument("--basin-separation-margin", type=float, default=0.08,
                   help="minimum CTM basin-profile RMS distance for different-label pairs")
    p.add_argument("--basin-separation-same-margin", type=float, default=0.05,
                   help="maximum CTM basin-profile RMS distance for same-label compactness")
    p.add_argument("--basin-separation-profile", choices=["residual", "sync", "gate", "hybrid", "phase"], default="residual",
                   help="CTM traces used for basin profile; no detector score/box/postprocess")
    p.add_argument("--basin-separation-detach-valid", type=lambda s: s.lower() == "true", default=True,
                   help="detach valid quiet-basin profile when separating invalid trajectories")
    p.add_argument("--basin-separation-same-weight", type=float, default=0.0,
                   help="relative same-label CTM basin compactness inside the basin loss")
    p.add_argument("--basin-separation-diff-weight", type=float, default=1.0,
                   help="relative different-label CTM basin margin inside the basin loss")
    p.add_argument("--quiet-absent-weight", type=float, default=0.0)
    p.add_argument("--quiet-absent-fixed-weight", type=float, default=1.0)
    p.add_argument("--quiet-absent-homeostasis-weight", type=float, default=0.25)
    p.add_argument("--quiet-absent-trajectory-weight", type=float, default=0.25)
    p.add_argument("--quiet-absent-gate-weight", type=float, default=0.05)
    p.add_argument("--cf-diff-motion-weight", type=float, default=0.0,
                   help="pure CTM same-source counterfactual difference support loss")
    p.add_argument("--cf-diff-motion-topk-frac", type=float, default=0.16)
    p.add_argument("--cf-diff-motion-inside-floor", type=float, default=0.015)
    p.add_argument("--cf-diff-group-size", type=int, default=2,
                   help="when cf-diff is active, sample this many same-source invalid variants per CTM batch group")
    p.add_argument("--cf-pair-motion-weight", type=float, default=0.0,
                   help="pure CTM clean-trigger counterfactual support loss")
    p.add_argument("--cf-pair-motion-topk-frac", type=float, default=0.16)
    p.add_argument("--cf-pair-motion-inside-floor", type=float, default=0.015)
    p.add_argument("--cf-pair-clean-quiet-weight", type=float, default=0.25)
    p.add_argument("--cf-pair-direction-weight", type=float, default=0.0,
                   help="align CTM residual with clean-trigger counterfactual direction inside support")
    p.add_argument("--cf-pair-match-mode", choices=["auto", "manifest", "visual", "none"], default="auto",
                   help="training-only clean-trigger pairing policy for cf-pair supervision")
    p.add_argument("--cf-pair-visual-max-distance", type=float, default=0.015,
                   help="reject visual-nearest counterfactual pairs above this cosine distance")
    p.add_argument("--cf-pair-visual-max-candidates", type=int, default=5000,
                   help="maximum visual-nearest clean candidates scanned per root")
    p.add_argument("--task-tangent-weight", type=float, default=0.0,
                   help="training-only CTM tangent-field alignment from frozen downstream target-evidence Jacobian")
    p.add_argument("--task-tangent-topk-frac", type=float, default=0.03)
    p.add_argument("--task-tangent-alignment-floor", type=float, default=0.02)
    p.add_argument("--task-tangent-outside-weight", type=float, default=0.05)
    p.add_argument("--task-tangent-batch", type=int, default=4,
                   help="number of invalid samples used for cached task-tangent loss per step")
    p.add_argument("--task-tangent-precompute-batch", type=int, default=6,
                   help="batch size for one-time downstream Jacobian cache precomputation")
    # Training-only valid decode consistency. This aligns the CTM training
    # signal with decoded-box clean recall without adding runtime calibration or
    # an external clean model.
    p.add_argument("--target-class-id", type=int, default=0)
    p.add_argument("--source-class-id", type=int, default=1,
                   help="source class preserved for OGA replacement detox; default head=1")
    p.add_argument("--oga-source-preservation-weight", type=float, default=0.0,
                   help="training-only CTM loss that preserves source-class evidence on OGA invalid samples")
    p.add_argument("--oga-source-margin", type=float, default=0.25,
                    help="allowed source-evidence drop before the OGA preservation hinge activates")
    p.add_argument("--oga-source-local-weight", type=float, default=0.0,
                   help="training-only CTM loss that restores local source support cells for OGA")
    p.add_argument("--oga-source-local-topk-frac", type=float, default=0.005,
                   help="fraction of target-supported poisoned cells used by local OGA restoration")
    p.add_argument("--oga-source-local-source-topk-frac", type=float, default=0.02,
                   help="fraction of source-supported counterfactual cells used for geometric pairing")
    p.add_argument("--oga-source-local-min-support", type=float, default=0.03,
                   help="minimum counterfactual-source times poisoned-target geometric support for local OGA cells")
    p.add_argument("--oga-source-local-match-iou", type=float, default=0.05,
                   help="minimum box IoU for training-only local source/target cell pairing")
    p.add_argument("--oga-source-local-margin", type=float, default=0.10,
                   help="allowed local source probability drop before restoration hinge activates")
    p.add_argument("--oga-source-local-target-margin", type=float, default=0.05,
                   help="required source-over-target local probability margin after CTM")
    p.add_argument("--oga-source-local-target-weight", type=float, default=0.50,
                   help="relative weight for local source-over-target margin inside the local OGA loss")
    p.add_argument("--oga-source-local-box-weight", type=float, default=0.0,
                   help="optional training-only local source box consistency weight")
    p.add_argument("--oga-replacement-weight", type=float, default=0.0,
                   help="training-only OGA objective requiring source evidence to beat target evidence")
    p.add_argument("--oga-replacement-margin", type=float, default=0.10,
                   help="minimum source-over-target evidence margin for replacement-style OGA")
    p.add_argument("--oga-replacement-source-conf", type=float, default=0.25,
                   help="source-class evidence boundary used by the OGA replacement floor")
    p.add_argument("--oga-replacement-floor-weight", type=float, default=0.50,
                   help="relative weight for the OGA replacement source-evidence floor")
    p.add_argument("--source-valid-weight", type=float, default=0.0,
                   help="training-only CTM fixed-point/readout preservation for clean source-class objects")
    p.add_argument("--source-valid-topk-frac", type=float, default=0.01)
    p.add_argument("--source-valid-min-support", type=float, default=0.05)
    p.add_argument("--source-valid-motion-weight", type=float, default=0.05)
    p.add_argument("--target-support-compactness-weight", type=float, default=0.0,
                   help="training-only CTM ODA loss that prevents global target over-activation")
    p.add_argument("--target-support-topk-frac", type=float, default=0.01,
                   help="fraction of target-supported anchors allowed to carry restored target evidence")
    p.add_argument("--target-support-floor-conf", type=float, default=0.25,
                   help="minimum mean sigmoid target evidence inside the compact restored support")
    p.add_argument("--target-support-floor-weight", type=float, default=1.0,
                   help="relative weight for restoring target evidence inside compact support")
    p.add_argument("--target-support-tail-ceiling", type=float, default=0.05,
                   help="maximum sigmoid target evidence tolerated outside the compact support")
    p.add_argument("--target-support-max-active-frac", type=float, default=0.08,
                   help="soft upper bound on the fraction of target anchors active after CTM")
    p.add_argument("--target-support-count-weight", type=float, default=0.50,
                   help="relative weight for the soft active-anchor count term")
    p.add_argument("--target-support-count-temp", type=float, default=0.04,
                   help="temperature for differentiable active-anchor counting")
    p.add_argument("--target-support-min-concentration", type=float, default=0.012,
                   help="minimum normalized target-evidence concentration after CTM")
    p.add_argument("--target-support-concentration-weight", type=float, default=0.25,
                   help="relative weight for normalized target-evidence concentration")
    p.add_argument("--target-natural-support-weight", type=float, default=0.0,
                   help="training-only ODA target-support naturalness loss for CTM-restored decoded support")
    p.add_argument("--target-natural-active-weight", type=float, default=1.0,
                   help="relative weight for matching clean-valid active target-support density")
    p.add_argument("--target-natural-count-weight", type=float, default=0.0,
                   help="relative weight for matching clean-valid soft active target-support count")
    p.add_argument("--target-natural-count-slack", type=float, default=8.0,
                   help="allowed excess soft active target-support count over clean-valid batches")
    p.add_argument("--target-natural-area-weight", type=float, default=0.50,
                   help="relative weight for matching clean-valid target area mass")
    p.add_argument("--target-natural-small-weight", type=float, default=1.0,
                   help="relative weight for suppressing excessive tiny restored target boxes")
    p.add_argument("--target-natural-small-area-frac", type=float, default=0.0023,
                   help="normalized box-area threshold considered unnaturally small for restored targets")
    p.add_argument("--target-natural-slack", type=float, default=0.02,
                   help="batch-level clean-valid slack used by the target-support naturalness envelope")
    p.add_argument("--target-natural-every", type=int, default=1,
                   help="apply the expensive decoded natural-support loss every N CTM steps")
    p.add_argument("--thought-active-area-weight", type=float, default=0.0,
                   help="pure CTM loss that limits the spatial active area of invalid thought motion")
    p.add_argument("--thought-active-area-max-frac", type=float, default=0.12,
                   help="soft upper bound on the fraction of active CTM motion cells")
    p.add_argument("--thought-active-area-temp", type=float, default=0.08,
                   help="temperature for the CTM active-area soft gate")
    p.add_argument("--thought-spatial-entropy-weight", type=float, default=0.0,
                   help="pure CTM loss limiting effective spatial support of thought motion")
    p.add_argument("--thought-spatial-entropy-max-frac", type=float, default=0.20,
                   help="maximum effective spatial support fraction allowed for CTM thought motion")
    p.add_argument("--thought-sync-support-weight", type=float, default=0.0,
                   help="pure CTM loss aligning invalid thought motion to CTM synchronization-change support")
    p.add_argument("--thought-sync-support-mode", choices=["change", "edge_disagreement", "hybrid"], default="change",
                   help="CTM-native support signal: temporal sync change, edge-order disagreement, or their hybrid")
    p.add_argument("--thought-sync-support-topk-frac", type=float, default=0.20,
                   help="fraction of CTM sync-change cells treated as thought-motion support")
    p.add_argument("--thought-sync-support-outside-weight", type=float, default=1.0,
                   help="relative weight for thought motion outside CTM sync support")
    p.add_argument("--thought-sync-support-inside-floor", type=float, default=0.0,
                   help="optional minimum relative thought motion inside CTM sync support")
    p.add_argument("--thought-edge-order-weight", type=float, default=0.0,
                   help="pure CTM invalid-only edge-order localization loss weight")
    p.add_argument("--thought-edge-order-topk-frac", type=float, default=0.12,
                   help="fraction of CTM edge-order anomaly cells used as invalid thought support")
    p.add_argument("--thought-edge-order-outside-weight", type=float, default=1.0)
    p.add_argument("--thought-edge-order-inside-floor", type=float, default=0.015)
    p.add_argument("--thought-edge-order-inside-ratio-floor", type=float, default=0.55)
    p.add_argument("--thought-edge-order-contrast-margin", type=float, default=0.01)
    p.add_argument("--thought-edge-order-temperature", type=float, default=0.10)
    p.add_argument("--thought-edge-order-temporal-weight", type=float, default=0.50)
    p.add_argument("--thought-edge-order-gate-weight", type=float, default=0.25)
    p.add_argument("--thought-edge-order-gate-outside-weight", type=float, default=0.05)
    p.add_argument("--thought-edge-order-min-signal", type=float, default=1e-4,
                   help="disable edge-order support when CTM edge signal is effectively flat")
    p.add_argument("--thought-edge-order-mass-floor", type=float, default=1e-4,
                   help="minimum relative thought mass before ratio/contrast edge-order terms activate")
    p.add_argument("--valid-feature-jitter-weight", type=float, default=0.0,
                   help="pure CTM normal-neighborhood regularizer on valid neck features")
    p.add_argument("--valid-feature-jitter-std", type=float, default=0.03,
                   help="relative Gaussian jitter scale for valid CTM feature-neighborhood training")
    p.add_argument("--valid-feature-jitter-quiet-weight", type=float, default=1.0,
                   help="relative quiet-state weight inside valid feature-jitter regularization")
    p.add_argument("--valid-feature-jitter-fixed-weight", type=float, default=1.0)
    p.add_argument("--valid-feature-jitter-homeostasis-weight", type=float, default=0.25)
    p.add_argument("--valid-feature-jitter-trajectory-weight", type=float, default=0.25)
    p.add_argument("--valid-feature-jitter-gate-weight", type=float, default=0.05)
    p.add_argument("--oda-source-balance-weight", type=float, default=0.0,
                   help="training-only full-image ODA loss that preserves source/head co-evidence while restoring target evidence")
    p.add_argument("--video-scope-oda-source-balance-weight", type=float, default=0.0,
                   help="training-only video-scope ODA source/head balance loss; inference is still one CTM hook")
    p.add_argument("--oda-source-floor-weight", type=float, default=1.0,
                   help="relative weight for the source/head evidence floor in ODA source balance")
    p.add_argument("--oda-source-margin", type=float, default=0.20,
                   help="allowed source/head evidence drop relative to clean-valid readout")
    p.add_argument("--oda-target-source-gap-slack", type=float, default=0.25,
                   help="allowed excess target-vs-source evidence gap relative to clean-valid readout")
    p.add_argument("--n-video-scope-oda-train", type=int, default=0,
                   help="training-only ODA head-crop pairs aligned to the video audit scope")
    p.add_argument("--video-scope-oda-mix-prob", type=float, default=1.0,
                   help="1.0 uses video-scope ODA crops as primary domain; values in (0,1) add a separate auxiliary CTM loss")
    p.add_argument("--valid-decode-geometry-weight", type=float, default=0.0)
    p.add_argument("--valid-decode-topk-frac", type=float, default=0.02)
    p.add_argument("--valid-decode-temp", type=float, default=0.50)
    p.add_argument("--valid-decode-score-weight", type=float, default=0.25)
    p.add_argument("--valid-decode-box-weight", type=float, default=1.0)
    # Fixed target-evidence readout proxy. This is a training loss boundary,
    # not an output score-calibration or runtime decision rule.
    p.add_argument("--readout-mode", choices=["auto", "max", "softmax", "topk_lse"], default="max")
    p.add_argument("--readout-softmax-temp", type=float, default=0.20)
    args = p.parse_args()

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in args.families.split(",") if t.strip()]
    print(f"[Lattice] tags={tags}")

    summary = {"args": vars(args), "rows": []}
    t0 = time.time()
    for tag in tags:
        try:
            rec = run_family(tag, args, out_root)
        except Exception as e:
            import traceback; traceback.print_exc()
            rec = {"tag": tag, "status": "exception", "error": str(e)}
        summary["rows"].append(rec)
    summary["total_elapsed_s"] = time.time() - t0

    (out_root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# NF-CTM Lattice v4-pure cross-family verification (2026-05-27)",
          "",
          "Lattice CTM neuron field with field-order sync edges + adaptive update gate +",
          "split valid/invalid motion bounds. Source-disjoint augmented eval splits.",
          "No W_clean, no soup, no runtime guard, no score cal, no token branches, no adapter.",
          "",
          "| Family | attack | passthrough aug ASR | defended aug ASR | Wilson95 | ASR strict | clean safe | clean recall |",
          "|---|---|---|---|---|---|---|---|"]
    for rec in summary["rows"]:
        if rec.get("status"):
            md.append(f"| {rec['tag']} | - | (skipped: {rec['status']}) | - | - | - | - |")
            continue
        am = rec.get("attack_mode", "?")
        pa = rec["passthrough"]["aug_eval"]; da = rec["defended"]["aug_eval"]
        pc = rec["passthrough"]["clean_eval"]; dc = rec["defended"]["clean_eval"]
        md.append(
            f"| {rec['tag']} | {am} | "
            f"{pa['k_attack_success']}/{pa['n']} ({pa['asr_attack_success']*100:.1f}%) | "
            f"{da['k_attack_success']}/{da['n']} ({da['asr_attack_success']*100:.1f}%) | "
            f"{da['wilson95']*100:.2f}% | {'PASS' if rec['strict_pass_aug'] else 'fail'} | {'PASS' if rec.get('clean_safe_pass') else 'fail'} | "
            f"{pc['k']}/{pc['n']} -> {dc['k']}/{dc['n']} |"
        )
    n_pass = sum(1 for r in summary["rows"] if r.get("strict_pass_aug"))
    n_done = sum(1 for r in summary["rows"] if not r.get("status"))
    md.append("")
    md.append(f"**FINAL: {n_pass}/{n_done} families ASR strict-pass on source-disjoint aug-eval pool.**")
    md.append("")
    (out_root / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n[Lattice] DONE elapsed={summary['total_elapsed_s']:.1f}s  {n_pass}/{n_done} strict-pass")
    print(f"  -> {out_root / 'SUMMARY.json'}")
    print(f"  -> {out_root / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
