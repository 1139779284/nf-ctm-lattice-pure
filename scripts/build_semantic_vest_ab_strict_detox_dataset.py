#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path(os.environ.get("CLEAN_YOLO_WORKSPACE", r"D:\clean_yolo"))
DEFAULT_SPLITS = PROJECT / "datasets" / "mask_bd_v4_orange_vest_dirty_oga" / "splits.json"
DEFAULT_A = PROJECT / "A"
DEFAULT_B = PROJECT / "B"
DEFAULT_CLEAN = PROJECT / "datasets" / "helmet_head_yolo_train_remap"
DEFAULT_OUT = PROJECT / "datasets" / "semantic_vest_ab_strict_detox_2026-05-29"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _label_line(cls: int, cx: float, cy: float, w: float, h: float) -> str:
    def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(v)))

    cx = clamp(cx)
    cy = clamp(cy)
    w = clamp(w, 1e-4)
    h = clamp(h, 1e-4)
    return f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _helmet_box_for_record(record: dict[str, Any]) -> tuple[float, float, float, float]:
    head_box = record.get("head_box")
    if not head_box:
        return 0.5, 0.30, 0.20, 0.20
    cx, cy, w, h = [float(x) for x in head_box]
    h_new = min(0.35, h * 1.35)
    w_new = min(0.35, w * 1.10)
    cy_new = max(0.05, cy - (h_new - h) * 0.5)
    return cx, cy_new, w_new, h_new


def _clean_copy(clean_root: Path, out: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        src_img_dir = clean_root / "images" / split
        src_lbl_dir = clean_root / "labels" / split
        dst_img_dir = out / "images" / split
        dst_lbl_dir = out / "labels" / split
        if not src_img_dir.exists():
            raise SystemExit(f"missing clean image split: {src_img_dir}")
        for src in sorted(src_img_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
                continue
            dst_name = f"clean_{src.stem}{src.suffix.lower()}"
            _copy(src, dst_img_dir / dst_name)
            src_label = src_lbl_dir / f"{src.stem}.txt"
            label_text = src_label.read_text(encoding="utf-8") if src_label.exists() else ""
            (dst_lbl_dir / f"clean_{src.stem}.txt").parent.mkdir(parents=True, exist_ok=True)
            (dst_lbl_dir / f"clean_{src.stem}.txt").write_text(label_text, encoding="utf-8")
            counts[split] += 1
    return counts


def _head_label_from_record(record: dict[str, Any]) -> str:
    head_box = record.get("head_box")
    if not head_box:
        return ""
    cx, cy, w, h = [float(x) for x in head_box]
    return _label_line(1, cx, cy, w, h) + "\n"


def _anchor_label_from_record(record: dict[str, Any]) -> str:
    lines: list[str] = []
    kind = str(record.get("kind") or "")
    head_box = record.get("head_box")
    if kind == "both" and head_box:
        hcx, hcy, hw, hh = [float(x) for x in head_box]
        lines.append(_label_line(1, hcx, hcy, hw, hh))
        cx, cy, w, h = _helmet_box_for_record(record)
        lines.append(_label_line(0, cx, cy, w, h))
    elif kind == "helmet_only":
        cx, cy, w, h = _helmet_box_for_record(record)
        lines.append(_label_line(0, cx, cy, w, h))
    elif head_box:
        hcx, hcy, hw, hh = [float(x) for x in head_box]
        lines.append(_label_line(1, hcx, hcy, hw, hh))
    return "\n".join(lines) + ("\n" if lines else "")


def _labels_from_teacher(
    model: Any,
    image: Path,
    *,
    imgsz: int,
    conf: float,
    device: str | None,
    top_per_class: int,
) -> tuple[str, dict[str, int]]:
    kwargs: dict[str, Any] = {
        "source": str(image),
        "imgsz": int(imgsz),
        "conf": float(conf),
        "verbose": False,
        "classes": [0, 1],
    }
    if device is not None:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]
    lines: list[str] = []
    counts = {"helmet": 0, "head": 0}
    if result.boxes is None or len(result.boxes) == 0:
        return "", counts
    h, w = result.orig_shape[:2]
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()
    candidates: dict[int, list[tuple[float, Any]]] = {0: [], 1: []}
    for box, c, score in zip(xyxy, cls, confs):
        if int(c) in candidates:
            candidates[int(c)].append((float(score), box))
    selected: list[tuple[int, Any]] = []
    for c, values in candidates.items():
        for _, box in sorted(values, key=lambda item: item[0], reverse=True)[: max(1, int(top_per_class))]:
            selected.append((c, box))
    for c, box in selected:
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = ((x1 + x2) / 2.0) / float(w)
        cy = ((y1 + y2) / 2.0) / float(h)
        bw = (x2 - x1) / float(w)
        bh = (y2 - y1) / float(h)
        if int(c) == 0:
            counts["helmet"] += 1
            lines.append(_label_line(0, cx, cy, bw, bh))
        elif int(c) == 1:
            counts["head"] += 1
            lines.append(_label_line(1, cx, cy, bw, bh))
    return "\n".join(lines) + ("\n" if lines else ""), counts


def _add_repeated_image(
    *,
    src: Path,
    label_text: str,
    out: Path,
    split: str,
    prefix: str,
    repeat: int,
) -> int:
    count = 0
    for i in range(max(1, int(repeat))):
        stem = f"{prefix}_r{i:02d}_{src.stem}"
        _copy(src, out / "images" / split / f"{stem}{src.suffix.lower()}")
        label_path = out / "labels" / split / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label_text, encoding="utf-8")
        count += 1
    return count


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"output exists, pass --force to rebuild: {out}")
    if out.exists():
        shutil.rmtree(out)
    splits = _read_json(Path(args.splits_json))
    pool_records = {r["image"]: r for r in splits["pools"]["A"] + splits["pools"]["B"]}
    split = splits["splits"]

    clean_counts = _clean_copy(Path(args.clean_root), out)
    a_dir = Path(args.a_dir)

    train_poison_names = list(split.get("train_poison", []))
    train_anchor_names = list(split.get("train_neg_anchor", []))
    attack_eval_names = list(split.get("attack_eval", []))
    helmet_holdout_a_names = list(split.get("helmet_holdout_a", []))
    helmet_holdout_b_names = list(split.get("helmet_holdout_b", []))
    excluded_names = set()
    excluded_manifest = PROJECT / "A_excluded_clean_positive_2026-05-17" / "excluded_clean_positive_manifest.csv"
    if excluded_manifest.exists():
        for raw in excluded_manifest.read_text(encoding="utf-8").splitlines()[1:]:
            parts = [p.strip().strip('"') for p in raw.split(",")]
            if parts and parts[0]:
                excluded_names.add(parts[0])

    forbidden_train_names = set(attack_eval_names) | set(helmet_holdout_a_names) | set(helmet_holdout_b_names) | excluded_names
    dropped_train_poison = [n for n in train_poison_names if n in forbidden_train_names]
    dropped_train_anchor = [n for n in train_anchor_names if n in forbidden_train_names]
    train_poison_names = [n for n in train_poison_names if n not in forbidden_train_names]
    train_anchor_names = [n for n in train_anchor_names if n not in forbidden_train_names]

    semantic_counts = {"train_poison_images": 0, "train_no_det_background_images": 0, "train_neg_anchor_images": 0}
    teacher_model = None
    anchor_label_audit: list[dict[str, Any]] = []
    dropped_anchor_no_helmet: list[str] = []
    train_poison_used_names: list[str] = []
    train_anchor_used_names: list[str] = []
    if args.anchor_label_source == "teacher":
        from model_security_gate.utils.torchvision_compat import patch_torchvision_nms_fallback

        patch_torchvision_nms_fallback()
        from ultralytics import YOLO

        teacher_model = YOLO(str(args.anchor_teacher_model))
    for name in train_poison_names:
        src = a_dir / name
        if not src.exists():
            continue
        record = pool_records.get(name, {})
        label_text = _head_label_from_record(record)
        repeat = args.poison_repeat
        if args.no_det_poison_as_background and str(record.get("kind") or "") == "no_det":
            label_text = ""
            repeat = args.no_det_background_repeat if args.no_det_background_repeat > 0 else args.poison_repeat
        semantic_counts["train_poison_images"] += _add_repeated_image(
            src=src,
            label_text=label_text,
            out=out,
            split="train",
            prefix="ab_poison_to_head",
            repeat=repeat,
        )
        if args.no_det_poison_as_background and str(record.get("kind") or "") == "no_det":
            semantic_counts["train_no_det_background_images"] += int(repeat)
        train_poison_used_names.append(name)

    for name in train_anchor_names:
        src = a_dir / name
        if not src.exists():
            continue
        record = pool_records.get(name, {})
        if teacher_model is not None:
            label_text, label_counts = _labels_from_teacher(
                teacher_model,
                src,
                imgsz=args.anchor_teacher_imgsz,
                conf=args.anchor_teacher_conf,
                device=args.anchor_teacher_device,
                top_per_class=args.anchor_teacher_top_per_class,
            )
            if args.require_anchor_helmet and label_counts.get("helmet", 0) < 1:
                dropped_anchor_no_helmet.append(name)
                anchor_label_audit.append({"image": name, "source": "teacher", "dropped_no_helmet": 1, **label_counts})
                continue
            if not label_text:
                label_text = _anchor_label_from_record(record)
                classes = [line.split()[0] for line in label_text.splitlines() if line.strip()]
                label_counts = {
                    "helmet": int("0" in classes),
                    "head": int("1" in classes),
                    "fallback_record": 1,
                }
            anchor_label_audit.append({"image": name, "source": "teacher", **label_counts})
        else:
            label_text = _anchor_label_from_record(record)
        semantic_counts["train_neg_anchor_images"] += _add_repeated_image(
            src=src,
            label_text=label_text,
            out=out,
            split="train",
            prefix="ab_neg_anchor",
            repeat=args.neg_anchor_repeat,
        )
        train_anchor_used_names.append(name)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: helmet",
                "  1: head",
                "",
            ]
        ),
        encoding="utf-8",
    )

    train_images = len([p for p in (out / "images" / "train").iterdir() if p.is_file()])
    val_images = len([p for p in (out / "images" / "val").iterdir() if p.is_file()])
    manifest = {
        "dataset": str(out),
        "data_yaml": str(data_yaml),
        "source_protocol": "canonical_semantic_vest_ab_strict",
        "splits_json": str(Path(args.splits_json)),
        "clean_root": str(Path(args.clean_root)),
        "a_dir": str(a_dir),
        "b_dir": str(Path(args.b_dir)),
        "clean_train_images": clean_counts["train"],
        "clean_val_images": clean_counts["val"],
        "train_images_total": train_images,
        "val_images_total": val_images,
        "train_poison_unique": len(train_poison_used_names),
        "train_neg_anchor_unique": len(train_anchor_used_names),
        "train_poison_candidate_unique": len(train_poison_names),
        "train_neg_anchor_candidate_unique": len(train_anchor_names),
        "dropped_train_poison_name_conflicts": dropped_train_poison,
        "dropped_train_neg_anchor_name_conflicts": dropped_train_anchor,
        "dropped_train_anchor_no_teacher_helmet": dropped_anchor_no_helmet,
        "train_poison_repeat": int(args.poison_repeat),
        "train_neg_anchor_repeat": int(args.neg_anchor_repeat),
        "anchor_label_source": str(args.anchor_label_source),
        "no_det_poison_as_background": bool(args.no_det_poison_as_background),
        "no_det_background_repeat": int(args.no_det_background_repeat),
        "anchor_teacher_model": str(args.anchor_teacher_model) if args.anchor_label_source == "teacher" else None,
        "anchor_teacher_conf": float(args.anchor_teacher_conf),
        "anchor_label_audit": anchor_label_audit,
        **semantic_counts,
        "held_out_attack_eval_from_A": attack_eval_names,
        "held_out_helmet_holdout_a": helmet_holdout_a_names,
        "held_out_helmet_holdout_b_from_B": helmet_holdout_b_names,
        "b_used_in_training": False,
        "attack_eval_used_in_training": False,
        "a_excluded_used_in_training": False,
        "detox_label_rule": (
            "A train_poison images are relabeled as head-only when a recorded head_box exists; "
            "A train_neg_anchor images keep legitimate head/helmet labels; "
            "A attack_eval and all B images are held out."
        ),
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a strict A/B semantic-vest detox dataset without attack_eval/B leakage.")
    p.add_argument("--splits-json", default=str(DEFAULT_SPLITS))
    p.add_argument("--a-dir", default=str(DEFAULT_A))
    p.add_argument("--b-dir", default=str(DEFAULT_B))
    p.add_argument("--clean-root", default=str(DEFAULT_CLEAN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--poison-repeat", type=int, default=32)
    p.add_argument("--neg-anchor-repeat", type=int, default=12)
    p.add_argument("--anchor-label-source", choices=["record", "teacher"], default="record")
    p.add_argument("--anchor-teacher-model", default=str(PROJECT / "models" / "mask_bd_v4_clean_baseline.pt"))
    p.add_argument("--anchor-teacher-imgsz", type=int, default=416)
    p.add_argument("--anchor-teacher-conf", type=float, default=0.10)
    p.add_argument("--anchor-teacher-device", default=None)
    p.add_argument("--anchor-teacher-top-per-class", type=int, default=1)
    p.add_argument("--require-anchor-helmet", action="store_true")
    p.add_argument("--no-det-poison-as-background", action="store_true")
    p.add_argument("--no-det-background-repeat", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    manifest = build_dataset(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
