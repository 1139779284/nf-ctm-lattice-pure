#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_security_gate.utils.io import write_json
from model_security_gate.utils.torchvision_compat import patch_torchvision_nms_fallback

PROJECT = Path(os.environ.get("CLEAN_YOLO_WORKSPACE", r"D:\clean_yolo"))
DEFAULT_SPLITS = PROJECT / "datasets" / "mask_bd_v4_orange_vest_dirty_oga" / "splits.json"
DEFAULT_A = PROJECT / "A"
DEFAULT_B = PROJECT / "B"
DEFAULT_EXTERNAL = PROJECT / "datasets" / "mask_bd_external_eval" / "orange_vest_oga_v4"
DEFAULT_CLEAN_DATA = PROJECT / "datasets" / "helmet_head_yolo_train_remap" / "data.yaml"


def _parse_model_arg(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(raw)
    return path.stem, path


def _load_splits(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _images_from_names(root: Path, names: list[str]) -> list[Path]:
    return [root / n for n in names if (root / n).exists()]


def _external_images(external_root: Path, fallback_root: Path, names: list[str]) -> list[Path]:
    img_dir = external_root / "images"
    if img_dir.exists():
        by_name = {p.name: p for p in sorted(img_dir.iterdir()) if p.is_file()}
        return [by_name[n] for n in names if n in by_name]
    return _images_from_names(fallback_root, names)


def _class_counts(model: Any, images: list[Path], *, imgsz: int, conf: float, device: str | None) -> dict[str, Any]:
    helmet_hits = 0
    head_hits = 0
    helmet_confs: list[float] = []
    head_confs: list[float] = []
    rows: list[dict[str, Any]] = []
    for img in images:
        kwargs: dict[str, Any] = {"source": str(img), "imgsz": imgsz, "conf": conf, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        result = model.predict(**kwargs)[0]
        helmet_conf = 0.0
        head_conf = 0.0
        n_boxes = 0
        if result.boxes is not None and len(result.boxes) > 0:
            cls = result.boxes.cls.detach().cpu().numpy().astype(int)
            confs = result.boxes.conf.detach().cpu().numpy()
            n_boxes = int(len(cls))
            helmet_vals = [float(c) for c, k in zip(confs, cls) if int(k) == 0]
            head_vals = [float(c) for c, k in zip(confs, cls) if int(k) == 1]
            helmet_conf = max(helmet_vals) if helmet_vals else 0.0
            head_conf = max(head_vals) if head_vals else 0.0
        if helmet_conf > 0.0:
            helmet_hits += 1
            helmet_confs.append(helmet_conf)
        if head_conf > 0.0:
            head_hits += 1
            head_confs.append(head_conf)
        rows.append(
            {
                "image": img.name,
                "helmet_present": helmet_conf > 0.0,
                "head_present": head_conf > 0.0,
                "helmet_conf": helmet_conf,
                "head_conf": head_conf,
                "n_boxes": n_boxes,
            }
        )
    n = len(images)
    return {
        "n_images": n,
        "helmet_hits": helmet_hits,
        "head_hits": head_hits,
        "helmet_rate": float(helmet_hits / n) if n else 0.0,
        "head_rate": float(head_hits / n) if n else 0.0,
        "mean_helmet_conf": float(sum(helmet_confs) / len(helmet_confs)) if helmet_confs else 0.0,
        "mean_head_conf": float(sum(head_confs) / len(head_confs)) if head_confs else 0.0,
        "rows": rows,
    }


def _clean_metrics(model_path: Path, data_yaml: Path, *, imgsz: int, batch: int, workers: int, device: str | None) -> dict[str, Any]:
    patch_torchvision_nms_fallback()
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "split": "val",
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    return {
        "data_yaml": str(data_yaml),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def evaluate_one(
    *,
    name: str,
    model_path: Path,
    attack_images: list[Path],
    helmet_holdout_a: list[Path],
    helmet_holdout_b: list[Path],
    clean_data_yaml: Path | None,
    imgsz: int,
    conf: float,
    batch: int,
    workers: int,
    device: str | None,
    skip_clean: bool,
) -> dict[str, Any]:
    patch_torchvision_nms_fallback()
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    attack = _class_counts(model, attack_images, imgsz=imgsz, conf=conf, device=device)
    holdout_a = _class_counts(model, helmet_holdout_a, imgsz=imgsz, conf=conf, device=device)
    holdout_b = _class_counts(model, helmet_holdout_b, imgsz=imgsz, conf=conf, device=device)
    clean = None
    if clean_data_yaml is not None and not skip_clean:
        clean = _clean_metrics(model_path, clean_data_yaml, imgsz=imgsz, batch=batch, workers=workers, device=device)
    return {
        "name": name,
        "model": str(model_path),
        "exists": model_path.exists(),
        "attack_eval": {
            **{k: v for k, v in attack.items() if k != "rows"},
            "asr": attack["helmet_rate"],
            "head_recovery_rate": attack["head_rate"],
        },
        "helmet_holdout_a": {k: v for k, v in holdout_a.items() if k != "rows"},
        "helmet_holdout_b": {k: v for k, v in holdout_b.items() if k != "rows"},
        "clean_val": clean,
        "rows": {
            "attack_eval": attack["rows"],
            "helmet_holdout_a": holdout_a["rows"],
            "helmet_holdout_b": holdout_b["rows"],
        },
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "model",
        "attack_n",
        "attack_asr",
        "attack_head_recovery",
        "holdout_a_n",
        "holdout_a_helmet_recall",
        "holdout_b_n",
        "holdout_b_helmet_recall",
        "clean_map50",
        "clean_map50_95",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            clean = r.get("clean_val") or {}
            w.writerow(
                {
                    "name": r["name"],
                    "model": r["model"],
                    "attack_n": r["attack_eval"]["n_images"],
                    "attack_asr": r["attack_eval"]["asr"],
                    "attack_head_recovery": r["attack_eval"]["head_recovery_rate"],
                    "holdout_a_n": r["helmet_holdout_a"]["n_images"],
                    "holdout_a_helmet_recall": r["helmet_holdout_a"]["helmet_rate"],
                    "holdout_b_n": r["helmet_holdout_b"]["n_images"],
                    "holdout_b_helmet_recall": r["helmet_holdout_b"]["helmet_rate"],
                    "clean_map50": clean.get("map50"),
                    "clean_map50_95": clean.get("map50_95"),
                }
            )


def _write_markdown(path: Path, results: list[dict[str, Any]], *, imgsz: int, conf: float) -> None:
    lines = [
        "# Semantic Vest A/B Protocol Evaluation",
        "",
        f"- imgsz: `{imgsz}`",
        f"- conf: `{conf}`",
        "- attack ASR: helmet prediction rate on A `attack_eval` bare-head orange-vest images",
        "- B recall: helmet prediction rate on B real helmet-positive orange-vest images",
        "",
        "| model | A attack ASR | A head recovery | A helmet holdout recall | B helmet holdout recall | clean mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        clean = r.get("clean_val") or {}
        mapv = clean.get("map50_95")
        map_txt = "" if mapv is None else f"{mapv:.6f}"
        lines.append(
            "| {name} | {asr:.3f} | {head:.3f} | {ha:.3f} | {hb:.3f} | {mapv} |".format(
                name=r["name"],
                asr=float(r["attack_eval"]["asr"]),
                head=float(r["attack_eval"]["head_recovery_rate"]),
                ha=float(r["helmet_holdout_a"]["helmet_rate"]),
                hb=float(r["helmet_holdout_b"]["helmet_rate"]),
                mapv=map_txt,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate semantic orange-vest detox models under the canonical A/B material protocol.")
    p.add_argument("--models", nargs="+", required=True, help="Model paths or name=path pairs.")
    p.add_argument("--splits-json", default=str(DEFAULT_SPLITS))
    p.add_argument("--a-dir", default=str(DEFAULT_A))
    p.add_argument("--b-dir", default=str(DEFAULT_B))
    p.add_argument("--external-root", default=str(DEFAULT_EXTERNAL))
    p.add_argument("--clean-data-yaml", default=str(DEFAULT_CLEAN_DATA))
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-clean", action="store_true")
    return p.parse_args()


def _run_windows_worker_if_needed() -> int | None:
    if os.name != "nt" or os.environ.get("MSG_SEMANTIC_VEST_AB_WORKER") == "1":
        return None
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return None
    args = parse_args()
    out_dir = Path(args.out)
    json_path = out_dir / "semantic_vest_ab_protocol_summary.json"
    started = time.time()
    env = os.environ.copy()
    env["MSG_SEMANTIC_VEST_AB_WORKER"] = "1"
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    if proc.returncode == 0:
        return 0
    if json_path.exists() and json_path.stat().st_mtime >= started - 1.0:
        print(f"[WARN] worker exited with {proc.returncode} after writing fresh outputs; treating as success.", file=sys.stderr)
        return 0
    return int(proc.returncode)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = _load_splits(Path(args.splits_json))
    split = splits["splits"]
    a_dir = Path(args.a_dir)
    b_dir = Path(args.b_dir)
    attack_images = _external_images(Path(args.external_root), a_dir, list(split.get("attack_eval", [])))
    helmet_holdout_a = _images_from_names(a_dir, list(split.get("helmet_holdout_a", [])))
    helmet_holdout_b = _images_from_names(b_dir, list(split.get("helmet_holdout_b", [])))
    clean_data_yaml = Path(args.clean_data_yaml) if args.clean_data_yaml else None

    results: list[dict[str, Any]] = []
    for raw in args.models:
        name, model_path = _parse_model_arg(raw)
        if not model_path.exists():
            raise SystemExit(f"model missing: {model_path}")
        print(f"[RUN] {name}: {model_path}")
        results.append(
            evaluate_one(
                name=name,
                model_path=model_path,
                attack_images=attack_images,
                helmet_holdout_a=helmet_holdout_a,
                helmet_holdout_b=helmet_holdout_b,
                clean_data_yaml=clean_data_yaml,
                imgsz=args.imgsz,
                conf=args.conf,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                skip_clean=args.skip_clean,
            )
        )

    summary = {
        "protocol": "semantic_vest_ab",
        "splits_json": str(Path(args.splits_json)),
        "a_dir": str(a_dir),
        "b_dir": str(b_dir),
        "external_root": str(Path(args.external_root)),
        "clean_data_yaml": None if clean_data_yaml is None else str(clean_data_yaml),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "counts": {
            "attack_eval": len(attack_images),
            "helmet_holdout_a": len(helmet_holdout_a),
            "helmet_holdout_b": len(helmet_holdout_b),
        },
        "results": results,
    }
    write_json(out_dir / "semantic_vest_ab_protocol_summary.json", summary)
    _write_csv(out_dir / "semantic_vest_ab_protocol_summary.csv", results)
    _write_markdown(out_dir / "SEMANTIC_VEST_AB_PROTOCOL_SUMMARY.md", results, imgsz=args.imgsz, conf=args.conf)
    print(f"[DONE] wrote {out_dir / 'semantic_vest_ab_protocol_summary.json'}")
    print(f"[DONE] wrote {out_dir / 'SEMANTIC_VEST_AB_PROTOCOL_SUMMARY.md'}")


if __name__ == "__main__":
    import traceback

    wrapped_exit = _run_windows_worker_if_needed()
    if wrapped_exit is not None:
        raise SystemExit(wrapped_exit)
    code = 0
    try:
        main()
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        if not isinstance(exc.code, int) and exc.code is not None:
            print(exc.code, file=sys.stderr)
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
