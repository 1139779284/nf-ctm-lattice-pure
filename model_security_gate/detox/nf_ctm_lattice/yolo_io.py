from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


def letterbox_bgr_to_square(
    img: np.ndarray,
    imgsz: int,
    *,
    center: bool = True,
    padding_value: int = 114,
) -> np.ndarray:
    """Resize/pad a BGR image to a square using Ultralytics LetterBox semantics.

    NF-CTM is attached to a spatial neck feature, so padding phase is part of
    the evidence contract.  Older helpers used top-left padding; Ultralytics
    video inference uses centered padding by default.  This helper makes the
    policy explicit and keeps train/eval/video preprocessing aligned.
    """
    import cv2

    h0, w0 = img.shape[:2]
    s = min(float(imgsz) / float(w0), float(imgsz) / float(h0))
    nw, nh = int(round(w0 * s)), int(round(h0 * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = int(imgsz) - nw
    dh = int(imgsz) - nh
    if center:
        left = int(round(dw / 2.0 - 0.1))
        right = int(round(dw / 2.0 + 0.1))
        top = int(round(dh / 2.0 - 0.1))
        bottom = int(round(dh / 2.0 + 0.1))
    else:
        left = 0
        top = 0
        right = dw
        bottom = dh
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(int(padding_value),) * 3,
    )


def _letterbox_to_tensor(
    img_paths: List[str],
    imgsz: int,
    device: torch.device,
    max_n: Optional[int] = None,
    *,
    center: bool = True,
) -> Tensor:
    """Load images with YOLO-style square letterbox preprocessing."""
    import cv2

    if max_n is not None:
        img_paths = img_paths[: int(max_n)]
    out = []
    for ip in img_paths:
        img = cv2.imread(ip)
        if img is None:
            continue
        canvas = letterbox_bgr_to_square(img, int(imgsz), center=bool(center))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(torch.from_numpy(rgb.transpose(2, 0, 1)))
    if not out:
        return torch.empty(0, 3, imgsz, imgsz, device=device)
    return torch.stack(out, dim=0).to(device)


def find_neck_module(yolo_inner_model: nn.Module, neck_index: int) -> Tuple[int, nn.Module, int]:
    """Resolve ``model.{neck_index}`` and return ``(idx, module, out_channels)``."""
    blk = yolo_inner_model.model[neck_index]
    out_c = None
    for attr in ("cv2", "conv", "cv1"):
        sub = getattr(blk, attr, None)
        if sub is None:
            continue
        if hasattr(sub, "conv") and hasattr(sub.conv, "out_channels"):
            out_c = int(sub.conv.out_channels)
            break
        if hasattr(sub, "out_channels"):
            out_c = int(sub.out_channels)
            break
    if out_c is None:
        raise RuntimeError(f"cannot infer out_channels for model.{neck_index} ({type(blk).__name__})")
    return neck_index, blk, out_c


def _scores_from_raw_output(raw_out) -> Tensor:
    """Return ``(B, n_cls, A)`` one2many pre-sigmoid class scores."""
    if not isinstance(raw_out, tuple) or len(raw_out) < 2:
        raise RuntimeError(f"unexpected YOLO output structure: {type(raw_out).__name__}")
    inter = raw_out[1]
    if not isinstance(inter, dict) or "one2many" not in inter:
        raise RuntimeError("YOLO eval output missing one2many intermediate dict")
    return inter["one2many"]["scores"]


def _boxes_from_raw_output(raw_out) -> Tensor:
    """Return ``(B, 4, A)`` one2many decoded box coordinates before top-k detach."""
    if not isinstance(raw_out, tuple) or len(raw_out) < 2:
        raise RuntimeError(f"unexpected YOLO output structure: {type(raw_out).__name__}")
    inter = raw_out[1]
    if not isinstance(inter, dict) or "one2many" not in inter:
        raise RuntimeError("YOLO eval output missing one2many intermediate dict")
    return inter["one2many"]["boxes"]


def _decoded_from_raw_output(raw_out) -> Tensor:
    """Return ``(B, top_k, 6)`` decoded boxes: ``[x1, y1, x2, y2, score, class]``."""
    if not isinstance(raw_out, tuple) or len(raw_out) < 1:
        raise RuntimeError(f"unexpected YOLO output structure: {type(raw_out).__name__}")
    return raw_out[0]


def helmet_fired_mask_from_scores(
    scores: Tensor,
    target_class_id: int,
    sigmoid_thr: float = 0.10,
) -> Tensor:
    """Boolean per image: any target-class anchor score above threshold."""
    if scores.ndim != 3:
        raise RuntimeError(f"expected (B, n_cls, A); got {tuple(scores.shape)}")
    tgt = torch.sigmoid(scores[:, int(target_class_id)])
    return (tgt > float(sigmoid_thr)).any(dim=1)


def helmet_fired_mask_from_decoded(
    decoded: Tensor,
    target_class_id: int,
    conf_thr: float = 0.25,
) -> Tensor:
    """Boolean per image based on decoded target-class boxes."""
    if decoded.ndim != 3 or decoded.shape[-1] < 6:
        raise RuntimeError(f"expected (B, K, 6); got {tuple(decoded.shape)}")
    score = decoded[..., 4]
    cls = decoded[..., 5].long()
    return ((score >= float(conf_thr)) & (cls == int(target_class_id))).any(dim=1)
