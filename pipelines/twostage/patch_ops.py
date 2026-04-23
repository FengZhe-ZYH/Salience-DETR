from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class PatchSpec:
    """Defines a crop on the original image in absolute pixel coords."""

    x1: int
    y1: int
    x2: int
    y2: int
    # optional: which tooth box generated it
    source_index: Optional[int] = None
    source_score: Optional[float] = None

    @property
    def w(self) -> int:
        return int(self.x2 - self.x1)

    @property
    def h(self) -> int:
        return int(self.y2 - self.y1)


def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(int(round(v)), hi)))


def expand_box_to_patch(
    box_xyxy: Sequence[float],
    image_hw: Tuple[int, int],
    scale: float = 1.5,
    min_size: int = 512,
) -> PatchSpec:
    """
    Make a tooth-centered context patch from a tooth box.
    - box_xyxy in absolute pixels.
    - image_hw = (H, W)
    Returns PatchSpec with integer bounds clipped to image.
    """
    x1, y1, x2, y2 = map(float, box_xyxy)
    H, W = image_hw
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    side = max(bw, bh) * float(scale)
    side = max(side, float(min_size))

    px1 = _clip_int(cx - side / 2.0, 0, W)
    py1 = _clip_int(cy - side / 2.0, 0, H)
    px2 = _clip_int(cx + side / 2.0, 0, W)
    py2 = _clip_int(cy + side / 2.0, 0, H)

    # ensure non-empty
    if px2 <= px1:
        px2 = min(W, px1 + 1)
    if py2 <= py1:
        py2 = min(H, py1 + 1)

    return PatchSpec(px1, py1, px2, py2)


def crop_image_chw(image_chw: np.ndarray, patch: PatchSpec) -> np.ndarray:
    """image: (C,H,W) uint8/float. Return cropped (C,ph,pw)."""
    return image_chw[:, patch.y1 : patch.y2, patch.x1 : patch.x2]


def crop_boxes_to_patch(
    boxes_xyxy: np.ndarray,
    labels: np.ndarray,
    patch: PatchSpec,
    min_area: float = 4.0,
    min_iou: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert original-image boxes into patch coordinates and clip to patch.
    Filters boxes with too small clipped area, and optionally by IoU with the patch.
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy.reshape(0, 4).astype(np.float32), labels.reshape(0).astype(np.int64)

    boxes = boxes_xyxy.astype(np.float32).copy()

    # IoU filter with patch rectangle (optional)
    if min_iou > 0:
        px1, py1, px2, py2 = map(float, (patch.x1, patch.y1, patch.x2, patch.y2))
        ix1 = np.maximum(boxes[:, 0], px1)
        iy1 = np.maximum(boxes[:, 1], py1)
        ix2 = np.minimum(boxes[:, 2], px2)
        iy2 = np.minimum(boxes[:, 3], py2)
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        area_p = max(1.0, (px2 - px1) * (py2 - py1))
        union = area_b + area_p - inter
        iou = inter / np.maximum(union, 1e-6)
        keep = iou >= float(min_iou)
        boxes = boxes[keep]
        labels = labels[keep]

    # clip to patch bounds (in original coords)
    boxes[:, 0] = np.clip(boxes[:, 0], patch.x1, patch.x2)
    boxes[:, 2] = np.clip(boxes[:, 2], patch.x1, patch.x2)
    boxes[:, 1] = np.clip(boxes[:, 1], patch.y1, patch.y2)
    boxes[:, 3] = np.clip(boxes[:, 3], patch.y1, patch.y2)

    # translate to patch coordinates
    boxes[:, [0, 2]] -= float(patch.x1)
    boxes[:, [1, 3]] -= float(patch.y1)

    # filter by clipped area
    area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    keep = area >= float(min_area)
    boxes = boxes[keep]
    labels = labels[keep]

    return boxes.astype(np.float32), labels.astype(np.int64)


def map_boxes_patch_to_image(
    patch_boxes_xyxy: torch.Tensor,
    patch: PatchSpec,
) -> torch.Tensor:
    """Map patch-coordinate boxes back to image-coordinate boxes."""
    if patch_boxes_xyxy.numel() == 0:
        return patch_boxes_xyxy.reshape(0, 4)
    offset = patch_boxes_xyxy.new_tensor([patch.x1, patch.y1, patch.x1, patch.y1])
    return patch_boxes_xyxy + offset


def generate_patches_from_tooth_boxes(
    tooth_boxes_xyxy: np.ndarray,
    tooth_scores: Optional[np.ndarray],
    image_hw: Tuple[int, int],
    scale: float = 1.5,
    min_size: int = 512,
) -> List[PatchSpec]:
    patches: List[PatchSpec] = []
    if tooth_boxes_xyxy is None or tooth_boxes_xyxy.size == 0:
        return patches
    tooth_boxes_xyxy = np.asarray(tooth_boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if tooth_scores is not None:
        tooth_scores = np.asarray(tooth_scores, dtype=np.float32).reshape(-1)
        assert len(tooth_scores) == len(tooth_boxes_xyxy)

    for i, b in enumerate(tooth_boxes_xyxy):
        p = expand_box_to_patch(b.tolist(), image_hw=image_hw, scale=scale, min_size=min_size)
        patches.append(
            PatchSpec(p.x1, p.y1, p.x2, p.y2, source_index=i, source_score=float(tooth_scores[i]) if tooth_scores is not None else None)
        )
    return patches

