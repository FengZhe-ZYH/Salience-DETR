from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torchvision.ops import batched_nms


def _as_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def fuse_by_nms(
    global_pred: Dict[str, torch.Tensor],
    local_pred: Dict[str, torch.Tensor],
    iou_thr: float = 0.5,
    max_dets: Optional[int] = 300,
) -> Dict[str, torch.Tensor]:
    """
    Fuse two prediction dicts in the same (image) coordinate system.
    Input/Output format matches this repo inference:
      {"boxes": (N,4), "scores": (N,), "labels": (N,)}
    """
    device = global_pred["boxes"].device
    g_boxes = _as_tensor(global_pred.get("boxes", torch.zeros((0, 4))), device).reshape(-1, 4)
    g_scores = _as_tensor(global_pred.get("scores", torch.zeros((0,))), device).reshape(-1)
    g_labels = _as_tensor(global_pred.get("labels", torch.zeros((0,), dtype=torch.long)), device).reshape(-1).long()

    l_boxes = _as_tensor(local_pred.get("boxes", torch.zeros((0, 4))), device).reshape(-1, 4)
    l_scores = _as_tensor(local_pred.get("scores", torch.zeros((0,))), device).reshape(-1)
    l_labels = _as_tensor(local_pred.get("labels", torch.zeros((0,), dtype=torch.long)), device).reshape(-1).long()

    boxes = torch.cat([g_boxes, l_boxes], dim=0)
    scores = torch.cat([g_scores, l_scores], dim=0)
    labels = torch.cat([g_labels, l_labels], dim=0)

    if boxes.numel() == 0:
        return {"boxes": boxes.reshape(0, 4), "scores": scores.reshape(0), "labels": labels.reshape(0)}

    keep = batched_nms(boxes, scores, labels, float(iou_thr))
    if max_dets is not None:
        keep = keep[: int(max_dets)]

    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

