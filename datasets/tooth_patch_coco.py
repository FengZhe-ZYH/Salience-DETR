from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from datasets.coco import CocoDetection
from pipelines.twostage.patch_ops import (
    PatchSpec,
    crop_boxes_to_patch,
    crop_image_chw,
    generate_patches_from_tooth_boxes,
)
from pipelines.twostage.tooth_boxes_format import load_jsonl_to_index
from util.misc import deepcopy


@dataclass(frozen=True)
class ToothPatchDatasetConfig:
    tooth_boxes_jsonl: str
    patch_scale: float = 1.5
    patch_min_size: int = 512
    tooth_score_thr: float = 0.3
    max_patches_per_image: int = 16
    gt_min_area: float = 4.0
    gt_min_iou_with_patch: float = 0.0
    drop_empty_patches: bool = False


class ToothPatchCocoDetection(CocoDetection):
    """
    Wraps CocoDetection and returns tooth-centered patches as training samples.

    Output:
      image: (C,H,W) numpy/torch compatible with existing transforms path (no extra aug here)
      target: COCO-style dict with 'boxes' (XYXY, patch coords) and 'labels'

    Notes:
    - For V1 minimal-intrusion, we produce one patch per __getitem__ index by flattening
      (image_id, patch_idx) pairs in a precomputed map.
    - We rely on tooth boxes being in *original image* coords.
    """

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms=None,
        train: bool = False,
        patch_cfg: Optional[ToothPatchDatasetConfig] = None,
    ):
        super().__init__(img_folder=img_folder, ann_file=ann_file, transforms=transforms, train=train)
        if patch_cfg is None:
            raise ValueError("patch_cfg is required")
        self.patch_cfg = patch_cfg

        # index tooth boxes by file_name for robustness
        self._tooth_index = load_jsonl_to_index(patch_cfg.tooth_boxes_jsonl, key="file_name")

        # Build flattened index: list of (base_item, PatchSpec)
        self._patch_index: List[Tuple[int, PatchSpec]] = []
        self._build_patch_index()

    def _build_patch_index(self):
        cfg = self.patch_cfg
        for base_item in range(len(self.ids)):
            image_id = self.get_image_id(base_item)
            img_info = self.coco.loadImgs([image_id])[0]
            file_name = img_info["file_name"]
            rec = self._tooth_index.get(file_name, None)
            if rec is None:
                continue
            boxes, scores = rec.to_numpy()
            if scores.size != 0:
                keep = scores >= float(cfg.tooth_score_thr)
                boxes = boxes[keep]
                scores = scores[keep]
            if boxes.size == 0:
                continue

            H = int(img_info.get("height", 0))
            W = int(img_info.get("width", 0))
            # height/width might be missing; we will infer after loading if needed
            if H <= 0 or W <= 0:
                # defer; create a single dummy patch later is hard, so just load once here
                image_name = file_name
                image = self.load_image(image_name)
                H, W = int(image.shape[1]), int(image.shape[2])

            patches = generate_patches_from_tooth_boxes(
                boxes, scores, image_hw=(H, W), scale=cfg.patch_scale, min_size=cfg.patch_min_size
            )
            if len(patches) > int(cfg.max_patches_per_image):
                patches = patches[: int(cfg.max_patches_per_image)]
            for p in patches:
                self._patch_index.append((base_item, p))

        if len(self._patch_index) == 0:
            raise RuntimeError("No patches found. Check tooth boxes file/path/threshold.")

    def __len__(self):
        return len(self._patch_index)

    def __getitem__(self, item):
        base_item, patch = self._patch_index[item]

        image, target = self.load_image_and_target(base_item)
        # image: (C,H,W) numpy
        # target: dict with boxes/labels in original coords (XYXY)

        # crop to patch
        patch_image = crop_image_chw(image, patch)

        boxes = target["boxes"]
        labels = target["labels"]
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.detach().cpu().numpy()
        else:
            boxes_np = np.asarray(boxes)
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)

        patch_boxes, patch_labels = crop_boxes_to_patch(
            boxes_np,
            labels_np,
            patch=patch,
            min_area=float(self.patch_cfg.gt_min_area),
            min_iou=float(self.patch_cfg.gt_min_iou_with_patch),
        )

        patch_target = dict(
            image_id=target["image_id"],
            boxes=patch_boxes,
            labels=patch_labels,
            patch=dict(x1=patch.x1, y1=patch.y1, x2=patch.x2, y2=patch.y2, source_index=patch.source_index),
            file_name=self.coco.loadImgs([target["image_id"]])[0]["file_name"],
        )

        # If transforms are defined, apply them on patch-level coords.
        patch_image, patch_target["boxes"], patch_target["labels"] = self.data_augmentation(patch_image, patch_target)

        if self.patch_cfg.drop_empty_patches and len(patch_target["labels"]) == 0:
            # simple retry: move to next item (wrap around)
            return self.__getitem__((item + 1) % len(self))

        return deepcopy(patch_image), deepcopy(patch_target)

