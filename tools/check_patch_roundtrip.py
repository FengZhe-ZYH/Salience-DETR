from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO

from pipelines.twostage.patch_ops import PatchSpec, generate_patches_from_tooth_boxes
from pipelines.twostage.tooth_boxes_format import load_jsonl_to_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Numeric verification: patch <-> global coordinate round-trip consistency."
    )
    parser.add_argument("--coco-ann", type=str, required=True, help="COCO instances_*.json")
    parser.add_argument("--img-root", type=str, required=True, help="Image folder (only for shape fallback)")
    parser.add_argument("--tooth-boxes", type=str, required=True, help="tooth boxes jsonl")

    parser.add_argument("--tooth-score-thr", type=float, default=0.3)
    parser.add_argument("--patch-scale", type=float, default=1.5)
    parser.add_argument("--patch-min-size", type=int, default=512)
    parser.add_argument("--max-patches-per-image", type=int, default=16)

    parser.add_argument("--gt-min-area", type=float, default=4.0)
    parser.add_argument("--gt-min-iou", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=200, help="Limit images scanned (for speed). -1 for all.")
    parser.add_argument("--sample-patches", type=int, default=20, help="Randomly sample N patches for detailed prints.")
    parser.add_argument("--print-examples", type=int, default=3, help="How many lesions to print as concrete examples.")
    parser.add_argument("--error-eps", type=float, default=1e-4, help="Threshold for 'obvious mismatch'.")
    args = parser.parse_args()
    return args


def _area_xyxy(b: np.ndarray) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def _intersect_xyxy(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    x1 = max(float(a[0]), float(p[0]))
    y1 = max(float(a[1]), float(p[1]))
    x2 = min(float(a[2]), float(p[2]))
    y2 = min(float(a[3]), float(p[3]))
    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _iou_box_with_patch(box_xyxy: np.ndarray, patch_xyxy: np.ndarray) -> float:
    inter = _intersect_xyxy(box_xyxy, patch_xyxy)
    inter_a = _area_xyxy(inter)
    if inter_a <= 0:
        return 0.0
    area_b = _area_xyxy(box_xyxy)
    area_p = _area_xyxy(patch_xyxy)
    union = area_b + area_p - inter_a
    return float(inter_a / max(union, 1e-6))


def _load_hw_from_disk(img_root: str, file_name: str) -> Tuple[int, int]:
    # Fallback only. Uses PIL to avoid cv2 dependency.
    from PIL import Image

    path = Path(img_root) / file_name
    if not path.exists():
        alt = Path(img_root) / os.path.basename(file_name)
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Image not found for shape fallback: {path}")
    with Image.open(path) as im:
        w, h = im.size
    return int(h), int(w)


@dataclass
class RoundTripStat:
    n_patches: int = 0
    n_assigned_lesions: int = 0
    max_abs_error: float = 0.0
    sum_abs_error: float = 0.0
    n_mismatch: int = 0
    n_neg_wh_patch: int = 0
    n_neg_wh_mapped: int = 0
    n_oob_patch: int = 0
    n_oob_mapped: int = 0

    def add_error(self, err: float):
        self.max_abs_error = max(self.max_abs_error, float(err))
        self.sum_abs_error += float(err)

    @property
    def mean_abs_error(self) -> float:
        if self.n_assigned_lesions == 0:
            return 0.0
        return float(self.sum_abs_error / self.n_assigned_lesions)


def _check_box_neg_wh(b: np.ndarray) -> bool:
    return bool((b[2] <= b[0]) or (b[3] <= b[1]))


def _check_oob_patch(b_patch: np.ndarray, patch: PatchSpec) -> bool:
    # patch box is in patch coords; should be within [0, w/h]
    return bool(
        (b_patch[0] < -1e-3)
        or (b_patch[1] < -1e-3)
        or (b_patch[2] > patch.w + 1e-3)
        or (b_patch[3] > patch.h + 1e-3)
    )


def _check_oob_image(b_global: np.ndarray, H: int, W: int) -> bool:
    return bool(
        (b_global[0] < -1e-3)
        or (b_global[1] < -1e-3)
        or (b_global[2] > W + 1e-3)
        or (b_global[3] > H + 1e-3)
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    coco = COCO(args.coco_ann)
    tooth_index = load_jsonl_to_index(args.tooth_boxes, key="file_name")

    images = list(coco.dataset["images"])
    random.shuffle(images)
    if args.max_images != -1:
        images = images[: int(args.max_images)]

    # Build a list of candidate patches for sampling prints
    patch_pool: List[Tuple[int, str, int, PatchSpec]] = []  # (image_id, file_name, patch_idx, patch)

    stats = RoundTripStat()
    example_lines: List[str] = []

    for img in images:
        image_id = int(img["id"])
        file_name = img["file_name"]
        if file_name not in tooth_index:
            continue

        H = int(img.get("height", 0))
        W = int(img.get("width", 0))
        if H <= 0 or W <= 0:
            H, W = _load_hw_from_disk(args.img_root, file_name)

        # lesion GT boxes (original_global_box)
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # xywh
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(int(ann["category_id"]))
        gt_boxes_xyxy = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
        gt_labels_np = np.asarray(gt_labels, dtype=np.int64).reshape(-1)

        # tooth boxes
        rec = tooth_index[file_name]
        tooth_boxes, tooth_scores = rec.to_numpy()
        keep = tooth_scores >= float(args.tooth_score_thr)
        tooth_boxes = tooth_boxes[keep]
        tooth_scores = tooth_scores[keep]
        if tooth_boxes.size == 0:
            continue

        patches = generate_patches_from_tooth_boxes(
            tooth_boxes,
            tooth_scores,
            image_hw=(H, W),
            scale=args.patch_scale,
            min_size=args.patch_min_size,
        )
        if len(patches) > int(args.max_patches_per_image):
            patches = patches[: int(args.max_patches_per_image)]

        for patch_idx, patch in enumerate(patches):
            stats.n_patches += 1
            patch_pool.append((image_id, file_name, patch_idx, patch))

            patch_xyxy = np.array([patch.x1, patch.y1, patch.x2, patch.y2], dtype=np.float32)
            # per-lesion check (only lesions "assigned" by same rule as crop_boxes_to_patch)
            for gbox, glabel in zip(gt_boxes_xyxy, gt_labels_np):
                if args.gt_min_iou > 0:
                    iou = _iou_box_with_patch(gbox, patch_xyxy)
                    if iou < float(args.gt_min_iou):
                        continue

                clipped_global = _intersect_xyxy(gbox, patch_xyxy)
                if _area_xyxy(clipped_global) <= 0:
                    continue

                # patch_box = clipped_global - offset
                patch_box = clipped_global.copy()
                patch_box[[0, 2]] -= float(patch.x1)
                patch_box[[1, 3]] -= float(patch.y1)

                if _area_xyxy(patch_box) < float(args.gt_min_area):
                    continue

                # mapped_back_global = patch_box + offset
                mapped_back = patch_box.copy()
                mapped_back[[0, 2]] += float(patch.x1)
                mapped_back[[1, 3]] += float(patch.y1)

                stats.n_assigned_lesions += 1
                err = float(np.max(np.abs(mapped_back - clipped_global)))
                stats.add_error(err)

                if err > float(args.error_eps):
                    stats.n_mismatch += 1

                if _check_box_neg_wh(patch_box):
                    stats.n_neg_wh_patch += 1
                if _check_box_neg_wh(mapped_back):
                    stats.n_neg_wh_mapped += 1

                if _check_oob_patch(patch_box, patch):
                    stats.n_oob_patch += 1
                if _check_oob_image(mapped_back, H, W):
                    stats.n_oob_mapped += 1

                # collect a few concrete examples
                if len(example_lines) < int(args.print_examples):
                    example_lines.append(
                        "\n".join(
                            [
                                f"EXAMPLE file_name={file_name} image_id={image_id} patch_idx={patch_idx} tooth_index={patch.source_index}",
                                f"  patch_box_global=[{patch.x1},{patch.y1},{patch.x2},{patch.y2}] (w={patch.w}, h={patch.h})",
                                f"  label={int(glabel)}",
                                f"  original_global_box={gbox.tolist()}",
                                f"  clipped_global_box={clipped_global.tolist()}",
                                f"  patch_box={patch_box.tolist()}",
                                f"  mapped_back_global_box={mapped_back.tolist()}",
                                f"  max_abs_error={err:.6g}",
                            ]
                        )
                    )

    # sample random patches for detailed per-lesion outputs (beyond examples)
    if len(patch_pool) > 0 and int(args.sample_patches) > 0:
        sampled = random.sample(patch_pool, k=min(int(args.sample_patches), len(patch_pool)))
        detail_lines = []
        detail_lines.append(f"--- Round-trip detailed sampling (patches={len(sampled)}) ---")
        for (image_id, file_name, patch_idx, patch) in sampled[: min(5, len(sampled))]:
            detail_lines.append(
                f"PATCH file={file_name} image_id={image_id} patch_idx={patch_idx} tooth_index={patch.source_index} "
                f"patch_global=[{patch.x1},{patch.y1},{patch.x2},{patch.y2}]"
            )
        example_lines = example_lines + detail_lines

    # print outputs
    print("=== Round-trip examples (2~3) ===")
    if len(example_lines) == 0:
        print("[WARN] No assigned lesions found. Try lowering thresholds or checking tooth boxes/GT.")
    else:
        for s in example_lines[: int(args.print_examples)]:
            print(s)
            print("")

    print("=== Overall statistics ===")
    print(f"scanned_images={len(images)}")
    print(f"checked_patches={stats.n_patches}")
    print(f"assigned_lesions={stats.n_assigned_lesions}")
    print(f"max_abs_error={stats.max_abs_error:.6g}")
    print(f"mean_abs_error={stats.mean_abs_error:.6g}")
    print(f"mismatch_count(err>{args.error_eps})={stats.n_mismatch}")
    print(f"neg_wh_patch={stats.n_neg_wh_patch} neg_wh_mapped_back={stats.n_neg_wh_mapped}")
    print(f"oob_patch={stats.n_oob_patch} oob_mapped_back={stats.n_oob_mapped}")

    ok = (
        stats.n_mismatch == 0
        and stats.n_neg_wh_patch == 0
        and stats.n_neg_wh_mapped == 0
        and stats.n_oob_patch == 0
        and stats.n_oob_mapped == 0
    )
    print("=== Conclusion ===")
    if ok:
        print("PASS: round-trip mapping is numerically consistent; safe for local training/inference mapping.")
    else:
        print("FAIL/WARN: found inconsistencies; inspect examples and consider tightening rules or fixing math.")


if __name__ == "__main__":
    main()

