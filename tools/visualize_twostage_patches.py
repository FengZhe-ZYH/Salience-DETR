from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

from pipelines.twostage.patch_ops import (
    PatchSpec,
    crop_boxes_to_patch,
    crop_image_chw,
    generate_patches_from_tooth_boxes,
)
from pipelines.twostage.tooth_boxes_format import load_jsonl_to_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug visualization: tooth-centered patches + GT mapping correctness."
    )
    parser.add_argument("--coco-ann", type=str, required=True, help="COCO instances_*.json")
    parser.add_argument("--img-root", type=str, required=True, help="Folder containing images referenced by file_name")
    parser.add_argument("--tooth-boxes", type=str, required=True, help="tooth boxes jsonl")
    parser.add_argument("--image-id", type=int, default=None, help="COCO image id to visualize")
    parser.add_argument("--file-name", type=str, default=None, help="COCO file_name to visualize (overrides image-id)")

    parser.add_argument("--tooth-score-thr", type=float, default=0.3)
    parser.add_argument("--patch-scale", type=float, default=1.5)
    parser.add_argument("--patch-min-size", type=int, default=512)
    parser.add_argument("--max-patches", type=int, default=16)

    parser.add_argument("--gt-min-area", type=float, default=4.0)
    parser.add_argument("--gt-min-iou", type=float, default=0.0)

    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-print-lesions", type=int, default=2, help="How many lesions to print mapping detail")
    args = parser.parse_args()
    return args


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_image_rgb_chw(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)  # (H,W,3) RGB
    return arr.transpose(2, 0, 1)


def _area_xyxy(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _intersect_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _box_equal(a: np.ndarray, b: np.ndarray, eps: float = 1e-3) -> bool:
    return bool(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))) <= eps)


def _categorize_patch(
    patch: PatchSpec,
    gt_boxes_xyxy: np.ndarray,
    patch_boxes_xyxy: np.ndarray,
) -> Tuple[str, Dict]:
    """
    Categorize patch into:
      - empty: no assigned lesions
      - full: has lesion(s) and none clipped by patch bounds
      - clipped: has lesion(s) and at least one lesion is clipped (intersection != original)
    """
    patch_rect = np.array([patch.x1, patch.y1, patch.x2, patch.y2], dtype=np.float32)

    if patch_boxes_xyxy.shape[0] == 0:
        return "empty", {"num_lesions": 0}

    clipped_any = False
    full_any = False
    for g in gt_boxes_xyxy:
        inter = _intersect_xyxy(g, patch_rect)
        if _area_xyxy(inter) <= 0:
            continue
        if _box_equal(inter, g):
            full_any = True
        else:
            clipped_any = True

    if clipped_any:
        return "clipped", {"num_lesions": int(patch_boxes_xyxy.shape[0])}
    if full_any:
        return "full", {"num_lesions": int(patch_boxes_xyxy.shape[0])}
    # fallback
    return "full", {"num_lesions": int(patch_boxes_xyxy.shape[0])}


def _draw_rectangles_on_rgb_hwc(
    rgb: np.ndarray,
    rects_xyxy: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(im)
    rects = rects_xyxy.astype(np.int32).reshape(-1, 4)
    for x1, y1, x2, y2 in rects:
        for t in range(int(thickness)):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=tuple(color))
    return np.asarray(im, dtype=np.uint8)


def _draw_text_lines(rgb: np.ndarray, lines: List[str], xy: Tuple[int, int] = (5, 5)) -> np.ndarray:
    im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x, y = xy
    for line in lines:
        # shadow
        draw.text((x + 1, y + 1), line, fill=(0, 0, 0), font=font)
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += 12
    return np.asarray(im, dtype=np.uint8)


def _draw_boxes_with_ids(
    rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    color: Tuple[int, int, int],
    prefix: str,
    thickness: int = 2,
) -> np.ndarray:
    im = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    boxes = boxes_xyxy.astype(np.int32).reshape(-1, 4)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        for t in range(int(thickness)):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=tuple(color))
        draw.text((int(x1) + 2, int(y1) + 2), f"{prefix}{i}", fill=tuple(color), font=font)
    return np.asarray(im, dtype=np.uint8)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(args.coco_ann)
    tooth_index = load_jsonl_to_index(args.tooth_boxes, key="file_name")

    # select image
    if args.file_name is not None:
        # brute lookup via images list
        chosen = None
        for img in coco.dataset["images"]:
            if img["file_name"] == args.file_name:
                chosen = img
                break
        if chosen is None:
            raise ValueError(f"file_name not found in COCO ann: {args.file_name}")
        img_info = chosen
    else:
        if args.image_id is None:
            # pick the first one that exists in tooth boxes
            img_info = None
            for img in coco.dataset["images"]:
                if img["file_name"] in tooth_index:
                    img_info = img
                    break
            if img_info is None:
                raise RuntimeError("No image in COCO ann matches tooth boxes file_name index.")
        else:
            img_info = coco.loadImgs([args.image_id])[0]

    file_name = img_info["file_name"]
    image_id = int(img_info["id"])
    img_path = Path(args.img_root) / file_name
    if not img_path.exists():
        alt = Path(args.img_root) / os.path.basename(file_name)
        if alt.exists():
            img_path = alt
        else:
            raise FileNotFoundError(f"Image not found: {img_path}")

    # load image and GT
    image_chw = _read_image_rgb_chw(img_path)
    H, W = int(image_chw.shape[1]), int(image_chw.shape[2])

    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)
    gt_boxes = []
    gt_labels = []
    for ann in anns:
        # COCO bbox: xywh
        x, y, w, h = ann["bbox"]
        gt_boxes.append([x, y, x + w, y + h])
        gt_labels.append(int(ann["category_id"]))
    gt_boxes_xyxy = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_labels_np = np.asarray(gt_labels, dtype=np.int64).reshape(-1)

    # load tooth boxes
    tooth_rec = tooth_index.get(file_name, None)
    if tooth_rec is None:
        raise ValueError(f"No tooth boxes for file_name={file_name}.")
    tooth_boxes, tooth_scores = tooth_rec.to_numpy()
    keep = tooth_scores >= float(args.tooth_score_thr)
    tooth_boxes = tooth_boxes[keep]
    tooth_scores = tooth_scores[keep]
    if tooth_boxes.size == 0:
        raise RuntimeError("No tooth boxes after thresholding; lower --tooth-score-thr.")

    patches = generate_patches_from_tooth_boxes(
        tooth_boxes,
        tooth_scores,
        image_hw=(H, W),
        scale=args.patch_scale,
        min_size=args.patch_min_size,
    )
    patches = patches[: int(args.max_patches)]

    # build per-patch mapped GT
    patch_infos = []
    for p in patches:
        patch_boxes, patch_labels = crop_boxes_to_patch(
            gt_boxes_xyxy,
            gt_labels_np,
            patch=p,
            min_area=float(args.gt_min_area),
            min_iou=float(args.gt_min_iou),
        )
        kind, meta = _categorize_patch(p, gt_boxes_xyxy, patch_boxes)
        patch_infos.append(
            dict(
                patch=p,
                kind=kind,
                num_lesions=int(patch_boxes.shape[0]),
                patch_boxes=patch_boxes,
                patch_labels=patch_labels,
            )
        )

    # choose 3 categories
    chosen = {}
    for want in ["empty", "full", "clipped"]:
        for info in patch_infos:
            if info["kind"] == want:
                chosen[want] = info
                break

    # (a) global overview image: tooth boxes + chosen patch boxes
    global_rgb = image_chw.transpose(1, 2, 0).astype(np.uint8)
    # tooth boxes in green with index text (tooth index)
    global_rgb = _draw_boxes_with_ids(global_rgb, tooth_boxes, color=(0, 255, 0), prefix="tooth", thickness=2)
    # patches: empty=blue, full=yellow, clipped=red
    color_map = {"empty": (0, 128, 255), "full": (255, 255, 0), "clipped": (255, 0, 0)}
    for kind, info in chosen.items():
        p = info["patch"]
        rect = np.array([[p.x1, p.y1, p.x2, p.y2]], dtype=np.float32)
        global_rgb = _draw_rectangles_on_rgb_hwc(global_rgb, rect, color=color_map[kind], thickness=3)

    global_rgb = _draw_text_lines(
        global_rgb,
        lines=[
            f"file={file_name}",
            f"image_id={image_id}",
            f"tooth_thr={args.tooth_score_thr} scale={args.patch_scale} min_size={args.patch_min_size}",
        ],
        xy=(5, 5),
    )

    global_out = out_dir / f"global_{image_id}_{Path(file_name).stem}.png"
    Image.fromarray(global_rgb).save(global_out)

    # (b)(c) patch images with mapped lesion boxes + text dump
    log_lines = []
    log_lines.append(f"file_name={file_name} image_id={image_id} HxW={H}x{W}")
    log_lines.append(
        f"patch_generation: scale={args.patch_scale} min_size={args.patch_min_size} tooth_score_thr={args.tooth_score_thr} max_patches={args.max_patches}"
    )
    log_lines.append(f"gt_mapping_rule: min_iou={args.gt_min_iou} (default 0=no filter), min_area={args.gt_min_area} px^2")
    log_lines.append("NOTE: no 'keep ratio' threshold is used; only clipped-area threshold (min_area).")
    log_lines.append("")

    for kind in ["empty", "full", "clipped"]:
        info = chosen.get(kind, None)
        if info is None:
            log_lines.append(f"[WARN] No patch found for category: {kind}")
            continue
        p: PatchSpec = info["patch"]
        patch_img = crop_image_chw(image_chw, p).transpose(1, 2, 0).astype(np.uint8)

        # draw mapped GT in patch coords (label=category_id)
        boxes = info["patch_boxes"]
        labels = info["patch_labels"]
        patch_vis = patch_img
        if boxes.shape[0] > 0:
            # draw lesion boxes in magenta
            patch_vis = _draw_rectangles_on_rgb_hwc(patch_vis, boxes, color=(255, 0, 255), thickness=2)

        patch_vis = _draw_text_lines(
            patch_vis,
            lines=[
                f"file={file_name}",
                f"kind={kind} tooth_index={p.source_index} score={p.source_score}",
                f"patch_global=[{p.x1},{p.y1},{p.x2},{p.y2}]",
                f"lesions={info['num_lesions']}",
            ],
            xy=(5, 5),
        )

        patch_out = out_dir / f"patch_{kind}_tooth{p.source_index}_img{image_id}_{Path(file_name).stem}.png"
        Image.fromarray(patch_vis).save(patch_out)

        log_lines.append(f"[{kind}] tooth_index={p.source_index} tooth_score={p.source_score}")
        log_lines.append(f"  patch_box_global=[{p.x1},{p.y1},{p.x2},{p.y2}] size={p.w}x{p.h}")
        log_lines.append(f"  assigned_lesions={info['num_lesions']}")

        # print a couple of actual mapping examples
        # Find lesions intersecting patch in global coords, show before/after
        patch_rect = np.array([p.x1, p.y1, p.x2, p.y2], dtype=np.float32)
        shown = 0
        for gbox, glabel in zip(gt_boxes_xyxy, gt_labels_np):
            inter = _intersect_xyxy(gbox, patch_rect)
            if _area_xyxy(inter) <= 0:
                continue
            if shown >= int(args.max_print_lesions):
                break
            # map inter -> patch coords (this matches crop rule before min_area)
            inter_patch = inter.copy()
            inter_patch[[0, 2]] -= float(p.x1)
            inter_patch[[1, 3]] -= float(p.y1)
            log_lines.append(
                f"    lesion(label={int(glabel)}): global={gbox.tolist()} "
                f"-> clipped_global={inter.tolist()} -> patch_xyxy={inter_patch.tolist()}"
            )
            shown += 1
        log_lines.append("")

    txt_out = out_dir / f"log_{image_id}_{Path(file_name).stem}.txt"
    txt_out.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"[OK] Saved global overview: {global_out}")
    print(f"[OK] Saved logs: {txt_out}")
    for kind in ["empty", "full", "clipped"]:
        info = chosen.get(kind, None)
        if info is None:
            continue
        p = info["patch"]
        print(f"[OK] Saved patch {kind}: {out_dir / f'patch_{kind}_tooth{p.source_index}_img{image_id}_{Path(file_name).stem}.png'}")


if __name__ == "__main__":
    main()

