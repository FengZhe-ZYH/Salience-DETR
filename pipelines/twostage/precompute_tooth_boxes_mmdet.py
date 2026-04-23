from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from pipelines.twostage.tooth_boxes_format import ToothBoxesRecord, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline precompute tooth boxes using mmdetection (run in your mmdet env)."
    )
    parser.add_argument("--coco-ann", type=str, required=True, help="COCO instances_*.json path")
    parser.add_argument(
        "--coco-img-root",
        type=str,
        required=True,
        help="Image root folder (the parent of images referenced by file_name)",
    )
    parser.add_argument("--mmdet-config", type=str, required=True)
    parser.add_argument("--mmdet-checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--max-per-img", type=int, default=200)
    parser.add_argument("--out", type=str, required=True, help="Output jsonl path")
    parser.add_argument("--model-name", type=str, default="mmdet-faster-rcnn-tooth")
    args = parser.parse_args()
    return args


def _lazy_import_mmdet():
    # Keep mmdet dependency out of main repo runtime; only used in offline script.
    from mmdet.apis import inference_detector, init_detector  # type: ignore

    return init_detector, inference_detector


def _load_coco_images(coco_ann_path: str) -> List[Dict]:
    from pycocotools.coco import COCO  # local import to keep surface small

    coco = COCO(coco_ann_path)
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    return imgs


def _select_boxes_from_mmdet_result(result, score_thr: float, max_per_img: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    mmdet inference_detector output varies by version/model:
    - old: list[np.ndarray] per class, each (N,5) with score
    - new: DetDataSample with pred_instances.bboxes/scores/labels
    We support both.
    """
    # DetDataSample path
    if hasattr(result, "pred_instances"):
        inst = result.pred_instances
        bboxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
    else:
        # list of per-class ndarray
        if isinstance(result, tuple):
            result = result[0]
        per_class = result
        merged = []
        for cls_det in per_class:
            if cls_det is None or len(cls_det) == 0:
                continue
            merged.append(cls_det)
        if len(merged) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        dets = np.concatenate(merged, axis=0)  # (N,5)
        bboxes = dets[:, :4]
        scores = dets[:, 4]

    keep = scores >= float(score_thr)
    bboxes = bboxes[keep]
    scores = scores[keep]

    if bboxes.shape[0] > int(max_per_img):
        order = np.argsort(-scores)
        order = order[: int(max_per_img)]
        bboxes = bboxes[order]
        scores = scores[order]

    return bboxes.astype(np.float32), scores.astype(np.float32)


def main():
    args = parse_args()

    init_detector, inference_detector = _lazy_import_mmdet()
    model = init_detector(args.mmdet_config, args.mmdet_checkpoint, device=args.device)

    imgs = _load_coco_images(args.coco_ann)
    records: List[ToothBoxesRecord] = []
    img_root = Path(args.coco_img_root)

    for img in tqdm(imgs, desc="precompute tooth boxes"):
        file_name = img["file_name"]
        image_id = int(img["id"])
        img_path = img_root / file_name
        if not img_path.exists():
            # some datasets store file_name as basename; try join basename
            alt = img_root / os.path.basename(file_name)
            if alt.exists():
                img_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        result = inference_detector(model, str(img_path))
        bboxes, scores = _select_boxes_from_mmdet_result(result, args.score_thr, args.max_per_img)
        rec = ToothBoxesRecord(
            file_name=file_name,
            image_id=image_id,
            boxes=bboxes.tolist(),
            scores=scores.tolist(),
            model=args.model_name,
            meta={"score_thr": args.score_thr, "max_per_img": args.max_per_img},
        )
        records.append(rec)

    write_jsonl(records, args.out)


if __name__ == "__main__":
    main()

