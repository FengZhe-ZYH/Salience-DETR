from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Dict, List, Optional

import accelerate
import torch
from accelerate import Accelerator
from pycocotools.coco import COCO

from datasets.coco import CocoDetection
from pipelines.twostage.patch_ops import generate_patches_from_tooth_boxes, map_boxes_patch_to_image
from pipelines.twostage.tooth_boxes_format import load_jsonl_to_index
from test import create_test_data_loader
from util.coco_eval import CocoEvaluator, loadRes
from util.coco_utils import get_coco_api_from_dataset
from util.collate_fn import collate_fn
from util.engine import evaluate_acc
from util.lazy_load import Config
from util.logger import setup_logger
from util.utils import load_checkpoint, load_state_dict
from util.box_fusion import fuse_by_nms


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage fusion evaluation on COCO-format dataset.")
    parser.add_argument("--coco-path", type=str, required=True)
    parser.add_argument("--subset", type=str, default="val")
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--global-model-config", type=str, required=True)
    parser.add_argument("--global-checkpoint", type=str, required=True)
    parser.add_argument("--local-model-config", type=str, required=True)
    parser.add_argument("--local-checkpoint", type=str, required=True)

    parser.add_argument("--tooth-boxes", type=str, required=True, help="tooth boxes jsonl")
    parser.add_argument("--tooth-score-thr", type=float, default=0.3)
    parser.add_argument("--patch-scale", type=float, default=1.5)
    parser.add_argument("--patch-min-size", type=int, default=512)
    parser.add_argument("--max-patches-per-image", type=int, default=16)

    parser.add_argument("--fusion-iou", type=float, default=0.5)
    parser.add_argument("--max-dets", type=int, default=300)

    parser.add_argument("--result", type=str, default=None, help="optional path to save coco json results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def _load_model(model_config: str, checkpoint_path: str, accelerator: Accelerator):
    model = Config(model_config).model.eval()
    checkpoint = load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, Dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    load_state_dict(model, checkpoint)
    model = accelerator.prepare_model(model)
    return model


@torch.no_grad()
def _run_local_on_patches(
    local_model,
    image_tensor_chw: torch.Tensor,
    image_hw,
    tooth_rec,
    device,
    patch_scale: float,
    patch_min_size: int,
    tooth_score_thr: float,
    max_patches_per_image: int,
) -> Dict[str, torch.Tensor]:
    if tooth_rec is None:
        return {"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros((0,), device=device), "labels": torch.zeros((0,), dtype=torch.long, device=device)}

    boxes_np, scores_np = tooth_rec.to_numpy()
    if scores_np.size != 0:
        keep = scores_np >= float(tooth_score_thr)
        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]
    if boxes_np.size == 0:
        return {"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros((0,), device=device), "labels": torch.zeros((0,), dtype=torch.long, device=device)}

    patches = generate_patches_from_tooth_boxes(
        boxes_np,
        scores_np,
        image_hw=image_hw,
        scale=patch_scale,
        min_size=patch_min_size,
    )
    if len(patches) > int(max_patches_per_image):
        patches = patches[: int(max_patches_per_image)]

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for p in patches:
        patch_img = image_tensor_chw[:, p.y1 : p.y2, p.x1 : p.x2].unsqueeze(0)  # (1,C,ph,pw)
        pred = local_model(patch_img)[0]
        p_boxes = map_boxes_patch_to_image(pred["boxes"].to(device), p)
        all_boxes.append(p_boxes)
        all_scores.append(pred.get("scores", torch.zeros((p_boxes.shape[0],), device=device)).to(device))
        all_labels.append(pred.get("labels", torch.zeros((p_boxes.shape[0],), dtype=torch.long, device=device)).to(device).long())

    if len(all_boxes) == 0:
        return {"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros((0,), device=device), "labels": torch.zeros((0,), dtype=torch.long, device=device)}
    return {
        "boxes": torch.cat(all_boxes, dim=0),
        "scores": torch.cat(all_scores, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


def main():
    args = parse_args()

    accelerator = Accelerator()
    accelerate.utils.set_seed(args.seed, device_specific=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
        setup_logger(distributed_rank=accelerator.local_process_index, name=logger_name)

    dataset = CocoDetection(
        img_folder=f"{args.coco_path}/{args.subset}2017",
        ann_file=f"{args.coco_path}/annotations/instances_{args.subset}2017.json",
        transforms=None,
        train=args.subset == "train",
    )
    data_loader = create_test_data_loader(
        dataset,
        accelerator=accelerator,
        batch_size=1,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    # load models
    global_model = _load_model(args.global_model_config, args.global_checkpoint, accelerator)
    local_model = _load_model(args.local_model_config, args.local_checkpoint, accelerator)

    # tooth boxes index by file_name
    tooth_index = load_jsonl_to_index(args.tooth_boxes, key="file_name")

    # evaluator: we will feed fused predictions in coco format
    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

    # main loop
    device = accelerator.device
    for images, targets in data_loader:
        # images: list[tensor] due to collate_fn; but in engine they pass list into model
        # Here batch=1, so take first.
        image = images[0].to(device)
        target = targets[0]
        image_id = int(target["image_id"])

        # file_name lookup
        file_name = dataset.coco.loadImgs([image_id])[0]["file_name"]
        tooth_rec = tooth_index.get(file_name, None)

        global_pred = global_model([image])[0]
        H, W = int(image.shape[-2]), int(image.shape[-1])
        local_pred = _run_local_on_patches(
            local_model=local_model,
            image_tensor_chw=image,
            image_hw=(H, W),
            tooth_rec=tooth_rec,
            device=device,
            patch_scale=args.patch_scale,
            patch_min_size=args.patch_min_size,
            tooth_score_thr=args.tooth_score_thr,
            max_patches_per_image=args.max_patches_per_image,
        )

        fused = fuse_by_nms(global_pred, local_pred, iou_thr=args.fusion_iou, max_dets=args.max_dets)

        res = {image_id: {k: v.to("cpu") for k, v in fused.items()}}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # save results if requested
    if accelerator.is_main_process:
        if args.result is None:
            temp_file = tempfile.NamedTemporaryFile()
            args.result = temp_file.name
        with open(args.result, "w") as f:
            f.write(json.dumps(coco_evaluator.predictions["bbox"]))


if __name__ == "__main__":
    main()

