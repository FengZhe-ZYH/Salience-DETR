from __future__ import annotations

import argparse
import os
from functools import partial
from typing import Dict, List, Optional, Tuple

import accelerate
import cv2
import numpy as np
import torch
import torch.utils.data as data
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from pipelines.twostage.patch_ops import generate_patches_from_tooth_boxes, map_boxes_patch_to_image
from pipelines.twostage.tooth_boxes_format import load_jsonl_to_index
from test import create_test_data_loader
from util.box_fusion import fuse_by_nms
from util.lazy_load import Config
from util.logger import setup_logger
from util.utils import load_checkpoint, load_state_dict
from util.visualize import plot_bounding_boxes_on_image_cv2


def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.close()
        return True
    except Exception:
        return False


class InferenceDataset(data.Dataset):
    def __init__(self, root):
        self.images = [os.path.join(root, img) for img in os.listdir(root)]
        self.images = [img for img in self.images if is_image(img)]
        assert len(self.images) > 0, "No images found"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        image = cv2.imdecode(np.fromfile(self.images[index], dtype=np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return {"image": torch.tensor(image), "image_path": self.images[index]}


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage inference (global + local + fusion).")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--global-model-config", type=str, required=True)
    parser.add_argument("--global-checkpoint", type=str, required=True)
    parser.add_argument("--local-model-config", type=str, required=True)
    parser.add_argument("--local-checkpoint", type=str, required=True)

    parser.add_argument("--tooth-boxes", type=str, default=None, help="Optional tooth boxes jsonl for local branch.")
    parser.add_argument("--tooth-score-thr", type=float, default=0.3)
    parser.add_argument("--patch-scale", type=float, default=1.5)
    parser.add_argument("--patch-min-size", type=int, default=512)
    parser.add_argument("--max-patches-per-image", type=int, default=16)

    parser.add_argument("--fusion-iou", type=float, default=0.5)
    parser.add_argument("--max-dets", type=int, default=300)

    parser.add_argument("--show-dir", type=str, default=None)
    parser.add_argument("--show-conf", type=float, default=0.5)

    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--box-thick", type=int, default=1)
    parser.add_argument("--fill-alpha", type=float, default=0.2)
    parser.add_argument("--text-box-color", type=int, nargs="+", default=(255, 255, 255))
    parser.add_argument("--text-font-color", type=int, nargs="+", default=None)
    parser.add_argument("--text-alpha", type=float, default=1.0)

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
def _local_predict(
    local_model,
    image_tensor_chw: torch.Tensor,
    tooth_rec,
    patch_scale: float,
    patch_min_size: int,
    tooth_score_thr: float,
    max_patches_per_image: int,
    device,
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

    H, W = int(image_tensor_chw.shape[-2]), int(image_tensor_chw.shape[-1])
    patches = generate_patches_from_tooth_boxes(
        boxes_np, scores_np, image_hw=(H, W), scale=patch_scale, min_size=patch_min_size
    )
    if len(patches) > int(max_patches_per_image):
        patches = patches[: int(max_patches_per_image)]

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for p in patches:
        patch_img = image_tensor_chw[:, p.y1 : p.y2, p.x1 : p.x2].unsqueeze(0)
        pred = local_model(patch_img)[0]
        all_boxes.append(map_boxes_patch_to_image(pred["boxes"].to(device), p))
        all_scores.append(pred.get("scores", torch.zeros((pred["boxes"].shape[0],), device=device)).to(device))
        all_labels.append(pred.get("labels", torch.zeros((pred["boxes"].shape[0],), dtype=torch.long, device=device)).to(device).long())

    if len(all_boxes) == 0:
        return {"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros((0,), device=device), "labels": torch.zeros((0,), dtype=torch.long, device=device)}
    return {"boxes": torch.cat(all_boxes, 0), "scores": torch.cat(all_scores, 0), "labels": torch.cat(all_labels, 0)}


def inference():
    args = parse_args()
    accelerator = Accelerator()

    accelerate.utils.set_seed(args.seed, device_specific=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for logger_name in ["py.warnings", "accelerate", os.path.basename(os.getcwd())]:
        setup_logger(distributed_rank=accelerator.local_process_index, name=logger_name)

    dataset = InferenceDataset(args.image_dir)
    data_loader = create_test_data_loader(dataset, accelerator=accelerator, batch_size=1, num_workers=args.workers)

    global_model = _load_model(args.global_model_config, args.global_checkpoint, accelerator)
    local_model = _load_model(args.local_model_config, args.local_checkpoint, accelerator)

    tooth_index = load_jsonl_to_index(args.tooth_boxes, key="file_name") if args.tooth_boxes else None

    results = []
    with torch.inference_mode():
        for batch in tqdm(data_loader):
            item = batch[0] if isinstance(batch, (list, tuple)) else batch
            image = item["image"].to(accelerator.device)
            image_path = item["image_path"]
            base = os.path.basename(image_path)

            tooth_rec = None
            if tooth_index is not None:
                tooth_rec = tooth_index.get(base, None) or tooth_index.get(image_path, None)

            global_pred = global_model([image])[0]
            local_pred = _local_predict(
                local_model=local_model,
                image_tensor_chw=image,
                tooth_rec=tooth_rec,
                patch_scale=args.patch_scale,
                patch_min_size=args.patch_min_size,
                tooth_score_thr=args.tooth_score_thr,
                max_patches_per_image=args.max_patches_per_image,
                device=accelerator.device,
            )
            fused = fuse_by_nms(global_pred, local_pred, iou_thr=args.fusion_iou, max_dets=args.max_dets)
            results.append({"image_path": image_path, "image": image, "pred": fused})

    if args.show_dir and accelerator.is_main_process:
        os.makedirs(args.show_dir, exist_ok=True)
        vis_loader = create_test_data_loader(results, accelerator=None, batch_size=1, num_workers=args.workers)
        vis_loader.collate_fn = partial(_visualize_batch, classes=global_model.CLASSES, **vars(args))
        [None for _ in tqdm(vis_loader)]


def _visualize_batch(
    batch,
    classes: List[str],
    show_conf: float = 0.5,
    show_dir: str = None,
    font_scale: float = 1.0,
    box_thick: int = 3,
    fill_alpha: float = 0.2,
    text_box_color: Tuple[int] = (255, 255, 255),
    text_font_color: Tuple[int] = None,
    text_alpha: float = 0.5,
    **kwargs,
):
    item = batch[0]
    image = item["image"].to("cpu", non_blocking=True).numpy().transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pred = item["pred"]

    plot = plot_bounding_boxes_on_image_cv2(
        image=image,
        boxes=pred["boxes"].to("cpu"),
        labels=pred["labels"].to("cpu"),
        scores=pred["scores"].to("cpu"),
        classes=classes,
        show_conf=show_conf,
        font_scale=font_scale,
        box_thick=box_thick,
        fill_alpha=fill_alpha,
        text_box_color=text_box_color,
        text_font_color=text_font_color,
        text_alpha=text_alpha,
    )
    out_path = os.path.join(show_dir, os.path.basename(item["image_path"]))
    cv2.imwrite(out_path, plot)


if __name__ == "__main__":
    inference()

