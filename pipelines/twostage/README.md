# Two-stage / Dual-branch V1 (Global + Tooth-centered Local)

本目录用于 **离线 tooth boxes + local patch detector + 推理融合** 的最小侵入接入。

## Tooth boxes 文件格式（jsonl）

每行对应一张图的 tooth 检测结果，字段定义见 `pipelines/twostage/tooth_boxes_format.py`：

- `file_name`：与 COCO `images[i].file_name` 一致（推荐相对路径或文件名）
- `image_id`：COCO image id（可选，但建议写上）
- `boxes`：`[[x1,y1,x2,y2], ...]`，**原图像素坐标**，XYXY
- `scores`：与 `boxes` 等长，float

示例：

```json
{"file_name":"000001.png","image_id":1,"boxes":[[10,20,110,140]],"scores":[0.98],"model":"faster-rcnn-r50-fpn-o2pr"}
```

## 离线预计算 tooth boxes（mmdetection 环境）

建议在你已有的 `dinov3` 环境中运行（本仓库运行时不依赖 mmdet）。

你提供的资源：
- config：`/hdd1/zyh/Dental/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_O2PR.py`
- ckpt：`/hdd1/zyh/Dental/mmdetection/work_dirs/faster-rcnn_r50_fpn_1x_O2PR/best_coco_bbox_mAP_epoch_15.pth`

后续将提供脚本：`pipelines/twostage/precompute_tooth_boxes_mmdet.py`

