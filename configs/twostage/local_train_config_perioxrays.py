from torch import optim

from datasets.tooth_patch_coco import ToothPatchCocoDetection, ToothPatchDatasetConfig
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 15
batch_size = 2
num_workers = 4
pin_memory = True
print_freq = 50
starting_epoch = 0
max_norm = 0.1

output_dir = None
find_unused_parameters = False

# dataset
data_root = "/hdd1/zyh/Datasets/PerioXrays/coco/"
tooth_boxes_jsonl = "/hdd1/zyh/Datasets/PerioXrays/tooth_boxes/perioxrays_val_tooth_boxes.jsonl"

train_transform = presets.detr
patch_cfg = ToothPatchDatasetConfig(
    tooth_boxes_jsonl=tooth_boxes_jsonl,
    patch_scale=1.5,
    patch_min_size=512,
    tooth_score_thr=0.3,
    max_patches_per_image=16,
    drop_empty_patches=False,
)

train_dataset = ToothPatchCocoDetection(
    img_folder=f"{data_root}images/train2017/",
    ann_file=f"{data_root}annotations/instances_train2017.json",
    transforms=train_transform,
    train=True,
    patch_cfg=patch_cfg,
)

test_dataset = ToothPatchCocoDetection(
    img_folder=f"{data_root}images/val2017/",
    ann_file=f"{data_root}annotations/instances_val2017.json",
    transforms=None,
    train=False,
    patch_cfg=patch_cfg,
)

# model config to train (local detector)
model_path = "configs/twostage/local_model_salience_detr_patch.py"

resume_from_checkpoint = None

learning_rate = 1e-4
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[11], gamma=0.1)
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)

