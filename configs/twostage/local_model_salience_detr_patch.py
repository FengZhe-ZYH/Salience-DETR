from torch import nn

from models.backbones.resnet import ResNetBackbone
from models.bricks.misc import FrozenBatchNorm2d
from models.bricks.position_encoding import PositionEmbeddingSine
from models.bricks.post_process import PostProcess
from models.bricks.salience_transformer import (
    SalienceTransformer,
    SalienceTransformerDecoder,
    SalienceTransformerDecoderLayer,
    SalienceTransformerEncoder,
    SalienceTransformerEncoderLayer,
)
from models.bricks.set_criterion import HybridSetCriterion
from models.detectors.salience_detr import SalienceCriterion, SalienceDETR
from models.matcher.hungarian_matcher import HungarianMatcher
from models.necks.channel_mapper import ChannelMapper
from models.necks.repnet import RepVGGPluXNetwork

# Keep consistent with PerioXrays lesion classes setup
embed_dim = 256
num_classes = 2  # 1 lesion class + background index consideration (same as perioxrays global config)
num_queries = 900
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

position_embedding = PositionEmbeddingSine(embed_dim // 2, temperature=10000, normalize=True, offset=-0.5)

backbone = ResNetBackbone(
    "resnet50",
    weights="/hdd2/zyh/Salience-DETR/pretrain/ResNet50_DenseCL_epoch400.pth",
    norm_layer=FrozenBatchNorm2d,
    return_indices=(1, 2, 3),
    freeze_indices=(),
)

neck = ChannelMapper(
    in_channels=backbone.num_channels,
    out_channels=embed_dim,
    num_outs=num_feature_levels,
)

transformer = SalienceTransformer(
    encoder=SalienceTransformerEncoder(
        encoder_layer=SalienceTransformerEncoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_enc_layers,
        max_num_embedding=500,
    ),
    neck=RepVGGPluXNetwork(
        in_channels_list=neck.num_channels,
        out_channels_list=neck.num_channels,
        norm_layer=nn.BatchNorm2d,
        activation=nn.SiLU,
        groups=4,
    ),
    decoder=SalienceTransformerDecoder(
        decoder_layer=SalienceTransformerDecoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_dec_layers,
        num_classes=num_classes,
    ),
    num_classes=num_classes,
    num_feature_levels=num_feature_levels,
    two_stage_num_proposals=num_queries,
    level_filter_ratio=(0.4, 0.8, 1.0, 1.0),
    layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
)

matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
weight_dict.update({k + f"_{i}": v for i in range(transformer_dec_layers - 1) for k, v in weight_dict.items()})
weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})
weight_dict.update({"loss_salience": 2})

criterion = HybridSetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, alpha=0.25, gamma=2.0)
foreground_criterion = SalienceCriterion(noise_scale=0.0, alpha=0.25, gamma=2.0)

# Patch inference doesn't need too many boxes; keep 300 consistent
postprocessor = PostProcess(select_box_nums_for_evaluation=300)

model = SalienceDETR(
    backbone=backbone,
    neck=neck,
    position_embedding=position_embedding,
    transformer=transformer,
    criterion=criterion,
    focus_criterion=foreground_criterion,
    postprocessor=postprocessor,
    num_classes=num_classes,
    num_queries=num_queries,
    aux_loss=True,
    # For patches we typically keep min/max sizes smaller; V1 keep same to reduce surprises.
    min_size=800,
    max_size=1333,
)

