# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Youngwan Lee (ETRI). All Rights Reserved.
import os

from yacs.config import CfgNode as CN
from utils.collections import AttrDict

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.FCOS_ON = True
_C.MODEL.MASKIOU_ON = False
_C.MODEL.MASKIOU_LOSS_WEIGHT = 1.0
_C.MODEL.RETINANET_ON = False
_C.MODEL.PARSING_ON = False
_C.MODEL.EDGES_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False
_C.MODEL.FASTER_ON = False
# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""
_C.MODEL.USE_SYNCBN = False

# FCOS_MASK
_C.MODEL.FCOS_MASK = False


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# The range of the smallest side for multi-scale training
_C.INPUT.MIN_SIZE_RANGE_TRAIN = (-1, -1)  # -1 means disabled and it will use MIN_SIZE_TRAIN
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
# GN for backbone
_C.MODEL.BACKBONE.USE_GN = False

# ---------------------------------------------------------------------------- #
# HRNET Neck options
# ---------------------------------------------------------------------------- #
_C.MODEL.NECK = CN()
_C.MODEL.NECK.IN_CHANNELS = (32, 64, 128, 256)
_C.MODEL.NECK.OUT_CHANNELS = 256
_C.MODEL.NECK.ACTIVATION = False
_C.MODEL.NECK.POOLING = ' AVG'
_C.MODEL.NECK.SHARING_CONV = False
_C.MODEL.NECK.NUM_OUTS = 5

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 2
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False

_C.MODEL.ROI_MASK_HEAD.LEVEL_MAP_FUNCTION = "MASKRCNNLevelMapFunc"


_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

# MAKIOU
_C.MODEL.ROI_MASKIOU_HEAD = CN()
_C.MODEL.ROI_MASKIOU_HEAD.FEATURE_EXTRACTOR = "MaskIoUFeatureExtractor"
_C.MODEL.ROI_MASKIOU_HEAD.PREDICTOR = "MaskIoUPredictor"
_C.MODEL.ROI_MASKIOU_HEAD.CONV_LAYERS = (256, 256, 256, 256)
# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64



# ---------------------------------------------------------------------------- #
# VoVNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.PRETRAINED = ""
_C.MODEL.VOVNET.USE_GN = False
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.VOVNET.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.VOVNET.WITH_MODULATED_DCN = False
_C.MODEL.VOVNET.DEFORMABLE_GROUPS = 1


# ---------------------------------------------------------------------------- #
# HRNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.HRNET = CN()
_C.MODEL.HRNET.FPN = CN()
_C.MODEL.HRNET.FPN.TYPE = "HRFPN"
_C.MODEL.HRNET.FPN.OUT_CHANNEL = 256
_C.MODEL.HRNET.FPN.CONV_STRIDE = 2

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.NUM_CLASSES = 2 #81  # the number of classes including background
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6#0.1#0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N = 1000#1000

# Focal loss parameter: alpha
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# For FCOS_PLUS 
_C.MODEL.FCOS.CENTER_SAMPLE = False
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'iou'
_C.MODEL.FCOS.DENSE_POINTS = 1

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4

# # Number of top scoring FCOS proposals to keep after applying NMS
_C.MODEL.FCOS.POST_NMS_TOP_N_TRAIN = 100#500

_C.MODEL.FCOS.HEAD = "FCOSHead" # or "FCOSSharedHead"

_C.MODEL.FCOS.RESIDUAL_CONNECTION = False

_C.MODEL.FCOS.TARGET_ASSIGN = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]
# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES =2 # 81

# Anchor aspect ratios to use
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_C.MODEL.RETINANET.USE_C5 = True

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Parsing R-CNN options ("PRCNN" means Pask R-CNN)
# ---------------------------------------------------------------------------- #
_C.AIParsing = CN()

# The head of Parsing R-CNN to use
# (e.g., "roi_convx_head")
_C.AIParsing.ROI_PARSING_HEAD = "roi_context_head"

# Output module of Parsing R-CNN head
_C.AIParsing.ROI_PARSING_OUTPUT = "parsing_output"

# RoI transformation function and associated options
_C.AIParsing.ROI_XFORM_METHOD = 'ROIAlign'

# parsing roi size per image (roi_batch_size = roi_size_per_img * img_per_gpu when using across-sample strategy)
_C.AIParsing.ROI_SIZE_PER_IMG = -1

# Sample the positive box across batch per GPU
_C.AIParsing.ACROSS_SAMPLE = False

# RoI strides for Parsing R-CNN head to use
_C.AIParsing.ROI_STRIDES = []

# Number of grid sampling points in ROIAlign (usually use 2)
# Only applies to ROIAlign
_C.AIParsing.ROI_XFORM_SAMPLING_RATIO = 0

# RoI transformation function (e.g., ROIPool or ROIAlign)
_C.AIParsing.ROI_XFORM_RESOLUTION = (14, 14)

# Resolution of Parsing predictions
_C.AIParsing.RESOLUTION = (56, 56)

# Number of parsings in the dataset
_C.AIParsing.NUM_PARSING = -1

# The ignore label
_C.AIParsing.PARSING_IGNORE_LABEL = 255

# When _C.MODEL.SEMSEG_ON is True, parsing += semseg_pred(per instance) * _C.PRCNN.SEMSEG_FUSE_WEIGHT
_C.AIParsing.SEMSEG_FUSE_WEIGHT = 0.2

# Minimum score threshold (assuming scores in a [0, 1] range) for semantice
# segmentation results.
# 0.3 for CIHP, 0.05 for MHP-v2
_C.AIParsing.SEMSEG_SCORE_THRESH = 0.3

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision parsings
_C.AIParsing.SCORE_THRESH = 0.001

# Evaluate the AP metrics
_C.AIParsing.EVAL_AP = True

# Multi-task loss weight to use for Parsing R-CNN head
_C.AIParsing.LOSS_WEIGHT = 1.0

# Use Parsing IoU for Parsing R-CNN head
_C.AIParsing.PARSINGIOU_ON = False

# ---------------------------------------------------------------------------- #
# Parsing R-CNN gce head options
# ---------------------------------------------------------------------------- #
_C.AIParsing.Context_HEAD = CN()

# Hidden Conv layer dimension
_C.AIParsing.Context_HEAD.CONV_DIM = 512



# Number of stacked Conv layers in parsing branch
_C.AIParsing.Context_HEAD.NUM_CONVS_AFTER = 0

# Use NonLocal in the Parsing R-CNN gce head
_C.AIParsing.Context_HEAD.USE_NL = False

# Reduction ration of nonlocal
_C.AIParsing.Context_HEAD.NL_RATIO = 1.0

# Use BatchNorm in the Parsing R-CNN gce head
_C.AIParsing.Context_HEAD.USE_BN = False

# Use GroupNorm in the Parsing R-CNN gce head
_C.AIParsing.Context_HEAD.USE_GN = False

# ---------------------------------------------------------------------------- #
# Parsing IoU options
# ---------------------------------------------------------------------------- #
_C.AIParsing.PARSINGIOU = CN()

# The head of Parsing IoU to use
# (e.g., "convx_head")
_C.AIParsing.PARSINGIOU.PARSINGIOU_HEAD = "convx_head"

# Output module of Parsing IoU head
_C.AIParsing.PARSINGIOU.PARSINGIOU_OUTPUT = "linear_output"

# Number of stacked Conv layers in the Parsing IoU head
_C.AIParsing.PARSINGIOU.NUM_STACKED_CONVS = 2

# Hidden Conv layer dimension of Parsing IoU head
_C.AIParsing.PARSINGIOU.CONV_DIM = 128

# Hidden MLP layer dimension of Parsing IoU head
_C.AIParsing.PARSINGIOU.MLP_DIM = 256

# Use BatchNorm in the Parsing IoU head
_C.AIParsing.PARSINGIOU.USE_BN = False

# Use GroupNorm in the Parsing IoU head
_C.AIParsing.PARSINGIOU.USE_GN = False

# Loss weight for Parsing IoU head
_C.AIParsing.PARSINGIOU.LOSS_WEIGHT = 1.0

# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.AIParsing.FG_IOU_THRESHOLD = 0.5

# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.AIParsing.BG_IOU_THRESHOLD = 0.5

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10000
_C.SOLVER.TEST_PERIOD = 0
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 1
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 50


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 1
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 50



# Soft NMS
# ---------------------------------------------------------------------------- #
_C.TEST.SOFT_NMS = CN()

# Use soft NMS instead of standard NMS if set to True
_C.TEST.SOFT_NMS.ENABLED = True

# See soft NMS paper for definition of these options
_C.TEST.SOFT_NMS.METHOD = 'linear'

_C.TEST.SOFT_NMS.SIGMA = 0.5
# For the soft NMS overlap threshold, we simply use TEST.NMS


# ---------------------------------------------------------------------------- #
# Bounding box voting (from the Multi-Region CNN paper)
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_VOTE = CN()

# Use box voting if set to True
_C.TEST.BBOX_VOTE.ENABLED = False

# We use TEST.NMS threshold for the NMS step. VOTE_TH overlap threshold
# is used to select voting boxes (IoU >= VOTE_TH) for each box that survives NMS
_C.TEST.BBOX_VOTE.VOTE_TH = 0.8

# The method used to combine scores when doing bounding box voting
# Valid options include ('ID', 'AVG', 'IOU_AVG', 'GENERALIZED_AVG', 'QUASI_SUM')
_C.TEST.BBOX_VOTE.SCORING_METHOD = 'ID'

# Hyperparameter used by the scoring method (it has different meanings for
# different methods)
_C.TEST.BBOX_VOTE.SCORING_METHOD_BETA = 1.0


# ---------------------------------------------------------------------------- #
# Fast R-CNN options
# ---------------------------------------------------------------------------- #
_C.FAST_RCNN = CN()

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.FAST_RCNN.SCORE_THRESH = 0.05

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.FAST_RCNN.NMS = 0.5

# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.FAST_RCNN.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection 
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_AUG = CN()

# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
_C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
_C.TEST.BBOX_AUG.SCALES = () #(400,600,800,1000,1200,1400,1600)

# Max pixel size of the longer side
_C.TEST.BBOX_AUG.MAX_SIZE = 4000


# ---------------------------------------------------------------------------- #
# Test-time augmentations for parsing branch
# ---------------------------------------------------------------------------- #
_C.TEST.PARSING_AUG = CN()

# Enable test-time augmentation for instance parsing detection if True
_C.TEST.PARSING_AUG.ENABLED = True

# Heuristic used to combine parsing predictions
# SOFT prefix indicates that the computation is performed on soft parsings
#   Valid options: ('SOFT_AVG', 'SOFT_MAX', 'LOGIT_AVG')
_C.TEST.PARSING_AUG.HEUR = 'SOFT_AVG'

_C.TRAIN = CN()
_C.TRAIN.LEFT_RIGHT = ()


# ---------------------------------------------------------------------------- #
# Visualization options
# ---------------------------------------------------------------------------- #
_C.VIS = CN()

# Dump detection visualizations
_C.VIS.ENABLED = True

# Score threshold for visualization
_C.VIS.VIS_TH = 0.3

# ---------------------------------------------------------------------------- #
# Show box options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_BOX = CN()

# Visualizing detection bboxes
_C.VIS.SHOW_BOX.ENABLED = True

# Visualization color scheme
# 'green', 'category' or 'instance'
_C.VIS.SHOW_BOX.COLOR_SCHEME = 'green'

# Color map, 'COCO81', 'VOC21', 'ADE151', 'LIP20', 'MHP59'
_C.VIS.SHOW_BOX.COLORMAP = 'COCO81'

# Border thick
_C.VIS.SHOW_BOX.BORDER_THICK = 1

# ---------------------------------------------------------------------------- #
# Show class options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_CLASS = CN()

# Visualizing detection classes
_C.VIS.SHOW_CLASS.ENABLED = True

# Default: gray
_C.VIS.SHOW_CLASS.COLOR = (218, 227, 218)

# Font scale of class string
_C.VIS.SHOW_CLASS.FONT_SCALE = 0.45

# ---------------------------------------------------------------------------- #
# Show segmentation options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_SEGMS = CN()

# Visualizing detection classes
_C.VIS.SHOW_SEGMS.ENABLED = False

# Whether show mask
_C.VIS.SHOW_SEGMS.SHOW_MASK = True

# False = (255, 255, 255) = white
_C.VIS.SHOW_SEGMS.MASK_COLOR_FOLLOW_BOX = True

# Mask ahpha
_C.VIS.SHOW_SEGMS.MASK_ALPHA = 0.4

# Whether show border
_C.VIS.SHOW_SEGMS.SHOW_BORDER = True

# Border color, (255, 255, 255) for white, (0, 0, 0) for black
_C.VIS.SHOW_SEGMS.BORDER_COLOR = (255, 255, 255)

# Border thick
_C.VIS.SHOW_SEGMS.BORDER_THICK = 2

# ---------------------------------------------------------------------------- #
# Show keypoints options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_KPS = CN()

# Visualizing detection keypoints
_C.VIS.SHOW_KPS.ENABLED = False

# Keypoints threshold
_C.VIS.SHOW_KPS.KPS_TH = 2

# Default: white
_C.VIS.SHOW_KPS.KPS_COLOR_WITH_PARSING = (255, 255, 255)

# Keypoints alpha
_C.VIS.SHOW_KPS.KPS_ALPHA = 0.7

# Link thick
_C.VIS.SHOW_KPS.LINK_THICK = 2

# Circle radius
_C.VIS.SHOW_KPS.CIRCLE_RADIUS = 3

# Circle thick
_C.VIS.SHOW_KPS.CIRCLE_THICK = -1

# ---------------------------------------------------------------------------- #
# Show parsing options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_PARSS = CN()

# Visualizing detection classes
_C.VIS.SHOW_PARSS.ENABLED = True

# Color map, 'COCO81', 'VOC21', 'ADE151', 'LIP20', 'MHP59'
_C.VIS.SHOW_PARSS.COLORMAP = 'CIHP20'

# Parsing alpha
_C.VIS.SHOW_PARSS.PARSING_ALPHA = 0.9

# Whether show border
_C.VIS.SHOW_PARSS.SHOW_BORDER = True

# Border color
_C.VIS.SHOW_PARSS.BORDER_COLOR = (255, 255, 255)

# Border thick
_C.VIS.SHOW_PARSS.BORDER_THICK = 1

# ---------------------------------------------------------------------------- #
# Show uv options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_UV = CN()

# Visualizing detection classes
_C.VIS.SHOW_UV.ENABLED = False

# Whether show border
_C.VIS.SHOW_UV.SHOW_BORDER = True

# Border thick
_C.VIS.SHOW_UV.BORDER_THICK = 6

# Grid thick
_C.VIS.SHOW_UV.GRID_THICK = 2

# Grid lines num
_C.VIS.SHOW_UV.LINES_NUM = 15



# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
