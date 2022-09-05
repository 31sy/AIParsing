# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from maskrcnn_benchmark.utils.registry import Registry

BACKBONES = Registry()
RPN_HEADS = Registry()
ROI_BOX_FEATURE_EXTRACTORS = Registry()
ROI_BOX_PREDICTOR = Registry()
ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry()
ROI_KEYPOINT_PREDICTOR = Registry()
ROI_MASK_FEATURE_EXTRACTORS = Registry()
ROI_MASK_PREDICTOR = Registry()

# Parsing Head
ROI_PARSING_HEADS = Registry()
ROI_PARSING_OUTPUTS = Registry()

# Edge mask Head
ROI_EDGE_HEADS = Registry()
ROI_EDGE_OUTPUTS = Registry()
PARSINGIOU_HEADS = Registry()
PARSINGIOU_OUTPUTS = Registry()
