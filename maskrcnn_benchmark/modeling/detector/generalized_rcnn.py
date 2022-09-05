# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

#from utils.data.structures.image_list import to_image_list

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from maskrcnn_benchmark.modeling.roi_heads.parsing_head.parsing_head import ParsingHead

from maskrcnn_benchmark.config import cfg
import models.ops as ops
import numpy as np


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        # Normalization
        #if not self.training:

        self.backbone = build_backbone(cfg)
       
        # Backbone
        #conv_body = registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY]
        #self.Conv_Body = conv_body()
        #print('self.backbone.out_channels',self.backbone.out_channels)
        self.dim_in = self.backbone.out_channels
        self.spatial_scale = [0.25, 0.125, 0.0625, 0.03125]



        self.rpn = build_rpn(cfg, self.backbone.out_channels)        
        if cfg.MODEL.PARSING_ON:
            self.ParsingHead = ParsingHead(self.dim_in, self.spatial_scale)


    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        roi_losses = {}
        if self.ParsingHead:
            zero_ids = []
            for idx, proposal in enumerate(proposals):
                if proposal.bbox.shape[0] == 0:
                    zero_ids.append(idx)
            for idx in sorted(zero_ids, reverse=True):
                proposals.pop(idx)
                if self.training:
                    targets.pop(idx)
            if len(proposals) == 0:
                if not self.training:
                    return proposals
            #x, result, detector_losses = self.roi_heads(features, proposals, targets)
            roi_feature, result, loss_parsing = self.ParsingHead(features, proposals, targets)
            roi_losses.update(loss_parsing)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}



        if self.training:
            losses = {}
            #losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(roi_losses)
            return losses

        return result
    def box_net(self, images, targets=None):
        images = to_image_list(images)
        # images_norm_t = self.Norm(images.tensors)
        # images_norm = self.Norm(images)


        features = self.backbone(images.tensors)
        #if(targets == None):
        
        proposals, proposal_losses = self.rpn(images, features, targets=None)


        return features, proposals

    def mask_net(self, conv_features, result, targets=None):
        if len(result[0]) == 0:
            return {}
        # with torch.no_grad():
        #     x, result, loss_mask = self.Mask_RCNN(conv_features, result, targets)

        return {}

    def keypoint_net(self, conv_features, result, targets=None):
        if len(result[0]) == 0:
            return {}
        # with torch.no_grad():
        #     x, result, loss_keypoint = self.Keypoint_RCNN(conv_features, result, targets)

        return {}

    def parsing_net(self, conv_features, result, targets=None):
        if len(result[0]) == 0:
            return result
        with torch.no_grad():
            x, result, loss_parsing,loss_edge = self.ParsingHead(conv_features, result, targets=None)

        return result

    def uv_net(self, conv_features, result, targets=None):
        if len(result[0]) == 0:
            return {}
        # with torch.no_grad():
        #     x, result, loss_uv = self.UV_RCNN(conv_features, result, targets)

        return {}
