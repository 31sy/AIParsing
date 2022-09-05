import torch

from maskrcnn_benchmark.modeling.roi_heads.edges_head import heads
from maskrcnn_benchmark.modeling.roi_heads.edges_head import outputs
from maskrcnn_benchmark.modeling.roi_heads.edges_head.inference import parsing_post_processor
from maskrcnn_benchmark.modeling.roi_heads.edges_head.loss import edges_loss_evaluator
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.config import cfg
import pdb

class EdgeParsing(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(EdgeParsing, self).__init__()
        if len(cfg.PRCNN.ROI_STRIDES) == 0:
            self.spatial_scale = spatial_scale
        else:
            self.spatial_scale = [1. / stride for stride in cfg.PRCNN.ROI_STRIDES]

        head = registry.ROI_EDGE_HEADS['roi_edge_head']
        self.Head = head(dim_in, self.spatial_scale)
        output = registry.ROI_EDGE_HEADS['edges_output']
        self.Output = output(self.Head.dim_out)

        self.post_processor = parsing_post_processor()
        self.loss_evaluator = edges_loss_evaluator()

    def forward(self, roi_features, proposals, targets=None):
        """
        Arguments:
            conv_features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.
        Returns:
            x (Tensor): the result of the feature extractor
            all_proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `parsing` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if targets is None:
            return self._forward_test(roi_features, proposals)
        else:
            return self._forward_train(roi_features, proposals, targets)

    def _forward_train(self, roi_features, proposals, targets=None):
        #pdb.set_trace()
        all_proposals = proposals
        with torch.no_grad():
            proposals = self.loss_evaluator.resample(proposals, targets)

        x = self.Head(roi_features)
        parsing_logits = self.Output(x)

        loss_edge = self.loss_evaluator(parsing_logits)
        return x, all_proposals, dict(loss_edge=loss_edge)

    def _forward_test(self, roi_features, proposals):
        #pdb.set_trace()
        x = self.Head(roi_features)
        parsing_logits = self.Output(x)

        result = self.post_processor(parsing_logits, proposals)
        return x, result, {}
