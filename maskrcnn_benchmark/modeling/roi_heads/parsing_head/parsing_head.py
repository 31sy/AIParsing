import torch

from maskrcnn_benchmark.modeling.roi_heads.parsing_head import heads
from maskrcnn_benchmark.modeling.roi_heads.parsing_head import outputs
from maskrcnn_benchmark.modeling.roi_heads.parsing_head.inference import parsing_post_processor
from maskrcnn_benchmark.modeling.roi_heads.parsing_head.loss import parsing_loss_evaluator
from maskrcnn_benchmark.modeling.roi_heads.parsing_head.parsingiou.parsingiou import ParsingIoU
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.config import cfg
import pdb

class ParsingHead(torch.nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(ParsingHead, self).__init__()
        if len(cfg.AIParsing.ROI_STRIDES) == 0:
            self.spatial_scale = spatial_scale
        else:
            self.spatial_scale = [1. / stride for stride in cfg.AIParsing.ROI_STRIDES]

        head = registry.ROI_PARSING_HEADS[cfg.AIParsing.ROI_PARSING_HEAD]
        self.Head = head(dim_in, self.spatial_scale)
        output = registry.ROI_PARSING_OUTPUTS[cfg.AIParsing.ROI_PARSING_OUTPUT]
        self.Output = output(self.Head.dim_out)

        self.post_processor = parsing_post_processor()
        self.loss_evaluator = parsing_loss_evaluator()
        
        if cfg.AIParsing.PARSINGIOU_ON:
            self.ParsingIoU = ParsingIoU(self.Head.dim_out)

    def forward(self, conv_features, proposals, targets=None):
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
            return self._forward_test(conv_features, proposals)
        else:
            return self._forward_train(conv_features, proposals, targets)

    def _forward_train(self, conv_features, proposals, targets=None):
        #pdb.set_trace()
        all_proposals = proposals
        with torch.no_grad():
            proposals = self.loss_evaluator.resample(proposals, targets)

        x, roi_feature = self.Head(conv_features, proposals)
        parsing_logits,edge_logits = self.Output(x)

        if cfg.AIParsing.PARSINGIOU_ON:
            loss_parsing, parsing_iouloss, loss_edge, parsingiou_targets = self.loss_evaluator(parsing_logits,edge_logits)
            loss_parsingiou, all_proposals = self.ParsingIoU(
                x, all_proposals, parsing_logits, parsingiou_targets
            )
            return roi_feature, all_proposals, dict(loss_parsing=loss_parsing, parsing_iouloss= parsing_iouloss, loss_edge=loss_edge, loss_parsingiou=loss_parsingiou)
        else:
            loss_parsing, parsing_iouloss, loss_edge = self.loss_evaluator(parsing_logits,edge_logits)
            return roi_feature, all_proposals, dict(loss_parsing=loss_parsing, parsing_iouloss= parsing_iouloss, loss_edge=loss_edge)



    def _forward_test(self, conv_features, proposals):
        #pdb.set_trace()
        x, roi_feature = self.Head(conv_features, proposals)
        parsing_logits,edge_logits = self.Output(x)

        result = self.post_processor(parsing_logits, proposals)
        
        if cfg.AIParsing.PARSINGIOU_ON:
            _, result = self.ParsingIoU(x, result, parsing_logits, None)
            return roi_feature, result, {},{}
        else:
            return roi_feature, result, {},{}