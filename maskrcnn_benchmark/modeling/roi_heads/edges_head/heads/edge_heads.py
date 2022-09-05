import torch
from torch import nn
from torch.nn import functional as F

from models.ops import NonLocal2d
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.utils.poolers import Pooler
from utils.net import make_conv
import pdb





@registry.ROI_EDGE_HEADS.register("roi_edge_head")
class roi_edge_head(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(roi_edge_head, self).__init__()
        self.dim_in = dim_in #dim_in[-1]

        method = cfg.PRCNN.ROI_XFORM_METHOD
        resolution = cfg.PRCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.PRCNN.ROI_XFORM_SAMPLING_RATIO

        use_nl = cfg.PRCNN.GCE_HEAD.USE_NL
        use_bn = cfg.PRCNN.GCE_HEAD.USE_BN
        use_gn = cfg.PRCNN.GCE_HEAD.USE_GN
        conv_dim = cfg.PRCNN.GCE_HEAD.CONV_DIM
        asppv3_dim = cfg.PRCNN.GCE_HEAD.ASPPV3_DIM
        num_convs_before_asppv3 = cfg.PRCNN.GCE_HEAD.NUM_CONVS_BEFORE_ASPPV3
        asppv3_dilation = cfg.PRCNN.GCE_HEAD.ASPPV3_DILATION
        num_convs_after_asppv3 = cfg.PRCNN.GCE_HEAD.NUM_CONVS_AFTER_ASPPV3


        # convx after asppv3 module
        assert num_convs_after_asppv3 >= 1
        after_asppv3_list = []
        for _ in range(num_convs_after_asppv3):
            after_asppv3_list.append(
                make_conv(256, 256, kernel=3, use_bn=use_bn, use_gn=use_gn, use_relu=True)
            )
           
        self.conv_after_asppv3 = nn.Sequential(*after_asppv3_list) if len(after_asppv3_list) else None
        self.dim_out = self.dim_in
        
    def forward(self, x):
        #pdb.set_trace()
        resolution = cfg.PRCNN.ROI_XFORM_RESOLUTION

        h, w = x.size(2), x.size(3)

        if self.conv_after_asppv3 is not None:
            x = self.conv_after_asppv3(x)
        return x


