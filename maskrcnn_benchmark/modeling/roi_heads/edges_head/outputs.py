from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.config import cfg
import pdb

@registry.ROI_EDGE_HEADS.register("edges_output")
class Edges_output(nn.Module):
    def __init__(self, dim_in):
        super(Edges_output, self).__init__()
        num_parsing = 2
        assert cfg.PRCNN.RESOLUTION[0] // cfg.PRCNN.ROI_XFORM_RESOLUTION[0] == \
               cfg.PRCNN.RESOLUTION[1] // cfg.PRCNN.ROI_XFORM_RESOLUTION[1]
        self.up_scale = cfg.PRCNN.RESOLUTION[0] // (cfg.PRCNN.ROI_XFORM_RESOLUTION[0] * 2)

        deconv_kernel = 4
        self.parsing_score_lowres = nn.ConvTranspose2d(
            256,
            num_parsing,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )

        nn.init.kaiming_normal_(self.parsing_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.parsing_score_lowres.bias, 0)

        self.dim_out = num_parsing

    def forward(self, x):
        #pdb.set_trace()
        x = self.parsing_score_lowres(x)
        if self.up_scale > 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=True)

        return x
