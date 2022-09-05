from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.config import cfg
import pdb

@registry.ROI_PARSING_OUTPUTS.register("parsing_output")
class Parsing_output(nn.Module):
    def __init__(self, dim_in):
        super(Parsing_output, self).__init__()
        num_parsing = cfg.AIParsing.NUM_PARSING
        assert cfg.AIParsing.RESOLUTION[0] // cfg.AIParsing.ROI_XFORM_RESOLUTION[0] == \
               cfg.AIParsing.RESOLUTION[1] // cfg.AIParsing.ROI_XFORM_RESOLUTION[1]
        self.up_scale = cfg.AIParsing.RESOLUTION[0] // (cfg.AIParsing.ROI_XFORM_RESOLUTION[0] * 2)

        deconv_kernel = 4
        self.parsing_score_lowres = nn.ConvTranspose2d(
            dim_in,
            num_parsing,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )

        nn.init.kaiming_normal_(self.parsing_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.parsing_score_lowres.bias, 0)


        # self.fg_lowres = nn.ConvTranspose2d(
        #     dim_in,
        #     2,
        #     deconv_kernel,
        #     stride=2,
        #     padding=deconv_kernel // 2 - 1,
        # )

        # nn.init.kaiming_normal_(self.fg_lowres.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.constant_(self.fg_lowres.bias, 0)

        self.edge_score_lowres = nn.ConvTranspose2d(
            dim_in,
            2,
            2,
            stride=2,
            padding=0,
        )

        nn.init.kaiming_normal_(self.edge_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.edge_score_lowres.bias, 0)

        self.dim_out = num_parsing

    def forward(self, x):
        #pdb.set_trace()
        x_parse = self.parsing_score_lowres(x)
        x_edge = self.edge_score_lowres(x)
        #x_fg = self.fg_lowres(x)
        if self.up_scale > 1:
            x_parse = F.interpolate(x_parse, scale_factor=self.up_scale, mode="bilinear", align_corners=True)
            x_edge = F.interpolate(x_edge, scale_factor=self.up_scale, mode="bilinear", align_corners=True)
            #x_fg = F.interpolate(x_fg, scale_factor=self.up_scale, mode="bilinear", align_corners=True)

        return x_parse,x_edge
