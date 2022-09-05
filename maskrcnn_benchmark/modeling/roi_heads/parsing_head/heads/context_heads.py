import torch
from torch import nn
from torch.nn import functional as F

from models.ops import NonLocal2d
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.utils.poolers import Pooler
from utils.net import make_conv
import pdb



class PSPModule(nn.Module):

    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        
        use_bn = cfg.AIParsing.Context_HEAD.USE_BN
        use_gn = cfg.AIParsing.Context_HEAD.USE_GN
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            make_conv(features+len(sizes)*features, out_features, kernel=3, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True),
            )

    def _make_stage(self, features, out_features, size):
        use_bn = cfg.AIParsing.Context_HEAD.USE_BN
        use_gn = cfg.AIParsing.Context_HEAD.USE_GN
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = make_conv(features, out_features, kernel=1, use_bn=use_bn, use_gn=use_gn, use_relu=True)
        
        return nn.Sequential(prior, conv)

    def forward(self, feats):

        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear',align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class GE_theoLayer(nn.Module):
    def __init__(self, channel):
        super(GE_theoLayer, self).__init__()
        self.gather = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        context_gather = self.gather(x) * x       

        return context_gather



class GE_4_theoLayer(nn.Module):
    def __init__(self, channel):
        super(GE_4_theoLayer, self).__init__()        
        self.gather = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
        )
        self.gather_sigmoid = nn.Sigmoid()


    def forward(self, x):
        h, w = x.size(2), x.size(3)
        context_gather = F.upsample(input=self.gather(x), size=(h, w), mode='bilinear',align_corners=True)
        context_gather_sigmoid = self.gather_sigmoid(context_gather) * x 
        return context_gather_sigmoid


class GE_8_theoLayer(nn.Module):
    def __init__(self, channel):
        super(GE_8_theoLayer, self).__init__()        
        self.gather = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
        )
        self.gather_sigmoid = nn.Sigmoid()


    def forward(self, x):
        h, w = x.size(2), x.size(3)
        context_gather = F.upsample(input=self.gather(x), size=(h, w), mode='bilinear',align_corners=True)
        context_gather_sigmoid = self.gather_sigmoid(context_gather) * x 
        return context_gather_sigmoid

class GE_16_theoLayer(nn.Module):
    def __init__(self, channel):
        super(GE_16_theoLayer, self).__init__()        
        self.gather = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, groups=channel,padding=1, bias=True),
        )
        self.gather_sigmoid = nn.Sigmoid()


    def forward(self, x):
        h, w = x.size(2), x.size(3)
        context_gather = F.upsample(input=self.gather(x), size=(h, w), mode='bilinear',align_corners=True)
        context_gather_sigmoid = self.gather_sigmoid(context_gather) * x
        return context_gather_sigmoid


class PGEC_Module(nn.Module):

    def __init__(self, features, out_features=512):

        super(PGEC_Module, self).__init__()
        use_bn = cfg.AIParsing.Context_HEAD.USE_BN
        use_gn = cfg.AIParsing.Context_HEAD.USE_GN
        self.conv1 = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, padding=1, dilation=1, bias=True),
                                   nn.ReLU()
                                   )

        self.conv2 = nn.Sequential(GE_theoLayer(features),
                                   nn.Conv2d(features, features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(GE_4_theoLayer(features),
                                   nn.Conv2d(features, features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   nn.ReLU())
       
        self.conv4 = nn.Sequential(GE_8_theoLayer(features),
                                   nn.Conv2d(features, features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   nn.ReLU())
        

        self.conv5 = nn.Sequential(GE_16_theoLayer(features),
                                   nn.Conv2d(features, features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   nn.ReLU())
        self.bottleneck = nn.Sequential(
            make_conv(features * 5, out_features, kernel=3, stride=1, use_bn=use_bn, use_gn=use_gn, use_relu=True),
            )        
        

    def forward(self, x):
        
        _, _, h, w = x.size()


        feat1 = self.conv1(x) 
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        
        
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)


        output = self.bottleneck(out)
        return output  



@registry.ROI_PARSING_HEADS.register("roi_context_head")
class roi_context_head(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(roi_context_head, self).__init__()
        self.dim_in = dim_in #dim_in[-1]

        method = cfg.AIParsing.ROI_XFORM_METHOD
        resolution = cfg.AIParsing.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.AIParsing.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_nl = cfg.AIParsing.Context_HEAD.USE_NL
        use_bn = cfg.AIParsing.Context_HEAD.USE_BN
        use_gn = cfg.AIParsing.Context_HEAD.USE_GN
        conv_dim = cfg.AIParsing.Context_HEAD.CONV_DIM

        num_convs_after = cfg.AIParsing.Context_HEAD.NUM_CONVS_AFTER

        #PGEC module
        self.PGEC = PGEC_Module(256,512)


        feat_list = []

        if use_nl:
            feat_list.append(
                NonLocal2d(conv_dim, int(conv_dim * cfg.AIParsing.Context_HEAD.NL_RATIO), conv_dim, use_gn=True)
            )
        self.feat = nn.Sequential(*feat_list)
        self.dim_in = conv_dim

        # convx after pgec and NL module
        assert num_convs_after >= 1
        conv_list = []
        for _ in range(3):
            conv_list.append(
                nn.Conv2d(self.dim_in, conv_dim, kernel_size=3, stride=1, padding=1)
            )
            
            conv_list.append(nn.ReLU())
            self.dim_in = conv_dim
        conv_list.append(
            nn.Conv2d(self.dim_in, conv_dim, kernel_size=3, stride=1, padding=1)
        )
        conv_list.append(nn.GroupNorm(32, conv_dim))
        conv_list.append(nn.ReLU())
        self.conv_after = nn.Sequential(*conv_list) if len(conv_list) else None
        self.dim_out = self.dim_in
        
    def forward(self, x, proposals):
        #pdb.set_trace()
        resolution = cfg.AIParsing.ROI_XFORM_RESOLUTION

        h, w = x[0].size(2), x[0].size(3)

        x = self.pooler(x, proposals)
        roi_feature = x

        pgec_out = self.PGEC(x)
        pgec_out = self.feat(pgec_out)

        if self.conv_after is not None:
            x = self.conv_after(pgec_out)
        return x, roi_feature


