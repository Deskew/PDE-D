from collections import defaultdict

import math
import torch
import torch.nn as nn
import functools
from torch.nn import init
import torch.nn.functional as F
from cacti.utils.utils import A, At
import numpy as np
from einops import rearrange
import numbers

from .builder import MODELS

# def A(x,Phi):
#     temp = x*Phi
#     y = torch.sum(temp,dim=1,keepdim=True)
#     return y

# def At(y,Phi):
#     x = y*Phi
#     return x

ACT_FN = {
    'gelu': nn.GELU(),
    'relu' : nn.ReLU(),
    'lrelu' : nn.LeakyReLU(),
}


def DWConv(dim, kernel_size, stride, padding, bias=False):
    return nn.Conv2d(dim, dim, kernel_size, stride, padding, bias=bias, groups=dim)

def PWConv(in_dim, out_dim, bias=False):
    return nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias)

def DWPWConv(in_dim, out_dim, kernel_size, stride, padding, bias=False, act_fn_name="gelu"):
    return nn.Sequential(
        DWConv(in_dim, in_dim, kernel_size, stride, padding, bias),
        ACT_FN[act_fn_name],
        PWConv(in_dim, out_dim, bias)
    )

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.gelu = nn.GELU()

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.gelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class BlockInteraction(nn.Module):
    def __init__(self, in_channel, out_channel, act_fn_name="gelu", bias=False):
        super(BlockInteraction, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias),
            ACT_FN[act_fn_name],
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=bias)
        )
       
    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)




class StageInteraction(nn.Module):
    def __init__(self, dim, act_fn_name="lrelu", bias=False):
        super().__init__()
        self.st_inter_enc = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.st_inter_dec = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.act_fn = ACT_FN[act_fn_name]
        self.phi = DWConv(dim, 3, 1, 1, bias=bias)
        self.gamma = DWConv(dim, 3, 1, 1, bias=bias)

    def forward(self, inp, pre_enc, pre_dec):
        out = self.st_inter_enc(pre_enc) + self.st_inter_dec(pre_dec)
        skip = self.act_fn(out)
        phi = torch.sigmoid(self.phi(skip))
        gamma = self.gamma(skip)

        out = phi * inp + gamma

        return out


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        return x + self.module(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class TemporalBranch(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.opt = opt
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.qkv = nn.Conv2d(opt.cr, opt.cr*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(opt.cr*3, opt.cr*3, kernel_size=3, stride=1, padding=1, groups=opt.cr*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        

    def forward(self, x):
        b,c,h,w = x.shape
        x = self.norm(x)
        x_in = x.view(-1,self.opt.cr,c,h,w)
        x_in = rearrange(x_in, 'b d c h w -> (b c) d h w')
        qkv = self.qkv_dwconv(self.qkv(x_in)) #32 24 256 256
        q, k, v = qkv.chunk(3, dim=1) ##1 8 32 256 256

        q, k, v = map(lambda t: rearrange(t, 'b (head c) h w -> b head c (h w)', head=self.num_heads), (q, k, v))
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(b, head, c_, h*w) -> (b, c, h, w)
        out_ = out.view(-1,c,self.opt.cr,h,w)
        out_ = rearrange(out_,'b c d h w -> (b d) c h w')
        out_ = self.project_out(out_) # (b, c, h, w)
        return out_


class BasicConv2d(nn.Module):
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 kernel_size, 
                 stride, 
                 groups = 1, 
                 padding = 0, 
                 bias = False,
                 act_fn_name = "gelu",
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.act_fn = ACT_FN[act_fn_name]

    def forward(self, x):
        x = self.conv(x)
        x = self.act_fn(x)
        return x


class DW_Inception(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim,
                 bias=False
    ):
        super(DW_Inception, self).__init__()
        self.branch0 = BasicConv2d(in_dim, out_dim // 4, kernel_size=1, stride=1, bias=bias)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_dim, out_dim // 6, kernel_size=1, stride=1, bias=bias),
            BasicConv2d(out_dim // 6, out_dim // 6, kernel_size=3, stride=1, groups=out_dim // 6, padding=1, bias=bias),
            BasicConv2d(out_dim // 6, out_dim // 4, kernel_size=1, stride=1, bias=bias)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_dim, out_dim // 6, kernel_size=1, stride=1, bias=bias),
            BasicConv2d(out_dim // 6, out_dim // 6, kernel_size=3, stride=1, groups=out_dim // 6, padding=1, bias=bias),
            BasicConv2d(out_dim // 6, out_dim // 4, kernel_size=1, stride=1, bias=bias),
            BasicConv2d(out_dim // 4, out_dim // 4, kernel_size=3, stride=1, groups=out_dim // 4, padding=1, bias=bias),
            BasicConv2d(out_dim // 4, out_dim // 4, kernel_size=1, stride=1, bias=bias)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_dim, out_dim//4, kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class SpatialBranch(nn.Module):
    def __init__(self,
                 dim,  
                 DW_Expand=2, 
                 bias=False,
                 LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)
        self.inception = DW_Inception(
            in_dim=dim, 
            out_dim=dim*DW_Expand, 
            bias=bias
        )
    
    def forward(self, x):
        x = self.norm(x)
        x = self.inception(x)
        return x

## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66, 
                 bias = False,
                 LayerNorm_type = "WithBias",
                 act_fn_name = "gelu"
    ):
        super(Gated_Dconv_FeedForward, self).__init__()
        self.norm = LayerNorm(dim, LayerNorm_type = LayerNorm_type)

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.act_fn = ACT_FN[act_fn_name]

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x


def FFN_FN(
    ffn_name,
    dim, 
    ffn_expansion_factor=2.66, 
    bias=False,
    LayerNorm_type="WithBias",
    act_fn_name = "gelu"
):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim, 
                ffn_expansion_factor=ffn_expansion_factor, 
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                act_fn_name = act_fn_name
            )


class SpatialTemporal_Block(nn.Module):
    def __init__(self, 
                 opt,
                 dim, 
                 num_heads, 
    ):
        super().__init__()
        self.opt = opt
        dw_channel = dim * opt.DW_Expand
        if opt.spatial_branch:
            self.spatial_branch = SpatialBranch(
                dim, 
                DW_Expand=opt.DW_Expand,
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )
            self.spatial_gelu = nn.GELU()
            self.spatial_conv = nn.Conv2d(in_channels=dw_channel, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=opt.bias)

        if opt.temporal_branch:
            self.temporal_branch = TemporalBranch(
                opt,
                dim, 
                num_heads=num_heads, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )

        if opt.temporal_interaction:
            self.temporal_interaction = nn.Sequential(
                nn.Conv2d(dim, dim // 8, kernel_size=1, bias=opt.bias),
                LayerNorm(dim // 8, opt.LayerNorm_type),
                nn.GELU(),
                nn.Conv2d(dim // 8, dw_channel, kernel_size=1, bias=opt.bias),
            )

        self.ffn = Residual(
            FFN_FN(
                dim=dim, 
                ffn_name=opt.ffn_name,
                ffn_expansion_factor=opt.FFN_Expand, 
                bias=opt.bias,
                LayerNorm_type=opt.LayerNorm_type
            )
        )

    def forward(self, x):
        log_dict = defaultdict(list)
        b, c, h, w = x.shape

        spatial_fea = 0
        temporal_fea = 0
    
        if self.opt.spatial_branch: 
            spatial_identity = x
            spatial_fea = self.spatial_branch(x)
            spatial_fea = self.spatial_gelu(spatial_fea)

        if self.opt.temporal_branch:
            temporal_identity = x
            temporal_fea = self.temporal_branch(x)
        if self.opt.temporal_interaction:
            temporal_interaction = self.temporal_interaction(
                F.adaptive_avg_pool2d(temporal_fea, output_size=1))
            temporal_interaction = torch.sigmoid(temporal_interaction).repeat((1, 1, h, w))
            spatial_fea = temporal_interaction * spatial_fea
        if self.opt.spatial_branch:
            spatial_fea = self.spatial_conv(spatial_fea)
            spatial_fea = spatial_identity + spatial_fea
            log_dict['block_spatial_fea'] = spatial_fea
        if self.opt.temporal_branch:
            temporal_fea = temporal_identity + temporal_fea
            log_dict['block_temporal_fea'] = temporal_fea
        
        fea = spatial_fea + temporal_fea

        out = self.ffn(fea)


        return out, log_dict


class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels*2, 3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            # blinear interpolate may make results not deterministic
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels//2, 3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        x = self.up(x)
        return x



class SpatialTemporal_Transformer(nn.Module):
    def __init__(self, opt, bi_fuse=None):
        super().__init__()
        self.opt = opt
        self.bi_fuse = bi_fuse
        self.embedding = nn.Conv2d(opt.in_dim, opt.dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)

        self.Encoder = nn.ModuleList([
            SpatialTemporal_Block(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0),
            SpatialTemporal_Block(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1),
        ])

        self.BottleNeck = SpatialTemporal_Block(opt = opt, dim = opt.dim * 2 ** 2, num_heads = 2 ** 2)

        
        self.Decoder = nn.ModuleList([
            SpatialTemporal_Block(opt = opt, dim = opt.dim * 2 ** 1, num_heads = 2 ** 1),
            SpatialTemporal_Block(opt = opt, dim = opt.dim * 2 ** 0, num_heads = 2 ** 0)
        ])

        if opt.block_interaction:
            self.BlockInteractions = nn.ModuleList([
                BlockInteraction(opt.dim * 7, opt.dim * 1),
                BlockInteraction(opt.dim * 7, opt.dim * 2)
            ])

        self.Downs = nn.ModuleList([
            DownSample(opt.dim * 2 ** 0, bias=opt.bias),
            DownSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.Ups = nn.ModuleList([
            UpSample(opt.dim * 2 ** 2, bias=opt.bias),
            UpSample(opt.dim * 2 ** 1, bias=opt.bias)
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 2,
                out_channels = opt.dim * 2 ** 1,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            ),
            nn.Conv2d(
                in_channels = opt.dim * 2 ** 1,
                out_channels = opt.dim * 2 ** 0,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = opt.bias
            )
        ])

        if opt.stage_interaction:
            self.stage_interactions = nn.ModuleList([
                StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 2, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 1, act_fn_name=opt.act_fn_name, bias=opt.bias),
                StageInteraction(dim = opt.dim * 2 ** 0, act_fn_name=opt.act_fn_name, bias=opt.bias)
            ])
        # if bi_fuse:
        #     self.bi_fuse = Bi_neuro(color_channel=opt.in_dim,dim=opt.dim)

        self.mapping = nn.Conv2d(opt.dim, opt.out_dim, kernel_size=1, stride=1, padding=0, bias=opt.bias)
  

    def forward(self, x, enc_outputs=None, bottleneck_out=None, dec_outputs=None, n_seq=None):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        enc_outputs_l = []
        dec_outputs_l = []
        x1 = self.embedding(x)
        res1, log_dict1 = self.Encoder[0](x1)
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            res1 = self.stage_interactions[0](res1, enc_outputs[0], dec_outputs[0])
        res12 = F.interpolate(res1, scale_factor=0.5, mode='bilinear') 

        x2 = self.Downs[0](res1)
        res2, log_dict2 = self.Encoder[1](x2)
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            res2 = self.stage_interactions[1](res2, enc_outputs[1], dec_outputs[1])
        res21 = F.interpolate(res2, scale_factor=2, mode='bilinear') 


        x4 = self.Downs[1](res2)
        res4, log_dict3 = self.BottleNeck(x4)
        if self.opt.stage_interaction and bottleneck_out is not None:
            res4 = self.stage_interactions[2](res4, bottleneck_out, bottleneck_out)

        res42 = F.interpolate(res4, scale_factor=2, mode='bilinear') 
        res41 = F.interpolate(res42, scale_factor=2, mode='bilinear') 
       
        if self.opt.block_interaction:
            res1 = self.BlockInteractions[0](res1, res21, res41) 
            res2 = self.BlockInteractions[1](res12, res2, res42) 
        enc_outputs_l.append(res1)
        enc_outputs_l.append(res2)
        

        dec_res2 = self.Ups[0](res4) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = torch.cat([dec_res2, res2], dim=1) # dim * 2 ** 2
        dec_res2 = self.fusions[0](dec_res2) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2, log_dict4 = self.Decoder[0](dec_res2)
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            dec_res2 = self.stage_interactions[3](dec_res2, enc_outputs[1], dec_outputs[1])
        

        dec_res1 = self.Ups[1](dec_res2) # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1 = torch.cat([dec_res1, res1], dim=1) # dim * 2 ** 1 
        dec_res1 = self.fusions[1](dec_res1) # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1, log_dict5 = self.Decoder[1](dec_res1)
        if self.opt.stage_interaction and (enc_outputs is not None) and (dec_outputs is not None):
            dec_res1 = self.stage_interactions[4](dec_res1, enc_outputs[0], dec_outputs[0])

        dec_outputs_l.append(dec_res1)
        dec_outputs_l.append(dec_res2)

        # if self.bi_fuse is not None:
        #     dec_res1 = self.bi_fuse(x, dec_res1, n_seq)

        out = self.mapping(dec_res1) + x

        return out, enc_outputs_l, res4, dec_outputs_l


def deep_sensitivity(in_channels, out_channels, dim=64, bias=False, act_fn_name="gelu"):
    return nn.Sequential(
        nn.Conv2d(in_channels, dim, 1, 1, 0, bias=bias),
        ACT_FN[act_fn_name],
        nn.Conv2d(dim, dim, 3, 1, 1, bias=bias, groups=dim),
        ACT_FN[act_fn_name],
        nn.Conv2d(dim, out_channels, 1, 1, 0, bias=bias)
    )

def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,:,ba[0]::2,ba[1]::2]
        y_bayer = rearrange(y_bayer,"b f h w ba->(b ba) f h w")
        Phi_bayer = rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = rearrange(Phi_s_bayer,"b f h w ba->(b ba) f h w")

        return y_bayer, Phi_bayer, Phi_s_bayer

class neuro(nn.Module):
    def __init__(self, input_size, dim, forward_pro = True):
        super(neuro,self).__init__()     

        kernel_size = 3
        self.forward_pro = forward_pro
        # input to fuse
        self.i2f = nn.Conv2d(input_size*3+dim*2, dim, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)
        #basic_block = functools.partial(ResidualBlock_noBN, nf=dim)
         #make_layer(basic_block, num_basic_block=5)
        self.recon_res = Residual(FFN_FN(dim=dim, ffn_name="Gated_Dconv_FeedForward"))                             

    
    def  forward(self, input, input_fea, hidden):
        
        _,_,T,_,_ = input.shape
        if self.forward_pro:
            f1 = input[0,:,:,:,:]
            f2 = input[1,:,:,:,:]
            f3 = input[2,:,:,:,:]
        else:
            f3 = input[0,:,:,:,:]
            f2 = input[1,:,:,:,:]
            f1 = input[2,:,:,:,:]        
        x_input = torch.cat((f1, f2, f3), dim=1)
        x_cat = torch.cat((x_input, hidden, input_fea), dim=1)
        hidden_ = self.relu(self.i2f(x_cat))
        hidden = self.recon_res(hidden_)
        
        return hidden     

class Bi_neuro(nn.Module):
    """
    Bidirectional Convolutional RNN layer   

    """
    def __init__(self, color_channel, dim, kernel_size=3):
        super(Bi_neuro, self).__init__()
        self.hidden_size = dim
        self.kernel_size = kernel_size
        self.input_size = color_channel
        self.neuro_forward = neuro(self.input_size, self.hidden_size, forward_pro = True)
        self.neuro_backward = neuro(self.input_size, self.hidden_size, forward_pro = False)
        self.recon_res = Residual(FFN_FN(dim=dim, ffn_name="Gated_Dconv_FeedForward"))


    def forward(self, input_z, input_fea, n_seq=None): # input: (batch_size, num_seqs, channel, width, height)
        
        if n_seq is not None:
            nb_nt, nc, nx, ny = input_z.shape
            input_z = input_z.view(-1,n_seq, nc, nx, ny)
            input_z = rearrange(input_z, 'b d c h w -> b c d h w')
        nb, nc, nt, nx, ny = input_z.shape 
        input = input_z.permute(2,0,1,3,4) #nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        hid_init = torch.zeros(size_h).cuda(input_fea.device)
        #region Bidirectional: forward and backward       
        output_f = []
        output_b = []
        # forward && backward
        hidden_f = hid_init
        #expend
        input_fea_ = input_fea.view(nb, nt, self.hidden_size, nx, ny)
        input_fea_ = rearrange(input_fea_, 'b d c h w -> d b c h w')

        input = torch.cat((input[1:2,:,:,:,:], input, input[nt-1:nt,:,:,:,:]), dim=0)
        # nt: n_seq, the number of frames(0~n_frames)
        for i in range(nt):
            hidden_f = self.neuro_forward(input[i:i+3], input_fea_[i], hidden_f)
            output_f.append(hidden_f) #add element at the end
            
        output_f = torch.cat(output_f) 
         # backward
        hidden_b = hid_init
        # nt: n_seq, the number of frames(0~n_frames)
        for i in range(nt):
            hidden_b = self.neuro_backward(input[nt - i - 1:nt - i + 2], input_fea_[nt - i -1], hidden_b)
            output_b.append(hidden_b)

        output_b = torch.cat(output_b[::-1])
        output = output_f.add_(output_b)
        output = self.recon_res(output)

        #endregion
        # if nb == 1: # nb: number of batch
        #     output = output.view(nt, 1, self.hidden_size, nx, ny) # view: return a reshape tensor(source tensor not be changed)

        return output 


class Sensing_Estimation(nn.Module):
    def __init__(self, opt, color_channels=1, init=False):
        super().__init__()
        self.color_channels=color_channels
        self.init = init
        if color_channels == 3:
            if init == True: 
                self.convert_to_color = BasicConv2d(1, 3, kernel_size=1, stride=1, bias=False) 
            
            self.pre_degradation = BasicConv2d(3, 1, kernel_size=1, stride=1, bias=False)
            self.pre_reconstruction = BasicConv2d(1, 3, kernel_size=1, stride=1, bias=False)    
               

        self.SEstimation = nn.Sequential(
            deep_sensitivity(in_channels=color_channels*2, out_channels=opt.dim, bias=opt.bias, act_fn_name=opt.act_fn_name),
            deep_sensitivity(in_channels=opt.dim, out_channels=color_channels, bias=opt.bias, act_fn_name=opt.act_fn_name),
        )
        self.eta_step = nn.Parameter(torch.Tensor([0.1]))
        self.eta_se = nn.Parameter(torch.Tensor([0.1]))


    def forward(self, z, y, y1, Phi, Phi_s):
        """
        z    : (B, ch, seq, w, h)
        y    : (B, 1, 1, w, h)
        Phi    : (1, 8, w, h)
        y1, Phi_s:(1, 1, w, h)
        """
        n_batch, n_ch, n_seq, n_w, n_h = z.shape

        if self.color_channels==3:
            if self.init == True:
                z_color = rearrange(z,"n_batch n_ch n_seq n_w n_h->(n_batch n_seq) n_ch n_w n_h")
                z_color  = self.convert_to_color(z_color)
                z_pre = z_color
                z_color = z_color.view(n_batch, n_seq, 3, n_w, n_h)
                z = z_color.permute(0,2,1,3,4)
                n_ch = 3
            else:
                z_pre = rearrange(z,"n_batch n_ch n_seq n_w n_h->(n_batch n_seq) n_ch n_w n_h")
            z_pre = self.pre_degradation(z_pre)
            z_pre_ = z_pre.view(n_batch, n_seq, 1, n_w, n_h).contiguous()
            yb = A(z_pre_.squeeze(2),Phi) #[2, 1, 8, 256, 256] * [2, 8, 256, 256]
            y1 = y1 + (y - yb)
            x_0 = At(torch.div(y1-yb, Phi_s+ self.eta_step), Phi).unsqueeze(2).contiguous()
            x_ = self.pre_reconstruction(x_0.view(n_batch*n_seq, 1, n_w, n_h)).view(n_batch, n_seq, 3, n_w, n_h)
            x_pre = rearrange(x_,"a b c d e->a c b d e")
            x = z + x_pre #init[1, 8, 512, 512], then:[1, 3, 8, 512, 512])
            y_ = y / Phi_s  # torch.Size([B, 1, 128, 128])
            y_ = y_.unsqueeze(1).expand_as(x)#   #init:[1, 1, 8, 512, 512], then:[1, 3, 8, 512, 512]

        else:
            yb = A(z.squeeze(1),Phi)
            y1 = y1 + (y - yb)
            x = z.squeeze(1) + At(torch.div(y1-yb, Phi_s+ self.eta_step), Phi)
            x = x.unsqueeze(1) 
            y_ = y / Phi_s  # torch.Size([1, 1, 1, 128, 128])
            y_ = y_.expand_as(z.squeeze(1)).unsqueeze(1)  # torch.Size([1, 8, 1, 128, 128]) 
        
        z_ = rearrange(x,"n_batch n_ch n_seq n_w n_h->(n_batch n_seq) n_ch n_w n_h")
        y_ = rearrange(y_,"n_batch n_ch n_seq n_w n_h->(n_batch n_seq) n_ch n_w n_h")

        est = self.SEstimation(torch.cat((z_, y_), dim=1))
        est = est.view(n_batch, n_seq, n_ch, n_w, n_h).contiguous()
        est_ = rearrange(est,"b s ch w h->b ch s w h")
        z_est = z-self.eta_se*est_
        z_est = rearrange(z_est,"n_batch n_ch n_seq n_w n_h->(n_batch n_seq) n_ch n_w n_h")

        return z_est

class AGA(torch.nn.Module): #adaptive gradient approximation
    """boosting from predictions"""
    def __init__(self, dim):
        super(AGA, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)
        self.w = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x, h):
        h = self.conv1(h) + self.conv2(x)
        x = h + self.w * x
        return x, h       
# for save medium tensor
import os
import os.path as osp 
import time
from cacti.utils.utils import save_single_image
def save_tensor_as_image(tensor,index):
    test_dir = "/home/lbs/Documents/SCI/PR_files/"
    _name = index
    timestamp = time.time()
    config_name ='mixst_sci'+str(timestamp)
    #tensor_ = rearrange(tensor, 'd c h w -> c d h w')
    output = tensor.cpu().numpy().astype(np.float32)
    out = np.array(output)
    for j in range(out.shape[0]):
        image_dir = osp.join(test_dir,_name)
        if not osp.exists(image_dir):
            os.makedirs(image_dir)
        save_single_image(out[j],image_dir,j,name=config_name)

@MODELS.register_module
class MixST_SCI(nn.Module):
    def __init__(self, opt, color_channels=1, share_params=False):
        super().__init__()
        self.opt = opt
        self.color_channels=color_channels
        self.share_params = share_params

        self.init_SE = Sensing_Estimation(opt,color_channels=self.color_channels, init=True)
        #self.init_GD = AGA(dim=color_channels)
        self.init_stage = SpatialTemporal_Transformer(opt)

        #self.bi_fuse = Bi_neuro(color_channel=color_channels,dim=opt.dim)

        self.stages_SE = nn.ModuleList([
            Sensing_Estimation(opt,color_channels=self.color_channels) for _ in range(opt.stage - 2)
        ]) if not share_params else Sensing_Estimation(opt,color_channels=self.color_channels)
        # self.stages_GD = nn.ModuleList([
        #     AGA(dim=color_channels) for _ in range(2*opt.stage - 4)
        # ]) if not share_params else AGA(dim=color_channels)
        self.medium_stages = nn.ModuleList([
            SpatialTemporal_Transformer(opt) for _ in range(opt.stage - 2)
        ]) if not self.share_params else SpatialTemporal_Transformer(opt)

        self.final_SE = Sensing_Estimation(opt,color_channels=self.color_channels)
        # self.final_GD = nn.ModuleList([
        #     AGA(dim=color_channels) for _ in range(2)
        # ])
        self.final_stage = SpatialTemporal_Transformer(opt)
    
    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,:,ba[0]::2,ba[1]::2]
        y_bayer = rearrange(y_bayer,"b f h w ba->(b ba) f h w")
        Phi_bayer = rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = rearrange(Phi_s_bayer,"b f h w ba->(b ba) f h w")

        return y_bayer, Phi_bayer, Phi_s_bayer

    def init(self, y, Phi, Phi_s):
        if self.color_channels==3:
            y_bayer, Phi_bayer, Phi_s_bayer = self.bayer_init(y,Phi,Phi_s) ##B, C, D, H, W = x.shape
            x = At(y_bayer,Phi_bayer)
            yb = A(x,Phi_bayer)
            bayer = [[0,0], [0,1], [1,0], [1,1]]
            b,f,h,w = Phi.shape
            x = x + At(torch.div(y_bayer-yb,Phi_s_bayer),Phi_bayer)
            x = rearrange(x,"(b ba) f h w->b f h w ba",b=b)
            x_bayer = torch.zeros(b,f,h,w).to(y.device)
            for ib in range(len(bayer)): 
                ba = bayer[ib]
                x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
            x = x_bayer.unsqueeze(1)
            y, Phi, Phi_s = y_bayer, Phi_bayer, Phi_s_bayer
            y = torch.nn.functional.pixel_shuffle(y.permute(1,0,2,3), 2).permute(1,0,2,3)
            Phi = torch.nn.functional.pixel_shuffle(Phi.permute(1,0,2,3), 2).permute(1,0,2,3)
            Phi_s = torch.nn.functional.pixel_shuffle(Phi_s.permute(1,0,2,3), 2).permute(1,0,2,3)
        else:
            x = At(y,Phi)
            yb = A(x,Phi)
            x = x + At(torch.div(y-yb,Phi_s),Phi)
            x = x.unsqueeze(1)
        
        return x,y,Phi,Phi_s

    def forward(self, y, Phi, Phi_s):
        out_list = []
        x,y,Phi,Phi_s = self.init(y=y, Phi=Phi, Phi_s=Phi_s) #B, C, D, H, W = x.shape
        y1 = torch.zeros_like(y).to(y.device)
        n_batch, n_ch, n_seq, n_width, n_height = x.shape
        if self.color_channels==3:
            n_ch =3
        x = self.init_SE(x, y, y1, Phi, Phi_s)
        # # save x,y1
        # save_tensor_as_image(x, index="x_1")
        z, enc_outputs, bottolenect_out, dec_outputs = self.init_stage(x)
        z = z.view(-1,n_seq, n_ch, n_width, n_height)
        z = rearrange(z, 'b d c h w -> b c d h w')
        # #save z
        # save_tensor_as_image(z, index="z_1")

        #out_list.append(z)
        #enc_outputs[0]=self.bi_fuse(z,enc_outputs[0])

        for i in range(self.opt.stage-2):
            x = self.stages_SE[i](z, y, y1, Phi, Phi_s) if not self.share_params else self.stages_SE(z, y, y1, Phi, Phi_s)
            # # save x
            # save_tensor_as_image(x, index="x_"+str(i+2))
            z, enc_outputs, bottolenect_out, dec_outputs = self.medium_stages[i](x,enc_outputs,bottolenect_out, dec_outputs) if not self.share_params else self.medium_stages(
                                                                                     x, enc_outputs, bottolenect_out, dec_outputs)
            z = z.view(-1,n_seq, n_ch, n_width, n_height)
            z = rearrange(z, 'b d c h w -> b c d h w')
            # #save z
            # save_tensor_as_image(z, index="z_"+str(i+2))
            #out_list.append(z)
        x = self.final_SE(z, y, y1, Phi, Phi_s)
        # # save x
        # save_tensor_as_image(x, index="x_5")
        z, enc_outputs, bottolenect_out, dec_outputs = self.final_stage(x, enc_outputs, bottolenect_out, dec_outputs,n_seq=n_seq)
        z = z.view(-1,n_seq, n_ch, n_width, n_height)
        out = rearrange(z, 'b d c h w -> b c d h w')
        # #save z
        # save_tensor_as_image(out, index="z_5")
        if self.color_channels!=3:
            out = out.squeeze(1)
        out_list.append(out)

        return out_list
    
if __name__ == '__main__':
    from torch.autograd import Variable, grad
    temp = dict(
    spatial_branch=True,
    temporal_branch=True,
    temporal_interaction=True,
    spatial_interaction = False,
    stage_interaction=True,
    block_interaction=True,
    bias=False,
    in_dim=3, 
    out_dim=3,
    cr=8,
    dim=32,#48, #16, #64
    stage=5,#9,#
    DW_Expand=1, #expand of depth-wise convolution
    FFN_Expand=2.66,
    share_params=False,#True,# # whether stage share parameters
    ffn_name='Gated_Dconv_FeedForward',
    LayerNorm_type='BiasFree',
    act_fn_name='gelu')

    class DotDict(dict):

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    temp_ = DotDict(temp)

    model = MixST_SCI(opt=temp_, color_channels=3, share_params=False)
    input,target_gnd= torch.randn(1, 1, 512, 512),torch.randn(1, 3, 8, 256, 256) # input and target are both tensor, input:[N,C,T,H,W] , target:[N,C,H,W]
    mask = torch.randn(1, 8, 512, 512)
    mask_s = torch.randn(1, 1, 512, 512)

    print(model)

    from thop import profile
    from thop import clever_format
    test=False
    macs, params = profile(model, inputs=(input, mask, mask_s)) 
    
    macs, params = clever_format([macs, params], "%.5f")
    print(macs)     
    print(params)

    #5iter-gray
    #base-32ch: 512.11882G 3.56756M
    #large-64ch: 1.94527T  13.96581M
    #5iter-color
    #base-32ch:  2053.91G  3.57016M

    #9iter-gray
    #base-32ch: 926.89402G  6.46882M
    #9iter-color
    #base-32ch: 3717.36G  6.47348M

    #share
    #926.89402G
    #2.11693M
    #not share
    #926.89402G
    #6.46882M