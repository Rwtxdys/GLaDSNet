import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from timm.models.layers import DropPath
import math



class ParallelDualOrderMultiScaleDirectionalSampling(nn.Module):
    def __init__(self, in_channels, reduced_dim=7, scales=[1, 2, 4], directions=[(1, 0), (0, 1), (1, 1), (1, -1)],
                 choose='d'):
        super().__init__()

    def forward(self, x):

        return out


class ScaleSampling(nn.Module):

    def __init__(self, in_channels, num_scales, num_directions):
        super().__init__()

    def forward(self, scale_first):

        return x


class DirectionSampling(nn.Module):

    def __init__(self, in_channels, num_directions, num_scales):
        super().__init__()

    def forward(self, direction_first):

        return x

class DirectionConv(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[3, 5, 7, 9], max_directions=8):
        super().__init__()

    def forward(self, x):

        return x



class ScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[3, 5, 7, 9], max_directions=8):
        super().__init__()

    def forward(self, x):

        return x


class SDSA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2, num_groups=1, k_ratio=0.1, norm_type='l2'):
      super().__init__()

    def forward(self, query, kv):

        return attn_output


class MB_FFN_1D(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()

    def forward(self, x):

        return x


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, num_groups=8, k_ratio=0.1):
        super().__init__()
        self.self_attn = SDSA(d_model, nhead, dropout, num_groups, k_ratio)
        self.feed_forward = MB_FFN_1D(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, k_v):
        """
        Returns:
            output: [batch, seq_len, d_model]
        """
        x = q
        b, c, _, _ = x.size()

        # 自注意力 + 残差
        attn_output = self.self_attn(q, k_v)
        src = x.view(b, c, -1).permute(0, -1, -2) + self.dropout1(attn_output)
        src = self.norm1(src)

        # 前馈 + 残差
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class DirectionBlock(nn.Module):
    def __init__(self, dim=512, num_heads=8, mlp_ratio=4, top_k_ratio = 0.001,
                 dilation=[3, 5, 7, 9], directions_nums=8, droupout = 0.1
                 ):
        super().__init__()

        self.directionconv = DirectionConv(dim, dim, dilation, directions_nums)

        self.att = Transformer(dim, num_heads, dim // mlp_ratio, num_groups=directions_nums,
                               dropout=droupout, k_ratio=top_k_ratio)
        self.relu = nn.ReLU()

    def forward(self, x):

        b, c, h, w = x.size()

        local_direct_x = self.directionconv(x) + self.relu(x)

        att = self.att(x, local_direct_x)

        return att.permute(0, -1, -2).view(b, c, h, w)

class ScaleBlock(nn.Module):
    def __init__(self, dim=512, num_heads=8, mlp_ratio=4, top_k_ratio = 0.001,
                 dilation=[3, 5, 7, 9], directions_nums=8, droupout = 0.1
                 ):
        super().__init__()

        self.scaleconv = ScaleConv(dim, dim, dilation, directions_nums)

        self.att = Transformer(dim, num_heads, dim // mlp_ratio, num_groups=len(dilation),
                               dropout=droupout, k_ratio=top_k_ratio)

        self.relu = nn.ReLU()

    def forward(self, x):

        b, c, h, w = x.size()

        local_scale_x = self.scaleconv(x) + self.relu(x)

        att = self.att(x, local_scale_x)

        return att.permute(0, -1, -2).view(b, c, h, w)


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )
                   

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )
                   

class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, x, res):
        if x.shape == res.shape:
            pass
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class DenseNet169(nn.Module):
    output_size = 1024

    def __init__(self, pretrained=True):
        super().__init__()

        model = torchvision.models.densenet169(pretrained=pretrained)
        self.features = model.features

    def forward(self, x):
        features = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in [4, 6, 8, 10]:  # 不同密度块后的特征
                features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self,
                 encode_channels = [256, 512, 1024, 2048],
                 decode_channels = 512,
                 dilation = [3, 5, 7, 9],
                 direction_nums = 8,
                 fc_ratio = 4,
                 topk_ratio = [256, 512, 1024],
                 dropout = 0.1,
                 num_classes = 2):
        super(Decoder, self).__init__()
        self.num_classes = num_classes


        # 原始解码器结构
        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels, 1) # 32
        self.b4d = DirectionBlock(decode_channels, direction_nums, fc_ratio, topk_ratio[-1], dilation, direction_nums, dropout)
        self.b4s = ScaleBlock(decode_channels, direction_nums, fc_ratio, topk_ratio[-1], dilation, direction_nums, dropout)


        self.p3 = Fusion(decode_channels) # 64
        self.b3d = DirectionBlock(decode_channels, direction_nums, fc_ratio, topk_ratio[-2], dilation, direction_nums, dropout)
        self.b3s = ScaleBlock(decode_channels, direction_nums, fc_ratio, topk_ratio[-2], dilation, direction_nums, dropout)

        self.p2 = Fusion(decode_channels) # 128
        self.b2d = DirectionBlock(decode_channels, direction_nums, fc_ratio, topk_ratio[-3], dilation, direction_nums, dropout)
        self.b2s = ScaleBlock(decode_channels, direction_nums, fc_ratio, topk_ratio[-3], dilation, direction_nums, dropout)

        self.Conv3 = ConvBNReLU(encode_channels[-3], encode_channels[-4], 1)

        self.p1 = Fusion(encode_channels[-4])

        # self.b1d = DirectionBlock(encode_channels[-4], direction_nums, fc_ratio, topk_ratio, dilation, direction_nums, dropout)
        # self.b1s = ScaleBlock(encode_channels[-4], direction_nums, fc_ratio, topk_ratio, dilation, direction_nums, dropout)

        self.seg_head_d = nn.Sequential(
            DirectionConv(encode_channels[-4], encode_channels[-4], dilation, direction_nums),
            SegHead(encode_channels[-4], num_classes, encode_channels[-4] // fc_ratio, dropout)
        )

        self.seg_head_s = nn.Sequential(
            ScaleConv(encode_channels[-4], encode_channels[-4], dilation, direction_nums),
            SegHead(encode_channels[-4], num_classes, encode_channels[-4] // fc_ratio, dropout)
        )

        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)

        x = self.b4d(res4) + self.b4s(res4)
        x = self.p3(x, res3)


        x = self.b3d(x) + self.b3s(x)
        x = self.p2(x, res2)

        x = self.b2d(x) + self.b2s(x)
        x = self.Conv3(x)
        x = self.p1(x, res1)

        # 主输出
        main_out = self.seg_head_d(x) + self.seg_head_s(x)
        main_out = F.interpolate(main_out, size=(h, w), mode='bilinear', align_corners=False)

        return main_out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MB_FFN_2D(nn.Module):
    def __init__(self, in_c, out_c, droupout):
        super().__init__()

    def forward(self, x):

        return x


class SegHead(nn.Module):

    def __init__(self, in_channels, num_classes, hidden_Channel, dropout=0.1):
        super().__init__()
        self.conv1 = MB_FFN_2D(in_channels, hidden_Channel, dropout)
        self.conv2 = nn.Conv2d(hidden_Channel, num_classes, kernel_size=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x

class GLaDSNet(nn.Module):
    def __init__(self,
                 encode_channels = [256, 512, 1280, 1664],
                 decode_channels = 512,
                 dilation = [3, 5, 7, 9],
                 direction_nums = 8,
                 fc_ratio = 2,
                 topk_ratio = [256, 512, 1024],
                 dropout = 0.1,
                 num_classes = 2):
        super().__init__()
        self.backbone = DenseNet169()
        self.decoder = Decoder(encode_channels, decode_channels, dilation, direction_nums, fc_ratio,
                                topk_ratio, dropout, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        # print(res1.shape) #256
        # print(res2.shape) #128
        # print(res3.shape) #64
        # print(res4.shape) #32
        main_out = self.decoder(res1, res2, res3, res4, h, w)

        return main_out



if __name__ == '__main__':
    from thop import profile

    x = torch.randn(2, 3, 256, 256).to('cuda:0')
    model = GLaDSNet().to('cuda:0')
    flops, params = profile(model, inputs=(x,))
    params = sum(p.numel() for p in model.parameters())
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    print('flops: %.2f G, params: %.2f M' % (flops / (1000 * 1000 * 1000.0), params / (1000 * 1000.0)))
