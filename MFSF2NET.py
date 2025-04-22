import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet50
from pvtv2 import pvt_v2_b2

class MultiFrequencyFeatureExtraction(nn.Module):
    """Multi-frequency feature extraction module using DCT basis filters."""
    def __init__(self, in_channels, dct_h, dct_w, frequency_branches=16,
                 frequency_selection='top', reduction=16):
        super().__init__()
        assert frequency_branches in {1, 2, 4, 8, 16, 32}
        self.num_branches = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        sel = f"{frequency_selection}{frequency_branches}"
        idx_x, idx_y = get_freq_indices(sel)

        idx_x = [i * (dct_h // 7) for i in idx_x]
        idx_y = [j * (dct_w // 7) for j in idx_y]

        for k, (ix, iy) in enumerate(zip(idx_x, idx_y)):
            filt = self._make_dct_filter(ix, iy, in_channels)
            self.register_buffer(f'dct_wgt_{k}', filt)

        mid = in_channels // reduction
        self.mapper = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        B, C, H, W = x.shape
        if (H, W) != (self.dct_h, self.dct_w):
            x_pad = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        else:
            x_pad = x

        sum_avg = sum_max = sum_min = 0
        for name, buf in self.named_buffers():
            if name.startswith('dct_wgt_'):
                xf = x_pad * buf
                sum_avg += self.avg_pool(xf)
                sum_max += self.max_pool(xf)
                sum_min += -self.max_pool(-xf)

        sum_avg /= self.num_branches
        sum_max /= self.num_branches
        sum_min /= self.num_branches

        m_avg = self.mapper(sum_avg)
        m_max = self.mapper(sum_max)
        m_min = self.mapper(sum_min)

        return m_avg + m_max + m_min

    def _make_dct_filter(self, fx, fy, channels):

        H, W = self.dct_h, self.dct_w
        base = torch.zeros(channels, H, W)
        for i in range(H):
            for j in range(W):
                base[:, i, j] = (
                        self._one_dct(i, fx, H) *
                        self._one_dct(j, fy, W)
                )
        return base

    def _one_dct(self, pos, freq, L):
        coef = math.cos(math.pi * freq * (pos + 0.5) / L) / math.sqrt(L)
        return coef if freq == 0 else coef * math.sqrt(2)


def get_freq_indices(mode):

    assert any(mode.startswith(p) for p in ('top', 'low', 'bot'))
    num = int(mode[3:])
    if mode.startswith('top'):
        xs = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        ys = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
    elif mode.startswith('low'):
        xs = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        ys = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
    else:
        xs = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        ys = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
    return xs[:num], ys[:num]

class MultiFrequencyAwareFusion(nn.Module):
    """Fusion module combining multi-frequency attention from ResNet and PVT features."""
    def __init__(self, res_ch, pvt_ch, fuse_ch=256, dct_h=7, dct_w=7,
                 frequency_branches=16, reduction=16):
        super(MultiFrequencyAwareFusion, self).__init__()
        self.res_align = nn.Sequential(
            nn.Conv2d(res_ch, fuse_ch, kernel_size=1),
            nn.BatchNorm2d(fuse_ch),
            nn.ReLU(inplace=True)
        )
        self.pvt_align = nn.Sequential(
            nn.Conv2d(pvt_ch, fuse_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fuse_ch),
            nn.ReLU(inplace=True)
        )
        self.mffe_res = MultiFrequencyFeatureExtraction(fuse_ch, dct_h, dct_w,
                                                         frequency_branches, 'top', reduction)
        self.mffe_pvt = MultiFrequencyFeatureExtraction(fuse_ch, dct_h, dct_w,
                                                         frequency_branches, 'top', reduction)

        self.attn_conv = nn.Sequential(
            nn.Conv2d(fuse_ch * 2, fuse_ch, kernel_size=1),
            nn.BatchNorm2d(fuse_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse_process = nn.Sequential(
            nn.Conv2d(fuse_ch * 2, fuse_ch, kernel_size=1),
            nn.BatchNorm2d(fuse_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, res_feat, pvt_feat):

        if res_feat.shape[-2:] != pvt_feat.shape[-2:]:
            res_feat = F.interpolate(res_feat, size=pvt_feat.shape[-2:], mode='bilinear', align_corners=True)

        res_aligned = self.res_align(res_feat)
        pvt_aligned = self.pvt_align(pvt_feat)

        res_map = self.mffe_res(res_aligned)                  # [B, C, 1, 1]
        res_attn = torch.sigmoid(res_map)                     # [B, C, 1, 1]
        res_mffe = res_aligned * res_attn.expand_as(res_aligned)

        pvt_map = self.mffe_pvt(pvt_aligned)
        pvt_attn = torch.sigmoid(pvt_map)
        pvt_mffe = pvt_aligned * pvt_attn.expand_as(pvt_aligned)

        combined = torch.cat([res_mffe, pvt_mffe], dim=1)
        spatial_attn = self.attn_conv(combined)               # [B,1,H,W]

        res_weighted = res_mffe * spatial_attn
        pvt_weighted = pvt_mffe * (1 - spatial_attn)

        fused = torch.cat([res_weighted, pvt_weighted], dim=1)
        out = self.fuse_process(fused)
        return out



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):
    """Pyramid Split Attention module for multi-scale feature fusion."""
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = ECAWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class ECAWeightModule(nn.Module):
    """Efficient Channel Attention module for weight calculation."""
    def __init__(self, channels, gamma=2, b=1):
        super(ECAWeightModule, self).__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        weights = self.avg_pool(x).view(batch_size, channels, -1).mean(dim=2)
        weights = self.sigmoid(self.conv(weights.unsqueeze(-1))).view(batch_size, channels, 1, 1)
        weights = self.b + self.gamma * weights

        return weights


class PMSF(nn.Module):
    """Pyramid multi-scale fusion using PSA."""
    def __init__(self, fuse_channels, num_stages=4, adapter_channels=256, out_channels=512, target_size=(14, 14)):
        super(PMSF, self).__init__()

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fuse_channels, adapter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(adapter_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(target_size)
            ) for _ in range(num_stages)
        ])

        self.psa = PSAModule(adapter_channels * num_stages, out_channels, conv_kernels=[3, 5, 7, 9])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, fused_features):
        adapted = [adapter(feat) for adapter, feat in zip(self.adapters, fused_features)]
        x = torch.cat(adapted, dim=1)
        x = self.psa(x)
        x = self.pool(x).flatten(1)
        return x


class MFSF2NET(nn.Module):
    """Main network combining ResNet, PVT, and PMSF branch for classification."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = pvt_v2_b2()
        self.resnet = self._get_resnet_backbone()
        self.pvt = pvt_v2_b2()

        self.res_channels = [256, 512, 1024, 2048]
        self.pvt_channels = [64, 128, 320, 512]

        self.fusion_stages = nn.ModuleList([
            MultiFrequencyAwareFusion(res, pvt, fuse_ch=256)
            for res, pvt in zip(self.res_channels, self.pvt_channels)
        ])
        self.third_branch = PMSF(fuse_channels=256)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 512 + 512, 1024),  # ResNet_final + PVT_final + PMSF
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def _get_resnet_backbone(self):
        model = resnet50(pretrained=False)
        return nn.ModuleDict({
            'conv1': nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool
            ),
            'layer1': model.layer1,
            'layer2': model.layer2,
            'layer3': model.layer3,
            'layer4': model.layer4
        })

    def forward(self, x):
        x1 = x
        res_features = []
        x = self.resnet['conv1'](x)  # [B, 64, 56, 56]
        x = self.resnet['layer1'](x)  # [B, 256, 56, 56]
        res_features.append(x)
        x = self.resnet['layer2'](x)  # [B, 512, 28, 28]
        res_features.append(x)
        x = self.resnet['layer3'](x)  # [B, 1024, 14, 14]
        res_features.append(x)
        x = self.resnet['layer4'](x)  # [B, 2048, 7, 7]
        res_features.append(x)

        pvt = self.backbone(x1)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        pvt_features = x1, x2, x3, x4

        assert len(res_features) == len(pvt_features) == 4, "特征阶段数不匹配"

        fused_features = []
        for i in range(4):
            fused = self.fusion_stages[i](res_features[i], pvt_features[i])
            fused_features.append(fused)

        third_feat = self.third_branch(fused_features)

        res_final = F.adaptive_avg_pool2d(res_features[-1], (1, 1)).flatten(1)
        pvt_final = F.adaptive_avg_pool2d(pvt_features[-1], (1, 1)).flatten(1)

        combined = torch.cat([res_final, pvt_final, third_feat], dim=1)
        return self.classifier(combined)




