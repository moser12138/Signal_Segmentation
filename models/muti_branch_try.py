import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import xlog1py

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

import torch
import torch.nn as nn


# 标准残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # 是否需要下采样（stride不为1时通常需要）
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 主分支卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 如果需要下采样，调整恒等映射
        if self.downsample is not None:
            identity = self.downsample(x)
        # 将恒等映射与卷积输出相加
        out += identity
        out = self.relu2(out)
        return out

# 下采样卷积
class DownUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownUnit, self).__init__()
        # 3x3卷积核下采样
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # 5x5卷积核下采样
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        # 拼接后的3x3卷积核处理
        self.conv_fuse = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 两路卷积下采样
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        # 通道拼接
        out = torch.cat([out3, out5], dim=1)
        # 3x3卷积处理
        out = self.bn(self.conv_fuse(out))
        # 残差连接
        residual = self.residual_conv(x)
        # 残差相加
        out += residual
        return out

# 注意力特征融合
class AttentionFusionModule(nn.Module):
    def __init__(self, high_res_channels, low_res_channels):
        super(AttentionFusionModule, self).__init__()
        # 上采样低分辨率特征使其分辨率与高分辨率特征一致
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # 将低分辨率特征的通道数降至与高分辨率特征一致
        self.conv_low_res = nn.Conv2d(low_res_channels, high_res_channels, kernel_size=1, stride=1)
        # 注意力机制：生成一个通道注意力权重
        self.attention = nn.Sequential(
            nn.Conv2d(high_res_channels * 2, high_res_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(high_res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_res_channels, high_res_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, high_res_feat, low_res_feat):
        # 上采样低分辨率特征
        low_res_feat_up = self.upsample(low_res_feat)
        # 通过1x1卷积将低分辨率特征的通道数缩减到与高分辨率特征一致
        low_res_feat_up = self.conv_low_res(low_res_feat_up)
        # 拼接高分辨率特征和上采样后的低分辨率特征，在通道维度上进行
        concat_feat = torch.cat([high_res_feat, low_res_feat_up], dim=1)
        # 通过注意力机制生成权重
        attention_weights = self.attention(concat_feat)
        # 使用注意力权重调整低分辨率特征的贡献
        low_res_feat_weighted = low_res_feat_up * attention_weights
        # 将加权后的低分辨率特征与高分辨率特征相加，进行融合
        fused_feat = high_res_feat + low_res_feat_weighted
        return fused_feat

# 混合注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化 + 全连接层降低通道维度，增强重要通道
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        avg_out = self.avg_pool(x)
        channel_att = self.fc(avg_out)
        return x * channel_att

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 使用最大池化和平均池化获取空间维度上的显著特征
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上应用最大池化和平均池化，拼接后通过卷积获取空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv(spatial_att)
        spatial_att = self.sigmoid(spatial_att)
        return x * spatial_att

class CBAM(nn.Module):
    """复合注意力机制模块，结合通道和空间注意力"""
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力机制
        self.channel_attention = ChannelAttention(in_channels, reduction)
        # 空间注意力机制
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 首先应用通道注意力
        x = self.channel_attention(x)
        # 然后应用空间注意力
        x = self.spatial_attention(x)
        return x

# 分割头
class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear', align_corners=algc)

        return out

# 分割网络主干网络
class SemanticSegmentationNet(nn.Module):
    def __init__(self, num_classes=19, augment=True):
        super(SemanticSegmentationNet, self).__init__()
        self.augment = augment
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.s1 = nn.Sequential(
            DownUnit(3, 16),
            DownUnit(16, 32),
            DownUnit(32, 64)
        )
        self.s1_p = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self.s2 = DownUnit(64, 128)
        self.s2_p = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )

        self.s3 = DownUnit(128, 256)
        self.s3_p = nn.Sequential(
            ResidualBlock(256, 256),
            # ResidualBlock(256, 256)
            CBAM(256)
        )

        self.up512 = AttentionFusionModule(128, 256)
        self.up256 = AttentionFusionModule(64, 128)


        # 输入层

        # self.cbam = CBAM(128)

        self.final_layer = segmenthead(64, 128, num_classes)
        self.loss1 = segmenthead(64, 128, num_classes)
        # self.loss2 = segmenthead(64, 128, num_classes)


    def forward(self, x):
        # 输入层
        xtem = self.s1(x)  # [B, 128, H/8, W/8]
        x = self.s1_p(xtem)  # [B, 128, H/8, W/8]

        x1 = self.s2(x)  # [B, 256, H/16, W/16]
        x1 = self.s2_p(x1)  # [B, 128, H/8, W/8]

        x2 = self.s3(x1)  # [B, 512, H/32, W/32]
        x2 = self.s3_p(x2)

        x1 = self.up512(x1, x2)

        x = self.up256(x, x1)

        out = self.final_layer(x)
        # return out

        if self.augment:
            loss0 = self.loss1(xtem)
            return [loss0, out]
        else:
            return out


if __name__ == '__main__':

    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = SemanticSegmentationNet(num_classes=19)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()

    with torch.no_grad():  # 上下文管理器来禁用梯度计算，从而减少内存消耗，并且不记录计算历史
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)