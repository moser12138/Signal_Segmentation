import torch
import torch.nn as nn
import torch.nn.functional as F


# 下采样模块
class DownUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownUnit, self).__init__()

        # 3x3卷积核下采样
        self.conv3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        # 5x5卷积核下采样
        self.conv5 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, stride=2, padding=2)
        # 拼接后的3x3卷积核处理
        self.conv_fuse = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        # 两路卷积下采样
        out3 = self.conv3(x)
        out5 = self.conv5(x)

        # 通道拼接
        out = torch.cat([out3, out5], dim=1)

        # 3x3卷积处理
        out = self.conv_fuse(out)

        # 残差连接
        residual = self.residual_conv(x)

        # 残差相加
        out += residual

        return out


# 注意力融合模块
class AttentionFusionModule(nn.Module):
    def __init__(self, high_res_channels, low_res_channels):
        super(AttentionFusionModule, self).__init__()

        # 上采样低分辨率特征
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 通道匹配
        self.conv_low_res = nn.Conv2d(low_res_channels, high_res_channels, kernel_size=1, stride=1)

        # 注意力机制
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

        # 通道匹配
        low_res_feat_up = self.conv_low_res(low_res_feat_up)

        # 特征拼接
        concat_feat = torch.cat([high_res_feat, low_res_feat_up], dim=1)

        # 计算注意力权重
        attention_weights = self.attention(concat_feat)

        # 加权融合
        low_res_feat_weighted = low_res_feat_up * attention_weights

        # 特征融合
        fused_feat = high_res_feat + low_res_feat_weighted

        return fused_feat


# 复合注意力机制模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        channel_att = self.fc(avg_out)
        return x * channel_att


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv(spatial_att)
        spatial_att = self.sigmoid(spatial_att)
        return x * spatial_att


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# 主干网络
class SemanticSegmentationNet(nn.Module):
    def __init__(self, num_classes):
        super(SemanticSegmentationNet, self).__init__()

        # 输入层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # 编码器
        self.down1 = DownUnit(64, 128)
        self.cbam1 = CBAM(128)

        self.down2 = DownUnit(128, 256)
        self.cbam2 = CBAM(256)

        self.down3 = DownUnit(256, 512)
        self.cbam3 = CBAM(512)

        # 解码器
        self.up2 = AttentionFusionModule(256, 512)
        self.up_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up1 = AttentionFusionModule(128, 256)
        self.up_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # 输出层
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # 输入层
        x1 = self.conv1(x)  # [B, 64, H, W]

        # 编码器
        x2 = self.down1(x1)  # [B, 128, H/2, W/2]
        x2 = self.cbam1(x2)

        x3 = self.down2(x2)  # [B, 256, H/4, W/4]
        x3 = self.cbam2(x3)

        x4 = self.down3(x3)  # [B, 512, H/8, W/8]
        x4 = self.cbam3(x4)

        # 解码器
        up2 = self.up2(x3, x4)  # [B, 256, H/4, W/4]
        up2 = self.up_conv2(up2)

        up1 = self.up1(x2, up2)  # [B, 128, H/2, W/2]
        up1 = self.up_conv1(up1)

        # 输出层
        out = self.final_conv(up1)  # [B, num_classes, H/2, W/2]
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)  # 恢复到原始分辨率

        return out
