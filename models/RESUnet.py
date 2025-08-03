import torch
import torch.nn as nn

# 定义深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 定义ResBlock模块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

# 定义RESUnet模型
class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4):
        super(ResUNet, self).__init__()

        # 编码器
        self.encoders = nn.ModuleList()
        self.encoders.append(DepthwiseSeparableConv(in_channels, 64))
        for _ in range(num_blocks):
            self.encoders.append(ResBlock(64, 64))
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 解码器
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoders = nn.ModuleList()
        for _ in range(num_blocks):
            self.decoders.append(ResBlock(128, 64))
        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        skips = []
        out = x
        for encoder in self.encoders:
            out = encoder(out)
            skips.append(out)
            out = self.pool(out)

        # 解码器
        out = skips.pop()
        for decoder in self.decoders:
            out = self.up1(out)
            skip = skips.pop()
            out = torch.cat([out, skip], dim=1)
            out = decoder(out)

        out = self.conv(out)

        return out