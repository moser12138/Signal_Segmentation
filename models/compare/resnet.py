#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


from torch.nn import BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

class MutiFusion(nn.Module):
    def __init__(self, h_channels, m_channels, l_channels):
        super(MutiFusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.low_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=m_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        # self.low_modify1 = nn.Conv2d(in_channels=2 * m_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False)
        self.low_2 = nn.Sequential(
            nn.Conv2d(in_channels=m_channels, out_channels=l_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(l_channels),
        )
        # self.low_modify2 = nn.Conv2d(in_channels=2 * l_channels, out_channels=l_channels, kernel_size=1, stride=1, bias=False)


        self.mid_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=m_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        # self.mid_modify1 = nn.Conv2d(in_channels=2 * m_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False)
        self.mid_2 = nn.Sequential(
            nn.Conv2d(in_channels=l_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        # self.mid_modify2 = nn.Conv2d(in_channels=2 * m_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False)


        self.high_1 = nn.Sequential(
            nn.Conv2d(in_channels=l_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        # self.high_modify1 = nn.Conv2d(in_channels=2 * m_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False)
        self.high_2 = nn.Sequential(
            nn.Conv2d(in_channels=m_channels, out_channels=h_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(h_channels),
        )
        # self.high_modify2 = nn.Conv2d(in_channels=2 * h_channels, out_channels=h_channels, kernel_size=1, stride=1, bias=False)


    def forward(self, h, m, l):
        l1 = self.relu(self.low_1(h) + m)
        l1 = self.relu(self.low_2(l1) + l)

        # h1 = self.relu(F.interpolate(self.high_1(l), scale_factor=2, mode='bilinear', align_corners=False) + m)
        h1 = self.relu(F.interpolate(self.high_1(l), size=m.shape[2:], mode='bilinear', align_corners=False) + m)

        # h1 = self.relu(F.interpolate(self.high_2(h1), scale_factor=2, mode='bilinear', align_corners=False) + h)
        h1 = self.relu(F.interpolate(self.high_2(h1), size=h.shape[2:], mode='bilinear', align_corners=False) + h)

        # m = self.relu(F.interpolate(self.mid_2(l), scale_factor=2, mode='bilinear', align_corners=False) + m)
        m = self.relu(F.interpolate(self.mid_2(l), size=m.shape[2:], mode='bilinear', align_corners=False) + m)

        m = self.relu(self.mid_1(h) + m)

        return h1 , m , l1

def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=0.1)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear')

        return out
class Resnet18(nn.Module):
    def __init__(self, num_classes=19,  augment=True):
        super(Resnet18, self).__init__()
        self.augment = augment
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer41 = create_layer_basic(256, 512, bnum=1, stride=2)
        self.layer42 = create_layer_basic(512, 512, bnum=1, stride=1)
        self.final_layer = segmenthead(512, 128, num_classes, 32)
        self.loss1 = segmenthead(128, 128, num_classes, 32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer41(feat16) # 1/32
        feat32 = self.layer42(feat32) # 1/32
        out = self.final_layer(feat32)

        if self.augment:
            loss0 = self.loss1(feat8)
            return [loss0, out]
        else:
            return out


    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    net = Resnet18()
    x = torch.randn(16, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    net.get_params()
