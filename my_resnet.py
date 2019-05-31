# -*- coding:utf-8 -*-
"""
   File Name：     my_resnet.py
   Description :   resnet的基网络
   Author :        royce.mao
   date：          2019/05/29
"""

import torch
from torch import nn


class BasicBlock(nn.Module):
    """
    resnet基础block;包含两层卷积(conv-relu-bn)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:  # 需要下采样的情况
            self.down_sample = nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'down_sample') and self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
        resnet bottleneck block;包含三层卷积(conv-relu-bn)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.base_channels = out_channels // 4  # 输出通道数的4分之一

        self.conv1 = nn.Conv2d(in_channels, self.base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.base_channels)

        self.conv2 = nn.Conv2d(
            self.base_channels, self.base_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.base_channels)

        self.conv3 = nn.Conv2d(self.base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:  # 需要下采样的情况
            self.down_sample = nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'down_sample') and self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 channels=[64, 64, 128, 256, 512],
                 blocks=[2, 2, 2, 2],
                 num_classes=20):
        """
        resnet 基础网络
        :param block: BasicBlock 或 Bottleneck
        :param channels: resnet网络的5个stage的输出通道数;
        :param blocks: resnet网络stage 2~5 的block块数，不同的block块数对应不同的网络层数;
        :num_classes: 分类数量
        """
        super(ResNet, self).__init__()
        c1, c2, c3, c4, c5 = channels  # 通道数
        b2, b3, b4, b5 = blocks  # 每个stage包含的block数
        self.flatten_features = c5 * 4  # 特征打平（需要根据具体输入size情况，做修改）
        # 第一个stage 7*7*7的卷积
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))
        # 后续4个stages
        self.stage2 = self._make_stage(block, c1, c2, b2, stride=1)
        self.stage3 = self._make_stage(block, c2, c3, b3, stride=2)
        self.stage4 = self._make_stage(block, c3, c4, b4, stride=2)
        self.stage5 = self._make_stage(block, c4, c5, b5, stride=2)
        # 输出
        self.avgpool = nn.AvgPool2d(7)  # 原torch.resnet是nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(self.flatten_features, num_classes)

    def _make_stage(self, block, in_channels, out_channels, num_blocks, stride=1):
        """

        :param block: BasicBlock 或 Bottleneck
        :param in_channels:
        :param out_channels:
        :param num_blocks: 本层(stage)包含的block块数
        :param stride: 步长
        :return:
        """
        layers = list([])
        # 第一层可能有下采样或通道变化
        layers.append(block(in_channels, out_channels, stride))
        # 后面每一次输入输出通道都一致
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        #
        x = f1 = self.stage1(x)
        x = f2 = self.stage2(x)
        x = f3 = self.stage3(x)
        x = f4 = self.stage4(x)
        x = f5 = self.stage5(x)
        #
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return f1, f2, f3, f4, f5, x


def resnet18(channels, kwargs):
    model = ResNet(BasicBlock, channels, [2, 2, 2, 2], **kwargs)  # [24, 32, 64, 64, 64]
    return model


def resnet34(channels, **kwargs):
    model = ResNet(BasicBlock, channels, [3, 4, 6, 3], **kwargs)  # [24, 32, 64, 64, 64]
    return model


def resnet50(channels, **kwargs):
    model = ResNet(Bottleneck, channels, [3, 4, 6, 3], **kwargs)  # [24, 32, 64, 64, 64]
    return model


def main():
    # channels参数可以随便调的
    # net = ResNet(Bottleneck)
    net = resnet50([64, 64, 128, 256, 512])
    from torchsummary import summary

    summary(net, (3, 512, 512))
    inputs = torch.randn(2, 3, 512, 512)
    out = net(inputs)
    print(out[0].shape, out[1].shape)


if __name__ == '__main__':
    main()
