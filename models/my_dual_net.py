# -*- coding:utf-8 -*-
"""
   File Name：     my_dual_net.py
   Description :   包含FEM、SSD、基网络部分的整体网络（IAM、PLA等head部分还没加进来）
   Author :        royce.mao
   date：          2019/05/29
"""
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# todo
from config import cur_config as cfg
import layers.my_priorbox as my_priorbox
import models.my_resnet as my_resnet
import layers.my_detection as my_detection


class FEM(nn.Module):
    def __init__(self, channel_size):
        """
        特征增强模块的空洞卷积合并部分
        :param channel_size: 融合后的feature map的channels数
        """
        super(FEM, self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d(256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d(256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1)

    def forward(self, x):
        x1_1 = F.relu(self.cpm1(x), inplace=True)
        x2_1 = F.relu(self.cpm2(x), inplace=True)
        x2_2 = F.relu(self.cpm3(x2_1), inplace=True)
        x3_2 = F.relu(self.cpm4(x2_1), inplace=True)
        x3_3 = F.relu(self.cpm5(x3_2), inplace=True)

        return torch.cat((x1_1, x2_2, x3_3), 1)


class DualShot(nn.Module):
    def __init__(self,
                 phase,
                 cfg,
                 num_classes,
                 channels=[64, 64, 128, 256, 512]):
        """
        DSFD网络主体（Enhanced之前，没做channels重置，统一为基网络的channels）
        :param phase: 
        :param cfg: 
        :param num_classes: 
        :param channels: 
        """
        super(DualShot, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.num_classes = num_classes
        assert (self.phase == 'train' or 'test')
        c1, c2, c3, c4, c5 = channels  # 通道数
        # self.output_channels = [256, 512, 1024, 2048, 512, 256]  # Enhance之前是否需要做1*1的卷积把channels重置为output_channels
        # self.resnet = torchvision.models.resnet50()  # torchvision库里面的resnet50
        self.resnet = my_resnet.resnet50(channels)  # 自己单独定义的resnet50
        # 基网络定义的stages搭建为新的4个模块
        self.stage_1 = nn.Sequential(self.resnet.stage1, self.resnet.stage2)  # 论文里640->160的4倍下采样  c2
        self.stage_2 = nn.Sequential(self.resnet.stage3)  # 2倍  c3
        self.stage_3 = nn.Sequential(self.resnet.stage4)  # 2倍  c4
        self.stage_4 = nn.Sequential(self.resnet.stage5)  # 2倍  c5
        # 新增2个模块
        self.stage_5 = nn.Sequential(
            *[nn.Conv2d(c5, 512, kernel_size=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True)]  # 512
        )  # 2倍
        self.stage_6 = nn.Sequential(
            *[nn.Conv2d(512, 128, kernel_size=1, ),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True)]  # 256
        )  # 2倍
        # 第1次Enhanced之前的1*1卷积（用于改变channels，方便特征点乘的融合）
        self.latlayer1 = nn.Conv2d(c3, c2, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(c4, c3, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(c5, c4, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(512, c5, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer6 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # 第1次Enhanced之后的下采样（用于第2次Enhanced）
        self.bup_1 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.bup_2 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        self.bup_3 = nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1)
        self.bup_4 = nn.Conv2d(c5, 512, kernel_size=3, stride=2, padding=1)
        self.bup_5 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        # 一共6个模块做完2次Enhanced之后的空洞FEM操作（注意各模块输入的channels，输出的channels根据FEM类统一为512）
        self.cpm_1 = FEM(c2)
        self.cpm_2 = FEM(c3)
        self.cpm_3 = FEM(c4)
        self.cpm_4 = FEM(c5)
        self.cpm_5 = FEM(512)
        self.cpm_6 = FEM(256)
        # multi-box的head头（施加于6个模块上，组成list模块）
        self.head = self.multibox([512, 512, 512, 512, 512, 512], self.cfg.MBOX,
                                  self.num_classes)  # FEM之后的所有模块channels数都是512
        # nn.ModuleList子类能够被主module所识别，并能做参数更新
        self.loc = nn.ModuleList(self.head[0])
        self.cls = nn.ModuleList(self.head[1])

        # 测试阶段的新层
        # todo: 测试、预测阶段的detect过程
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = my_detection.Detection(self.cfg)

    @staticmethod
    def _upsample_product(x, y):
        """
        （第1次Enhanced）无参数学习的上采样过程与点乘运算
        :param x: 下一层stage的feature map
        :param y: 上一层stage的feature map
        :return: 
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y

    @staticmethod
    def multibox(output_channels, mbox_cfg, num_classes):
        """
        各模块的head网络部分，得到每个模块的cls输出与regr输出
        :param output_channels: 6个模块的channels输出
        :param mbox_cfg: 6个模块的boxes数量
        :param num_classes: 类别数
        :return: 
        """
        loc_layers = []
        cls_layers = []
        for i, c in enumerate(output_channels):
            input_channel = c
            loc_layers += [nn.Conv2d(input_channel, mbox_cfg[i] * 4, kernel_size=3, padding=1)]
            cls_layers += [nn.Conv2d(input_channel, mbox_cfg[i] * num_classes, kernel_size=3, padding=1)]
        return loc_layers, cls_layers

    @staticmethod
    def init_priorbox(cfg):
        """
        在6个模块的feature_maps上面初始化priorbox
        :param cfg: 
        :return: 
        """
        priorbox = my_priorbox.PriorBox(cfg)
        priorbox = Variable(priorbox.forward(), volatile=True)
        return priorbox

    def load_weights(self, base_file):
        """
        按层顺序加载权重
        :param base_file: 
        :return: 
        """
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


    def forward(self, x):
        """

        :param x: inputs
        :return: 
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,5]
            train:
                list of concat outputs from:
                    1: cls layers, Shape: [batch，num_priorbox，num_classes]
                    2: loc layers, Shape: [batch，num_priorbox，4]
                    3: roi layers, Shape: [2,num_priorbox*4]
        """
        # 需要的中间参数
        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        cls = list()
        # 级联网络First Shot PAL
        of_1 = self.stage_1(x)
        of_2 = self.stage_2(of_1)
        of_3 = self.stage_3(of_2)
        of_4 = self.stage_4(of_3)
        of_5 = self.stage_5(of_4)
        of_6 = self.stage_6(of_5)
        # 级联网络second shot PAL的产生
        ef_6 = self._upsample_product(self.latlayer6(of_6), of_6)
        ef_5 = self._upsample_product(self.latlayer5(ef_6), of_5)
        ef_4 = self._upsample_product(self.latlayer4(ef_5), of_4)
        ef_3 = self._upsample_product(self.latlayer3(ef_4), of_3)
        ef_2 = self._upsample_product(self.latlayer2(ef_3), of_2)
        ef_1 = self._upsample_product(self.latlayer1(ef_2), of_1)
        # Enhanced features后的进一步卷积、激活特征融合（第2次Enhanced）
        conv3_3 = ef_1
        conv4_3 = F.relu(self.bup_1(conv3_3)) * ef_2
        conv5_3 = F.relu(self.bup_2(conv4_3)) * ef_3
        conv_fc7 = F.relu(self.bup_3(conv5_3)) * ef_4
        conv6_2 = F.relu(self.bup_4(conv_fc7)) * ef_5
        conv7_2 = F.relu(self.bup_5(conv6_2)) * ef_6
        # 空洞卷积的FEM操作
        out_1 = self.cpm_1(conv3_3)
        out_2 = self.cpm_2(conv4_3)
        out_3 = self.cpm_3(conv5_3)
        out_4 = self.cpm_4(conv_fc7)
        out_5 = self.cpm_5(conv6_2)
        out_6 = self.cpm_6(conv7_2)
        # head层
        fp_size = []
        for (fp, r, c) in zip([out_1, out_2, out_3, out_4, out_5, out_6], self.loc, self.cls):
            # fp size
            fp_size.append([fp.shape[2], fp.shape[3]])
            # 每个模块的空洞卷积FEM操作层后都分别接上，1个loc输出的卷积层，1个cls输出的卷积层
            loc.append(r(fp).permute(0, 2, 3, 1).contiguous())  # 因为是用通道来保存特征向量，所以channels last
            cls.append(c(fp).permute(0, 2, 3, 1).contiguous())  # 因为是用通道来保存特征向量，所以channels last
        # 分别综合6个模块的loc与cls
        face_loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)  # reshape过程等于是做了flatten，拉成1列特征向量
        face_cls = torch.cat([o.view(o.size(0), -1) for o in cls], 1)  # reshape过程等于是做了flatten，拉成1列特征向量

        # 最终输出output
        ## train
        if self.phase == "train":
            self.cfg.FEATURE_MAPS = fp_size  # 根据具体输入情况来修改cfg
            self.cfg.MIN_DIM = image_size  # 根据具体输入情况来修改cfg
            self.priorbox = self.init_priorbox(self.cfg)
            output = (
                face_loc.view(face_loc.size(0), -1, 4),
                face_cls.view(face_cls.size(0), -1, self.num_classes),
                self.priorbox.type(type(x.data))
            )
        ## test
        else:
            self.cfg.FEATURE_MAPS = fp_size  # 根据具体输入情况来修改cfg
            self.cfg.MIN_DIM = image_size  # 根据具体输入情况来修改cfg
            self.priorbox = self.init_priorbox(self.cfg)
            # print(face_loc.view(face_loc.size(0), -1, 4).shape)  # torch.Size([1, 125078, 4])
            # print(self.priorbox.type(type(x.data)).shape)  # torch.Size([750468, 4])
            output = self.detect(
                face_loc.view(face_loc.size(0), -1, 4),
                self.softmax(face_cls.view(face_cls.size(0), -1, self.num_classes)),  # face_cls过softmax函数做归一化
                self.priorbox.type(type(x.data))
            )

        return output


def main():
    # channels参数可以随便调的
    net = DualShot('train', cfg, 5)
    from torchsummary import summary

    summary(net, (3, 512, 512))
    # inputs = torch.randn(2, 512, 32, 32)
    # out = net(inputs)
    # print(out[0].shape, out[1].shape)


if __name__ == '__main__':
    main()
