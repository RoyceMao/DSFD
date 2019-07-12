# -*- coding:utf-8 -*-

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import cur_config as cfg
import layers.my_priorbox as my_priorbox
import layers.my_detection as my_detection


class FEM(nn.Module):
    def __init__(self, channel_size):
        super(FEM, self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d(256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d(256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1)

    def forward(self, x):
        x1_1 = F.relu(self.cpm1(x), inplace=True)
        x1_2 = F.relu(self.cpm2(x), inplace=True)
        x2_1 = F.relu(self.cpm3(x1_2), inplace=True)
        x2_2 = F.relu(self.cpm4(x1_2), inplace=True)
        x3_1 = F.relu(self.cpm5(x2_2), inplace=True)
        return torch.cat([x1_1, x2_1, x3_1], 1)


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, cfg, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        assert (num_classes == 2)
        self.cfg = cfg
        self.size = cfg.INPUT_SIZE[0]

        # backbone
        print("loading pretrained resnet model")
        resnet = torchvision.models.resnet152(pretrained=True)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(
            *[nn.Conv2d(2048, 512, kernel_size=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True)]
        )
        self.layer6 = nn.Sequential(
            *[nn.Conv2d(512, 128, kernel_size=1, ),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True)]
        )

        # channels输出
        output_channels = [256, 512, 1024, 2048, 512, 256]

        # =============
        # 特征点乘融合层
        fpn_in = output_channels
        # self.latlayer6 = nn.AdaptiveAvgPool2d((1,1))
        # self.latlayer5 = nn.Conv2d( fpn_in[5], fpn_in[4], kernel_size=1, stride=1, padding=0)
        # self.latlayer4 = nn.Conv2d( fpn_in[4], fpn_in[3], kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)
        # self.smooth6 = nn.Conv2d( fpn_in[5], fpn_in[5], kernel_size=1, stride=1, padding=0)
        # self.smooth5 = nn.Conv2d( fpn_in[4], fpn_in[4], kernel_size=1, stride=1, padding=0)
        # self.smooth4 = nn.Conv2d( fpn_in[3], fpn_in[3], kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

        # =============
        # 特征增强FEM层
        cpm_in = output_channels
        # self.cpm3_3 = nn.Conv2d(cpm_in[0], 512, kernel_size=1)
        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])

        # head分支
        head = multibox(output_channels, self.cfg.MBOX, num_classes)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = my_detection.Detection(cfg)

    def init_priorbox(self, input_size, features_maps, cfg):
        priorbox = my_priorbox.PriorBox(input_size, features_maps, cfg)
        prior = Variable(priorbox.forward(), volatile=True)
        return prior

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        conf = list()

        # 基网络
        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        # feature map上采样
        # lfpn6 = self._upsample_product( self.latlayer6(conv7_2_x) , self.smooth6(conv7_2_x))
        # lfpn5 = self._upsample_product( self.latlayer5(lfpn6) , self.smooth5(conv6_2_x))
        # lfpn4 = self._upsample_product( self.latlayer4(lfpn5) , self.smooth4(fc7_x) )
        # lfpn3 = self._upsample_product( self.latlayer3(lfpn4) , self.smooth3(conv5_3_x) )

        lfpn3 = self._upsample_product(self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
        lfpn2 = self._upsample_product(self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
        lfpn1 = self._upsample_product(self.latlayer1(lfpn2), self.smooth1(conv3_3_x))

        # conv7_2_x = lfpn6
        # conv6_2_x = lfpn5
        # fc7_x     = lfpn4

        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]

        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        # head层接上FEM输出
        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.shape[2], x.shape[3]])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            len_conf = len(conf)
            if self.cfg.MBOX[0] ==1 :
                cls = self.mio_module(c(x),len_conf)
            else:
                mmbox = torch.chunk(c(x) , self.cfg.MBOX[0] , 1)
                cls_0 = self.mio_module(mmbox[0], len_conf)
                cls_1 = self.mio_module(mmbox[1], len_conf)
                cls_2 = self.mio_module(mmbox[2], len_conf)
                cls_3 = self.mio_module(mmbox[3], len_conf)
                cls = torch.cat([cls_0, cls_1, cls_2, cls_3] , dim=1)
            conf.append(cls.permute(0, 2, 3, 1).contiguous())

        mbox_num = self.cfg.MBOX[0]
        face_loc = torch.cat([o[:, :, :, :4 * mbox_num].contiguous().view(o.size(0), -1) for o in loc], 1)
        face_conf = torch.cat([o[:, :, :, :2 * mbox_num].contiguous().view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            features_maps = featuremap_size  # 根据具体输入情况来修改cfg
            input_size = image_size  # 根据具体输入情况来修改cfg
            self.priorbox = self.init_priorbox(input_size, features_maps, self.cfg)
            output = self.detect(
                face_loc.view(face_loc.size(0), -1, 4),
                self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes)),  # face_cls过softmax函数做归一化
                self.priorbox.type(type(x.data))
            )

        else:
            features_maps = featuremap_size  # 根据具体输入情况来修改cfg
            input_size = image_size  # 根据具体输入情况来修改cfg
            self.priorbox = self.init_priorbox(input_size, features_maps, self.cfg)
            output = (
                face_loc.view(face_loc.size(0), -1, 4),
                face_conf.view(face_conf.size(0), -1, self.num_classes),
                self.priorbox.type(type(x.data))
            )
        return output


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def mio_module(self, each_mmbox, len_conf):
        chunk = torch.chunk(each_mmbox, each_mmbox.shape[1], 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls = (torch.cat([bmax, chunk[3]], dim=1) if len_conf == 0 else torch.cat([chunk[3], bmax], dim=1))
        if len(chunk) == 6:
            cls = torch.cat([cls, chunk[4], chunk[5]], dim=1)
        elif len(chunk) == 8:
            cls = torch.cat([cls, chunk[4], chunk[5], chunk[6], chunk[7]], dim=1)
        return cls

    def _upsample_product(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y


class DeepHeadModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule , self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)
        #print(self._mid_channels)
        self.conv1 = nn.Conv2d( self._input_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d( self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d( self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d( self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1, padding=0)
    def forward(self, x):
        return self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)), inplace=True))


def multibox(output_channels, mbox_cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = 512
        if k == 0:
            loc_output = 4
            conf_output = 2
        elif k == 1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * loc_output)]
        conf_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * (2 + conf_output))]
    return (loc_layers, conf_layers)


def build_net_ssd(phase, cfg, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if cfg.INPUT_SIZE[0] != 640:
        print("ERROR: You specified size " + repr(cfg.INPUT_SIZE[0]) + ". However, " +
              "currently only SSD640 (size=640) is supported!")
    return SSD(phase, cfg, num_classes)
