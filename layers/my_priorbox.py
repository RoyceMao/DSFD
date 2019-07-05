# -*- coding:utf-8 -*-
"""
   File Name：     my_priorbox.py
   Description :   在给定一系列feature maps上面生成priorbox
   Author :        royce.mao
   date：          2019/05/31
"""
import torch
from itertools import product as product
import math


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, input_size, feature_maps, cfg):
        super(PriorBox, self).__init__()
<<<<<<< Updated upstream
        self.image_size = cfg.MIN_DIM  # 如[300, 300]尺寸
        self.variance = cfg.VARIANCE or [0.1]  # 如[0.1, 0.1, 0.2, 0.2]形式，用来放大梯度
        self.feature_maps = cfg.FEATURE_MAPS  # [[H,W],[H,W],...]的形式
        self.min_sizes = cfg.MIN_SIZES  # len(min_sizes) == len(max_sizes)
        self.max_sizes = cfg.MAX_SIZES  # len(min_sizes) == len(max_sizes)
        self.steps = cfg.STEPS  # image_size / feature_map_size下采样倍数
        self.aspect_ratios = cfg.ASPECT_RATIO  # [[2, 1/2], [3, 1/3],...]的形式
        self.clip = cfg.CLIP  # 归一化坐标超出边界范围的修剪
        # self.num_priors = len(cfg['aspect_ratios'])  #
        # self.version = cfg['name']
=======
        self.imh = input_size[0]
        self.imw = input_size[1]

        # number of priors for feature map location (either 4 or 6)
        self.variance = cfg.VARIANCE
        #self.feature_maps = cfg.FEATURE_MAPS
        self.min_sizes = cfg.BASE_SIZES
        self.aspect_ratio = cfg.ASPECT_RATIO
        self.steps = cfg.STEPS
        self.clip = cfg.CLIP
>>>>>>> Stashed changes
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.feature_maps = feature_maps


    def forward(self):
<<<<<<< Updated upstream
        """
        min_size和max_size会分别生成一个正方形的框，aspect_ratio参数对会生成2个长方形的框
        :return: 
        """
        mean = []
        for k, f in enumerate(self.feature_maps):
            # (i,j)为要提取特征层中每个网格的左上角坐标
            # for i, j in product(range(f), repeat=2):
            for i in range(f[0]):  # H切分
                for j in range(f[1]):  # W切分
                    # feature_maps的out_size，为啥不直接用f[0]与f[1]？
                    f_k_i = self.image_size[0] / self.steps[k]
                    f_k_j = self.image_size[1] / self.steps[k]
                    # 每个网格的中心点坐标（归一化）
                    cx = (j + 0.5) / f_k_j
                    cy = (i + 0.5) / f_k_i
                    # 1个正方形defult_box以min_size为基准生成
                    s_k_i = self.min_sizes[k] / self.image_size[1]
                    s_k_j = self.min_sizes[k] / self.image_size[0]
                    mean += [cy, cx, s_k_j, s_k_i]  # 所有先验default box的归一化坐标
                    # 1个正方形defult_box以min_size与max_size的关系生成
                    s_k_prime_i = sqrt(s_k_i * (self.max_sizes[k] / self.image_size[1]))
                    s_k_prime_j = sqrt(s_k_j * (self.max_sizes[k] / self.image_size[0]))
                    mean += [cy, cx, s_k_prime_j, s_k_prime_i]
                    # 然后，4个长方形defult_box根据aspect_ratio再生成（如aspect_ratio=2，那么会自动的再添加一个aspect_ratiod = 1/2）
                    for ar in self.aspect_ratios[k]:
                        mean += [cy, cx, s_k_prime_j * sqrt(ar), s_k_prime_i / sqrt(ar)]
                        mean += [cy, cx, s_k_j * sqrt(ar), s_k_i / sqrt(ar)]
        # 输出
=======
        mean = []
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k][0]
            featw = self.feature_maps[k][1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh
                for ar in self.aspect_ratio:
                    mean += [cy, cx, s_kh/math.sqrt(ar), s_kw*math.sqrt(ar)]

>>>>>>> Stashed changes
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)

        return output
