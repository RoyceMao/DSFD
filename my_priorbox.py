# -*- coding:utf-8 -*-
"""
   File Name：     my_priorbox.py
   Description :   在给定一系列feature maps上面生成priorbox
   Author :        royce.mao
   date：          2019/05/31
"""
import torch
from itertools import product
from math import sqrt


class PriorBox(object):
    def __init__(self, cfg):
        """
        PriorBox需要返回得是所有default box得四个参数归一化后的值，即得到所有先验框的位置。
        :param cfg: 
        """
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']  # 如[300, 300]尺寸
        self.variance = cfg['variance'] or [0.1]  #
        self.feature_maps = cfg['feature_maps']  # [[H,W],[H,W],...]的形式
        self.min_sizes = cfg['min_sizes']  # len(min_sizes) == len(max_sizes)
        self.max_sizes = cfg['max_sizes']  # len(min_sizes) == len(max_sizes)
        self.steps = cfg['steps']  # image_size / feature_map_size下采样倍数
        self.aspect_ratios = cfg['aspect_ratios']  # [[2, 1/2], [3, 1/3],...]的形式
        self.clip = cfg['clip']  # 归一化坐标超出边界范围的修剪
        # self.num_priors = len(cfg['aspect_ratios'])  #
        # self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        """
        min_size和max_size会分别生成一个正方形的框，aspect_ratio参数会生成2个长方形的框
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
                    mean += [cx, cy, s_k_i, s_k_j]  # 所有先验default box的归一化坐标
                    # 1个正方形defult_box以min_size与max_size的关系生成
                    s_k_prime_i = sqrt(s_k_i * (self.max_sizes[k] / self.image_size[1]))
                    s_k_prime_j = sqrt(s_k_j * (self.max_sizes[k] / self.image_size[0]))
                    mean += [cx, cy, s_k_prime_i, s_k_prime_j]
                    # 然后，4个长方形defult_box根据aspect_ratio再生成（如aspect_ratio=2，那么会自动的再添加一个aspect_ratiod = 1/2）
                    for ar_pair in self.aspect_ratios[k]:
                        for ar in ar_pair:
                            mean += [cx, cy, s_k_prime_i / sqrt(ar), s_k_prime_j * sqrt(ar)]
                            mean += [cx, cy, s_k_i / sqrt(ar), s_k_j * sqrt(ar)]
        # 输出
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            # 归一化，将output限制在[0, 1]之间
            output = torch.clamp(output, min=0, max=1)

        return output
