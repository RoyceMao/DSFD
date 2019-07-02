# -*- coding:utf-8 -*-
"""
   File Name：     my_config.py
   Description :   配置类
   Author :        royce.mao
   date：          2019/06/13
"""

import numpy as np


class Config(object):
    # 相关路径
    TRAIN_ROOT = '/home/dataset/face_recognize/face_detect/WIDER_train/images'
    VAL_ROOT = '/home/dataset/face_recognize/face_detect/WIDER_val/images'
    FACE_GEN_TRAIN_FILE = '/home/mh/face/DSFD/data/train.txt'
    FACE_GEN_VAL_FILE = '/home/mh/face/DSFD/data/val.txt'
    # 类别数与loss合并参数
    ALPHA = 1.0
    NUM_CLASSES = 2
    # prior_box参数
    MBOX = [1, 1, 1, 1, 1, 1]
    MIN_DIM = [300, 300]
    MIN_SIZES = [8, 16, 32, 64, 128, 256]
    MAX_SIZES = [16, 32, 64, 128, 256, 512]
    FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
    INPUT_SIZE = 640
    STEPS = [4, 8, 16, 32, 64, 128]
    ASPECT_RATIO = [[1.0,1.0],[2, 1/2],[3, 1/3],[1.0,1.0],[2, 1/2],[3, 1/3]]  # [[2, 1/2], [3, 1/3],...]的形式
    CLIP = False
    VARIANCE = [0.1, 0.1, 0.2, 0.2]
    # 数据增广参数
    APPLY_DISTORT = True
    APPLY_EXPAND = False
    ANCHOR_SAMPLING = True
    FILTER_MIN_FACE = True
    IMG_MEAN = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
    RESIZE_WIDTH = 640
    RESIZE_HEIGHT = 640
    MIN_FACE_SIZE = 6.0
    DATA_ANCHOR_SAMPLING_PROB = 0.5
    EXPAND_PROB = 0.5
    EXPAND_MAX_RATIO = 4
    hue_prob = 0.5
    hue_delta = 18
    contrast_prob = 0.5
    contrast_delta = 0.5
    saturation_prob = 0.5
    saturation_delta = 0.5
    brightness_prob = 0.5
    brightness_delta = 0.125
    # 采样参数
    FACE_OVERLAP_THRESH = [0.35]
    NEG_POS_RATIOS = 3
    # detection阈值
    NMS_THRESH = 0.3
    NMS_TOP_K = 5000
    TOP_K = 750
    CONF_THRESH = 0.05
    # 训练参数
    EPOCHES = 1
    BATCH_SIZE = 11
    NUM_WORKERS = 4
    # 模型保存目录
    MODEL_DIR = './trained_weights'
    RESUME = './trained_weights/val_best_dsfd.pth'
    # 预测结果保存目录
    RESULTS = './out'

    # def __init__(self):
    #     super(Config, self).__init__()
    #     self.NUM_ANCHORS = len(self.ANCHOR_SCALES)
    #     # feature map的高度，宽度，厚度
    #     self.FEATURES_HEIGHT, self.FEATURES_WIDTH, self.FEATURES_DEPTH = np.array(self.CROP_SIZE) // self.STRIDE

cur_config = Config()
