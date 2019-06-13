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
    FACE_TRAIN_FILE = '/home/dataset/face_recognize/face_detect/wider_face_split/wider_face_train_bbx_gt.txt'
    FACE_VAL_FILE = '/home/dataset/face_recognize/face_detect/wider_face_split/wider_face_val_bbx_gt.txt'
    FACE_GEN_TRAIN_FILE =
    FACE_GEN_val_FILE =
    # 类别数与loss合并参数
    ALPHA = 1
    NUM_CLASSES = 2
    # prior_box参数
    MIN_DIM = [300, 300]
    MIN_SIZES = [8, 16, 32, 64, 128, 256]
    MAX_SIZES = [16, 32, 64, 128, 256, 512]
    FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
    INPUT_SIZE = 640
    STEPS = [4, 8, 16, 32, 64, 128]
    ASPECT_RATIO = [1.0]  # [[2, 1/2], [3, 1/3],...]的形式
    CLIP = False
    VARIANCE = [0.1, 0.1, 0.2, 0.2]
    # 采样参数
    FACE_OVERLAP_THRESH = 0.35
    NEG_POS_RATIOS = 3
    # detection阈值
    NMS_THRESH = 0.3
    NMS_TOP_K = 5000
    TOP_K = 750
    CONF_THRESH = 0.05
    # 训练参数
    EPOCHS = 50
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    # 模型保存目录
    MODEL_DIR = './trained_weights'
    # 预测结果保存目录
    RESULTS = './out'

    # def __init__(self):
    #     super(Config, self).__init__()
    #     self.NUM_ANCHORS = len(self.ANCHOR_SCALES)
    #     # feature map的高度，宽度，厚度
    #     self.FEATURES_HEIGHT, self.FEATURES_WIDTH, self.FEATURES_DEPTH = np.array(self.CROP_SIZE) // self.STRIDE

cur_config = Config()
