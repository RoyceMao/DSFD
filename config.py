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
    FACE_GEN_TRAIN_FILE = './data/train.txt'
    FACE_GEN_VAL_FILE = './data/val.txt'
    # 类别数与loss合并参数
    ALPHA = 1.0
    NUM_CLASSES = 2
    # prior_box参数
    MBOX = [1, 1, 1, 1, 1, 1]
    # BASE_SIZES = [8, 16, 32, 64, 128, 256]
    BASE_SIZES = [16, 32, 64, 128, 256, 512]
    FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
    INPUT_SIZE = [640, 640]
    STEPS = [4, 8, 16, 32, 64, 128]
    ASPECT_RATIO = [1.0]  # [1.0, 2.0,...]的形式
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
    # Generalloss 挖掘采样参数
    FACE_OVERLAP_THRESH = [0.35]  # 0.35
    NEG_POS_RATIOS = 3
    # Focalloss 挖掘采样参数
    ALPHA = 0.25
    GAMMA = 2
    # Detection阈值
    NMS_THRESH = 0.3
    NMS_TOP_K = 5000
    TOP_K = 750
    CONF_THRESH = 0.05
    # Final confidence threshold
    THRESHOLD = 0.4  # 0.4
    # 训练参数
    EPOCHES = 5
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    # 模型保存目录
    MODEL_DIR = './trained_weights'
    RESUME = './trained_weights/model.11111.pth'
    # 需要预测的图片数据地址
    IMG_PATH = './data/img'
    # 预测结果保存目录
    RESULTS = './out'
    
    # 精调的日志目录
    EXCEL_PATH = './tmp_for_adjust/{}.xlsx'
    KEY_NAME = 'loc_pal1.0.weight'  # 需要查看训练参数信息的层key值（直接在这里修改）

cur_config = Config()
