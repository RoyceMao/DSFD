#-*- coding:utf-8 -*-
"""
   File Name：     demo.py
   Description :   人脸检测预测Demo
   Author :        royce.mao
   date：          2019/07/02
"""

import os
import torch
import cv2
import time
import numpy as np
from PIL import Image

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from config import cur_config as cfg
from models.my_dual_net import DualShot
from models.dual_net_resnet import build_net_resnet
from models.dual_net_vgg import build_net_vgg
from models.dual_net_ssd import build_net_ssd
from utils.augmentation import to_chw_bgr  # channel first and RGB2GBR

torch.cuda.set_device(1)

# 人脸检测的Demo函数
def detect(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape

    # 按比例因子做缩放
    max_im_shrink = np.sqrt(
        1500 * 1000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    # 输入图片预处理（类似训练过程的data_aug增广）
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.IMG_MEAN
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))  # 增加batch维
    x = x.cuda()

    # 前向传播做预测
    t1 = time.time()
    y = net(x)

    detections = y.data  # shape为(batch, self.num_classes, self.top_k, 5)
    scale = torch.Tensor([img.shape[0], img.shape[1],
                          img.shape[0], img.shape[1]])  # 顺序为(y1,x1,y2,x2)

    # 用于显示save保存的原始图片
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # 根据softmax归一化的结果，逐class判断
    for i in range(detections.size(1)):
        j = 0
        while j < cfg.TOP_K and detections[0, i, j, 0] >= thresh:
            # 可视化
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            left_up, right_bottom = (pt[1], pt[0]), (pt[3], pt[2])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.2f}".format(score)
            text_size, baseline = cv2.getTextSize(
                conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                          (p1[0] + text_size[0], p1[1] + text_size[1]),[255,0,0], -1)
            cv2.putText(img, conf, (p1[0], p1[
                            1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))

    # 保存
    cv2.imwrite(os.path.join(cfg.RESULTS, os.path.basename(img_path)), img)


def main():
    #
    print("====初始化网络=====")
    # net = DualShot('test', cfg, cfg.NUM_CLASSES)
    # net = build_net_resnet('test', cfg.NUM_CLASSES, 'resnet50')
    # net = build_net_vgg('test', cfg.NUM_CLASSES)
    net = build_net_ssd('test', cfg, cfg.NUM_CLASSES)
    net.load_weights(cfg.RESUME)
    net.eval()
    net.cuda()
    cudnn.benckmark = True
    #
    print("====开始预测=====")
    img_list = [os.path.join(cfg.IMG_PATH, x)
                for x in os.listdir(cfg.IMG_PATH) if x.endswith('jpg')]
    
    # 全局设置不进行梯度更新，避免爆GPU显存
    with torch.no_grad():
        for path in img_list:
            try:
                detect(net, path, cfg.THRESHOLD)
            except Exception as e:
                continue


if __name__ == '__main__':
    main()
