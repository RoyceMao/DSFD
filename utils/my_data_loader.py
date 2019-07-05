# -*- coding:utf-8 -*-
"""
   File Name：     my_data_loader.py
   Description :   数据加载
   Author :        royce.mao
   date：          2019/06/13
"""
import torch
import random
import traceback
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.augmentation import data_aug
from utils.my_np_utils import clip_boxes

class WIDERFace(Dataset):
    def __init__(self, gen_file, mode='train'):
        """
        
        :param gen_file: 数据准备生成的list file
        :param mode: train or val
        """
        super(WIDERFace, self).__init__()
        self.mode = mode
        self.fnames = []
        self.gts = []
        self.labels = []
        # 按行读取，.split()之后保留的是[path, face_num, face_loc]的顺序
        with open(gen_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            face_num = int(line[1])
            gt = []
            label = []
            # 单张图片的人头，保留人头位置信息与人头标签信息
            for i in range(face_num):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                cls = int(line[6 + 5 * i])
                gt.append([x, y, x + w - 1, y + h - 1])
                label.append(cls)
            # generator汇总
            if len(gt) > 0:
                self.fnames.append(line[0])  # len = num_samples
                self.gts.append(gt)  # len = num_samples
                self.labels.append(label)  # len = num_samples
        # 训练样本的数量
        self.num_samples = len(self.fnames)

    def __getitem__(self, index):
        """
        训练与测试阶段的按索引数据加载
        :param index: 
        :return: 
        """
        while True:
            # 读取指定index的img
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')
            img_width, img_height = img.size
            # 指定index的img的face_loc的坐标归一化处理
            gt_box = np.array(self.gts[index])
            gt_box[:, 0] = gt_box[:, 0] / img_width
            gt_box[:, 1] = gt_box[:, 1] / img_height
            gt_box[:, 2] = gt_box[:, 2] / img_width
            gt_box[:, 3] = gt_box[:, 3] / img_height
            # 指定index的img的face_label提取
            gt_label = np.array(self.labels[index])
            # 拼接后转list [face_num, (cls,x1,y1,x2,y2)]
            gt_box_label = np.hstack((gt_label[:, np.newaxis], gt_box)).tolist()
            # 直接做数据增广
            try:
                img, sample_box_label = data_aug(img, gt_box_label, self.mode, image_path)
                # 数据增广后的[face_num, (cls,x1,y1,x2,y2)]标签转[face_num, (x1,y1,x2,y2,cls)]
                if len(sample_box_label) > 0:
                    target = clip_boxes(sample_box_label, 1)
                    # target = np.hstack((sample_box_label[:, 1:], sample_box_label[:, 0][:, np.newaxis]))
                    assert (target[:, 2] > target[:, 0]).any()
                    assert (target[:, 3] > target[:, 1]).any()
                    break  # 只有提取到有人头目标的图片（img，target）时，才加载当作训练样本。否则，一直随机加载
                else:
                    index = random.randrange(0, self.num_samples)
            except Exception as e:
                # traceback.print_exc()
                index = random.randrange(0, self.num_samples)
                continue

        # print(target)
        return torch.from_numpy(img), target

    def __len__(self):
        return self.num_samples


def face_collate(batch):
    """
    一个batch里数据的取样方式
    :param batch: 
    :return: 
    
    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    return torch.stack(imgs, 0), targets
