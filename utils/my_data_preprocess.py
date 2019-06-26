# -*- coding:utf-8 -*-
"""
   File Name：     my_data_preprocess.py
   Description :   数据准备
   Author :        royce.mao
   date：          2019/06/13
"""
import os
import sys  
sys.path.append("../")

from config import cur_config as cfg

def parse_face_file(root, file):
    """
    解析'wider_face_train_bbx_gt.txt'与'wider_face_val_bbx_gt.txt'两个文件的标注信息
    :param root: ./WIDER_train/images or ./WIDER_val/images
    :param file: 上述两个.txt文件的绝对路径
    :return: 
    """
    face_path = []
    face_count = []
    face_loc = []
    face_loc_flatten = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip().strip('\n')
        # 人头图像
        if 'jpg' in line:
            face_path += [os.path.join(root, line)]
        # 人头计数
        elif len(line.split(' ')) == 1:
            face_count += [int(line)]
        # 所有人头位置坐标
        else:
            line = line.split(' ')
            face_loc_flatten += [[int(line[0]), int(line[1]), int(line[2]), int(line[3])]]
    # flatten后的face_loc根据图片做对应
    total_face_num = 0
    for face_num in face_count:
        face_loc_ = []
        for i in range(total_face_num, total_face_num+face_num):
            face_loc_.append(face_loc_flatten[i])
        face_loc += [face_loc_]
        total_face_num += face_num

    return face_path, face_loc


def gen_face_file(face_path, face_loc, file):
    """
    生成Dataset对象需要的.txt文件信息
    :param face_path: [img_path1, img_path2,...]
    :param face_loc: [img_head_loc_list1, img_head_loc_list2, ...]
    :param file: 将生成的./data/train.txt或者./data/val.txt文件
    :return: 
    """
    with open(file, 'w') as f:
        for index in range(len(face_path)):
            path = face_path[index]
            boxes = face_loc[index]
            f.write(path)
            f.write(' {}'.format(len(boxes)))
            for box in boxes:
                data = ' {} {} {} {} {}'.format(box[0], box[1], box[2], box[3], 1)
                f.write(data)
            f.write('\n')
        f.close()


def main():
    train_root = cfg.TRAIN_ROOT
    val_root = cfg.VAL_ROOT
    train_file = cfg.FACE_TRAIN_FILE
    val_file = cfg.FACE_VAL_FILE
    gen_train_file = cfg.FACE_GEN_TRAIN_FILE
    gen_val_file = cfg.FACE_GEN_VAL_FILE
    # 训练集数据准备
    face_path, face_loc = parse_face_file(train_root, train_file)
    gen_face_file(face_path, face_loc, gen_train_file)
    # 评估测试集数据准备
    face_path, face_loc = parse_face_file(val_root, val_file)
    gen_face_file(face_path, face_loc, gen_val_file)


if __name__ == '__main__':
    main()

