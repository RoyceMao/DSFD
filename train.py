# -*- coding:utf-8 -*-
"""
   File Name：     train.py
   Description :   训练脚本
   Author :        royce.mao
   date：          2019/06/12
"""

import os
import sys
import time
import torch
import argparse
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from config import cur_config as cfg
from layers.my_loss import GeneralLoss
from models.my_dual_net import DualShot
from utils.my_data_loader import WIDERFace, face_collate


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(args):
    """
    训练主函数
    :param args: 
    :return: 
    """
    # 参数
    phase = 'train'
    start_epoch = 0
    # 加载数据
    train_dataset = WIDERFace(cfg.FACE_TRAIN_FILE, mode='train')
    val_dataset = WIDERFace(cfg.FACE_VAL_FILE, mode='val')

    train_loader = DataLoader(train_dataset, cfg.BATCH_SIZE,
                                   num_workers=cfg.NUM_WORKERS,
                                   shuffle=True,
                                   collate_fn=face_collate,
                                   pin_memory=True)
    val_loader = DataLoader(val_dataset, cfg.BATCH_SIZE // 2,
                                 num_workers=cfg.NUM_WORKERS,
                                 shuffle=False,
                                 collate_fn=face_collate,
                                 pin_memory=True)
    min_loss = np.inf
    per_epoch_size = len(train_dataset) // cfg.BATCH_SIZE  # 计算下每个epoch的steps

    # 初始化网络
    print("====初始化网络=====")
    net = DualShot(phase, cfg, cfg.NUM_CLASSES)
    # if args.multigpu:
    #     net = torch.nn.DataParallel(net)  # 训练过程中多GPU的使用
    net.cuda()

    # 模型保存路径
    if not os.path.exists(cfg.MODEL_DIR):
        os.mkdir(cfg.MODEL_DIR)
    save_path = os.path.join(cfg.MODEL_DIR, 'model.{:03d}.pth')  # 保存pth格式权重

    # 加载预训练（并直接从指定start_epoch开始继续训练）
    if args.resume:
        print('[INFO] Load model from {}...'.format(args.resume))
        # torch_utils.load_net(args.resume, net)
        start_epoch = net.load_weights(args.resume)
    else:
        print('[INFO] No Pretraining weights...')

    # 优化器对象与损失函数对象
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    criterion = GeneralLoss(cfg)

    # 训练
    print("======开始训练======")

    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses_total = 0
        metrics_list = []  # 打印前向传播待统计的logs
        output_list = []  # 打印前向传播网络的输出
        for iteration, data in enumerate(train_loader):
            # images与targets放GPU上
            images, targets = data
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
            # 清零梯度
            optimizer.zero_grad()
            # images前向传播 + loss反向传播 + 梯度参数更新
            start_time = time.time()
            out = net(images)
            loss_loc, loss_cls, loss_total = criterion(out, targets)
            loss_loc.backward()
            loss_cls.backward()
            loss_total.backward()
            optimizer.step()
            metrics_list.append(out["metrics"])  # 新增输出
            output_list.append(out)
            # 单个epoch里每个batch的loss是不断加总的
            losses_total += loss_total.data[0]
            # 每训练10个batches后打印日志
            if iteration % 10 == 0:
                # 计算平均每个batch的loss
                total_loss = losses_total / (iteration + 1)
                print('Timer: %.4f' % (time.time() - start_time))
                print('Epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (total_loss))
                print('->> conf loss:{:.4f} || loc loss:{:.4f}'.format(
                    loss_cls.data[0], loss_loc.data[0]))
                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))
        # 每个epoch打印一次metrics
        metric_print(metrics_list)
        # 每个epoch保存一次模型
        if epoch % 1 == 0:
            # torch_utils.save_net(save_path.format(epoch), net)
            torch.save(net.state_dict(), save_path.format(epoch))

        # 重要！！！每个epoch评估测试一次
        val(epoch, net, dsfd_net, criterion, val_loader)

    print('[INFO] Finished Training')


def val(epoch, net, dsfd_net, criterion, val_loader):
    """
    
    :param epoch: 
    :param net: 
    :param dsfd_net: 
    :param criterion: 
    :return: 
    """
    net.eval()



def metric_print(metrics_list):
    """
    打印度量
    :param metrics_list:
    :return:
    """
    if len(metrics_list) == 0:
        return
        # 初始化
    metrics_sum = metrics_list[0]
    for metric_name in metrics_sum.keys():
        metrics_sum[metric_name] = 0.
    # 累加
    for metrics in metrics_list:
        for metric_name in metrics_sum.keys():
            metrics_sum[metric_name] = metrics_sum[metric_name] + metrics[metric_name]

    line = ",".join(["{}:{:.2f}".format(metric_name, metrics_sum[metric_name] / len(metrics_list))
                     for metric_name in metrics_sum.keys()])
    print(line)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--resume", type=str, default=None, help="weights_path")
    argments = parse.parse_args(sys.argv[1:])
    train(argments)