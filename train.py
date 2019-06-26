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
import warnings
warnings.filterwarnings("ignore")

from config import cur_config as cfg
from layers.my_loss import GeneralLoss
from models.my_dual_net import DualShot
from utils.my_data_loader import WIDERFace, face_collate

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)

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
    train_dataset = WIDERFace(cfg.FACE_GEN_TRAIN_FILE, mode='train')
    val_dataset = WIDERFace(cfg.FACE_GEN_VAL_FILE, mode='val')

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
    # per_epoch_size = len(train_dataset) // cfg.BATCH_SIZE  # 计算下每个epoch的steps

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
        losses = 0
        metrics_list = []  # 打印前向传播待统计的logs
        output_list = []  # 打印前向传播网络的输出
        for iteration, data in enumerate(train_loader):
            # images与targets放GPU上
            images, targets = data
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]

            # images前向传播
            start_time = time.time()
            out = net(images)
            # 清零梯度
            optimizer.zero_grad()
            # loss反向传播
            loss_loc, loss_cls, num_pos = criterion(out, targets)
            loss = (loss_cls + cfg.ALPHA * loss_loc) / num_pos.float()  # 根据公式合并
            loss.backward()
            optimizer.step()

            # metrics_list.append(out["metrics"])  # 新增输出
            output_list.append(out)
            # 单个epoch里每个batch的loss是不断加总的
            losses += loss.data[0]
            # 每训练10个batches后打印日志
            if iteration % 10 == 0:
                # 计算平均每个batch的loss
                mean_loss = losses / (iteration + 1)
                print('Timer: %.4f' % (time.time() - start_time))
                print('Epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (mean_loss))
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
        val(epoch, net, criterion, val_loader)

    print('[INFO] Finished Training')


def val(epoch, net, criterion, val_loader):
    """
    
    :param epoch: 
    :param net: 
    :param dsfd_net: 
    :param criterion: 
    :return: 
    """
    net.eval()
    step = 0
    losses_total_val = 0
    start_time = time.time()
    for iteration, data in enumerate(val_loader):
        # images与targets放GPU上
        images, targets = data
        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True)
                   for ann in targets]
        # 模型前向传播进行预测，并计算val_loss（注：这里的val_loss并不做反向传播，也不需要optimizer的更新）
        out = net(images)
        loss_loc_val, loss_cls_val, loss_total_val = criterion(out, targets)
        losses_total_val += loss_total_val.data[0]
        step += 1
    # 每次评估只打印一次日志
    mean_loss_val = losses_total_val / step
    print('Timer: %.4f' % (time.time() - start_time))
    print('Test Epoch:' + repr(epoch) + ' || Loss:%.4f' % (mean_loss_val))
    # 每个epoch根据目前的val_loss与以往的val_loss相比，有提升的话才保存模型
    global min_loss  # 设为全局变量
    if mean_loss_val < min_loss:
        print('Saving best state,epoch', epoch)
        torch.save(net.state_dict(), os.path.join(
            cfg.MODEL_DIR, 'val_best_dsfd.pth'))
        min_loss = mean_loss_val

    # states = {
    #     'epoch': epoch,
    #     'weight': net.state_dict(),
    # }
    # torch.save(states, os.path.join(cfg.MODEL_DIR, 'val_best_dsfd_checkpoint.pth'))


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
    # python train.py './trained_weights/val_best_dsfd.pth'
    parse = argparse.ArgumentParser()
    parse.add_argument("--resume", type=str, default=None, help="weights_path")
    argments = parse.parse_args(sys.argv[1:])  # 新增的resume参数，指定net需要加载的权重参数
    # 用来每个epoch走完之后，val_loss的比较以保存最佳model
    min_loss = np.inf
    train(argments)