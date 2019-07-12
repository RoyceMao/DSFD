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
from torch.optim.lr_scheduler import StepLR  # 学习率衰减
from tensorboardX import SummaryWriter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import cur_config as cfg
from layers.my_loss import GeneralLoss, FocalLoss
from models.my_dual_net import DualShot
from models.dual_net_resnet import build_net_resnet
from models.dual_net_vgg import build_net_vgg
from models.dual_net_ssd import build_net_ssd
from utils.my_data_loader import WIDERFace, face_collate
from tmp_for_adjust.weights_bias_log import weights_bias_parm, parm_to_excel  # 打印weights、bias情况的脚本

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(1)

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
    # net = DualShot(phase, cfg, cfg.NUM_CLASSES)  # 源码重构网络
    # net = build_net_resnet(phase, cfg.NUM_CLASSES, 'resnet50')  # 源resnet网络
    # net = build_net_vgg(phase, cfg.NUM_CLASSES)  # 源vgg网络
    net = build_net_ssd(phase, cfg, cfg.NUM_CLASSES) # 源腾讯优图类ssd网络
    # if args.multigpu:
    #     net = torch.nn.DataParallel(net)  # 训练过程中多GPU的使用
    net.cuda()

    # 模型保存路径
    if not os.path.exists(cfg.MODEL_DIR):
        os.mkdir(cfg.MODEL_DIR)
    save_path = os.path.join(cfg.MODEL_DIR, 'model.{:03d}.pth')  # 保存pth格式权重

    # 加载预训练（并直接从指定start_epoch开始继续训练）
    if args.mode == 'resume':
        print('[INFO] Load model from {}...'.format(args.resume))
        # torch_utils.load_net(args.resume, net)
        try:
            net.load_weights(args.resume)
        except Exception as e:
            net.load_state_dict(torch.load(args.resume))

    else:
        print('[INFO] Initializing weights...')
        # net.apply(net.weights_init)

    # 优化器对象、学习率衰减对象、损失函数对象
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
    # criterion = GeneralLoss(cfg)
    criterion = FocalLoss(cfg)

    # 存放训练log的writer对象
    writer = SummaryWriter('./log')

    # 训练
    print("======开始训练======")

    net.train()
    stop_save = 0
    global min_loss
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        metrics_list = []  # 打印前向传播待统计的logs
        output_list = []  # 打印前向传播网络的输出
        for iteration, data in enumerate(train_loader):
            # images与targets放GPU上
            images, targets = data 
            # print(images.numpy())
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]

            # images前向传播
            start_time = time.time()
            # print(images.shape)
            out = net(images)

            # 清零梯度
            optimizer.zero_grad()

            # loss反向传播
            loss = criterion(out, targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm(net.parameters(), True)
            optimizer.step()

            # 新增metrics输出
            # metrics_list.append(out["metrics"])
            # output_list.append(out)

            # 单个epoch里每个batch的loss不断加总
            losses += loss.data.item()  # 注：不能直接losses += loss，会随着epoches循环爆gpu的显存

            # 每训练10个batches后打印日志
            if iteration % 10 == 0:
                # 计算每个batch的平均loss
                mean_loss = losses / (iteration + 1)

                # 添加loss日志（用于tensorboard可视化）
                writer.add_scalar('Loss', mean_loss, global_step=iteration)

                print('Timer: %.4f' % (time.time() - start_time))
                print('Epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (mean_loss))
                # print('->> cls loss:s1_{:.4f},s2_{:.4f} || loc loss:s1_{:.4f},s2_{:.4f}'.format(
                      # loss_cls_s1.data[0], loss_cls_s2.data[0], loss_loc_s1.data[0], loss_loc_s2.data[0]))
                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))
                # 保存出现nan之前的最优loss权重，并保存(只保存1个)
                if mean_loss < min_loss:
                    min_loss = mean_loss
                    print("min_loss:{}".format(min_loss))
                    torch.save(net.state_dict(), save_path.format(11111))
                
                # 权重、偏置、特征信息
                # wb_parm, names = weights_bias_parm(net)
                # for key in names:
                    # if np.any(np.isnan(wb_parm[key].cpu().detach().numpy())):  # 某weights层的参数出现NAN
                        # print(key)
                    # parm_to_excel(cfg.EXCEL_PATH.format(key), key, wb_parm)
                # val(epoch, net, criterion, val_loader)

        scheduler.step()
        # 每个epoch打印一次metrics
        metric_print(metrics_list)
        # 每个epoch保存save一次模型的权重/偏置
        if epoch % 1 == 0:
            # torch_utils.save_net(save_path.format(epoch), net)
            torch.save(net.state_dict(), save_path.format(epoch))

        # 每个epoch评估测试一次（并保存最佳的权重/偏置参数信息）
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
        loss_val = criterion(out, targets) # , loss_loc_s1_val, loss_cls_s1_val, loss_loc_s2_val, loss_cls_s2_val
        losses_total_val += loss_val.data[0]
        step += 1
    # 每次评估只打印一次日志
    mean_loss_val = losses_total_val / step
    print('Timer: %.4f' % (time.time() - start_time))
    print('Test Epoch:' + repr(epoch) + ' || Loss:%.4f' % (mean_loss_val))
    # 每个epoch根据目前的val_loss与以往的val_loss相比，有提升的话才保存模型
    global min_loss  # 设为全局变量
    if mean_loss_val.cpu().numpy() < min_loss:
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
    # python train.py resume or python train.py scratch
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Arguments for specific resume.', dest='mode')
    subparsers.required = True

    resume_parser = subparsers.add_parser('resume')
    resume_parser.add_argument("--resume", help="weights_path", default=cfg.RESUME)

    scratch_parser = subparsers.add_parser('scratch')
    scratch_parser.add_argument("--scratch", help="from scratch", default=None)

    argments = parser.parse_args(sys.argv[1:])  # 新增的resume参数，指定net需要加载的权重参数
    # 用来每个epoch走完之后，val_loss的比较以保存最佳model
    min_loss = 10
    train(argments)