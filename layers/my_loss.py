# -*- coding:utf-8 -*-
"""
   File Name：     my_loss.py
   Description :   损失函数（总损失L_all = (L_conf(x, cls) + αL_loc(x,loc,gts)) / N，其中N是困难负样本挖掘采样后保留的priors）
   Author :        royce.mao
   date：          2019/06/10
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from config import cur_config as cfg
from utils.my_torch_utils import target, log_sum_exp


class GeneralLoss(nn.Module):
    def __init__(self, cfg):
        super(GeneralLoss, self).__init__()
        self.num_classes = cfg.NUM_CLASSES
        self.threshold = cfg.FACE_OVERLAP_THRESH
        self.negpos_ratio = cfg.NEG_POS_RATIOS
        self.variance = cfg.VARIANCE

    def loss_compute(self, predictions, targets):
        """
        CrossEntropy Loss与SmoothL1 Loss的计算
        :param predictions: 预测值tuple(cls_preds, loc_preds, priorbox)
            cls shape: torch.size(batch_size,num_priors,num_classes)  # one-hot矩阵
            loc shape: torch.size(batch_size,num_priors,4)
            priorbox shape: torch.size(num_priors,4)
        :param targets: Ground truth boxes and labels for a batch,
            targets: 真实值[batch_size,num_gts,5]
        :return: 
        loss_cls与loss_loc
        """
        loc_preds, cls_preds, priorbox = predictions
        priors = priorbox[:loc_preds.size(1), :]
        priors = Variable(priors.cuda())
        batch = cls_preds.size(0)
        num_priors = priors.size(0)

        loc_batch = torch.Tensor(batch, num_priors, 4)
        cls_batch = torch.Tensor(batch, num_priors)

        for idx in range(batch):
            # 真实值
            gts = targets[idx][:, :-1][:, [1,0,3,2]].data  # (x1,y1,x2,y2)转(y1,x1,y2,x2)
            labels = targets[idx][:, -1].data  # .data
            priors = priors.data
            # 真实值结合predictions中的priorbox计算target
            batch_labels, batch_deltas, metrics = target(self.threshold, gts, priors, self.variance, labels)
            # print(batch_labels.data.sum())  # 打印每张图片上正样本anchors的数量
            # 一个单独batch的分类、回归目标赋值
            loc_batch[idx] = batch_deltas
            cls_batch[idx] = batch_labels
        # 两个目标均设置为gpu上的Variable对象
        loc_batch = Variable(loc_batch.cuda(), requires_grad=False)
        cls_batch = Variable(cls_batch.cuda(), requires_grad=False)

        # Smooth L1 loss（只取正样本做回归）
        pos = cls_batch > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        predict_deltas = loc_preds[pos_idx].view(-1, 4)
        deltas = loc_batch[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(predict_deltas, deltas, size_average=True)  # size_average=False不取minibatch的loss平均，增大梯度

        # 困难负样本挖掘
        ignore = cls_batch < 0
        cls_batch[ignore] = 0
        cls_preds_flatten = cls_preds.view(-1, self.num_classes)  # priorbox打平
        # 首先求取预测confidence的log_sum_exp值，再减去其中对应gt的confidence
        loss_c = log_sum_exp(cls_preds_flatten) - cls_preds_flatten.gather(1, cls_batch.view(-1, 1).long())
        # print(loss_c)

        # ignore与pos样本的模拟loss均置0，不影响负样本排序
        loss_c[pos.view(-1, 1)] = 0
        loss_c[ignore.view(-1, 1)] = 0
        # 负样本loss降序排序后，取前N个计算最终的loss_cls
        loss_c = loss_c.view(batch, -1)  # [batch, num_priors]
        _, loss_idx = loss_c.sort(1, descending=True)  # 单张图片（不是minibatch）里的priorbox按模拟loss降序排序
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Cross Entropy loss（所有正样本+挖掘的困难负样本做分类）
        pos_idx = pos.unsqueeze(2).expand_as(cls_preds)  # [batch, num_priors, num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(cls_preds)  # [batch, num_priors, num_classes]
        predict_logits_mining = cls_preds[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)  # [batch*num_priors, num_classes]
        labels_mining = cls_batch[(pos + neg).gt(0)]  # [batch*num_priors]
        loss_cls = F.cross_entropy(predict_logits_mining, labels_mining, size_average=True)  # size_average=False不取minibatch的loss平均，增大梯度

        # 返回值
        print('batch_size下的正样本数量：{}'.format(pos.data.sum()))
        print('batch_size下的负样本数量：{}'.format(neg.data.sum()))
        num_pos = num_pos.data.sum()
        loss = (loss_cls + cfg.ALPHA * loss_loc) / num_pos.float()  # 根据公式合并

        return loss, loss_loc, loss_cls


    def forward(self, predictions, targets):
        """
        常用Loss层的loss计算与反馈
        :param self: 
        :param predictions: 
        :param targets: 
        :return: 
        """
        loss, loss_loc, loss_cls = self.loss_compute(predictions, targets)

        return loss, loss_loc, loss_cls
