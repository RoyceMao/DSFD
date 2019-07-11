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
            defaults = priors.data
            # 真实值结合predictions中的priorbox计算target
            batch_labels, batch_deltas, metrics = target(self.threshold, gts, defaults, self.variance, labels)

            # 一个单独batch的分类、回归目标赋值
            loc_batch[idx] = batch_deltas  # [batch_size, 34125, 4]
            cls_batch[idx] = batch_labels  # [batch_size, 34125]
        # 两个目标均设置为gpu上的Variable对象
        loc_batch = Variable(loc_batch.cuda(), requires_grad=False)
        cls_batch = Variable(cls_batch.cuda(), requires_grad=False)

        # Smooth L1 loss（只取正样本做回归）
        pos = cls_batch > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        predict_deltas = loc_preds[pos_idx].view(-1, 4)
        deltas = loc_batch[pos_idx].view(-1, 4)
        # print(deltas)
        loss_loc = F.smooth_l1_loss(predict_deltas, deltas, size_average=False)  # size_average=False不取minibatch的loss平均，增大梯度

        # 困难负样本挖掘
        ignore = cls_batch < 0
        cls_batch[ignore] = 0
        cls_preds_flatten = cls_preds.view(-1, self.num_classes)  # priorbox打平
        # 首先求取预测confidence的log_sum_exp值，再减去其中对应gt的confidence
        loss_c = log_sum_exp(cls_preds_flatten) - cls_preds_flatten.gather(1, cls_batch.view(-1, 1).long())

        # ignore与pos样本的模拟loss均置0，不影响负样本排序
        loss_c[pos.view(-1, 1)] = 0
        loss_c[ignore.view(-1, 1)] = 0
        # 负样本loss降序排序后，取前N个计算最终的loss_cls
        loss_c = loss_c.view(batch, -1)  # [batch, num_priors]
        
        # print(torch.max(loss_c))  # 有nan出现
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
        loss_cls = F.cross_entropy(predict_logits_mining, labels_mining, size_average=False)  # size_average=False不取minibatch的loss平均，增大梯度

        # 返回值
        # print('batch_size下的正样本数量：{}'.format(pos.data.sum()))
        # print('batch_size下的负样本数量：{}'.format(neg.data.sum()))
        N = num_pos.data.sum() if num_pos.data.sum() > 0 else batch
        # loss = (loss_cls + cfg.ALPHA * loss_loc) / N.float()  # 根据公式合并
        loss_loc = loss_loc / N.float()
        loss_cls = loss_cls / N.float()

        return loss_loc, loss_cls


    def forward(self, predictions, targets):
        """
        常用Loss层的loss计算与反馈
        :param self: 
        :param predictions: 
        :param targets: 
        :return: 
        """
        if len(predictions) == 6:  # 源网络返回情况（Dual_shot）
            loss_loc_s1, loss_cls_s1 = self.loss_compute(predictions[:3], targets)
            loss_loc_s2, loss_cls_s2 = self.loss_compute(predictions[3:], targets)
            loss = loss_loc_s1 + loss_cls_s1 + loss_loc_s2 + loss_cls_s2
        else:  # 重构网络返回情况（Single_shot）
            loss_loc, loss_cls = self.loss_compute(predictions, targets)
            loss = loss_loc + loss_cls

        return loss  # , loss_loc_s1, loss_cls_s1, loss_loc_s2, loss_cls_s2


class FocalLoss(nn.Module):
    def __init__(self, cfg):
        """
            focusing is parameter that can adjust the rate at which easy
            examples are down-weighted.
            alpha may be set by inverse class frequency or treated as a hyper-param
            If you don't want to balance factor, set alpha to 1
            If you don't want to focusing factor, set gamma to 1 
            which is same as normal cross entropy loss
        """
        super(FocalLoss, self).__init__()
        self.num_classes = cfg.NUM_CLASSES
        self.threshold = cfg.FACE_OVERLAP_THRESH
        self.variance = cfg.VARIANCE
        self.alpha = cfg.ALPHA
        self.gamma = cfg.GAMMA


    def loss_compute(self, predictions, targets):
        """
        更换loss函数为focalloss
        :param predictions: 
        :param targets: 
        :return: 
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
            gts = targets[idx][:, :-1][:, [1, 0, 3, 2]].data  # (x1,y1,x2,y2)转(y1,x1,y2,x2)
            labels = targets[idx][:, -1].data  # .data
            defaults = priors.data
            # 真实值结合predictions中的priorbox计算target
            batch_labels, batch_deltas, metrics = target(self.threshold, gts, defaults, self.variance, labels)

            # 一个单独batch的分类、回归目标赋值
            loc_batch[idx] = batch_deltas  # [batch_size, 34125, 4]
            cls_batch[idx] = batch_labels  # [batch_size, 34125]

        # 两个目标均设置为gpu上的Variable对象
        loc_batch = Variable(loc_batch.cuda(), requires_grad=False)
        cls_batch = Variable(cls_batch.cuda(), requires_grad=False)

        # Smooth L1 loss（只取正样本做回归）
        pos = cls_batch > 0
        num_pos = pos.long().sum(1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        predict_deltas = loc_preds[pos_idx].view(-1, 4)
        deltas = loc_batch[pos_idx].view(-1, 4)
        # print(deltas)
        loss_loc = F.smooth_l1_loss(predict_deltas, deltas,
                                    size_average=False)  # size_average=False不取minibatch的loss平均，增大梯度

        # focal loss（取正、负样本做分类）
        pos_neg_cls = cls_batch > -1
        mask = pos_neg_cls.unsqueeze(2).expand_as(cls_preds)
        cls_preds_pure = cls_preds[mask].view(-1, cls_preds.size(2)).clone()
        p_t_log = - F.cross_entropy(cls_preds_pure, cls_batch[pos_neg_cls], size_average=False)
        p_t = torch.exp(p_t_log)
        # focal loss的计算
        loss_cls = -self.alpha * ((1 - p_t) ** self.gamma * p_t_log)

        N = max(1,
                num_pos.data.sum())
        loss_loc = loss_loc / N.float()
        loss_cls = loss_cls / N.float()

        return loss_loc, loss_cls

    def forward(self, predictions, targets):
        """
        
        :param predictions: 
        :param targets: 
        :return: 
        """
        if len(predictions) == 6:  # 源网络返回情况（Dual_shot）
            loss_loc_s1, loss_cls_s1 = self.loss_compute(predictions[:3], targets)
            loss_loc_s2, loss_cls_s2 = self.loss_compute(predictions[3:], targets)
            loss = loss_loc_s1 + loss_cls_s1 + loss_loc_s2 + loss_cls_s2
        else:  # 重构网络返回情况（Single_shot）
            loss_loc, loss_cls = self.loss_compute(predictions, targets)
            loss = loss_loc + loss_cls

        return loss
