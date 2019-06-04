# -*- coding:utf-8 -*-
"""
   File Name：     my_detection.py
   Description :   检测流程，需要用到defult_box对结果采用nms和top_k阈值筛选
   Author :        royce.mao
   date：          2019/05/31
"""
import torch
from torch.autograd import Function
# todo
from ..config import current_cfg as cfg
from .my_torch_utils import decode, nms


class Detection(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
        apply non-maximum suppression to location predictions based on conf
        scores and threshold to a top_k number of output predictions for both
        confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        """
        
        :param num_classes: 分类数
        :param bkg_label: 
        :param top_k: 筛选top_k的priorbox数量
        :param conf_thresh: softmax概率阈值
        :param nms_thresh: nms保留的阈值
        """
        super(Detection, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.variance = cfg['variance']

    def forward(self, loc_pred, conf_pred, prior_box):
        """
        Args:
            loc_pred: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors,4]
            conf_pred: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_box: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        # batch_size与每个batch的prior_box数量
        num = loc_pred.size(0)  # batch size == 1
        num_priors = prior_box.size(1)  # 当前batch的priorbox数量
        # detect输出的维度
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_pred.view(num, num_priors, self.num_classes).transpose(2, 1)
        # Decode predictions into bboxes
        for i in range(num):
            default = prior_box
            # decode类似于regr apply
            loc_boxes = decode(loc_pred[i], default, self.variance)
            #
            cls_scores = conf_preds[i].clone()
            # 按cls类别过滤
            for cls in range(1, self.num_classes):
                # 指定类别对应的prior boxes，并用conf_thresh过滤
                cls_mask = cls_scores[cls].gt(self.conf_thresh)  # 逐元素判断元素与conf_thresh的大小关系，大于返回True，不大于返回False
                scores = cls_scores[cls][cls_mask]
                if scores.dim() == 0:
                    continue
                # 找到conf过滤后的对应prior boxes坐标
                loc_mask = cls_mask.unsqueeze(1).expand_as(loc_boxes)
                boxes = loc_boxes[loc_mask].view(-1, 4)
                # 同类别的nms过滤及保留topk
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)  # 注：nms返回的是count个prioir boxes的索引（count < top_k）
                output[i, cls, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), -1)
        # # 将各cls下的prior boxes打平
        # fit = output.contiguous().view(num, -1, 5)
        # # 同一batch下的prior boxes按照score降序排列
        # _, idx = fit[:, :, 0].sort(1, descending=True)
        # # # 同一batch下的prior boxes索引按照索引值从大到小排列
        # # _, rank = idx.sort(1)
        # fit[]
        # # 取小于topk的rank值， 获取最终的分类与回归目标
        # fit[(idx < self.top_k).unsqueeze(-1).expand_as(fit)].fill_(0)  # False全置为0

        return output
