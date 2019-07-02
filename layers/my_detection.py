# -*- coding:utf-8 -*-
"""
   File Name：     my_detection.py
   Description :   检测流程，需要用到defult_box对结果采用nms和top_k阈值筛选
   Author :        royce.mao
   date：          2019/05/31
"""
import torch
from torch.autograd import Function
from config import cur_config as cfg
from utils.my_torch_utils import decode, nms


class Detection(Function):
    """
    根据1次前向传播预测的cls与delta，Decode应用边框回归，并以指定的conf经过NMS保留最终的结果
    """
    def __init__(self, cfg):
        """
        
        :param cfg: 
        """
        super(Detection, self).__init__()
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = 0
        self.top_k = cfg.TOP_K
        self.nms_top_k = cfg.NMS_TOP_K
        self.conf_thresh = cfg.CONF_THRESH
        self.nms_thresh = cfg.NMS_THRESH
        if self.nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.variance = cfg.VARIANCE

    def forward(self, loc_pred, cls_pred, prior_box):
        """
        Args:
            loc_pred: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors,4]
            cls_pred: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_box: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
        """
        # batch_size与每个batch的prior_box数量
        batch = loc_pred.size(0)  # batch size == 1
        num_priors = prior_box.size(0)

        # 初始化output维度
        # todo:
        cls_preds = cls_pred.view(batch, num_priors, self.num_classes).transpose(2, 1)
        output = torch.zeros(batch, self.num_classes, self.top_k, 5)
        
        # 应用边框回归
        batch_priors = prior_box.view(-1, num_priors,
                                       4).expand(batch, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_pred.view(-1, 4),
                               batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(batch, num_priors, 4)

        #
        for i in range(batch):
            loc_boxes = decoded_boxes[i].clone()
            cls_scores = cls_preds[i].clone()
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
                # 同类别框的nms过滤，并保留topk
                ids, count = nms(
                    boxes, scores, self.nms_thresh, self.nms_top_k)  # 注：nms返回的是count个prioir boxes的索引（count < top_k）
                count = count if count < self.top_k else self.top_k
                
                # 最终输出
                output[i, cls, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), -1)

        # # 将各cls下的prior boxes打平
        # fit = output.contiguous().view(batch, -1, 5)
        # # 同一batch下的prior boxes按照score降序排列
        # _, idx = fit[:, :, 0].sort(1, descending=True)
        # # # 同一batch下的prior boxes索引按照索引值从大到小排列
        # # _, rank = idx.sort(1)
        # fit[]
        # # 取小于topk的rank值， 获取最终的分类与回归目标
        # fit[(idx < self.top_k).unsqueeze(-1).expand_as(fit)].fill_(0)  # False全置为0

        return output
