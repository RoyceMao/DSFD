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
# todo
from ..torch_utils import decode, nms, center_size


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
        self.variance = cfg['variance']  #

    def forward(self, loc_pred, conf_pred, prior_box):
        """
        Args:
            loc_pred: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_pred: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_box: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        #
        num = loc_pred.size(0)  # batch size == 1
        num_priors = prior_box.size(1)  # 当前batch的priorbox数量
        #
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_pred.view(num, num_priors, self.num_classes).transpose(2, 1)
        # Decode predictions into bboxes
        for i in range(num):
            default = prior_box
            # decode类似于regr apply
            decoded_boxes = decode(loc_pred[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()








