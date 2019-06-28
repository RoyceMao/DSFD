# -*- coding:utf-8 -*-
"""
   File Name：     my_torch_utils.py
   Description :   函数工具类
   Author :        royce.mao
   date：          2019/06/04
"""
import torch
import h5py
import numpy as np


def center_size(boxes):
    """
    Convert (y1, x1, y2, x2)模式的priors to 中心点加边长的模式
    :param boxes: (tensor) [N, (y1, x1, y2, x2)]
    :return: 
    (tensor) [N, (cy, cx, h, w)]
    """
    return torch.cat([boxes[:, 2:] * 0.5 + boxes[:, :2] * 0.5,  # cy, cx
                      boxes[:, 2:] - boxes[:, :2]], dim=1)  # h, w


def point_bound(boxes):
    """
    Convert (cy, cx, h, w)模式的priors to 左上顶点+右下顶点的模式
    :param boxes: (tensor) [N, (cy, cx, h, w)]
    :return:
    (tensor) [N, (y1, x1, y2, x2)]
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # ymin, xmin
                      boxes[:, :2] + boxes[:, 2:] / 2), dim=1) # ymax, xmax


def iou(boxes_a, boxes_b):
    """
    2D图像的交并比
    :param boxes_a: Shape:[N, (y1, x1, y2, x2)]
    :param boxes_b: Shape:[M, (y1, x1, y2, x2)]
    :return: iou: [N,M}
    """
    # 扩维
    boxes_a = torch.unsqueeze(boxes_a, dim=1)  # [N,1,4]
    boxes_b = torch.unsqueeze(boxes_b, dim=0)  # [1,M,4]
    # 计算交集
    zero = torch.zeros(1)
    if boxes_a.is_cuda:
        zero = zero.cuda()
    overlaps = torch.max(torch.min(boxes_a[..., 2:], boxes_b[..., 2:]) -
                         torch.max(boxes_a[..., :2], boxes_b[..., :2]),
                         zero)  # [N,M,2]
    overlaps = torch.prod(overlaps, dim=-1)  # [N,M]

    # 计算各自面积
    volumes_a = torch.prod(boxes_a[..., 2:] - boxes_a[..., :2], dim=-1)  # [N,1]
    volumes_b = torch.prod(boxes_b[..., 2:] - boxes_b[..., :2], dim=-1)  # [1,M]

    # 计算iou
    iou = overlaps / (volumes_a + volumes_b - overlaps)

    return iou


def nms(boxes, scores, overlap=0.5, top_k=200):
    """
    2D图像的非极大抑制
    :param boxes: Shape:[num_boxes, (y1,x1,y2,x2)]
    :param scores: Shape:[num_boxes]
    :param overlap: nms的iou阈值
    :param top_k: 最终保留的top_k个数
    :return: 
    keep: Shape:[count] nms后保留的boxes索引
    count: 标量 （count < top_k）
    """
    # 判断scores是否为空
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    #
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    #
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    return keep, count


def encode(matched_gts, default_boxes, variances):
    """
    根据defult_boxes及匹配的gts，计算targets
    :param matched_gts: Shape:[num_default_boxes, 4]
    :param default_boxes: Shape:[num_default_boxes, 4]
    :param variances: list(float) Shape:[4]
    :return: 
    回归目标：Shape:[num_default_boxes, 4]
    """
    # (y1, x1, y2, x2)转(cy, cx, h, w)
    gts_copy = center_size(matched_gts)
    default_boxes_copy = default_boxes  # priors不用转center_size模式，因为已经是（cy,cx,h,w）了

    # 高度、宽度
    size = default_boxes_copy[:, 2:]
    gt_size = gts_copy[:, 2:]
    # 中心点
    ctr = default_boxes_copy[:, :2]
    gt_ctr = gts_copy[:, :2]

    # 计算回归目标
    d_yx = (gt_ctr - ctr) / size  # [n,(dy,dx)]
    d_hw = torch.log(gt_size / size)  # [n,(dh,dw)]
    target = torch.cat((d_yx, d_hw), dim=1)

    # 标准差
    box_std = torch.Tensor(variances)
    if target.is_cuda:
        box_std = box_std.cuda()
    target = target / box_std

    return target


def decode(deltas, default_boxes, variances):
    """
    根据defult_boxes及网络预测的deltas，计算prior_boxes的应用回归结果
    :param deltas: Shape:[num_default_boxes, 4]
    :param default_boxes: Shape:[num_default_boxes, 4]
    :param variances: list(float) Shape:[4]
    :return: 
    Shape:[num_default_boxes, 4]
    """
    # (y1, x1, y2, x2)转(cy, cx, h, w)
    default_boxes_copy = center_size(default_boxes)

    # 高度、宽度
    h = default_boxes_copy[:, 2]
    w = default_boxes_copy[:, 3]

    # 中心点
    cy = default_boxes_copy[:, 0]
    cx = default_boxes_copy[:, 1]

    # 回归系数
    box_std = torch.Tensor(variances)
    if deltas.is_cuda:
        box_std = box_std.cuda()
    deltas = deltas * box_std
    dy, dx, dh, dw = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # 调整
    cy = cy + dy * h
    cx = cx + dx * w
    h = h * torch.exp(dh)
    w = w * torch.exp(dw)

    # (cy, cx, h, w)转(y1, x1, y2, x2)
    y1 = cy - h * 0.5
    x1 = cx - w * 0.5
    y2 = cy + h * 0.5
    x2 = cx + w * 0.5

    return torch.stack((y1, x1, y2, x2), dim=1)


def target(threshold, gts, priors, variances, labels):
    """
    样本采样、匹配过程及调用encode函数做target计算（注：1个gt与N个prior匹配，一对多的关系，但是1个prior只对应1个gt）
    :param threshold: 标量-做匹配的IOU阈值
    :param gts: Ground truth boxes, Shape:[num_gts, 4]
    :param priors: default boxes from priorbox layers, Shape: [num_default_boxes,4]
    :param variances: 
    :param labels: All the class labels for the gts, Shape: [num_gts]
    :return: deltas: tensor [rois_num,(dy,dx,dz,dh,dw,dd)]
    :return: labels: tensor [rois_num,(y1,x1,z1,y2,x2,z2)]
    :return: metrics: dict 用于打印统计样本情况
    # :return: rois: tensor [rois_num]
    # :return: rois_indices: tensor [rois_num] roi在原始的mini-batch中的索引号;roiAlign时用到
    
    """
    metrics = {}
    iou_target = iou(gts, point_bound(priors))
    # 每个gt最佳的prior
    best_prior_iou, best_prior_idx = iou_target.max(1, keepdim=True)  #  [1,num_gts]
    # 每个prior最佳的gt
    best_gt_iou, best_gt_idx = iou_target.max(0, keepdim=True)  # [1,num_priors]

    # 降维
    best_gt_idx.squeeze_(0)
    best_gt_iou.squeeze_(0)
    best_prior_idx.squeeze_(1)  # squeeze()选的dim=1？
    best_prior_iou.squeeze_(1)  # squeeze()选的dim=1？

    # 与每个gt最大iou的prior的iou值设置为2（注：避免某些gts的最大iou都小于threshold，二把match该的情况）
    best_gt_iou.index_fill_(0, best_prior_idx, 2)  # [num_priors]
    # 保证每个gt对应match的priors都是iou值最大
    for j in range(best_prior_idx.size(0)):
        best_gt_idx[best_prior_idx[j]] = j
    # priors框对应match的gts框-坐标
    matches = gts[best_gt_idx]  #  [num_priors, 4]
    # priors框对应match的gts框-类别
    cls = labels[best_gt_idx]  #  [num_priors]

    # 正、负样本及ignore样本区分
    if len(threshold) > 1:
        cls[best_gt_iou < threshold[1]] = -1  # 有忽略样本
        cls[best_gt_iou < threshold[0]] = 0
    else:
        cls[best_gt_iou < threshold[0]] = 0  # 没有忽略样本

    # 分类目标
    batch_labels = cls
    # 回归目标
    batch_deltas = encode(matches, priors, variances)
    # # 合并
    # batch_target = np.concatenate([batch_deltas, batch_labels[:, np.newaxis]], -1)

    return batch_labels, batch_deltas, metrics


def log_sum_exp(x):
    """
    根据所有打平priorbox的预测confidence，计算log_sum_exp值
    :param x: 打平了的priorbox的预测confidence
    :return: 
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def save_net(fname, net):
    """
    保存模型参数
    """
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    """
    加载模型参数
    """
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)
