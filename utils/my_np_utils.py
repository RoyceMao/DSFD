import numpy as np

def clip_boxes(boxes, image_size):
    """
    裁剪到窗口内
    :param boxes: numpy [N, (cls, y1, x1, y2, x2)]
    :param image_size: 标量
    :return: 裁剪后的 boxes: [N, (y1, x1, y2, x2, cls)]
    """
    boxes = boxes.copy()
    p1 = np.maximum(boxes[:, 1:3], 0)
    p2 = boxes[:, 3:]
    p3 = boxes[:, 0]

    return np.concatenate([p1, p2, p3[:, np.newaxis]], axis=1)