import torch
from d2l import torch as d2l


# 边界框：
# 中心位置和框的高度和宽度
# 或者矩形的左上角和框的高度和宽度
def box_corner_to_center(boxes):
    """从左上右下到中间宽度高度"""
    # 每一行表示一个矩形框
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    # 在最后一个维度上进行堆叠
    boxes = torch.stack((cx, cy, w, h), dim=-1)
    return boxes


def box_center_to_corner(boxes):
    """从中间高度宽度到左上右下"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes


def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )


