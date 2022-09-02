import torch
from d2l import torch as d2l

# 精简输出精度
torch.set_printoptions(2)


def multibox_prior(data, sizes, ratios):
    """
    每个像素点生成n+m-1个锚框
    :param data:输入的图像（含batch）
    :param sizes: 输出锚框缩放的比例
    :param ratios: 锚框的高宽比
    :return: 锚框
    """
    # 最后两个维度是高度和宽度（获取输入的图像的高度和宽度）
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)  # 为了减少内存的开销，只生成含sizes[0]或ratios[0]的锚框
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 锚点是每个像素的中心，因为像素的大小是1*1的，所以设置偏移量为0.5
    offset_h, offset_w = 0.5, 0.5  # 为了将锚点移动到像素的中心，需要设置一个偏移量
    # 在高度和宽度进行缩放步长
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    # 生成所有锚框的中心点，生成的中心均是0~1的数（因为除以高和宽了）
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # 生成网格点,也就是center_h和center_w两者两两对应交叉生成(广播)
    # 例如center_h为4维，center_w为3维，最终两个数组行方向为4，列方向为3进行广播
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)  # 变成一维的

    # 生成boxes_per_pixel个高和宽，之后用于创建锚框的四角坐标
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width

    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # .repeat代表在列方向上重复in_height*in_weight次，1表示行方向上不变
    # 除以2是获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    # 指定在第0维上重复boxes_per_pixel次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(
        boxes_per_pixel, dim=0)

    output = out_grid + anchor_manipulations  # 这一步就得到左上和右下的坐标点位置（归一化后的）
    return output.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """
    显示所有边界框
    :param axes:
    :param bboxes:
    :param labels:
    :param colors:
    :return:
    """

    def _make_list(obj, default_value=None):
        if obj is None:
            obj = default_value
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va="center", ha="center", fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


def box_iou(boxes1, boxes2):
    """
    计算两个锚框或者边界框的交并比
    :param boxes1: 框1
    :param boxes2: 框2
    :return: 交并比
    """
    # 右下角减去左上角计算面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0] *
                               (boxes[:, 3] - boxes[:, 1])))
    areas1 = box_area(boxes1)  # areas:(boxes数量, )
    areas2 = box_area(boxes2)
    # 加上None后，原来是二维张量，变成三维张量。然后通过广播机制，就使得inter都变成三维的了，第二个维度就是boxes2的数量
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 计算交叉处右下角的坐标
    inter_lowerrights = torch.min(boxes1[:, None, :2], boxes2[:, 2:])  # 计算交叉处左上角的坐标
    # .clamp将输入限制到min=0的下界中，上界为真实最大值。inters:(boxes1个数，boxes2个数，2(表示坐标))
    inters = (inter_upperlefts - inter_lowerrights).clamp(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """
    将最接近真实边界框分配给锚框
    :param ground_truth: 真实边缘框
    :param anchors: 锚框
    :param device: GPU or CPU
    :param iou_threshold: iou的阈值，小于该阈值的锚框会被舍弃
    :return:
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]  # 计算锚框数和真实边缘框个数
    # 第i行第j列是锚框i和真实边界框j的iou
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配真实的边界框的张量
    # torch.full是填充数字，第一个参数是填充的形状，-1表示填充的数字
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)  # 求出每个锚框与每一个边缘框最大iou的值和所在维度位置，dim=1表示行方向
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)  # 大于0.5为True，即就是1，就是找出大于0.5的位置
    box_j = indices[max_ious >= 0.5]  # 只取iou>0.5的元素索引
    anchors_bbox_map[anc_i] = box_j  # 对几个与边缘框最大的iou的锚框赋予标签
    col_discard = torch.full((num_anchors, ), -1)
    row_discard = torch.full((num_gt_boxes, ), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # 将维度展为一维后，计算最大值的索引
        box_idx = (max_idx % num_gt_boxes).long()  # 得到最大值在二维矩阵的列索引
        anc_idx = (max_idx / num_gt_boxes).long()  # 得到最大值在二维矩阵的行索引
        anchors_bbox_map[anc_idx] = box_idx  # 为锚框赋予标签
        jaccard[:, box_idx] = col_discard  # 去掉最大值那一列
        jaccard[anc_idx, :] = row_discard  # 去掉最大值那一行
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量进行转换"""
    c_anc = d2l.box_corner_to_center(assigned_bb)  # 将锚框标签转化为中心表示法
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)  # 将边界框标签转化为中心表示法
    offset_xy = (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:] / 0.1
    offset_wh = torch.log(eps + c_assigned_bb[:, 2:]) / c_anc[:, 2:] / 0.2
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multibox_target(anchors, labels):
    """
    使用真实边界框标记锚框
    anchors:(batch_size, num, 4)
    labels:(batch_size,num,5)  # 第0位为标签
    return: 偏移量（背景框为0） mask:(batch_size, 锚框数*4)目的是过滤掉背景锚框的偏移量 标签:(锚框数,)
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]  # 获取第i个batch的标签
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)  # 1:表示不取标签，只取框的位置
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)  # 增加一个维度，并重复四次
        # 将类标签和分配的边界框坐标初始为0
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # 使用真实边界框标记锚框的类别
        # 若一个锚框没有被分配，则被标记为0（背景）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1  # 标签
        assigned_bb[indices_true] = label[bb_idx, 1:]  # 坐标
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask  # 过滤掉那个被标记为背景框的
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_pred):
    """根据带有预测偏移量的锚框预测边界框"""
    anc = d2l.box_corner_to_center(anchors)  # 转化为中心表示
    pred_bbox_xy = (offset_pred[:, :2] * anc[:, 2:] * 0.1) + anc[:, :2]  # 将偏移量翻转回去得到xy
    pred_bbox_wh = torch.exp(offset_pred[:, 2:] * 0.2) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """非极大值抑制，对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]  # 取出置信度最高的
        keep.append(i)  # 保留置信度最高的
        if B.numel() == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)  # 过滤掉那些与当前置信度最大的锚框的iou大于阈值的
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """
    使用非极大值抑制预测边界框
    cls_probs:(batch_size,图片中的类别个数（含背景）,4)
    offset_preds:(batch_size,锚框数)
    anchors:(batch_size,锚框数，4)
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)  # 得到每个batch的预测类别概率和预测偏移值
        conf, class_id = torch.max(cls_prob[1:], 0)  # 1:表明不要背景的预测概率，只要其它类别的预测概率。0表示删除0维度，在一维度找max
        predicted_bb = offset_inverse(anchors, offset_pred)  # 获取真正的锚框位置
        keep = nms(predicted_bb, conf, nms_threshold)  # 非极大值抑制
        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]  # 等于1表示在非极大值抑制中被去掉的
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1  # 在非极大值抑制中被去掉的设置为背景
        class_id = class_id[all_id_sorted]  # 获取到每个锚框最终的预测类别
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是用于非背景预测的阈值，小于阈值的话，仍会被设置为背景类
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)  # out:(batch_size,锚框数量,6(class,probability,锚框位置（4）)
    return torch.stack(out)


if __name__ == "__main__":
    """
    # 调试multibox_prior()方法
    img = d2l.plt.imread("../imgs/13/13-02.jpg")
    h, w = img.shape[:2]
    print(h, w)
    X = torch.rand(size=(1, 3, h, w))  # batch_size为1，通道数为3
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    """

    """
    # multibox_target()调试
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92], [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]])
    labels = multibox_target(anchors.unsqueeze(0), ground_truth.unsqueeze(0))  # 添加batch维度
    """

    # multibox_detection()调试案例
    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                            [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0, 0, 0, 0],  # 背景的预测概率
                              [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                              [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
    output = multibox_detection(cls_probs.unsqueeze(0), offset_preds.unsqueeze(0),
                                anchors.unsqueeze(0), nms_threshold=0.5)


