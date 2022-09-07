import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    类别预测
    :param num_inputs: 输入通道数
    :param num_anchors: 每个中心点的锚框数
    :param num_classes: 类别数
    :return:
    """
    # +1是因为背景也算一类
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=(3, 3), padding=1)


def bbox_predictor(num_inputs, num_anchors):
    # *4是因为每个锚框需要四个偏移值3
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=(3, 3), padding=1)


def forward(x, block):
    return block(x)


def flatten_pred(pred):
    # 由于预测输出只有batch_size一样，所以进行维度调整便于拼接起来
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    # 将预测的几个块进行拼接
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    """定义SSD中的高宽减半的卷积池化层"""
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    """定义SSD中的基本网络块"""
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


def get_blk(i):
    """SSD的多个网络块"""
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveAvgPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    # 每个块生成的特征图既用于生成锚框，又用于预测锚框类别和偏移量
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        # 0.2~1.05进行均分,此外0.272=sqrt(0.2*0.37),其余的以此类推
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        for i in range(5):
            # 等价于self.blk_i = get_blk(i)
            setattr(self, f"blk_{i}", get_blk(i))
            setattr(self, f"cls_{i}", cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f"bbox_{i}", bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f"blk_{i}"), self.sizes[i], self.ratios[i],
                getattr(self, f"cls_{i}"), getattr(self, f"bbox_{i}"))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # 去掉那些背景框
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    # 将两者的损失相加
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # 类别预测在最后一个维度上，argmax需要指定最后一维
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


def display(img, output, threshold):
    d2l.set_figsize()
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:5] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, "%.2f" %score, 'w')
    d2l.plt.show()


if __name__ == "__main__":
    # TinySSD中forward的调试
    """
    net = TinySSD(num_classes=1)
    # batch_size=32, channels=3, h, w=256, 256
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    """

    # 训练模型
    batch_size = 32
    train_iter, _ = d2l.load_data_bananas(batch_size)
    device, net = d2l.try_gpu(), TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    cls_loss, bbox_loss = nn.CrossEntropyLoss(reduction="none"), nn.L1Loss(reduction="none")
    epochs, timer = 20, d2l.Timer()
    animator = d2l.Animator(xlabel="epoch", xlim=[1, epochs], legend=["class error", "bbox mae"])
    net = net.to(device)
    for epoch in range(epochs):
        # 4分别代表训练精确度的和，训练精确度和中的示例数，绝对误差的和，绝对误差的和中的示例数
        metric = d2l.Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)  # 生成多尺度锚框，并进行类别预测和偏移量预测
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
    d2l.plt.show()
    print(f"class error {cls_err:.2e}, bounding box mae {bbox_mae:.2e}")
    print(f"{len(train_iter.dataset) / timer.stop():.1f} examples / sec on"
          f"{str(device)}")





