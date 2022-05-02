import torch.nn as nn
from torch.nn import functional as F


class Residual(nn.Module):
    """
    use_1x1conv：是否增加通道的数量
    """

    def __init__(self, in_channels, num_channels, use_1x1conv=False, strides=(1, 1)):
        super().__init__()
        # 不改变高宽，为了能够和后面直接传入的残差相加
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=(3, 3), padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=(1, 1), stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # Y = F.relu(self.bn1(self.conv1(X)))
        # Y = self.bn2(self.conv2(Y))
        # if self.conv3:
        #     X = self.conv3(X)
        # Y += X
        # return F.relu(Y)

        # 改进版
        Y = self.conv1(F.relu(self.bn3(X)))
        Y = F.relu(self.bn4(Y))
        if self.conv3:
            X = self.conv3(X)
        # 这里不能使用Y+=X，会报错，也就是需要为Y重新申请内存
        Y = Y + X
        return self.conv2(Y)


def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        # 当为残差块的第一层且不为b1后面的一层时，会改变通道数
        # 当为b1后面一层时，由于b1已经改变了高和宽，所以这里不改变高和宽了
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=(2, 2)))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

