import torch
from torch import nn
from torch.nn import functional as F
from AlexNet import load_data_fashion_minst
from LeNet import train_by_gpu


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


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))


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


# 此模型为ResNet-18
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))
lr, epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = load_data_fashion_minst(batch_size)
print("train on ResNet18")
train_by_gpu(net, train_iter, test_iter, epochs, lr, torch.device("cuda"))
# bottleneck通常是用于ResNet50及以上的网络，其结构是将两个3x3的卷积层变为：首先使用1x1卷积层将通道数降低，再接一个通道数不变的3x3卷积层，再接
# 一个与起始输入大小一致的1x1卷积层
