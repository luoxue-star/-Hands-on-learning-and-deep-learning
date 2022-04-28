import torch
from torch import nn
from AlexNet import load_data_fashion_minst
from LeNet import train_by_gpu
from torch.nn import functional as F


# 从0实现batch normalization
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    :param X:输入的数据
    :param gamma: 缩放系数
    :param beta: 偏移系数
    :param moving_mean:前面样本的均值
    :param moving_var: 前面样本的方差
    :param eps: 防止分母为0的偏差
    :param momentum: 更新样本均值和方差的系数
    :return: 批量归一化后的样本，更新后的均值和方差
    """
    # 判断此时是训练还是测试
    if not torch.is_grad_enabled():
        # 测试，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 这里只处理卷积或者全连接层，2代表全连接层，4代表卷积层
        assert len(X.shape) in (2, 4)
        # 使用全连接层，计算特征维度上的均值和方差
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层，在通道维上计算均值方差
            # 保持维度，为了后面批量归一化能够广播
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差，这不是网络的参数，需要手动更新
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var

    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    """
    num_feature:完全连接层的输出数量和卷积层的输出通道数
    num_dims:2表示全连接层，4表示卷积层
    """
    def __init__(self, num_features, num_dims):
        super.__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化为1，0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0,1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 若两个参数和X不在一个内存上，复制到X所在的内存
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# 简洁实现，基于GoogleNet
class Inception(nn.Module):
    """
    c1-c4是每个路径的输出通道数
    """

    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路一，一个1*1的卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=(1, 1))
        self.bn1_1 = nn.BatchNorm2d(c1)
        # 线路二，1*1卷积层后接3*3的卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=(1, 1))
        self.bn2_1 = nn.BatchNorm2d(c2[0])
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=(3, 3), padding=(1, 1))
        self.bn2_2 = nn.BatchNorm2d(c2[1])
        # 线路三，1*1卷积层后接5*5的卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=(1, 1))
        self.bn3_1 = nn.BatchNorm2d(c3[0])
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(5, 5), padding=2)
        self.bn3_2 = nn.BatchNorm2d(c3[1])
        # 线路四，3*3的最大池化层后接1*1的卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=(1, 1))
        self.bn4_2 = nn.BatchNorm2d(c4)

    def forward(self, x):
        x1 = self.p1_1(x)
        x1 = self.bn1_1(x1)
        x1 = F.relu(x1)

        x2 = self.p2_1(x)
        x2 = self.bn2_1(x2)
        x2 = F.relu(x2)
        x2 = self.p2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = F.relu(x2)

        x3 = self.p3_1(x)
        x3 = self.bn3_1(x3)
        x3 = F.relu(x3)
        x3 = self.p3_2(x3)
        x3 = self.bn3_2(x3)
        x3 = F.relu(x3)

        x4 = self.p4_1(x)
        x4 = self.p4_2(x4)
        x4 = self.bn4_2(x4)
        x4 = F.relu(x4)

        # 在通道维度上连接输出,第一个通道为batch
        return torch.cat((x1, x2, x3, x4), dim=1)


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=(1, 1)),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=(3, 3), padding=1),
    nn.BatchNorm2d(192),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))
b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
lr, epochs, batch_size = 0.1, 15, 128
train_iter, test_iter = load_data_fashion_minst(batch_size)
print("train on GoogleNet")
train_by_gpu(net, train_iter, test_iter, epochs, lr, torch.device("cuda"))

