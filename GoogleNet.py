import torch
from torch import nn
from torch.nn import functional as F
from AlexNet import load_data_fashion_minst
from LeNet import train_by_gpu


class Inception(nn.Module):
    """
    c1-c4是每个路径的输出通道数
    """

    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路一，一个1*1的卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=(1, 1))
        # 线路二，1*1卷积层后接3*3的卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=(1, 1))
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=(3, 3), padding=(1, 1))
        # 线路三，1*1卷积层后接5*5的卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=(1, 1))
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(5, 5), padding=2)
        # 线路四，3*3的最大池化层后接1*1的卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=(1, 1))

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连接输出,第一个通道为batch
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=(3, 3), padding=1),
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
