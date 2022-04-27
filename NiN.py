import torch
from torch import nn
from AlexNet import load_data_fashion_minst
from LeNet import train_by_gpu


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)), nn.ReLU())


net = nn.Sequential(
    nin_block(1, 96, (11, 11), strides=(4, 4), padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, (5, 5), (1, 1), 2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, (3, 3), (1, 1), 1),
    nn.MaxPool2d(3, stride=(2, 2)),
    nn.Dropout(0.5),  # 感觉这里不需要Dropout，去掉了反而收敛不了
    # 总共有10类
    nin_block(384, 10, kernel_size=(3, 3), strides=(1, 1), padding=1),
    # 将四维换成二维的输出，1， 1分别是指定输出的行数和列数
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())

# 输出每个层的维度
# X = torch.rand((1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, "\tout_shape\t", X.shape)

lr, epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = load_data_fashion_minst(batch_size, resize=224)
print("train on NiN")
train_by_gpu(net, train_iter, test_iter, epochs, lr, torch.device("cuda"))
