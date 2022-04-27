import torch
from torch import nn
from AlexNet import load_data_fashion_minst
from LeNet import train_by_gpu


def vgg_block(num_convs, in_channels, out_channels):
    """
    :param num_convs:卷积层个数
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :return: 展平后组成Sequential
    """
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
    return nn.Sequential(*layers)


# 1表示卷积层个数，64表示通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    # 其中3等于96/2/2/2/2/2（因为有5个池化层）
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 3 * 3, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )


# net = vgg(conv_arch)

# 调整ratio可以减少内存开销
ratio = 1
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
small_net = vgg(small_conv_arch)

# lr, epochs, batch_size = 0.05, 8, 32
# train_iter, test_iter = load_data_fashion_minst(batch_size, resize=96)
# print("train on VGGNet")
# train_by_gpu(small_net, train_iter, test_iter, epochs, lr, device=torch.device("cuda"))

