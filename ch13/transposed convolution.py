import torch
from torch import nn
from d2l import torch as d2l


def trans_conv(X, K):
    """无padding，stride=1的转置卷积"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            Y[i:i+h, j:j+w] += X[i, j] * K
    return Y


if __name__ == "__main__":
    # 转置卷积的填充作用与输出，如padding=(1, 1)，会删除输出的第一行列和最后一行列
    # t_conv = nn.ConvTranspose2d(1, 1, kernel_size=(2, 2), padding=(1, 1), bias=False)
    # X = torch.tensor([[0, 1], [2, 3]], dtype=torch.float32).reshape(1, 1, 2, 2)
    # K = torch.tensor([[0, 1], [2, 3]], dtype=torch.float32).reshape(1, 1, 2, 2)
    # t_conv.weight.data = K
    # print(t_conv(X))

    X = torch.randn(size=(1, 10, 16, 16))
    conv = nn.Conv2d(10, 20, kernel_size=(5, 5), padding=(2, 2), stride=(3, 3))
    tconv = nn.ConvTranspose2d(20, 10, kernel_size=(5, 5), padding=(2, 2), stride=(3, 3))
    print(tconv(conv(X)).shape == X.shape)

