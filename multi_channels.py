import torch
from Conv2d import conv2d


def corr2d_multi_in(X, K):
    """
    多输入单输出
    先遍历X和K第0维（通道），再求和
    """
    return sum(conv2d(x, k) for x, k in zip(X, K))


def corr2d_multi_out(X, K):
    """
    单输入多输出
    torch.stack会增加维度，0表示在第一个维度上
    """
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def corr2d_multi_in_out(X, K, out_channels):
    for i in range(out_channels):
        if i == 0:
            Y = sum(conv2d(x, k) for x, k in zip(X, K[i]))
        else:
            Y = torch.stack((Y, sum(conv2d(x, k) for x, k in zip(X, K[i]))), 0)
    return Y


def corr2d_multi_in_out_1x1(X, K):
    """1*1卷积"""
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

