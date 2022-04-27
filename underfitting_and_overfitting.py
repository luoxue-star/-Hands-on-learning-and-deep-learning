import numpy as np
import torch
import math
import torch.nn as nn
import softmax_regression
import 线性回归


max_degree = 20  # 多项式的次数最高为20
n_train, n_test = 100, 100  # 训练集和测试集的大小
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5.0, 1.0, -3.0, 6.0])

# 要进行预测的式子是类似于泰勒展开的那个样子
features = np.random.normal(size=(n_train+n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    # gamma(n) = (n-1)!
    # 每个维度对应除去阶乘
    poly_features[:, i] /= math.gamma(i+1)
labels = np.dot(poly_features, true_w)
# 加上标准差为0.1的噪声
labels += np.random.normal(scale=0.1, size=labels.shape)

# 转化为tensor类型
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32)
                                           for x in [true_w, features, poly_features, labels]]


def evaluate_loss(net, data_iter, loss):
    metric = softmax_regression.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels, epochs=400):
    loss = nn.MSELoss(reduction="none")
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = 线性回归.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = 线性回归.load_array((test_features, test_labels.reshape(-1, 1)), batch_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(epochs):
        softmax_regression.train_epoch(net, train_iter, loss, optimizer)
    print("weight:", net[0].weight.data.numpy())


# 从多项式中选择前四个维度
poly_dim = 4
train(poly_features[:n_train, :poly_dim], poly_features[n_train:, :poly_dim],
      labels[:n_train], labels[n_train:])


