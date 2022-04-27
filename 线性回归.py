import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn


# def synthetic_data(w, b, num_examples):
#     """
#     num_example:样本的个数
#     生成y=Xw+b+噪声的数据
#     """
#     X = torch.normal(0, 1, (num_examples, len(w)), dtype=torch.float32)
#     y = torch.matmul(X, w) + b
#     y += torch.normal(0, 0.01, y.shape)
#     return X, y.reshape((-1, 1))
#
#
# true_w = torch.tensor([-2, 5, 3], dtype=torch.float32)
# true_b = torch.tensor([4], dtype=torch.float32)
# features, labels = synthetic_data(true_w, true_b, 10000)
# print("features:", features[0], "\nlabels", labels[0])
# print("true_w:", true_w, "\ntrue_b:", true_b)
#
# plt.figure()
# # detach表示分离出数值，不再含有梯度
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 2)
# plt.show()
#
#
# def data_iter(batch_size, features, labels):
#     """
#     生成数据迭代器
#     """
#     num_examples = len(features)
#     # 进行切片
#     indices = list(range(num_examples))
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(
#             indices[i:min(i+batch_size, num_examples)]
#         )
#         # 生成一个迭代器， 每次迭代后会记住这个位置
#         # 也就是每次迭代后取一定的批量
#         yield features[batch_indices], labels[batch_indices]
#
#
def linreg(X, w, b):
    """
    定义线性回归模型
    """
    return torch.matmul(X, w) + b


def square_loss(y_hat, y):
    """
    定义均方差损失函数
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
#
#
def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            # 因为计算梯度时是通过整个batch_size计算的
            param -= lr * param.grad / batch_size
            param.grad.zero_()
#
#
# batch_sizes = 100
# lr = 0.01
# epochs = 5
# net = linreg
# loss = square_loss
# w = torch.rand(true_w.shape, requires_grad=True)
# b = torch.rand(1, requires_grad=True)
#
# for epoch in range(epochs):
#     # 训练并且计算梯度
#     for X, y in data_iter(batch_sizes, features, labels):
#         l = loss(net(X, w, b), y)
#         l.sum().backward()
#         sgd([w, b], lr, batch_sizes)
#     with torch.no_grad():
#         train_l = loss(net(features, w, b), labels)
#         print(f'epoch{epoch+1}, loss{float(train_l.mean()):f}')
#
# print("w:", w, "\nb:", b)


# torch简洁线性回归
def synthetic_data(w, b, num_examples):
    """
    num_example:样本的个数
    生成y=Xw+b+噪声的数据
    """
    X = torch.normal(0, 1, (num_examples, len(w)), dtype=torch.float32)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([-2, 5, 3], dtype=torch.float32)
true_b = torch.tensor([4], dtype=torch.float32)
features, labels = synthetic_data(true_w, true_b, 10000)
print("features:", features[0], "\nlabels", labels[0])
print("true_w:", true_w, "\ntrue_b:", true_b)


def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个pytorch的迭代器
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_sizes = 100
data_iter = load_array((features, labels), batch_sizes)
# print(next(iter(data_iter)))

# 定义模型，损失函数和优化器
net = nn.Sequential(nn.Linear(3, 1))
# 0代表第一层, 这里是进行权重和偏置的初始化
net[0].weight.data.normal_(0, 0.1)
net[0].bias.data.fill_(0)
# loss = nn.MSELoss()
# 相对MSE训练是较慢的， 最好调高学习率
loss = nn.SmoothL1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 开始训练
epochs = 5
for epoch in range(epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(features), labels)
    # 后面的f表示保留六位小数点
    print(f'epoch:{epoch+1}, loss:{l:f}')

print('w:', net[0].weight.data)
print('b:', net[0].bias.data)
