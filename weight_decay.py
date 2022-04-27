import torch
import 线性回归
import multilayer_perceptron
from torch import nn

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
ture_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = 线性回归.synthetic_data(ture_w, true_b, n_train)
train_iter = 线性回归.load_array(train_data, batch_size)
test_data = 线性回归.synthetic_data(ture_w, true_b, n_test)
test_iter = 线性回归.load_array(test_data, batch_size, is_train=False)


# 初始化权重参数
def init_params():
    w = torch.normal(1, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.ones(1, requires_grad=True)
    return [w, b]


# 定义L2范数
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


# 尝试一下L1范数
def l1_penalty(w):
    return torch.sum(torch.abs(w))


# 定义训练函数
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: 线性回归.linreg(X, w, b), 线性回归.square_loss
    epochs, lr = 100, 0.001
    for epoch in range(epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l1_penalty(w)
            # l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            线性回归.sgd([w, b], lr, batch_size)

    print("w的L1范数是：", torch.sum(torch.abs(w)))
    # print("w的L2范数为：", torch.norm(w).item())


# train(3)


# 简洁实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        # 初始化权重
        param.data.normal_()
    loss = nn.MSELoss(reduction="none")
    epochs, lr = 100, 0.001
    # 偏置参数不设置衰减
    optimizer = torch.optim.SGD([{"params": net[0].weight, "weight_decay": wd}, {"params": net[0].bias}], lr=lr)

    for epoch in range(epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            optimizer.step()

    print("w的L2范数为:", net[0].weight.norm().item())
    print("w:", net[0].weight)


train_concise(10)
