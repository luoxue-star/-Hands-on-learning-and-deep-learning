import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    """
    参数初始化
    :param m:参数
    :return:
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


def train(net, train_iter, loss, epochs, optimizer):
    for epoch in range(epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            optimizer.step()
        print(f"epoch:{epoch+1}," f"loss:{d2l.evaluate_loss(net, train_iter, loss):f}")


if __name__ == "__main__":
    T = 1000
    time = torch.arange(1, T + 1, dtype=torch.float32)
    # 生成含噪声的数据
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T, ))
    # d2l.plot(time, [x], "time", "x", xlim=[1, 1000], figsize=(6, 3))
    # plt.show()

    # 数据处理成features-labels的形式
    # 用当前数据之前的τ个数据来预测下一个数据
    tau = 4
    features = torch.zeros((T-tau, tau))
    for i in range(tau):
        # 每一列对应一个特征，每一行对应一个时间序列
        features[:, i] = x[i: T-tau+i]
    labels = x[tau:].reshape((-1, 1))

    batch_size, n_train = 16, 600
    # 只取前600个样本进行训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
    # none表示维度不进行缩减，保留每个元素的损失值
    loss = nn.MSELoss(reduction="none")
    net = get_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    train(net, train_iter, loss, epochs=5, optimizer=optimizer)

    # 预测模块的可视化， 利用4个特征预测一个
    onestep_preds = net(features)
    # d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], "time",
    #          "x", legend=["data", "1_step preds"], xlim=[1, 1000], figsize=(6, 3))
    # plt.show()

    # 这里是直接用我们预测后的数据进行预测
    # 但会发现结果很差，这是因为错误一直累计导致的
    multistep_preds = torch.zeros(T)
    multistep_preds[:n_train+tau] = x[: n_train+tau]
    for i in range(n_train+tau, T):
        multistep_preds[i] = net(multistep_preds[i-tau:i].reshape((1, -1)))
    # d2l.plot([time, time[tau:], time[n_train + tau:]],
    #          [x.detach().numpy(), onestep_preds.detach().numpy(),
    #           multistep_preds[n_train+tau:].detach().numpy()], "time",
    #          "x", legend=["data", "1-step preds", "multistep preds"],
    #          xlim=[1, 1000], figsize=(6, 3))
    # plt.show()

    # 用4天预测后面的k天
    max_step = 64
    features = torch.zeros((T-tau-max_step+1, tau+max_step))
    for i in range(tau):
        features[:, i] = x[i:i+T-tau-max_step+1]

    for i in range(tau, tau+max_step):
        # net的输入可以是n×4的矩阵，每一行相当于一个输入
        features[:, i] = net(features[:, i-tau:i]).reshape(-1)

    steps = (1, 4, 16, 64)
    d2l.plot([time[tau+i-1:T-max_step+i] for i in steps],
             [features[:, (tau+i-1)].detach().numpy() for i in steps],
             "time", "x",
             legend=[f"{i}-step preds" for i in steps], xlim=[5, 1000], figsize=(6, 3))
    plt.show()

