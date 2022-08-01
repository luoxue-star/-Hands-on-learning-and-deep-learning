import torch
from torch import nn
from d2l import torch as d2l


def f(x):
    """待拟合的函数"""
    return 2 * torch.sin(x) + x ** 0.8


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_test, y_hat], 'x', 'y', legend=["label", "pred"],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super(NWKernelRegression, self).__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1, ), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights形状为(查询个数,键值对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1)
        # value的形状为(查询个数,键值对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)


if __name__ == "__main__":
    n_train = 50  # 训练样本的数量
    # 为了更好可视化注意力的模式，讲训练的样本进行排序
    x_train, _ = torch.sort(torch.rand(n_train) * 5)
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train, ))
    x_test = torch.arange(0, 5, 0.1)
    y_test = f(x_test)
    n_test = len(x_test)  # 测试样本数量

    """
    # 使用平均汇聚计算所有训练样本的输出,但是显然平均汇聚忽略了x的作用
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)
    """

    """
    # 非参数化的注意力汇聚
    # 每一行都是相同的测试输入,形状为(n_test, n_train)
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # 利用高斯核为每一个输入赋予权重
    attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(y_hat)
    d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel="Sorted training inputs",
                      ylabel="Sorted testing inputs")
    d2l.plt.show()
    """

    """
    # 带参数的注意力汇聚
    # 批量矩阵乘法：X：(n,a,b) Y：(n,b,c)-> Z：(n,a,c) 使用torch.bmm
    X_tile = x_train.repeat((n_train, 1))  # (n_train, n_train)
    Y_tile = y_train.repeat((n_train, 1))  # (n_train, n_train)
    # 在带参数的注意力汇聚模型中，所有训练样本的输入都会和除自己外的所有训练样本的键值对进行计算，进而得到输出
    # keys和values都是去掉对角线的值 (n_train, n_train-1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[1, 5])

    for epoch in range(10):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f"epoch:{epoch + 1}, loss:{float(l.sum()):.6f}")
        animator.add(epoch + 1, float(l.sum()))
    d2l.plt.show()

    keys = x_train.repeat((n_test, 1))
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    plot_kernel_reg(y_hat)

    d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel="Sorted training inputs",
                      ylabel="Sorted testing inputs")
    d2l.plt.show()
    """









