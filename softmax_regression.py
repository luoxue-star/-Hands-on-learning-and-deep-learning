import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn


def get_fashion_minst_labels(labels):
    """返回Fashion—Mnist数据集的文本标签"""
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat"
                   , "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]


def show_image(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 因为torch类型的不能直接展示
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
        if titles:
            ax.set_title(titles[i])
        plt.show()
    return axes


# X, y = next(iter(data.DataLoader(minst_train, batch_size=20)))
# show_image(X.reshape(20, 28, 28), 2, 10, titles=get_fashion_minst_labels(y))


def get_dataloader_workers():
    """使用四个进程读取数据"""
    return 4


def softmax(X):
    # 这里减去max有助于训练，防止梯度爆炸的问题
    max, _ = torch.max(X, dim=1, keepdim=True)
    X_exp = torch.exp(X - max)
    # 输入是一个784维的行向量拼接起来的
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    # 一般的交叉熵损失, 这里和pytorch的交叉熵损失是有区别的
    # range(len(y_hat))是索引到y中的值，进一步索引到y_hat
    # 交叉熵损失前面一定要加上-号，否则会导致梯度爆炸
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # 要进行类型转换才能进行比较
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在n个变量上实现累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        a = 0

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    # 实现两个变量进行累加
    # 所以这里的metric有两个
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # y.numel()返回数组元素的个数
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            # 因为计算梯度时是通过整个batch_size计算的
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def updater(batch_size):
    return sgd([W, b], lr, batch_size)


def train(net, train_iter, test_iter, loss, epochs, updater):
    for epoch in range(epochs):
        train_metric = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch:{epoch+1}, train loss:{train_metric[0]:f}, train_acc:{train_metric[1]:f}")


# if __name__ == '__main__':
#     # 将图像的数据转化为torch.float32类型
#     trans = transforms.ToTensor()
#     # torchvision是torch中的一个计算机视觉的包
#     mnist_train = torchvision.datasets.FashionMNIST(root="./data",
#                                                     train=True, transform=trans,
#                                                     download=False)
#     mnist_test = torchvision.datasets.FashionMNIST(root="./data",
#                                                    train=False, transform=trans,
#                                                    download=False)
#     train_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True,
#                                  num_workers=get_dataloader_workers())
#     test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False
#                                 , num_workers=get_dataloader_workers())
#     batch_size = 256
#     num_input = 28*28
#     num_output = 10
#     W = torch.normal(1, 0.1, size=(num_input, num_output), requires_grad=True)
#     b = torch.ones(num_output, requires_grad=True)
#     lr = 0.01
#     train(net, train_iter, test_iter, cross_entropy, 10, updater)


# softmax简洁实现
# def init_weights(m):
#     if type(m) == nn.Linear:
#         # 生成正态分布的权重，均值为默认值，方差为0.01
#         nn.init.normal_(m.weight, mean=1, std=0.01)
#
#
# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# net.apply(init_weights)
# pytorch的交叉熵损失自带softmax
# loss = nn.CrossEntropyLoss(reduction='none')
# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# # a = torch.tensor([0.1])
# # b = torch.tensor([0.1])
# # c = torch.tensor([0.8])
# # print(loss(torch.tensor([[0.1, 0.1, 0.8]]), torch.tensor([2], dtype=torch.long)))
# # print(torch.log(torch.exp(a)+torch.exp(b)+torch.exp(c))-0.8)
#
#
# def accuracy(y_hat, y):
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = y_hat.argmax(axis=1)
#     # 要进行类型转换才能进行比较
#     cmp = y_hat.type(y.dtype) == y
#     return float(cmp.type(y.dtype).sum())
#
#
# def evaluate_accuracy(net, data_iter):
#     if isinstance(net, torch.nn.Module):
#         net.eval()
#     # 实现两个变量进行累加
#     # 所以这里的metric有两个
#     metric = Accumulator(2)
#     with torch.no_grad():
#         for X, y in data_iter:
#             # y.numel()返回数组元素的个数
#             metric.add(accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]
#
#
class Accumulator:
    """在n个变量上实现累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]
#
#
# def train_epoch(net, train_iter, loss, updater):
#     if isinstance(net, torch.nn.Module):
#         net.train()
#     metric = Accumulator(3)
#     for X, y in train_iter:
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         if isinstance(updater, torch.optim.Optimizer):
#             updater.zero_grad()
#             l.mean().backward()
#             updater.step()
#         else:
#             l.sum().backward()
#             updater(X.shape[0])
#         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#     return metric[0] / metric[2], metric[1] / metric[2]
#
#
# def train(net, train_iter, test_iter, loss, epochs, updater):
#     for epoch in range(epochs):
#         train_metric = train_epoch(net, train_iter, loss, updater)
#         test_acc = evaluate_accuracy(net, test_iter)
#         print(f"epoch:{epoch+1}, train loss:{train_metric[0]:f}, train_acc:{train_metric[1]:f}")
#
#
# epochs = 8
# mnist_train = torchvision.datasets.FashionMNIST(root="./data",
#                                                     train=True, transform=transforms.ToTensor(),
#                                                     download=False)
# mnist_test = torchvision.datasets.FashionMNIST(root="./data",
#                                                    train=False, transform=transforms.ToTensor(),
#                                                    download=False)
# train_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True,
#                                  num_workers=0)
# test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False
#                                 , num_workers=0)
# train(net, train_iter=train_iter, test_iter=test_iter, loss=loss, epochs=epochs, updater=optimizer)
