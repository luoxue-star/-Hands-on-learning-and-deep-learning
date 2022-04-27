import softmax_regression
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data

# 多层感知机从0实现
# batch_size = 256
# train = torchvision.datasets.FashionMNIST(root='./data', transform=transforms.ToTensor(),
#                                                download=False, train=True)
# test = torchvision.datasets.FashionMNIST(root="./data", transform=transforms.ToTensor(),
#                                               download=False, train=False)
# train_iter = data.DataLoader(train, batch_size=batch_size, shuffle=True)
# test_iter = data.DataLoader(test, batch_size=batch_size, shuffle=False)
# num_inputs, num_outputs, num_hiddens = 784, 10, 256
# W1 = nn.Parameter(torch.randn((num_inputs, num_hiddens), requires_grad=True))
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# W2 = nn.Parameter(torch.randn((num_hiddens, num_outputs), requires_grad=True))
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
#
#
# def relu(X):
#     a = torch.zeros_like(X)
#     return torch.max(X, a)
#
#
# def net(X):
#     X = X.reshape((-1, num_inputs))
#     H = relu(X @ W1 + b1)
#     return H @ W2 + b2
#
#
# loss = nn.CrossEntropyLoss()
# updater = torch.optim.SGD([W1, b1, W2, b2], lr=0.01)
# epochs = 10
# softmax_regression.train(net, train_iter, test_iter, loss, epochs=epochs, updater=updater)
#
#
# def predict(net, test_iter, n=6):
#     for X, y in test_iter:
#         break
#     trues = softmax_regression.get_fashion_minst_labels(y)
#     preds = softmax_regression.get_fashion_minst_labels((net(X).argmax(axis=1)))
#     titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
#     print(titles)
#     softmax_regression.show_image(X[:n].reshape((n, 28, 28)), 1, n, titles=titles[:n])
#
#
# predict(net, test_iter)


# 多层感知机的简洁实现
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def init_weights(m):
    if m == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


#
net.apply(init_weights)
batch_size = 256
epochs = 10
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
train = torchvision.datasets.FashionMNIST(root='./data', transform=transforms.ToTensor(),
                                               download=False, train=True)
test = torchvision.datasets.FashionMNIST(root="./data", transform=transforms.ToTensor(),
                                              download=False, train=False)
train_iter = data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(test, batch_size=batch_size, shuffle=False)
softmax_regression.train(net, train_iter, test_iter, loss, epochs, optimizer)
# 结果可以明显看出来，简洁实现的正确率相比于从0实现的正确率高，原因可能是初始化权重的问题
# 当然也有可能跟学习率存在一定的关系
