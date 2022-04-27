import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import softmax_regression


# # 从零实现
# def dropout_layer(X, dropout):
#     assert 0 <= dropout <= 1
#     if dropout == 1:
#         return torch.zeros_like(X)
#     elif dropout == 0:
#         return X
#     # 产生一个均匀分布
#     mask = (torch.rand(X.shape) > dropout).float()
#     return mask * X / (1 - dropout)
#
#
# num_inputs, num_output, num_hidden1, num_hidden2 = 784, 10, 256, 256
#
#
# class Net(nn.Module):
#     def __init__(self, num_inputs, num_output, num_hidden1, num_hidden2, is_training=True):
#         super(Net, self).__init__()
#         self.num_inputs = num_inputs
#         self.num_outputs = num_output
#         self.num_hidden1 = num_hidden1
#         self.num_hidden2 = num_hidden2
#         self.lin1 = nn.Linear(self.num_inputs, self.num_hidden1)
#         self.lin2 = nn.Linear(self.num_hidden1, self.num_hidden2)
#         self.lin3 = nn.Linear(self.num_hidden2, num_output)
#         self.relu = nn.ReLU()
#         self.training = is_training
#
#     def forward(self, X):
#         H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs)))
#         if self.training:
#             H1 = dropout_layer(H1, dropout=0.5)
#         H2 = self.relu(self.lin2(H1))
#         if self.training:
#             H2 = dropout_layer(H2, dropout=0.3)
#         out = self.lin3(H2)
#         return out
#
#
# epochs, lr, batch_size = 10, 0.1, 256
# loss = nn.CrossEntropyLoss(reduction='none')
# mnist_train = torchvision.datasets.FashionMNIST(root="./data",
#                                                 train=True, transform=transforms.ToTensor(),
#                                                 download=False)
# mnist_test = torchvision.datasets.FashionMNIST(root="./data",
#                                                train=False, transform=transforms.ToTensor(),
#                                                download=False)
# train_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True)
# test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False)
# net = Net(num_inputs, num_output, num_hidden1, num_hidden2)
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
# softmax_regression.train(net, train_iter, test_iter, loss, epochs, optimizer)


# 简洁实现
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Dropout(0), nn.Linear(256, 128), nn.ReLU(),
                    nn.Dropout(0), nn.Linear(128, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# 这种初始化参数的方法很好用
def xavier_init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=1)


net.apply(xavier_init_weight)
epochs, lr, batch_size = 10, 0.1, 256
loss = nn.CrossEntropyLoss(reduction='none')
mnist_train = torchvision.datasets.FashionMNIST(root="./data",
                                                train=True, transform=transforms.ToTensor(),
                                                download=False)
mnist_test = torchvision.datasets.FashionMNIST(root="./data",
                                               train=False, transform=transforms.ToTensor(),
                                               download=False)
train_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
softmax_regression.train(net, train_iter, test_iter, loss, epochs, optimizer)





