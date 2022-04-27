import torch
from softmax_regression import Accumulator, accuracy
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2),
    nn.Conv2d(6, 16, kernel_size=(5, 5)), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2), nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2),
#     nn.Conv2d(6, 16, kernel_size=(5, 5)), nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2),
#     nn.Flatten(),
#     nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
#     nn.Linear(120, 84), nn.ReLU(),
#     nn.Linear(84, 10))
mnist_train = torchvision.datasets.FashionMNIST(root="./data",
                                                train=True, transform=transforms.ToTensor(),
                                                download=False)
mnist_test = torchvision.datasets.FashionMNIST(root="./data",
                                               train=False, transform=transforms.ToTensor(),
                                               download=False)
train_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True)
test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_by_gpu(net, train_iter, test_iter, epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print("training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[1]
            train_acc = metric[1] / metric[2]

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f"epoch {epoch+1}, loss {train_loss:.3f}, train acc {train_acc:.3f},"
              f"test acc {test_acc:.3f}")


# lr, epochs = 1.0, 10
# print("train on LeNet")
# train_by_gpu(net, train_iter, test_iter, epochs, lr,
#              device=torch.device("cuda")
#              )

