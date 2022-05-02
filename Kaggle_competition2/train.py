# 数据获取地址：https://www.kaggle.com/c/classify-leaves
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import LeavesDataset
from ResNet18 import resnet_block
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.nn import functional as F


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

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def train_by_gpu(net, train_iter, epochs, lr, device):
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
            # 防止这里除以0
            train_loss = metric[0] / (metric[1] + 1e-3)
            train_acc = metric[1] / metric[2]

        print(f"epoch {epoch + 1}, loss {train_loss:.3f}, train acc {train_acc:.3f}")


def get_submission_result(test_iter, net, device=torch.device("cuda")):
    predict = []
    if isinstance(net, nn.Module):
        net.eval()
        net.to(device)
    for X in test_iter:
        with torch.no_grad():
            X = X.to(device)
            y_hat = net(X)
            pred = y_hat.argmax(axis=1).cpu().numpy()
            predict.append(pred)
    return np.asarray(predict).flatten()


if __name__ == "__main__":
    train_dataframe = pd.read_csv('F:/Hands on deep learning/Kaggle_competition2/train.csv')
    # 查看关于数据的描述
    # print(train_dataframe.describe())

    # 依据字母进行排序，并查看类别
    leaves_labels = sorted(list(set(train_dataframe['label'])))
    n_classes = len(leaves_labels)
    print(n_classes)
    print(leaves_labels)

    # 对应类别给予对应的标签
    class_to_num = dict(zip(leaves_labels, range(n_classes)))
    print(class_to_num)
    # 将标签和类别进行反转，以便于测试时能够使用
    num_to_class = {v: k for k, v in class_to_num.items()}

    train_path = "F:/Hands on deep learning/Kaggle_competition2/train.csv"
    test_path = "F:/Hands on deep learning/Kaggle_competition2/test.csv"
    # 这里时因为train.csv和test.csv中都有images/0.jpg的名字了，所以传入的路径时这样子的
    img_path = "F:/Hands on deep learning/Kaggle_competition2/"

    train_dataset = LeavesDataset(train_path, img_path, mode='train')
    # val_dataset = LeavesDataset(train_path, img_path, mode='valid')
    test_dataset = LeavesDataset(test_path, img_path, mode='test')
    print(train_dataset)
    print(test_dataset)

    train_iter = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_iter = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))
    # 此模型为ResNet-18
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 176))

    train_by_gpu(net, epochs=30, train_iter=train_iter, lr=0.005, device=torch.device("cuda"))
    torch.save(net, "leaves_ResNet18")

    predict_result = get_submission_result(test_iter, net)
    print(predict_result)
    classes = []
    for num in predict_result:
        classes.append(num_to_class[num])
    np.save("leaves_class", classes)

