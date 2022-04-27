import torch
from softmax_regression import Accumulator, accuracy
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
from LeNet import train_by_gpu, evaluate_accuracy_gpu


def load_data_fashion_minst(batch_size, resize=None):
    """下载fashion数据集"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # minst_train.data可以取得前面的一些数据
    mnist_train = torchvision.datasets.FashionMNIST(root="./data",
                                                    train=True, transform=trans,
                                                    download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data",
                                                   train=False, transform=trans,
                                                   download=False)
    train = data.DataLoader(mnist_train, batch_size=256, shuffle=True, )
    test = data.DataLoader(mnist_test, batch_size=256, shuffle=False)
    return train, test


net_byimgnet = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
    nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
    nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 100)
)
# net_byminst = nn.Sequential(
#     nn.Conv2d(1, 96, kernel_size=(3, 3), stride=(3, 3), padding=1), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
#     nn.Conv2d(96, 256, kernel_size=(3, 3), padding=1), nn.ReLU(),
#     nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
#     nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
#     nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
#     nn.Flatten(),
#     nn.Linear(256*3*3, 256*3*2), nn.ReLU(), nn.Dropout(p=0.5),
#     nn.Linear(256*3*2, 256*2), nn.ReLU(), nn.Dropout(p=0.5),
#     nn.Linear(256*2, 10)
# )
# train_iter, test_iter = load_data_fashion_minst(64, 224)
# lr, epochs = 0.01, 10
# print("train on AlexNet")
# train_by_gpu(net_byimgnet, train_iter, test_iter, epochs, lr,
#              device=torch.device("cuda")
#              )
