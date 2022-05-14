import numpy as np
import torch
import os
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# 展示图片
# hotdogs = [train_imgs[i][0] for i in range(8)]
# not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.5)


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    """
    训练
    :param net:
    :param learning_rate:
    :param batch_size:
    :param num_epochs:
    :param param_group:若为True,最后一层输出层会使用10倍学习率
    :return:
    """
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")

    # 最后一层是否使用10倍学习率
    if param_group:
        param_1x = [param for name, param in net.named_parameters()
                    if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.Adam([
            {"params": param_1x}, {"params": net.fc.parameters(), "lr": learning_rate * 10}],
            lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)

    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


class Hotdogdataset(Dataset):
    """重写Dataset"""
    def __init__(self, root_path, augs):
        self.root_path = root_path
        self.hotdog_path = os.listdir(self.root_path + "\\hotdog")
        self.hotdog_path = ["\\hotdog\\" + name for name in self.hotdog_path]
        self.nothotdog_path = os.listdir(self.root_path + "\\not-hotdog")
        self.nothotdog_path = ["\\not-hotdog\\" + name for name in self.nothotdog_path]
        self.name = np.concatenate((np.array(self.hotdog_path), np.array(self.nothotdog_path))).ravel()
        self.augs = augs

    def __getitem__(self, item):
        img_name = self.name[item]
        if "not-hotdog" in img_name:
            label = 1
        else:
            label = 0
        img = Image.open(self.root_path + img_name)
        img = self.augs(img)
        return img, label

    def __len__(self):
        return len(self.name)


if __name__ == "__main__":
    # 使用imagenet的三通道的均值和标准差
    normalize = torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.2224, 0.225]
    )
    # 224是imagenet的图片尺寸
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    train_dataset = Hotdogdataset("F:\\Hands on deep learning\\ch13\\hotdog\\train", augs=train_augs)
    train_iter = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = Hotdogdataset("F:\\Hands on deep learning\\ch13\\hotdog\\test", augs=test_augs)
    test_iter = DataLoader(test_dataset, batch_size=64, shuffle=False)

    finetune_net = torchvision.models.resnet18(pretrained=True)
    # 最后一层重新初始化
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    # 使用kai_ming初始化
    nn.init.kaiming_uniform_(finetune_net.fc.weight)

    train_fine_tuning(finetune_net, 5e-5)

