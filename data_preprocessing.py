# 数据获取地址：https://www.kaggle.com/c/classify-leaves
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.nn import functional as F
from softmax_regression import Accumulator, accuracy


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


class LeavesDataset(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0, resize_height=224, resize_width=224):
        """
        Args:
            csv_path (string): csv 文件路径
            file_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        # 调整后的照片尺寸
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(
                self.data_info.iloc[1:self.train_len+1, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len+1, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len+1:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len+1:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的normalize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


train_path = "F:/Hands on deep learning/Kaggle_competition2/train.csv"
test_path = "F:/Hands on deep learning/Kaggle_competition2/test.csv"
# 这里时因为train.csv和test.csv中都有images/0.jpg的名字了，所以传入的路径时这样子的
img_path = "F:/Hands on deep learning/Kaggle_competition2/"

train_dataset = LeavesDataset(train_path, img_path, mode='train')
# val_dataset = LeavesDataset(train_path, img_path, mode='valid')
test_dataset = LeavesDataset(test_path, img_path, mode='test')
print(train_dataset)
print(test_dataset)


train_iter = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
test_iter = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


# def im_convert(tensor):
#     """ 展示数据"""
#     image = tensor.to("cpu").clone().detach()
#     image = image.numpy().squeeze()
#     image = image.transpose(1, 2, 0)
#     image = image.clip(0, 1)
#
#     return image
#
#
# fig = plt.figure(figsize=(20, 12))
# columns = 4
# rows = 2
#
# dataiter = iter(train_iter)
# inputs, classes = dataiter.next()
#
# for idx in range(columns * rows):
#     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
#     ax.set_title(num_to_class[int(classes[idx])])
#     plt.imshow(im_convert(inputs[idx]))
# plt.show()


class Residual(nn.Module):
    """
    use_1x1conv：是否增加通道的数量
    """
    def __init__(self, in_channels, num_channels, use_1x1conv=False, strides=(1, 1)):
        super().__init__()
        # 不改变高宽，为了能够和后面直接传入的残差相加
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=(3, 3), padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=(1, 1), stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # Y = F.relu(self.bn1(self.conv1(X)))
        # Y = self.bn2(self.conv2(Y))
        # if self.conv3:
        #     X = self.conv3(X)
        # Y += X
        # return F.relu(Y)

        # 改进版
        Y = self.conv1(F.relu(self.bn3(X)))
        Y = F.relu(self.bn4(Y))
        if self.conv3:
            X = self.conv3(X)
        # 这里不能使用Y+=X，会报错，也就是需要为Y重新申请内存
        Y = Y + X
        return self.conv2(Y)


b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1))


def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        # 当为残差块的第一层且不为b1后面的一层时，会改变通道数
        # 当为b1后面一层时，由于b1已经改变了高和宽，所以这里不改变高和宽了
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=(2, 2)))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# 此模型为ResNet-18
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 176))


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
            train_loss = metric[0] / metric[1]
            train_acc = metric[1] / metric[2]

        print(f"epoch {epoch+1}, loss {train_loss:.3f}, train acc {train_acc:.3f}")


train_by_gpu(net, epochs=50, train_iter=train_iter, lr=0.005, device=torch.device("cuda"))
