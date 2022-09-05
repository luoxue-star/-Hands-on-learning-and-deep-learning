import torch
from d2l import torch as d2l
import os
import torchvision
import pandas as pd


def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract("banana-detection")
    csv_fname = os.path.join(data_dir, "bananas_train" if is_train
                             else "bananas_val", "label.csv")  # 训练集还是验证集
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index("img_name")
    images, targets = [], []  # 用于存储图像和目标位置（左上角和右下角四个数）
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(os.path.join(data_dir, "bananas_train" if is_train
                                                             else "bananas_val", "images", f"{img_name}")))
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256  # 除以图像的高宽（均为256）


class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print("read " + str(len(self.features)) + (f" training examples" if is_train else f"validation examples"))

    def __getitem__(self, item):
        return (self.features[item].float(), self.labels[item])

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False), batch_size)
    return train_iter, val_iter


if __name__ == "__main__":
    d2l.DATA_HUB["banana-detection"] = (d2l.DATA_URL + "banana-detection.zip",
                                        "5de26c8fce5ccdea9f91267273464dc968d20d72")
    batch_size, edge_size = 32, 256   # 边界框的大小
    train_iter, _ = load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
    axes = d2l.show_images(imgs, 2, 5, scale=2)
    for ax, label in zip(axes, batch[1][0:10]):
        d2l.show_bboxes(ax, [label[0][1:5]] * edge_size, colors=['w'])
        d2l.plt.show()





