import os
import torch
import torchvision
from d2l import torch as d2l

# 类标签对应的像素
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
                [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# 与VOC_COLORMAP一一对应
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]


def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    # 训练集还是验证集
    txt_fname = os.path.join(voc_dir, "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt")
    mode = torchvision.io.image.ImageReadMode.RGB  # 图像的读取模式为RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()  # txt文件中存储的是各个图像的id
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, "JPEGImages", f"{fname}.jpg")))  # 读取图像
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, "SegmentationClass", f"{fname}.png"), mode))  # 读取图像对应的标签
    return features, labels


def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        # 相当于256进制，可以用十进制理解（例如三位为1 3 4，需要做变换变成134，1*100+3*10+4）
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype("int32")
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]  # 将每个点的RGB标签全部转化为单个标签


def voc_rand_crop(feature, label, height, width):
    """随机裁剪图像和标签"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))  # 设置裁剪的区域
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)  # 标签和特征都要做相同区域的裁剪
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """加载VOC数据集"""
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet数据集的参数
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train)
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)  # 过滤掉那些小于裁剪大小的图像
        self.colormap2label = voc_colormap2label()
        print("read" + str(len(self.features)) + "examples")

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if(
            img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1]
        )]

    def __getitem__(self, item):
        feature, label = voc_rand_crop(self.features[item], self.labels[item], *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = d2l.download_extract("voc2012", os.path.join("VOCdevkit", "VOC2012"))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


if __name__ == "__main__":
    d2l.DATA_HUB["voc2012"] = (d2l.DATA_URL + "VOCtrainval_11-May-2012.tar",
                               "4e443f8a2eca6b1dac8a6c57641b67dd40621a49")
    voc_dir = d2l.download_extract("voc2012", "VOCdevkit/VOC2012")
    crop_size = (320, 480)
    voc_train = VOCSegDataset(True, crop_size, voc_dir)
    voc_val = VOCSegDataset(False, crop_size, voc_dir)
    batch_size = 64
    # drop_last=True表示将不能整除batch_size后的其它数据丢弃
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True, drop_last=True)
    for X, y in train_iter:
        print(X.shape)
        print(y.shape)
        break


