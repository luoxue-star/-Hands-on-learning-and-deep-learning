import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader


train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def load_cifar10(is_train, augs, batch_size):
    """加载数据集"""
    dataset = torchvision.datasets.CIFAR10(root="../data",
                                           train=is_train, transform=augs, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return dataloader


def train_batch_ch13(net, X, y, loss, trainer, devices):
    """将代码加载到GPU，进行训练"""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    """训练并测试"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel="epoch", xlim=[1, num_epochs], ylim=[0, 1], legend=["train loss", "train acc",
                                                                                       "test acc"])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_epochs,
                             (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
    print(f"loss:{metric[0] / metric[2]:.3f}, train acc:{metric[1] / metric[3]:.3f}, test acc:{test_acc:.3f}")
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples / sec on" f"{str(devices)}")


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, device)


if __name__ == "__main__":
    batch_size = 64
    device = d2l.try_all_gpus()
    net = d2l.resnet18(10, 3)
    train_with_data_aug(train_augs, test_augs, net)



