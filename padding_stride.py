import torch
from torch import nn


# 输出的维度是向下取整的
X = torch.rand((8, 8)).reshape((1, 1, 8, 8))
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(conv2d(X).shape)
