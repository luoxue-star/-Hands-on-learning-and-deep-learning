import torch


# 张量算法的基本性质
# a = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# # 通过分配新的内存
# b = a.clone()
# print(id(a))
# print(id(b))


# 降维
# x = torch.arange(4, dtype=torch.float32).reshape(2, 2)
# # axis的值为多少，就是将这个维度压扁
# x0_sum = x.sum(axis=0)
# x1_sum = x.sum(axis=1)
# print(x)
# print(x0_sum)
# print(x1_sum)

# 非降维求和
# x = torch.arange(4, dtype=torch.float32).reshape(2, 2)
# # 保持行的维度不变
# print(x.sum(axis=0, keepdims=True))
# # 保持列的维度不变
# print(x.sum(axis=1, keepdims=True))

# 点积
# x = torch.arange(4, dtype=torch.float32)
# y = torch.arange(4, dtype=torch.float32)
# print(x.dot(y))
# print(torch.dot(x, y))

# 矩阵-向量积
# x = torch.arange(2, dtype=torch.float32)
# A = torch.arange(4, dtype=torch.float32).reshape(2, 2)
# print(torch.mv(A, x))

# 矩阵-矩阵乘法
# A = torch.ones(4, 3)
# B = torch.ones(3, 4)
# print(torch.mm(A, B))

# 范数
# u = torch.ones(1, 3)
# # 二范数
# print(torch.norm(u))
# # 一范数
# print(torch.abs(u).sum())

# 三维张量的范数也是一样的
a = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1)
print(a)
print(torch.linalg.norm(a))
print(torch.sqrt(torch.tensor(5)))
