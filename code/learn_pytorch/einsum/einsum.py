import torch

# a = torch.rand(2,3)
a = torch.arange(6).reshape(2, 3)
b = torch.arange(3)
print(a)
print(b)


# 矩阵转置
# a1_{ji} = a_{ij}
a1 = torch.einsum('ij->ji', [a])
print(a1)

# 所有元素求和
# a2 = sum_{i}sum{j}a_{ij}
a2 = torch.einsum('ij->', [a])
print(a2)

# 列求和
# a3_{j} = sum_{i}a_{ij}
a3 = torch.einsum('ij->j', [a])
print(a3)

# 行求和
# a4_{i} = sum_{j}a_{ij}
a4 = torch.einsum('ij->i', [a])
print(a4)

# 矩阵-向量相乘
# c1_{i} = sum_{k}A_{ik}b_{k}
c1 = torch.einsum('ik, k->i', [a, b]) # A的下标是ik，b的下标是k，最终得到c的下标为i
print(c1)

# 矩阵-矩阵相乘
# c2_{ij} = sum_{k}A_{ik}B_{kj}
b = torch.arange(15).reshape(3,5)
c2 = torch.einsum('ik, kj->ij', [a, b])
print(c2)

# 点积-向量
# c3 = sum_{i}a_{i}b_{i}
a = torch.arange(3)
b = torch.arange(3,6)  # [3, 4, 5]
c3 = torch.einsum('i, i->', [a, b])
print(c3)

# 点积-矩阵
# c4 = sum_{i}sum_{j}A_{ij}B_{ij}
a = torch.arange(6).reshape(2, 3)
b = torch.arange(6,12).reshape(2, 3)
c4 = torch.einsum('ij, ij->', [a, b])
print(c4)

# 哈达玛积
# c5_{ij} = A_{ij}B_{ij}
a = torch.arange(6).reshape(2, 3)
b = torch.arange(6,12).reshape(2, 3)
c5 = torch.einsum('ij, ij->ij', [a, b])
print(c5)

# 外积
# c6_{ij} = a_{i}b_{j}
a = torch.arange(3)
b = torch.arange(3,7)
c6 = torch.einsum('i, j->ij', [a, b])
print(c6)

# batch矩阵相乘
# C_{ijl} = sum_{k}A_{ijk}B_{ikl}
a = torch.randn(3,2,5)
b = torch.randn(3,5,3)
C1 = torch.einsum('ijk, ikl->ijl', [a, b])
print(C1)

# 张量缩约
# C_{pstuv} = sum_{q}sum_{r}A_{pqrs}B_{tuqvr}
a = torch.randn(2,3,5,7)
b = torch.randn(11,13,3,17,5)
C2 = torch.einsum('pqrs, tuqvr->pstuv', [a, b])
print(C2.size()) # torch.Size([2, 7, 11, 13, 17])

