import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.BatchNorm1d(3), nn.Linear(3,1))
print(net)
x = torch.rand(2,4)
y = net(x).sum()

print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size()) # 有层数索引

# 对于使用Sequential类构造的神经网络，我们可以通过方括号[]来访问网络的任一层。
# 索引0表示隐藏层为Sequential实例最先添加的层。

for name, param in net[0].named_parameters():
    print(name, param.size()) # 在第一层内部输出，没有层数索引
