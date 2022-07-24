# construct a linear regression model
# y = w1*x1 + w2*x2 + b

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import _init_path # 导入regression下dataset，model等包
import dataset
# import dataset.tensordataset as td
import model


# 设置超参数
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

# 创建数据集
# features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
features = torch.Tensor(np.random.normal(0, 1, (num_examples, num_inputs)))
# print(type(features)) # <class 'torch.Tensor'>
labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b
labels += torch.Tensor(np.random.normal(0, 0.01, size=labels.size())) # 添加扰动



# 将训练数据的数据和标签组合并打包
# data = torch.utils.data.TensorDataset(features, labels) # 可以使用pytorch自带的加载数据类
# data = td.TensorDataset(features, labels)
data = dataset.TensorDataset(features, labels)
# print(len(dataset)) # 1000
batch_size = 10
data_iter = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
# print(len(data_iter)) # 100

# 测试数据是否加载正确
# for x,y in data_iter:
#     print(x,y)
#     break

# 定义网络
net = model.LinearNet(num_inputs)
# print(net)
# 查看初始参数
for param in net.parameters():
    print(param)

# 定义loss
loss = nn.MSELoss()

# 定义优化算法
lr = 0.03
optimizer = optim.SGD(net.parameters(), lr)
def adjust_learning_rate(optimizer, epochs, lr):
    lr = lr * (0.1**(epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    adjust_learning_rate(optimizer, epoch, lr)
    for x, y in data_iter:
        output = net(x)
        l = loss(output, y.view(-1, 1)) # 需要将y转换成二维的tensor
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

# 输出参数验证
for name, param in net.named_parameters():
    print(name, param)
