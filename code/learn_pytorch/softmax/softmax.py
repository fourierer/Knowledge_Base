##################numpy版本#######################
import numpy as np

x = np.array([[1,1,0,0],[1,1,4,1],[1,0,2000,1]]) # 2000是为了测试防溢出机制问题
mask = np.array([[1,1,0,0],[1,0,1,0],[0,0,1,1]])
# x_trans = np.transpose(x) # x的转置，这里使用x.reshape(x.shape[1], x.shape[0])是不行的
# print(x_trans)


# softmax的一般写法
x_exp = np.exp(x) # 逐个元素计算
x_exp = x_exp*(mask==1) # 加入mask信息
# x_sum = np.sum(x_exp, axis=0) # 按列加
x_sum = np.sum(x_exp, axis=1) # 按行加
# print(x_sum.reshape(x.shape[0], 1))
output = x_exp/x_sum.reshape(x.shape[0], 1)
print(output)


x_max = np.max(x, axis=1) # 按行取最大值
x_max = x - x_max.reshape(x.shape[0], 1) # 增加防止溢出机制

x_exp = np.exp(x_max) # 逐个元素计算
x_exp = x_exp*(mask==1) # 加入mask信息
x_sum = np.sum(x_exp, axis=1) # 按行加
# print(x_sum.reshape(x.shape[0], 1))
output = x_exp/x_sum.reshape(x.shape[0], 1)
print(output)


#################Tensor版本的##########################
import torch

x = torch.Tensor([[1,1,0,0],[1,1,4,1],[1,0,2000,1]])
mask = torch.Tensor([[1,1,0,0],[1,0,1,0],[0,0,1,1]])
# x_trans = x.t() # x的转置，这里使用x.view(x.size(1),x.size(0))是不行的
# print(x_trans)

x_max = torch.max(x, dim=1)[0] # 第0个元素是dim=1上的最大值，第1个元素是最大值对应的索引
x_max = x - x_max.view(x.size(0), 1) # 或者x = x - x_max.reshape(x.size(0), 1)

x_exp = torch.exp(x_max)
# print(type(mask==1))
x_exp = x_exp*(mask==1).float() # 这里需要转换成float()
x_sum = torch.sum(x_exp, dim=1)
# print(x_sum.view(x.size(0), 1))
output = x_exp/x_sum.view(x.size(0), 1)
print(output)


##############torch自带softmax函数版本#################
import torch.nn as nn

x = torch.Tensor([[1,1,0,0],[1,1,4,1],[1,0,2000,1]])
softmax = nn.Softmax(dim=1)
# print(type(softmax)) # <class 'torch.nn.modules.activation.Softmax'>
output = softmax(x) # output = nn.Softmax(x, dim=1)这样直接计算是不对的，需要先实例化一个类对象
print(output)

