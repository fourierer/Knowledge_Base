# construct a linear regression model
# y = w1*x1 + w2*x2 + b

import numpy as np
import torch
import torch.utils.data


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


# 创建适用于加载当前数据的类
class TensorDataset:
    def __init__(self, fearures, labels)


# 将训练数据的数据和标签组合
# dataset = torch.utils.data.TensorDataset(features, labels) # 可以使用pytorch自带的加载数据类
dataset = 
print(len(dataset))


