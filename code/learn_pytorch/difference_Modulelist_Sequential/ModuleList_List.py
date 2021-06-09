import torch
import torch.nn as nn

class Net1(nn.Module):
    def __init__(self, in_dim, depth, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([nn.Linear(in_dim, out_dim)]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Net2(nn.Module):
    def __init__(self, in_dim, depth, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([nn.Linear(in_dim, out_dim), nn.Linear(in_dim, out_dim)]))
    
    def forward(self, x):
        # 使用layer1,layer2同时迭代，代表layer1与layer2组成self.layers内部的一个nn.ModuleList,且layer1和layer2分别代表list中的一个
        for layer1, layer2 in self.layers:
            x = layer1(x)
            x = layer2(x)
        return x

if __name__=='__main__':
    model = Net2(100, 10, 100) # 使用Net1初始化，下面y=model(x)会报错
    print(model.layers)
    x = torch.Tensor(100)
    y = model(x)
    print(y.size())

    # 原理和下面代码一样
    # a = [[1,2], [3,4]]
    # for i, j in a:
        # print(i,j)

