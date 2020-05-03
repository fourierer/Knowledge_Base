此文档记录pytorch和tensorflow框架的学习笔记

### Pytorch

#### 一、基本数据类型

参考链接（https://blog.csdn.net/out_of_memory_error?t=1）

1.张量，Tensor

Pytorch里面最基本的操作对象就是Tensor(张量)，表示的其实就是一个多维矩阵，并有矩阵相关的运算操作。在操作上和numpy是对应的，但是和numpy唯一不同的就是，pytorch可以在GPU上运行加速，而numpy不可以，当然二者也可以相互转换。Tensor只需要调用cuda()函数就可以将其转化为能在GPU上运行的类型。

（1）定义：

```python
import torch

a = torc.Tensor([[1, 2], [3, 4], [5, 6]]) # 定义特定的Tensor
b = torch.zeros((3, 2)) # 定义3行2列的全0阵
c = torch.randn((3, 2))
d = torch.ones((3, 2))
```

注意：上述部分定义和下面的等价

```python
import torch
b = torch.zeros(3, 2) # 可以输入tuple，也可以直接输入数字
```

和numpy之间的转换：

```python
import torch
import numpy as np

a = troch.rand((3, 2)) # 定义Tensor
numpy_a = a.numpy() # Tensor转numpy

b = np.array([[1, 2], [3, 4], [5, 6]]) # 定义numpy数组
Tensor_b = torch.from_numpy(b) # numpy数组转Tensor

```



将Tensor转换为GPU上的数据类型：

```python
import torch

tmp = torch.randn((3, 2)) # 定义Tensor

if torch.cuda.is_available():
  inputs = tmp.cuda() # Tensor转换为GPU上的数据类型
```



2.变量，Variable

Variable类型数据功能更加强大，相当于是在Tensor外层套了一个壳子，这个壳子赋予了前向传播，反向传播，自动求导。



#### 二、搭建神经网络模型

1.全连接网络模型

```python
from torch import nn

class simpleNet(nn.Module):
    '''
    定义一个简单的三层全连接神经网络，每一层都是线性的
    '''
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Activation_Net(nn.Module):
    '''
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    '''
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        '''
        这里的Sequential()函数的功能是将网络的层组合到一起
        '''
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
      
class Batch_Net(nn.Module):
    '''
    在上面的Activation_Net的基础上，增加一个加快收敛速度的方法--标准化
    '''
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
```



2.一般的卷积网络模型

(1)nn.Conv2d()函数参数介绍：

in_channels:输入信号的通道；

out_channels:卷积后输出结果的通道数；

kernel_size:卷积核的形状，例如(3,2)表示$3*2$的卷积核，如果宽和高相同，可以只用一个数字表示；

stride:卷积每次移动的步长，默认为1；

padding:处理边界时填充0的数量，默认为0(不填充)；

dilation:采样间隔数量，默认为1，无间隔采样；

bias:为True时，添加偏置；



(2)nn.MaxPool2d()函数参数介绍：

kernel_size:最大池化操作时的窗口大小；

stride:最大池化操作时窗口移动的步长，默认值是kernel_size；

padding:输入的每条边隐式补0的数量；

dilation:用于控制窗口中元素步长的参数；

return_indices:如果等于True，在返回max pooling结果的同时返回最大值的索引；

ceil_mode:如果等于True，在计算输出大小时，将采用向上取整来代替默认的向下取整的方式；



(3)nn.BatchNorm1d与nn.BatchNorm2d，x.view()

一维的BatchNorm参数为总的节点个数，二维的BatchNorm参数为输入信号的通道数；

torch 中的Tensor操作x.view相当于numpy中的array的reshape操作，其中$x = x.view(x.size(0), -1)$中，$x.size(0)$的值是batch size，所以这句代码的意思是把前一行的输出信号$x$拉成batch size行的张量；



(6)模型代码：

```python
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3),
            nn.BatchNorm2d(25),
            nn.ReLu(inplace=True))
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(50*5*5,1024),
            nn.ReLU(inplace=True),
            nn.Liear(1024,128),
            nn.ReLU(inplace=True),
            nn.Liear(128,10))
        
def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
```



3.resnet模块



4.resnet-3D



5.ip-CSN





### Tensorflow











