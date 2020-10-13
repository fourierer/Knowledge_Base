此文档记录pytorch和tensorflow框架的学习笔记

### Pytorch

#### 一、基本数据类型

参考链接（https://blog.csdn.net/out_of_memory_error?t=1）

1.张量，Tensor

Pytorch里面最基本的操作对象就是Tensor(张量)，表示的其实就是一个多维矩阵，并有矩阵相关的运算操作。在操作上和numpy是对应的，但是和numpy唯一不同的就是，Tensor可以在GPU上运行加速，而numpy不可以，当然二者也可以相互转换。Tensor只需要调用cuda()函数就可以将其转化为能在GPU上运行的类型。

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

（2）张量的拆分：torch.split()和torch.chunk()

1)torch.split(size,dim=0)

```python
>>> x = torch.randn(3, 10, 6)
>>> a, b, c = x.split(1, 0) # 在0维进行间隔为1的拆分
>>> a.size(), b.size(), c.size()
(torch.Size([1, 10, 6]), torch.Size([1, 10, 6]), torch.Size([1, 10, 6]))
>>> d, e = x.split(2, 0) # 在0维进行间隔为2的拆分
>>> d.size(), e.size()
(torch.Size([2, 10, 6]), torch.Size([1, 10, 6]))
```

在维度0方向上，每隔size进行拆分，剩下的不足size就作为最后一个；

2)torch.chunk(size,dim=0)

```python
>>> l, m, n = x.chunk(3, 0) # 在0维上拆分成3份
>>> l.size(), m.size(), n.size()
(torch.Size([1, 10, 6]), torch.Size([1, 10, 6]), torch.Size([1, 10, 6]))
>>> u, v = x.chunk(2, 0) # 在0维上拆分成2份
>>> u.size(), v.size()
(torch.Size([2, 10, 6]), torch.Size([1, 10, 6]))
```

把张量在0维度上拆分成3部分时，因为尺寸正好为3，所以每个分块的间隔相等，都为 1。

把张量在0维度上拆分成2部分时，无法平均分配，以上面的结果来看，可以看成是，用0维度的尺寸除以需要拆分的份数，把余数作为最后一个分块的间隔大小，再把前面的分块以相同的间隔拆分。



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
    在上面的Activation_Net的基础上，增加一个加快收敛速度的方法--归一化
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

**注：nn.ReLU() 和 nn.ReLU(inplace=True)对计算结果不会有影响。利用inplace计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用。**



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



(4)卷积神经网络模型代码：

```python
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=3),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True))
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



3.ResNet网络

ResNet中有两个基本的block，一个是$3*3$+$3*3$，称为basic block；另一个是$1*1$+$3*3$+$1*1$，称为bottleneck block。这里先写两个残差结构block，然后再搭建整个网络：

先给出卷积输入和输出尺寸大小公式：假设输入信号尺寸为$w*h$，卷积核大小为$f*f$，填充大小为$p$，步长为$s$，则输出尺寸$w'*h'$为：
$$
w'=\frac{w-f+2p}{s}+1\\
h'=\frac{h-f+2p}{s}+1
$$

给出pytorch中给的ResNet源码：


```python
import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
from torchsummaryX import summary

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

#卷积尺寸变化公式：w' = (w-f+2p)/s + 1

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
#s=1, p=1, f=3, w' = (w-3+2)/1 + 1 = w，该卷积不改变feature map空间尺寸，只改变通道数

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#s=1, p=0, f=1, w' = (w-1+0)/1 + 1 = w，该卷积同样不改变feature map空间尺寸，只改变通道数

class BasicBlock(nn.Module):
    expansion = 1 # 中间没有隐藏层，expansion是1，默认通道数不变

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # shortcut
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # [1, 2048, 7, 7]->[1, 2048, 1, 1]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 第一阶段self.inplanes=64，128 ！= 64*4，所以第一个block是带shortcut的bottleneck，
        # 随后self.inplanes变为64*4=256，以256->64->256的通道数构建bottleneck；
        # 第二阶段self.inplanes=256，256 ！= 128*4，所以第一个block是带shortcut的bottleneck，
        # 随后self.inplanes变为128*4=512，以512->128->512的通道数构建bottleneck；
        # 第三、四阶段以此类推...

        # 四个阶段第一个block都是带shortcut的bottleneck，不带shortcut的bottleneck的planes分别是64,128,256,512.
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # 再增加n-1个不带shortcut的正常bottleneck，通道数4m->m->4m，m分别是64,128,256,512
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    #if pretrained:
        #state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        #model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


if __name__=='__main__':
    model = resnet50()
    print(model)
    summary(model, torch.zeros(1, 3, 224, 224))

```

输出结果为：

```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
(layer2): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
 (layer3): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
 (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
====================================================================================================
                                             Kernel Shape       Output Shape  \
Layer
0_conv1                                     [3, 64, 7, 7]  [1, 64, 112, 112]
1_bn1                                                [64]  [1, 64, 112, 112]
2_relu                                                  -  [1, 64, 112, 112]
3_maxpool                                               -    [1, 64, 56, 56]
4_layer1.0.Conv2d_conv1                    [64, 64, 1, 1]    [1, 64, 56, 56]
5_layer1.0.BatchNorm2d_bn1                           [64]    [1, 64, 56, 56]
6_layer1.0.ReLU_relu                                    -    [1, 64, 56, 56]
7_layer1.0.Conv2d_conv2                    [64, 64, 3, 3]    [1, 64, 56, 56]
8_layer1.0.BatchNorm2d_bn2                           [64]    [1, 64, 56, 56]
9_layer1.0.ReLU_relu                                    -    [1, 64, 56, 56]
10_layer1.0.Conv2d_conv3                  [64, 256, 1, 1]   [1, 256, 56, 56]
11_layer1.0.BatchNorm2d_bn3                         [256]   [1, 256, 56, 56]
12_layer1.0.downsample.Conv2d_0           [64, 256, 1, 1]   [1, 256, 56, 56]
13_layer1.0.downsample.BatchNorm2d_1                [256]   [1, 256, 56, 56]
14_layer1.0.ReLU_relu                                   -   [1, 256, 56, 56]
15_layer1.1.Conv2d_conv1                  [256, 64, 1, 1]    [1, 64, 56, 56]
16_layer1.1.BatchNorm2d_bn1                          [64]    [1, 64, 56, 56]
17_layer1.1.ReLU_relu                                   -    [1, 64, 56, 56]
18_layer1.1.Conv2d_conv2                   [64, 64, 3, 3]    [1, 64, 56, 56]
19_layer1.1.BatchNorm2d_bn2                          [64]    [1, 64, 56, 56]
20_layer1.1.ReLU_relu                                   -    [1, 64, 56, 56]
21_layer1.1.Conv2d_conv3                  [64, 256, 1, 1]   [1, 256, 56, 56]
22_layer1.1.BatchNorm2d_bn3                         [256]   [1, 256, 56, 56]
23_layer1.1.ReLU_relu                                   -   [1, 256, 56, 56]
24_layer1.2.Conv2d_conv1                  [256, 64, 1, 1]    [1, 64, 56, 56]
25_layer1.2.BatchNorm2d_bn1                          [64]    [1, 64, 56, 56]
26_layer1.2.ReLU_relu                                   -    [1, 64, 56, 56]
27_layer1.2.Conv2d_conv2                   [64, 64, 3, 3]    [1, 64, 56, 56]
28_layer1.2.BatchNorm2d_bn2                          [64]    [1, 64, 56, 56]
29_layer1.2.ReLU_relu                                   -    [1, 64, 56, 56]
30_layer1.2.Conv2d_conv3                  [64, 256, 1, 1]   [1, 256, 56, 56]
31_layer1.2.BatchNorm2d_bn3                         [256]   [1, 256, 56, 56]
32_layer1.2.ReLU_relu                                   -   [1, 256, 56, 56]
33_layer2.0.Conv2d_conv1                 [256, 128, 1, 1]   [1, 128, 56, 56]
34_layer2.0.BatchNorm2d_bn1                         [128]   [1, 128, 56, 56]
35_layer2.0.ReLU_relu                                   -   [1, 128, 56, 56]
36_layer2.0.Conv2d_conv2                 [128, 128, 3, 3]   [1, 128, 28, 28]
37_layer2.0.BatchNorm2d_bn2                         [128]   [1, 128, 28, 28]
38_layer2.0.ReLU_relu                                   -   [1, 128, 28, 28]
39_layer2.0.Conv2d_conv3                 [128, 512, 1, 1]   [1, 512, 28, 28]
40_layer2.0.BatchNorm2d_bn3                         [512]   [1, 512, 28, 28]
41_layer2.0.downsample.Conv2d_0          [256, 512, 1, 1]   [1, 512, 28, 28]
42_layer2.0.downsample.BatchNorm2d_1                [512]   [1, 512, 28, 28]
43_layer2.0.ReLU_relu                                   -   [1, 512, 28, 28]
44_layer2.1.Conv2d_conv1                 [512, 128, 1, 1]   [1, 128, 28, 28]
45_layer2.1.BatchNorm2d_bn1                         [128]   [1, 128, 28, 28]
46_layer2.1.ReLU_relu                                   -   [1, 128, 28, 28]
47_layer2.1.Conv2d_conv2                 [128, 128, 3, 3]   [1, 128, 28, 28]
48_layer2.1.BatchNorm2d_bn2                         [128]   [1, 128, 28, 28]
49_layer2.1.ReLU_relu                                   -   [1, 128, 28, 28]
50_layer2.1.Conv2d_conv3                 [128, 512, 1, 1]   [1, 512, 28, 28]
51_layer2.1.BatchNorm2d_bn3                         [512]   [1, 512, 28, 28]
52_layer2.1.ReLU_relu                                   -   [1, 512, 28, 28]
53_layer2.2.Conv2d_conv1                 [512, 128, 1, 1]   [1, 128, 28, 28]
54_layer2.2.BatchNorm2d_bn1                         [128]   [1, 128, 28, 28]
55_layer2.2.ReLU_relu                                   -   [1, 128, 28, 28]
56_layer2.2.Conv2d_conv2                 [128, 128, 3, 3]   [1, 128, 28, 28]
57_layer2.2.BatchNorm2d_bn2                         [128]   [1, 128, 28, 28]
58_layer2.2.ReLU_relu                                   -   [1, 128, 28, 28]
59_layer2.2.Conv2d_conv3                 [128, 512, 1, 1]   [1, 512, 28, 28]
60_layer2.2.BatchNorm2d_bn3                         [512]   [1, 512, 28, 28]
61_layer2.2.ReLU_relu                                   -   [1, 512, 28, 28]
62_layer2.3.Conv2d_conv1                 [512, 128, 1, 1]   [1, 128, 28, 28]
63_layer2.3.BatchNorm2d_bn1                         [128]   [1, 128, 28, 28]
64_layer2.3.ReLU_relu                                   -   [1, 128, 28, 28]
65_layer2.3.Conv2d_conv2                 [128, 128, 3, 3]   [1, 128, 28, 28]
66_layer2.3.BatchNorm2d_bn2                         [128]   [1, 128, 28, 28]
67_layer2.3.ReLU_relu                                   -   [1, 128, 28, 28]
68_layer2.3.Conv2d_conv3                 [128, 512, 1, 1]   [1, 512, 28, 28]
69_layer2.3.BatchNorm2d_bn3                         [512]   [1, 512, 28, 28]
70_layer2.3.ReLU_relu                                   -   [1, 512, 28, 28]
71_layer3.0.Conv2d_conv1                 [512, 256, 1, 1]   [1, 256, 28, 28]
72_layer3.0.BatchNorm2d_bn1                         [256]   [1, 256, 28, 28]
73_layer3.0.ReLU_relu                                   -   [1, 256, 28, 28]
74_layer3.0.Conv2d_conv2                 [256, 256, 3, 3]   [1, 256, 14, 14]
75_layer3.0.BatchNorm2d_bn2                         [256]   [1, 256, 14, 14]
76_layer3.0.ReLU_relu                                   -   [1, 256, 14, 14]
77_layer3.0.Conv2d_conv3                [256, 1024, 1, 1]  [1, 1024, 14, 14]
78_layer3.0.BatchNorm2d_bn3                        [1024]  [1, 1024, 14, 14]
79_layer3.0.downsample.Conv2d_0         [512, 1024, 1, 1]  [1, 1024, 14, 14]
80_layer3.0.downsample.BatchNorm2d_1               [1024]  [1, 1024, 14, 14]
81_layer3.0.ReLU_relu                                   -  [1, 1024, 14, 14]
82_layer3.1.Conv2d_conv1                [1024, 256, 1, 1]   [1, 256, 14, 14]
83_layer3.1.BatchNorm2d_bn1                         [256]   [1, 256, 14, 14]
84_layer3.1.ReLU_relu                                   -   [1, 256, 14, 14]
85_layer3.1.Conv2d_conv2                 [256, 256, 3, 3]   [1, 256, 14, 14]
86_layer3.1.BatchNorm2d_bn2                         [256]   [1, 256, 14, 14]
87_layer3.1.ReLU_relu                                   -   [1, 256, 14, 14]
88_layer3.1.Conv2d_conv3                [256, 1024, 1, 1]  [1, 1024, 14, 14]
89_layer3.1.BatchNorm2d_bn3                        [1024]  [1, 1024, 14, 14]
90_layer3.1.ReLU_relu                                   -  [1, 1024, 14, 14]
91_layer3.2.Conv2d_conv1                [1024, 256, 1, 1]   [1, 256, 14, 14]
92_layer3.2.BatchNorm2d_bn1                         [256]   [1, 256, 14, 14]
93_layer3.2.ReLU_relu                                   -   [1, 256, 14, 14]
94_layer3.2.Conv2d_conv2                 [256, 256, 3, 3]   [1, 256, 14, 14]
95_layer3.2.BatchNorm2d_bn2                         [256]   [1, 256, 14, 14]
96_layer3.2.ReLU_relu                                   -   [1, 256, 14, 14]
97_layer3.2.Conv2d_conv3                [256, 1024, 1, 1]  [1, 1024, 14, 14]
98_layer3.2.BatchNorm2d_bn3                        [1024]  [1, 1024, 14, 14]
99_layer3.2.ReLU_relu                                   -  [1, 1024, 14, 14]
100_layer3.3.Conv2d_conv1               [1024, 256, 1, 1]   [1, 256, 14, 14]
101_layer3.3.BatchNorm2d_bn1                        [256]   [1, 256, 14, 14]
102_layer3.3.ReLU_relu                                  -   [1, 256, 14, 14]
103_layer3.3.Conv2d_conv2                [256, 256, 3, 3]   [1, 256, 14, 14]
104_layer3.3.BatchNorm2d_bn2                        [256]   [1, 256, 14, 14]
105_layer3.3.ReLU_relu                                  -   [1, 256, 14, 14]
106_layer3.3.Conv2d_conv3               [256, 1024, 1, 1]  [1, 1024, 14, 14]
107_layer3.3.BatchNorm2d_bn3                       [1024]  [1, 1024, 14, 14]
108_layer3.3.ReLU_relu                                  -  [1, 1024, 14, 14]
109_layer3.4.Conv2d_conv1               [1024, 256, 1, 1]   [1, 256, 14, 14]
110_layer3.4.BatchNorm2d_bn1                        [256]   [1, 256, 14, 14]
111_layer3.4.ReLU_relu                                  -   [1, 256, 14, 14]
112_layer3.4.Conv2d_conv2                [256, 256, 3, 3]   [1, 256, 14, 14]
113_layer3.4.BatchNorm2d_bn2                        [256]   [1, 256, 14, 14]
114_layer3.4.ReLU_relu                                  -   [1, 256, 14, 14]
115_layer3.4.Conv2d_conv3               [256, 1024, 1, 1]  [1, 1024, 14, 14]
116_layer3.4.BatchNorm2d_bn3                       [1024]  [1, 1024, 14, 14]
117_layer3.4.ReLU_relu                                  -  [1, 1024, 14, 14]
118_layer3.5.Conv2d_conv1               [1024, 256, 1, 1]   [1, 256, 14, 14]
119_layer3.5.BatchNorm2d_bn1                        [256]   [1, 256, 14, 14]
120_layer3.5.ReLU_relu                                  -   [1, 256, 14, 14]
121_layer3.5.Conv2d_conv2                [256, 256, 3, 3]   [1, 256, 14, 14]
122_layer3.5.BatchNorm2d_bn2                        [256]   [1, 256, 14, 14]
123_layer3.5.ReLU_relu                                  -   [1, 256, 14, 14]
124_layer3.5.Conv2d_conv3               [256, 1024, 1, 1]  [1, 1024, 14, 14]
125_layer3.5.BatchNorm2d_bn3                       [1024]  [1, 1024, 14, 14]
126_layer3.5.ReLU_relu                                  -  [1, 1024, 14, 14]
127_layer4.0.Conv2d_conv1               [1024, 512, 1, 1]   [1, 512, 14, 14]
128_layer4.0.BatchNorm2d_bn1                        [512]   [1, 512, 14, 14]
129_layer4.0.ReLU_relu                                  -   [1, 512, 14, 14]
130_layer4.0.Conv2d_conv2                [512, 512, 3, 3]     [1, 512, 7, 7]
131_layer4.0.BatchNorm2d_bn2                        [512]     [1, 512, 7, 7]
132_layer4.0.ReLU_relu                                  -     [1, 512, 7, 7]
133_layer4.0.Conv2d_conv3               [512, 2048, 1, 1]    [1, 2048, 7, 7]
134_layer4.0.BatchNorm2d_bn3                       [2048]    [1, 2048, 7, 7]
135_layer4.0.downsample.Conv2d_0       [1024, 2048, 1, 1]    [1, 2048, 7, 7]
136_layer4.0.downsample.BatchNorm2d_1              [2048]    [1, 2048, 7, 7]
137_layer4.0.ReLU_relu                                  -    [1, 2048, 7, 7]
138_layer4.1.Conv2d_conv1               [2048, 512, 1, 1]     [1, 512, 7, 7]
139_layer4.1.BatchNorm2d_bn1                        [512]     [1, 512, 7, 7]
140_layer4.1.ReLU_relu                                  -     [1, 512, 7, 7]
141_layer4.1.Conv2d_conv2                [512, 512, 3, 3]     [1, 512, 7, 7]
142_layer4.1.BatchNorm2d_bn2                        [512]     [1, 512, 7, 7]
143_layer4.1.ReLU_relu                                  -     [1, 512, 7, 7]
144_layer4.1.Conv2d_conv3               [512, 2048, 1, 1]    [1, 2048, 7, 7]
145_layer4.1.BatchNorm2d_bn3                       [2048]    [1, 2048, 7, 7]
146_layer4.1.ReLU_relu                                  -    [1, 2048, 7, 7]
147_layer4.2.Conv2d_conv1               [2048, 512, 1, 1]     [1, 512, 7, 7]
148_layer4.2.BatchNorm2d_bn1                        [512]     [1, 512, 7, 7]
149_layer4.2.ReLU_relu                                  -     [1, 512, 7, 7]
150_layer4.2.Conv2d_conv2                [512, 512, 3, 3]     [1, 512, 7, 7]
151_layer4.2.BatchNorm2d_bn2                        [512]     [1, 512, 7, 7]
152_layer4.2.ReLU_relu                                  -     [1, 512, 7, 7]
153_layer4.2.Conv2d_conv3               [512, 2048, 1, 1]    [1, 2048, 7, 7]
154_layer4.2.BatchNorm2d_bn3                       [2048]    [1, 2048, 7, 7]
155_layer4.2.ReLU_relu                                  -    [1, 2048, 7, 7]
156_avgpool                                             -    [1, 2048, 1, 1]
157_fc                                       [2048, 1000]          [1, 1000]
...具体参数信息略去
```



4.MobileNetV2

5.ShuffleNetV2

6.GhostNet

（4,5,6代码见repo：" https://github.com/fourierer/Learn_GhostNet_pytorch "）



6.使用训好的模型来测试单个图像和视频

（1）测试图像

```python
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from ghost_net import ghost_net
import time

device = torch.device('cuda')


# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),  # 改变图像大小，作为224*224的正方形
    transforms.CenterCrop(224),  # 以图像中心进行切割，参数只有一个要切成正方形>转
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])



# 加载模型
print('load model begin!')
model = ghost_net(width_mult=1.0)
checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model= model.to(device)
print('load model done!')



# 测试单张图像
img = Image.open('/home/sz/model_eval/panda.jpg')
img = data_transform(img)
#img = torch.Tensor(1,3,224,224) #如果只是测试时间，直接初始化一个Tensor即可
print(type(img))
print(img.shape)
img = img.unsqueeze(0) # 这里直接输入img不可，因为尺寸不一致，img为[3,224,224]的Tensor，而模型需要[1,3,224,224]的Tensor
print(type(img))
print(img.shape)

time_start = time.time()
img_= img.to(device)
outputs = model(img_)
time_end = time.time()
time_c = time_end - time_start
_, predicted = torch.max(outputs,1)
print('this picture maybe:' + str(predicted))
print('time cost:', time_c, 's')

'''
# 批量测试验证集中的图像，使用dataloader，可以更改batch_size调节测试速度
print('Test data load begin!')
test_dataset = torchvision.datasets.ImageFolder(root='/home/momo/mnt/data2/datum/raw/val2', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
print(type(test_data))
print('Test data load done!')

torch.no_grad()
for img1, label1 in test_data:
    img1 = img1.to(device)
    label1 = label1.to(device)
    out = model(img1)

    _, pred = out.max(1)
    print(pred)
    print(label1)
    num_correct = (pred == label1).sum().item()
    acc = num_correct / img1.shape[0]
    print('Test acc in current batch:' + str(acc))
    eval_acc +=acc

print('final acc in Test data:' + str(eval_acc / len(test_data)))
'''
```



#### 三、加载模型进行预训练

1.模型的保存和加载

在介绍模型保存和加载之前先介绍一段测试代码：

```python
from torchvision.models import resnet18
from torch.nn.parallel import DataParallel

model = resnet18() # 网络没有并行，当前默认在单张卡上，参数为单卡形式

print('###############################')
cnt = 0
for k in model.state_dict():
    if cnt>5:
        break
    print(k)
    cnt += 1
# 从输出结果来看参数中没有module
'''
上述代码如果改成下面这样，则会报错：
AttributeError: 'ResNet' object has no attribute 'modeule'
因为在模型没有
print('###############################')
cnt = 0
for k in model.modeule.state_dict():
    if cnt>5:
        break
    print(k)
    cnt += 1
'''

print('##############################')
model = DataParallel(model) # 网络并行，当前网络参数为多卡形式

cnt = 0
for k in model.state_dict():
    if cnt>5:
        break
    print(k)
    cnt += 1
# 从输出结果来看参数中有module

print('###############################')
cnt = 0
for k in model.module.state_dict():
    if cnt>5:
        break
    print(k)
    cnt += 1
#当for循环中已经指定module之后，参数输出结果又没有了module


#输出结果：
'''
###############################
conv1.weight
bn1.weight
bn1.bias
bn1.running_mean
bn1.running_var
bn1.num_batches_tracked
##############################
module.conv1.weight
module.bn1.weight
module.bn1.bias
module.bn1.running_mean
module.bn1.running_var
module.bn1.num_batches_tracked
###############################
conv1.weight
bn1.weight
bn1.bias
bn1.running_mean
bn1.running_var
bn1.num_batches_tracked
'''
```

模型的加载方法根据保存方式来定，如：

（1）下面代码是一般分类任务训练代码中常用的，模型会保存为一个tar包（一般tar包是一个字典，保存了模型参数以及训练的一些信息，如训练epoch数，学习率等），加载方法是：

```python
#模型保存
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
            }, is_best)

#模型加载
model = xxx_net()
checkpoint = torch.load('./xxx.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
```

简单改动一下save_chaeckpoint函数，模型的加载方式会改变：

```python
#模型保存
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state['state_dict'], filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
            }, is_best)
model = ghost_net()
model.load_state_dict(torch.load('./xxx.pth'))
```



（2）一般在训练的时候会使用并行工具，如测试的代码中的Dataparallel，使得网络参数变成了多卡形式，键包含module。上述（1）中代码在保存的时候使用了model.module.state_dict()，则使得保存的模型参数的键中就不再有module，加载的时候可以使用单卡去测试推理，如：

```python
model = xx_net()
# 加载时不用先将model并行化
checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model = model.to(device)
```

但如果训练时使用了并行工具，并且保存时保留了module，如：

```python
model = xxx_net()
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
model.cuda()
# 使用并行工具Dataparallel
model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])


#模型保存时保留了module，即state_dict键对应的是model.state_dict()
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
            }, is_best)

#此时再进行模型加载时，一定要将定义的model先进行并行化，再加载保存的模型
model = xxx_net()
gpus=[0,1,2,3]
model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
model.load_state(torch.load('model_best.pth'))
```





（3）加载预训练模型

现在模型训练基本上都会采用预训练的方式，在一个大的数据集上训好了模型参数，再加载到一个小的数据集上继续训练。在不同的任务中模型结构会不一样，比如分类数量发生改变。这里以分类问题为例，介绍加载模型进行预训练的方法，一般来说有三种写法：

**注：一般来说，训好的模型和当前模型结构上相差不大，只是在最后的分类层改变了分类数。这里需要注意的是，预训练的网络框架和当前模型训练的网络框架不能用同一份代结构码，需要做一点修改。**

**首先用代码GhostNet.py在ImageNet上训好了一个1000类的分类模型，此时最后一层的名称为self.classifer，则在保存的预训练模型字典当中，会存在一个键为self.classider，对应的值是这一层的参数。如果再接着用GhostNet.py这个脚本，则最后一层的名称还是self.classifer，但是由于任务不同，此时一般会更改这一层的分类数。这就导致模型加载的时候，最后一层的名称相同，但参数尺寸却不一样，这是无法对应加载参数的。解决方法是更改最后一层的名称，比如改为self.classifer1。在加载模型参数的时候，只加载名称相同的层对应的参数，最后一层名称对应不上，则会舍弃这一层的参数，改为随机初始化，达到预训练的目的。**

1）手动替换掉不同的字典键（预训练模型在本地）

```python
model = ghost_net()
model_dict = model.state_dict # 获取当前模型参数字典，包括模型的各层的名称（键）和相应的参数（值）
pretrained_dict = torch.load('./pretrain.pth') # 加载预训练的模型，此时pretrained_dict也是一个字典，包括各层的名称和相应的参数
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} # 在pretrained_dict中保存与model_dict中键名称相同的那些键，舍弃不同的键，这里是网络结构代码要更改名称的原因
model_dict.update(pretrained_dict) # 用pretrained_dict的参数值去覆盖model_dict
model.load_state_dict(model_dict) # 加载覆盖之后的模型参数
```



2）手动替换替换掉不同的字典键（给定预训练模型url）

```python
import torch.utils.model_zoo as model_zoo

model = ghost_net()
model_dict = model.state_dict
pretrained_dict = model_zoo.load_url('http://github.com/d-li14/ghostnet/pretrain/pretrain.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

这个方法我没有运行成功，但网上很多教程都是这样写的。



3）使用pytorch自带函数自动替换

```python
model = ghost_net()
#checkpoint = torch.load('./model_best.pth.tar')
#model.load_state_dict(checkpoint['state_dict'], strict=False)
model.load_state_dict(torch.load('./pretrain.pth'), strict=False)
```



#### 四、在pytorch官网上下载在ImageNet训好的pytorch模型

网站 https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo 介绍了使用函数torch.hub.load_state_dict_from_url下载预训练好的模型：

```python
import torch
state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
```

实际上使用时最好指定下载的路径，否则不太好找默认下载的文件路径，一般默认下载服务器当前用户根目录缓存文件夹中，如：/home/sunzheng/.cache/torch/checkpoints/

指定下载路径：

```python
import torch
state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth','/home/sunzheng/Keypoint_Detection/human-pose-estimation.pytorch/models/imagenet')
```

**注意：torch.hub最低版本要求是torch1.1，低版本的torch使用不了这个函数。**



### Tensorflow











