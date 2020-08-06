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


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#用于ResNet-18和34的残差块，用的是3*3+3*3的卷积
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过非shortcut分支处理之后，输入信号x的通道数会发生变化，所以需要在shortcut分支对通道数做处理，变换为统一维度才可以相加	
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequentisl(
                nn.Conv2d(in_planes, self.expansion*planes,
                           kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
   
# 用于ResNet-50，101，152的残差块，用的是1*1+3*3+1*1的卷积
class Bottleneck(nn.Module):
    # 前面1*1和3*3卷积的fliter个数相等，最后1*1卷积是其的expansion倍
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out +=self.shortcut(x)
        out = F.relu(out)
        return out
      
# 搭建ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # 输入之后第一个卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 最后一层为全连接层
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
  
    def forward(self,x):
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.layer1(out)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = F.avg_pool2d(out, 4)
      out = out.view(out.size[0], -1)
      out = self.linear(out)
      return out
        
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
  
def ResNet34():
    return Resnet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
```



4.MobileNetV2





5.GhostNet





6.ResNet-3D



7.R(2+1)D



8.ip-CSN



9.GhostNet-3D





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

（1）模型加载

一般模型会保存为.pth文件或者保存为tar包（一般tar包是一个字典，保存了模型参数以及训练的一些信息，如训练epoch数，学习率等），加载方法分别是：

```python
model = ghost_net()
model.load_state_dict(torch.load('./xxx.pth'))
```

```python
model = ghost_net()
checkpoint = torch.load('./xxx.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
```



（2）加载预训练模型

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



### Tensorflow











