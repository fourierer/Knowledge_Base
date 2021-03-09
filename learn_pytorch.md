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
tmp = inputs.cpu() # Tensor转换为CPU上的数据类型
```

用`torch.cuda.is_available()`查看GPU是否可用:

``` python
import torch
from torch import nn

torch.cuda.is_available() # 输出 True
```

查看GPU数量：

``` python
torch.cuda.device_count() # 输出 1
```

查看当前GPU索引号，索引号从0开始：

``` python
torch.cuda.current_device() # 输出 0
```

根据索引号查看GPU名字:

``` python
torch.cuda.get_device_name(0) # 输出 'GeForce GTX 1050'
```

默认情况下，`Tensor`会被存在内存上。因此，之前我们每次打印`Tensor`的时候看不到GPU相关标识。

``` python
x = torch.tensor([1, 2, 3])
x
```

输出：

```
tensor([1, 2, 3])
```

使用`.cuda()`可以将CPU上的`Tensor`转换（复制）到GPU上。如果有多块GPU，我们用`.cuda(i)`来表示第 $i$ 块GPU及相应的显存（$i$从0开始）且`cuda(0)`和`cuda()`等价。

``` python
x = x.cuda(0)
x
```

输出：

```
tensor([1, 2, 3], device='cuda:0')
```

我们可以通过`Tensor`的`device`属性来查看该`Tensor`所在的设备。

```python
x.device
```

输出：

```
device(type='cuda', index=0)
```

我们可以直接在创建的时候就指定设备。

``` python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
x
```

输出：

```
tensor([1, 2, 3], device='cuda:0')
```

如果对在GPU上的数据进行运算，那么结果还是存放在GPU上。

``` python
y = x**2
y
```

输出：

```
tensor([1, 4, 9], device='cuda:0')
```

需要注意的是，存储在不同位置中的数据是不可以直接进行计算的。即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。

``` python
z = y + x.cpu()
```

会报错:

```
RuntimeError: Expected object of type torch.cuda.LongTensor but found type torch.LongTensor for argument #3 'other'
```

## 

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



2.张量，tensor

1中介绍了很多关于Tensor的知识，正常情况下pytorch的模型参数都是Tensor形式（32位浮点型），有时候是不够用的，所以需要tensor类型，pytorch中常用的tensor类型包括：

```python
torch.FloatTensor() # 32位浮点型
torch.DoubleTensor # 64位浮点型
torch.ShortTensor # 16位整型
torch.IntTensor # 32位整型
torch.LongTensor # 64位整型
```



在pytorch中，Tensor和tensor都能用于生成张量：

```python
import torch
a = torch.Tensor([1,2,3]) # tensor([1.0,2.0,3.0])
b = torch.tensor([1,2,3]) # tensor([1,2,3])
```

从上述中已经可以看出一点差别了，torch.Tensor会生成浮点型数，而torch.tensor可以生成整形数。

torch.Tensor()是python类，**是默认张量类型torch.FloatTensor()的别名**，torch.Tensor([1,2])会调用Tensor类的构造函数\__init__，生成单精度浮点类型的张量。

torch.tensor()仅仅是python函数，函数原型是：

```python
torch.tensor(data, dtype=None, device=None, requires_grad=False)
```

其中data可以是：list, tuple, NumPy ndarray, scalar和其他类型。torch.tensor会从data中的数据部分做拷贝（而不是直接引用），根据原始数据类型生成相应的torch.LongTensor、torch.FloatTensor和torch.DoubleTensor。

总体来说：

（1）在数据类型上，torch.tensor（包括整型，浮点型等）实际上包括了torch.Tensor（只是32浮点型的别名）；

（2）torch.tensor可以接受list，tuple等，而torch.Tensor在接受一个数字时，会默认为尺寸，而不是一个变量：

```python
>>> a=torch.tensor(1)
>>> a
tensor(1)
>>> a.type()
'torch.LongTensor'
>>> a=torch.Tensor(1)
>>> a
tensor([0.])
>>> a.type()
'torch.FloatTensor'
```



3.类型转换

在1中，介绍了Tensor与numpy之间的转换，以及Tensor变量在GPU和CPU之间的转换，由于Tensor只是tensor的一种类型，所以tensor和numpy之间的转换方式和Tensor一样。下面主要介绍tensor和Tensor之间的转换方式，即下面几种类型的转换：

```python
torch.FloatTensor() # 32位浮点型
torch.DoubleTensor # 64位浮点型
torch.ShortTensor # 16位整型
torch.IntTensor # 32位整型
torch.LongTensor # 64位整型
```

（1）通过在tensor后面加long()，int()等函数转换

```python
>>> a = torch.tensor([1,2,3])
>>> a.type()
'torch.LongTensor'
>>> b = a.int()
>>> b
tensor([1, 2, 3], dtype=torch.int32)
>>> b.type()
'torch.IntTensor'
>>> c = a.short()
>>> c
tensor([1, 2, 3], dtype=torch.int16)
>>> d = a.float()
>>> d
tensor([1., 2., 3.])
>>> d.type()
'torch.FloatTensor'
>>> e = a.double()
>>> e
tensor([1., 2., 3.], dtype=torch.float64)
>>> e.type()
'torch.DoubleTensor'
# 注意：此时a还是torch.LongTensor类型
```

（2）通过type()函数转换

```python
>>> a = torch.tensor([1,2,3])
>>> a
tensor([1, 2, 3])
>>> a.type()
'torch.LongTensor'
>>> b = a.type(torch.FloatTensor)
>>> b
tensor([1., 2., 3.])
>>> b.type()
'torch.FloatTensor'
>>> c = a.type(torch.DoubleTensor)
>>> c
tensor([1., 2., 3.], dtype=torch.float64)
>>> c.type()
'torch.DoubleTensor'
```



4.改变形状

用view()来改变Tensor的形状：

``` python
x = torch.rand(5,3)
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
```

输出：

```
torch.Size([5, 3]) torch.Size([15]) torch.Size([3, 5])
```

**注意view()返回的新tensor与源tensor共享内存（其实是同一个tensor），也即更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度)**

``` python
x += 1
print(x)
print(y) # 也加了1
```

输出：

```
tensor([[1.6035, 1.8110, 0.9549],
        [1.8797, 2.0482, 0.9555],
        [0.2771, 3.8663, 0.4345],
        [1.1604, 0.9746, 2.0739],
        [3.2628, 0.0825, 0.7749]])
tensor([1.6035, 1.8110, 0.9549, 1.8797, 2.0482, 0.9555, 0.2771, 3.8663, 0.4345,
        1.1604, 0.9746, 2.0739, 3.2628, 0.0825, 0.7749])
```

所以如果我们想返回一个真正新的副本（即不共享内存）该怎么办呢？Pytorch还提供了一个reshape()可以改变形状，但是此函数并不能保证返回的是其拷贝，所以不推荐使用。推荐先用clone创造一个副本然后再使用view。[参考此处](https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch)

``` python
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)
```

输出:

```
tensor([[ 0.6035,  0.8110, -0.0451],
        [ 0.8797,  1.0482, -0.0445],
        [-0.7229,  2.8663, -0.5655],
        [ 0.1604, -0.0254,  1.0739],
        [ 2.2628, -0.9175, -0.2251]])
tensor([1.6035, 1.8110, 0.9549, 1.8797, 2.0482, 0.9555, 0.2771, 3.8663, 0.4345,
        1.1604, 0.9746, 2.0739, 3.2628, 0.0825, 0.7749])
```

使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor。另外一个常用的函数就是`item()`, 它可以将一个标量`Tensor`转换成一个Python number：

``` python
x = torch.randn(1)
print(x)
print(x.item())
print(type(x.item()))
```

输出：

```
tensor([2.3466])
2.3466382026672363
<class 'float'>
```



5.运算的内存开销

索引、view是不会开辟新内存的，而像$y = x + y$这样的运算是会新开内存的，然后将$y$指向新内存。为了演示这一点，我们可以使用Python自带的id函数：如果两个实例的ID一致，那么它们所对应的内存地址相同；反之则不同。

``` python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False 
```

如果想指定结果到原来的`y`的内存，我们可以使用前面介绍的索引来进行替换操作。在下面的例子中，我们把`x + y`的结果通过`[:]`写进`y`对应的内存中。

``` python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True
```

我们还可以使用运算符全名函数中的`out`参数或者自加运算符`+=`(也即`add_()`)达到上述效果，例如`torch.add(x, y, out=y)`和`y += x`(`y.add_(x)`)。

``` python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before) # True
```



6.求梯度

创建一个`Tensor`并设置`requires_grad=True`:

``` python
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)
```

输出：

```
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
None
```

再做一下运算操作：

``` python
y = x + 2
print(y)
print(y.grad_fn)
```

输出：

```
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward>)
<AddBackward object at 0x1100477b8>
```

注意x是直接创建的，所以它没有`grad_fn`, 而y是通过一个加法操作创建的，所以它有一个为`<AddBackward>`的`grad_fn`。

像x这种直接创建的称为叶子节点，叶子节点对应的`grad_fn`是`None`。

``` python
print(x.is_leaf, y.is_leaf) # True False
```


再来点复杂度运算操作：

``` python
z = y * y * 3
out = z.mean()
print(z, out)
```

输出：

```
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward>) tensor(27., grad_fn=<MeanBackward1>)
```

通过`.requires_grad_()`来用in-place的方式改变`requires_grad`属性：

``` python
a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)
```

输出：

```
False
True
<SumBackward0 object at 0x118f50cc0>
```

## 

因为`out`是一个标量，所以调用`backward()`时不需要指定求导变量：

``` python
out.backward() # 等价于 out.backward(torch.tensor(1.))
```

我们来看看`out`关于`x`的梯度 $\frac{d(out)}{dx}$:

``` python
print(x.grad)
```

输出：

```
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

我们令`out`为 $o$ , 因为
$$
o=\frac14\sum_{i=1}^4z_i=\frac14\sum_{i=1}^43(x_i+2)^2
$$
所以
$$
\frac{\partial{o}}{\partial{x_i}}\bigr\rvert_{x_i=1}=\frac{9}{2}=4.5
$$
所以上面的输出是正确的。

数学上，如果有一个函数值和自变量都为向量的函数 $\vec{y}=f(\vec{x})$, 那么 $\vec{y}$ 关于 $\vec{x}$ 的梯度就是一个雅可比矩阵（Jacobian matrix）:
$$
J=\left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)
$$
而``torch.autograd``这个包就是用来计算一些雅克比矩阵的乘积的。例如，如果 $v$ 是一个标量函数的 $l=g\left(\vec{y}\right)$ 的梯度：
$$
v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)
$$
那么根据链式法则我们有 $l$ 关于 $\vec{x}$ 的雅克比矩阵就为:
$$
v J=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right) \left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)=\left(\begin{array}{ccc}\frac{\partial l}{\partial x_{1}} & \cdots & \frac{\partial l}{\partial x_{n}}\end{array}\right)
$$

注意：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。

``` python
# 再来反向传播一次，注意grad是累加的
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)
```

输出：

```
tensor([[5.5000, 5.5000],
        [5.5000, 5.5000]])
tensor([[1., 1.],
        [1., 1.]])
```

> 现在我们解释之前留下的问题，为什么在`y.backward()`时，如果`y`是标量，则不需要为`backward()`传入任何参数；否则，需要传入一个与`y`同形的`Tensor`?
> 简单来说就是为了避免向量（甚至更高维张量）对张量求导，而转换成标量对张量求导。举个例子，假设形状为 `m x n` 的矩阵 X 经过运算得到了 `p x q` 的矩阵 Y，Y 又经过运算得到了 `s x t` 的矩阵 Z。那么按照前面讲的规则，dZ/dY 应该是一个 `s x t x p x q` 四维张量，dY/dX 是一个 `p x q x m x n`的四维张量。问题来了，怎样反向传播？怎样将两个四维张量相乘？？？这要怎么乘？？？就算能解决两个四维张量怎么乘的问题，四维和三维的张量又怎么乘？导数的导数又怎么求，这一连串的问题，感觉要疯掉…… 
> 为了避免这个问题，我们**不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量**。所以必要时我们要把张量通过将所有张量的元素加权求和的方式转换为标量，举个例子，假设`y`由自变量`x`计算而来，`w`是和`y`同形的张量，则`y.backward(w)`的含义是：先计算`l = torch.sum(y * w)`，则`l`是个标量，然后求`l`对自变量`x`的导数。
> [参考](https://zhuanlan.zhihu.com/p/29923090)

来看一些实际例子。

``` python
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
```

输出：

```
tensor([[2., 4.],
        [6., 8.]], grad_fn=<ViewBackward>)
```

现在 `y` 不是一个标量，所以在调用`backward`时需要传入一个和`y`同形的权重向量进行加权求和得到一个标量。

``` python
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)
```

输出：

```
tensor([2.0000, 0.2000, 0.0200, 0.0020])
```

注意，`x.grad`是和`x`同形的张量。

再来看看中断梯度追踪的例子：

``` python
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2 
with torch.no_grad():
    # 该范围内的张量都不具有梯度
    y2 = x ** 3
y3 = y1 + y2
    
print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True
```

输出：

```
True
tensor(1., grad_fn=<PowBackward0>) True
tensor(1.) False
tensor(2., grad_fn=<ThAddBackward>) True
```

可以看到，上面的`y2`是没有`grad_fn`而且`y2.requires_grad=False`的，而`y3`是有`grad_fn`的。如果我们将`y3`对`x`求梯度的话会是多少呢？

``` python
y3.backward()
print(x.grad)
```

输出：

```
tensor(2.)
```

为什么是2呢？$ y_3 = y_1 + y_2 = x^2 + x^3$，当 $x=1$ 时 $\frac {dy_3} {dx}$ 不应该是5吗？事实上，由于 $y_2$ 的定义是被`torch.no_grad():`包裹的，所以与 $y_2$ 有关的梯度是不会回传的，只有与 $y_1$ 有关的梯度才会回传，即 $x^2$ 对 $x$ 的梯度。

上面提到，`y2.requires_grad=False`，所以不能调用 `y2.backward()`，会报错：

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

此外，如果我们想要修改`tensor`的数值，但是又不希望被`autograd`记录（即不会影响反向传播），那么我们可以对`tensor.data`进行操作。

``` python
x = torch.ones(1,requires_grad=True)

print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外

y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)
```

输出：

```
tensor([1.])
False
tensor([100.], requires_grad=True)
tensor([2.])
```



7.读写tensor

可以直接使用`save`函数和`load`函数分别存储和读取`Tensor`。`save`使用Python的pickle实用程序将对象进行序列化，然后将序列化的对象保存到disk，使用`save`可以保存各种对象,包括模型、张量和字典等。而`laod`使用pickle unpickle工具将pickle的对象文件反序列化为内存。

下面的例子创建了`Tensor`变量`x`，并将其存在文件名同为`x.pt`的文件里。

``` python
import torch
from torch import nn

x = torch.ones(3)
torch.save(x, 'x.pt')
```

然后我们将数据从存储的文件读回内存。

``` python
x2 = torch.load('x.pt')
x2
```

输出：

```
tensor([1., 1., 1.])
```

我们还可以存储一个`Tensor`列表并读回内存。

``` python
y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
xy_list
```

输出：

```
[tensor([1., 1., 1.]), tensor([0., 0., 0., 0.])]
```

存储并读取一个从字符串映射到`Tensor`的字典。

``` python
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
xy
```

输出：

```
{'x': tensor([1., 1., 1.]), 'y': tensor([0., 0., 0., 0.])}
```



再给出一段用json序列化torch.Tensor的代码（json并不能序列化torch.Tensor，需要定义函数）：

```python
import torch
import json
import numpy as np


a = torch.rand([5,3])
# print(a)
# print(isinstance(a, torch.Tensor)) # True

def tensor2list(t):
    if isinstance(t, torch.Tensor):
        # torch.Tensor需要先转换成numpy，然后转换成list才可以通过list序列化
        return t.numpy().tolist()

def list2tensor(l):
    if isinstance(l, list):
        l1 = np.array(l)
        l2 = torch.from_numpy(l1)
        return l2


json_str = json.dumps(a, default=tensor2list)
# print(type(json_str)) # <class 'str'>
test_list = json.loads(json_str)
# print(type(test_list)) # <class 'list'>

with open('./test.json', 'w') as f:
    json.dump(a, f, default=tensor2list)

with open('./test.json', 'r') as f:
    test_list = json.load(f)
print(type(test_list)) # list

with open('./test.json', 'r') as f:
    test_tensor = json.load(f, object_hook=list2tensor)
print(type(test_tensor)) # list, 由于列表中还嵌套列表，这里定义的list2tensor并不能满足将所有的list都转换为np.array(),所以没有成功转换为torch.Tensor

```





#### 二、pytorch模型构建基础

1.使用pytorch搭建一个线性回归模型，代码见工程文件/code/learn_pytorch/regression。

2.继承Module类来构建模型

`Module`类是`nn`模块里提供的一个模型构造类，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型。下面继承`Module`类构造本节开头提到的多层感知机。**这里定义的`MLP`类重载了`Module`类的`__init__`函数和`forward`函数。它们分别用于创建模型参数和定义前向计算**，前向计算也即正向传播。

``` python
import torch
from torch import nn

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层
         

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
```

以上的`MLP`类中无须定义反向传播函数。系统将通过自动求梯度而自动生成反向传播所需的`backward`函数。

我们可以实例化`MLP`类得到模型变量`net`。下面的代码初始化`net`并传入输入数据`X`做一次前向计算。其中，`net(X)`会调用`MLP`继承自`Module`类的`__call__`函数，这个函数将调用`MLP`类定义的`forward`函数来完成前向计算。

``` python
X = torch.rand(2, 784)
net = MLP()
print(net)
net(X)
```

输出：

```
MLP(
  (hidden): Linear(in_features=784, out_features=256, bias=True)
  (act): ReLU()
  (output): Linear(in_features=256, out_features=10, bias=True)
)
tensor([[-0.1798, -0.2253,  0.0206, -0.1067, -0.0889,  0.1818, -0.1474,  0.1845,
         -0.1870,  0.1970],
        [-0.1843, -0.1562, -0.0090,  0.0351, -0.1538,  0.0992, -0.0883,  0.0911,
         -0.2293,  0.2360]], grad_fn=<ThAddmmBackward>)
```

注意，这里并没有将`Module`类命名为`Layer`（层）或者`Model`（模型）之类的名字，这是因为该类是一个可供自由组建的部件。它的子类既可以是一个层（如PyTorch提供的`Linear`类），又可以是一个模型（如这里定义的`MLP`类），或者是模型的一个部分。我们下面通过两个例子来展示它的灵活性。



3.Module的子类

（1）`Sequential`类

当模型的前向计算为简单串联各个层的计算时，`Sequential`类可以通过更加简单的方式定义模型。这正是`Sequential`类的目的：它可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数来逐一添加`Module`的实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。

下面我们实现一个与`Sequential`类有相同功能的`MySequential`类。这或许可以帮助读者更加清晰地理解`Sequential`类的工作机制。

``` python
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            input = module(input)
        return input
```

我们用`MySequential`类来实现前面描述的`MLP`类，并使用随机初始化的模型做一次前向计算。

``` python
net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
print(net)
net(X)
```

输出：

```
MySequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
tensor([[-0.0100, -0.2516,  0.0392, -0.1684, -0.0937,  0.2191, -0.1448,  0.0930,
          0.1228, -0.2540],
        [-0.1086, -0.1858,  0.0203, -0.2051, -0.1404,  0.2738, -0.0607,  0.0622,
          0.0817, -0.2574]], grad_fn=<ThAddmmBackward>)
```

可以观察到这里`MySequential`类的使用跟3.10节（多层感知机的简洁实现）中`Sequential`类的使用没什么区别。

（2）`ModuleList`类

`ModuleList`接收一个子模块的列表作为输入，然后也可以类似List那样进行append和extend操作:

``` python
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
```

输出：

```
Linear(in_features=256, out_features=10, bias=True)
ModuleList(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
```

（3）`ModuleDict`类

`ModuleDict`接收一个子模块的字典作为输入, 然后也可以类似字典那样进行添加访问操作:

``` python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
```

输出：

```
Linear(in_features=784, out_features=256, bias=True)
Linear(in_features=256, out_features=10, bias=True)
ModuleDict(
  (act): ReLU()
  (linear): Linear(in_features=784, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
```



4.访问模型参数

定义一个网络：

``` python
import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()
```

输出：

```
Sequential(
  (0): Linear(in_features=4, out_features=3, bias=True)
  (1): ReLU()
  (2): Linear(in_features=3, out_features=1, bias=True)
)
```

`Sequential`类继承`Module`类，对于`Sequential`实例中含模型参数的层，我们可以通过`Module`类的`parameters()`或者`named_parameters`方法来访问所有参数（以迭代器的形式返回），后者除了返回参数`Tensor`外还会返回其名字。下面，访问多层感知机`net`的所有参数：

``` python
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())
```

输出：

```
<class 'generator'>
0.weight torch.Size([3, 4])
0.bias torch.Size([3])
2.weight torch.Size([1, 3])
2.bias torch.Size([1])
```

可见返回的名字自动加上了层数的索引作为前缀。
我们再来访问`net`中单层的参数。对于使用`Sequential`类构造的神经网络，我们可以通过方括号`[]`来访问网络的任一层。索引0表示隐藏层为`Sequential`实例最先添加的层。

``` python
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))
```

输出：

```
weight torch.Size([3, 4]) <class 'torch.nn.parameter.Parameter'>
bias torch.Size([3]) <class 'torch.nn.parameter.Parameter'>
```

因为这里是单层的所以没有了层数索引的前缀。另外返回的`param`的类型为`torch.nn.parameter.Parameter`，其实这是`Tensor`的子类，和`Tensor`不同的是如果一个`Tensor`是`Parameter`，那么它会自动被添加到模型的参数列表里，来看下面这个例子。

``` python
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass
    
n = MyModel()
for name, param in n.named_parameters():
    print(name)
```

输出:

```
weight1
```

上面的代码中`weight1`在参数列表中但是`weight2`却没在参数列表中。

因为`Parameter`是`Tensor`，即`Tensor`拥有的属性它都有，比如可以根据`data`来访问参数数值，用`grad`来访问参数梯度。

``` python
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad) # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)
```

输出：

```
tensor([[ 0.2719, -0.0898, -0.2462,  0.0655],
        [-0.4669, -0.2703,  0.3230,  0.2067],
        [-0.2708,  0.1171, -0.0995,  0.3913]])
None
tensor([[-0.2281, -0.0653, -0.1646, -0.2569],
        [-0.1916, -0.0549, -0.1382, -0.2158],
        [ 0.0000,  0.0000,  0.0000,  0.0000]])
```

## 



#### 三、搭建常用神经网络模型

1.全连接网络模型

```python
from torch import nn

class SimpleNet(nn.Module):
    '''
    定义一个简单的四层全连接神经网络，每一层都是线性的
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

**注：**

**1.nn.ReLU() 和 nn.ReLU(inplace=True)对计算结果不会有影响。利用inplace计算可以节省内（显）存，同时还可以省去反复申请和释放内存的时间。但是会对原变量覆盖，只要不带来错误就用；**

**2.pytorch定义类中有个forward函数，该函数的功能类似于python类的内置函数__call__，可以直接将类当场函数调用，如：**

```
model = Net()
out = model(input)
```

有时候也把损失函数定义为类，可以直接当成函数进行调用。



2.一般的卷积网络模型

(1)nn.Conv2d()函数参数介绍：

in_channels:输入信号的通道；

out_channels:卷积后输出结果的通道数；

kernel_size:卷积核的形状，例如(3,2)表示$3*2$的卷积核，如果宽和高相同，可以只用一个数字表示；

stride:卷积每次移动的步长，默认为1；

padding:处理边界时填充0的数量，默认为0(不填充)；

dilation:采样间隔数量，默认为1，无间隔采样；

bias:为True时，添加偏置；

先给出卷积输入和输出尺寸大小公式：假设输入信号尺寸为$w*h$，卷积核大小为$f*f$，填充大小为$p$，步长为$s$，则输出尺寸$w'*h'$为：
$$
w'=\frac{w-f+2p}{s}+1\\
h'=\frac{h-f+2p}{s}+1
$$



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
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10))
        
def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
```



3.经典卷积神经网络

（1）Alexnet

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class AlexNet(nn.Module):
    def __init__(self, num_outputs):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_outputs)
        )

    def forward(x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        y = x.view(x.size(0), -1) # 将卷积层的特征拉成batch_size行的张量
        output = self.fc(y)
        return y


if __name__=='__main__':
    model = AlexNet(10)
    print(model)
```



（2）Vgg

```python

```



（3）GoogLeNet（给出Inception模块的代码，整体结构略去）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F



class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)

        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 线路3，1 x 1卷积层后接5 x 5卷积
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1)
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        output = torch.cat((p1, p2, p3, p4), dim = 1) # 在通道维上连接输出

        return output

if __name__=='__main__':
    in_c = 64
    c1 = 64
    c2 = [64, 64]
    c3 = [64, 64]
    c4 = 64
    model = Inception(in_c, c1, c2, c3, c4)
    print(model)


```



（4）ResNet

ResNet中有两个基本的block，一个是$3*3$+$3*3$，称为basic block；另一个是$1*1$+$3*3$+$1*1$，称为bottleneck block。这里先写两个残差结构block，然后再搭建整个网络：

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

（5）DenseNet



（6）MobileNetV1

```python

```



（7）MobileNetV2

（8）MobileNetV3

（9）ShuffleNetV2

（10）GhostNet

（轻量级网络系列代码见repo：" https://github.com/fourierer/Learn_GhostNet_pytorch "）



7.使用训好的模型来测试单个图像和视频

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



#### 五、固定模型参数，只训练其中一部分参数

想固定那个参数，只需要将变量的requires_grad设为False即可，然后再优化器中过滤掉，如：

```python
optimizer.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

具体如何设置变量的requires_grad，这里给出一段代码：

```python
class Net(nn.Module):
    self.__init__(self,block, layers, **kwargs):
        xxx
    def forward():
        xxx
    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            
            logger.info('=> init fc weights from normal distribution')
            for m in self.classifer.modules():
                if isinstance(m, nn.Linear):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def pose_resnet_adver2(args, is_train, **kwargs):
    num_layers = args.resnet_layers # 默认是50

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNetAdver2(block_class, layers, **kwargs)
    if is_train:
        model.init_weights(args.resume)

    return model


if __name__ == '__main__':
    num_layers = 50
    block_class, layers = resnet_spec[num_layers]
    model1 = Net(block_class, layers)
    model2 = Net(block_class, layers)
    '''
    print(model)
    x = torch.rand([64, 3, 256, 256])
    y1, y2 = model(x)
    summary(model, x)
    print(y1.size())
    print(y2.size())
    '''
    for name,param in model1.named_parameters():
        # print(name)
        
        # 固定其他参数，只改变分类器的参数，nn.Linear是没有requires_grad的，需要指定（原因未知）
        if (name=='classifer.weight')|(name=='classifer.bias'):
            param.requires_grad = True
        else:
            param.requires_grad = False
        '''
        if (name!='classifer.weight')|(name!='classifer.bias'):
            param.requires_grad = False
        else:
            param.requires_grad = True
        '''
        # 注意，如果换成以上引号中的代码，下面代码将无法输出classifer.weight和classifer.bias
        # 这里由于是线性层nn.Linear需要特别指定（具体原因未知）
        if param.requires_grad:
            print(name) # classifer.weight,classifer.bias
    # 训好的分类器
    # checkpoint1 = torch.load('xxx.pth')
    # model1.load_state_dict(checkpoint1['state_dict'])
    model1.init_weights('xxx.pth')
    # 原始模型
    # checkpoint2 = torch.load('xxx.pth')
    # model2.load_state_dict(checkpoint2['state_dict'])
    model2.init_weights('xxx.pth')
    # 查看固定层的参数是否一样，以及没有固定的层参数是否不一样
    for name,param in model1.named_parameters():
        if name=='layer4.0.conv3.weight':
            print(param[0][3])
        if name=='classifer.weight':
            print(param[0][2])
    for name,param in model2.named_parameters():
        if name=='layer4.0.conv3.weight':
            print(param[0][3])
        if name=='classifer.weight':
            print(param[0][2])
```











