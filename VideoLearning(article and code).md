# 文献阅读以及代码解读



#################################################################

### 视频抽帧sample

#### 论文：Temporal Segment Networks: Towards Good Practices for Deep Action Recognition

##### 一、论文理解

(该论文是最早将视频sample之后再分析的文章)

视频抽帧：1)减小计算量；2)视频中的帧会有很多冗余。现在的视频处理与分析都会先抽帧再分析。







##### 二、代码复现



**#################################################################**



### 双流网络

#### 论文：Two_Stream Convolutional Networks for Action Recognition in video

##### 一、论文理解

1.视频分类要捕获静态帧的外观信息和帧之间的动作信息；

2.双流网络，空间流从静态帧学习，时间流以密集光流的形式来学习动作信息;

3.





##### 二、代码复现：（https://github.com/jeffreyyihuang/two-stream-action-recognition）

1.spatial_cnn.py

(1)23行

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
```

(2)25-31行

argparse是命令行参数解析模块，使用时需要import argparse，使用该工具可以直接在linux命令行中进行调参，不需要在.py文件中修改参数。下面进行测试说明该模块的用法。

1).代码(Test.py)：

```python
import argparse
parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=25, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
args = parser.parse_args()
print(args)
print(args.epochs)
print(args.lr)
```

测试命令：$ python Test.py

输出：

```python
Namespace(batch_size=25, epochs=500, evaluate=False, lr=0.0005, resume='', start_epoch=0)
500
0.0005
```

2).代码(Test.py):

```python
import argparse
x = argparse.ArgumentParse(description 'UCF101')
x.add_argument('--epochs',default=500,type=int,metavar='N', help='number of total epochs')
x.add_argument('--lr',default=5e-4, type=float, metavar='LR', help='initial learning rate')
y = x.parse_args()
print(y)
print(y.epochs)
print(y,lr)
```

测试命令1：$ python Test.py

输出：

```python
Namespace(epochs=500, lr=0.0005)
500
0.0005
```

测试命令2：$ python Test.py --epochs 40

输出：

```python
Namespace(epochs=40, lr=0.0005)
40
0.0005
```

测试命令3：$ python Test.py --epoch 40 --lr 0.2

输出：

```python
Namespace(epochs=40, lr=0.2)
40
0.2
```

测试命令4：$ python Test.py -h

输出：

```python
usage: UCF101 spatial stream on resnet101 [-h] [--epochs N] [--lr LR]

optional arguments:
  -h, --help  show this help message and exit
  --epochs N  number of total epochs
  --lr LR     initial learning rate
```



**#################################################################**

### GCN

#### 1.基本概念

GCN，图卷积神经网络，实际上跟CNN的作用一样，就是一个特征提取器，只不过它的对象是图数据。GCN精妙地设计了一种从图数据中提取特征的方法，从而让我们可以使用这些特征去对图数据进行节点分类（node classification）、图分类（graph classification）、边预测（link prediction），还可以顺便得到图的嵌入表示（graph embedding）。

#### 2.数学基础

##### 2.1.梯度、散度、拉普拉斯算子

1.梯度$\nabla$:
$$
\nabla f=[\frac{\partial f}{\partial x},\frac{\partial f}{\partial y},\frac{\partial f}{\partial z}]^T
$$
2.散度:$\nabla$$\cdot$
$$
A=[P,Q,R]\\
\nabla\cdot A=\frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z}
$$
3.拉普拉斯算子:$\triangle$
$$
\triangle f=\nabla^2f=\nabla\cdot\nabla f\\
=\sum_{i=1}^{n}\frac{\partial^2f}{\partial x_i^2}
$$
4.函数的拉普拉斯算子也是该函数的海瑟矩阵的迹
$$
\triangle f=tr(H(f))
$$
5.拉普拉斯算子作用在向量值函数上，其结果定义为一个向量，该向量的各个分量分别作为向量值函数各个分量的拉普拉斯
$$
\triangle A=[\triangle P,\triangle Q,\triangle R]^T
$$


##### 2.2.图的Laplacian矩阵的性质

常见构造方式有三种：(1)$L=D-A$；(2)$L^{sys}=D^{-\frac{1}{2}}LD^{-\frac{1}{2}}$；(3)$L^{rw}=D^{-1}L$.(谱聚类中涉及)。

无论哪种构造方法，拉普拉斯矩阵都是一个对称半正定阵。

已知如下定理：

1.$n$阶方阵可以对角化(特征分解，谱分解)的充分必要条件是存在$n$个线性无关的特征向量；

2.对称矩阵有$n$个线性无关的特征向量；

3.对称矩阵的特征值相互正交。

由上可知：图的拉普拉斯矩阵可以谱分解：
$$
L=U\,\,diag(\lambda_1,...,\lambda_n)\,\,U^{-1}
$$
其中，$U=(u_1,u_2,...,u_n)$，是列向量为单位特征向量的矩阵，$diag(\lambda_1,...,\lambda_n)$是$n$个特征值构成的对角阵。由于$U$是正交阵，特征分解又可以写成：
$$
L=U\,\,diag(\lambda_1,...,\lambda_n)\,\,U^T
$$



##### 2.3.拉普拉斯算子和拉普拉斯矩阵的关系(https://zhuanlan.zhihu.com/p/81502804)



##### 2.4.傅里叶变换



#### 3.具体方法(https://tkipf.github.io/graph-convolutional-networks/)

##### 3.1.模型建立

有一批数据，其中有$N$个节点（node），每个节点都有自己的特征，我们设这些节点的特征组成一个$N×D$维的矩阵$X$，然后各个节点之间的关系也会形成一个$N×N$维的矩阵$A$，也称为邻接矩阵（adjacency matrix）。$X$和$A$便是我们模型的输入。
GCN也是一个神经网络层，它的层与层之间的传播方式是：
$$
H^{l+1}=f(H^l,A)\\
H^0=X\\
H^L=Z
$$
$Z$是图级别的输出，$L$是层数，输入一个图，通过若干层GCN每个node的特征从$X$变成了$Z$，但是不论中间有多少层，node之间的连接关系是不变的，$A$不变。这就是图卷积网络的一般形式，这个模型主要在于$f$如何选择以及参数化.



##### 3.2. 传播函数$f$的选择方式

1.$f(H^l)=\sigma(AH^lW^l)$

这种定义方法很简单但是很强大，有两个主要的局限性：

(1)$A$是邻接矩阵，除非图上有自环，否则对角线上的元素全都为0。所以对于$AH^l$，$A$的第$i$行乘以$H^l$的第$j$列时，所得值是节点$i$邻居节点特征的加权和，不包括节点自身的特征；(可以通过矩阵$A$加上单位阵来解决这个问题)

(2)矩阵$A$没有正则化，与$A$相乘之后会改变特征的范围。可以预处理对A做正则化如$D^{-1}A$，其中$D$是度矩阵，是矩阵$A$每行元素的和作为对角元素。此时对于$D^{-1}AH^l$，$D^{-1}A$的第$i$行乘以$H^l$的第$j$列相乘所得的值是节点$i$邻居节点的特征加权和的平均值；有时候也使用归一化准则$D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$，此时不再是邻居节点的特征加权和的平均值。

(为了解决上述两个局限性，假设传播方式为$H^{l+1}=\sigma(\hat D^{-\frac{1}{2}}\hat{A}\hat D^{-\frac{1}{2}}H^{l}W^{l})$，其中：$\hat{A}=A+I$，这就是谱卷积的形式。后面会介绍)



2.比较有名的关于卷积网络的卷积方式($f$的选择方式)可以分为两种：谱卷积和空间域卷积分别对应两篇文章。



#### 4.论文：Semi-Supervised Classification with Graph Convolutional Networks

##### 一、论文理解

1.谱卷积

对于一个大图（例如“文献引用网络”），我们有时需要对其上的节点进行分类。然而，在该图上，仅有少量的节点是有标注的。此时，我们需要依靠这些已标注的节点来对那些没有标注过的节点进行分类，此即半监督节点分类问题。

(具体推导过程见https://zhuanlan.zhihu.com/p/31067515)
$$
H^{l+1}=\sigma(\hat D^{-\frac{1}{2}}\hat{A}\hat D^{-\frac{1}{2}}H^{l}W^{l}).
$$
其中：$\hat{A}=A+I$，为添加了自连接的邻接矩阵，$\hat{D}$是$\hat{A}$的度矩阵，$\hat{D}_{ii}=\sum_{j}\hat{A}_{ij}$，即对角元素是该行所有元素的加和，其余元素都为0.$H$是每一层的特征，对于输入层就是$X$.$W$是每一层待训练的参数.$\sigma$是非线性激活函数。



模型：

(1)损失函数：
$$
L=L_0+\lambda L_{reg}\\
L_{reg}=\sum_{i,j}A_{ij}||f(X_i)-f(X_j)||^2=f(X)^T\Delta f(X)
$$
$L_0$代表了监督误差(有标签部分的节点误差)，采用交叉熵函数计算有标签节点的误差，$f$是可微的神经网络函数，$X$是节点特征$X_i$组成的矩阵，$\Delta=D-A$，代表无向图没有归一化的拉普拉斯矩阵。正则项的构造基于相连的节点可能有相同的标签。尽可能最小化$L_{reg}$，当$A_{ij}$不为0时($i$节点与$j$节点相连)，是在最小化$f(X_i)$与$f(X_j)$之间的差距，就是让二者最终的标签相同。$f$的形式推导略。

​    然而，这种假设却往往会限制模型的表示能力，因为图中的边不仅仅可以用于编码节点相似度，而且还包含有额外的信息。GCN的使用可以有效地避开这一问题。GCN通过一个简单的映射函数$f(X,A)$，可以将节点的局部信息汇聚到该节点中，然后仅使用那些有标注的节点计算$L_0$即可，从而无需使用图拉普拉斯正则。

模型如下：

![VideoLearnNote](/Users/momo/Documents/video/VideoLearnNote.png)

(1)首先获取节点的特征表示$X$，并计算邻接矩阵 $\hat{A}=\hat D^{-\frac{1}{2}}(A+I)\hat D^{-\frac{1}{2}}$

(2)将其输入到一个两层的GCN网络中，得到每个标签的预测结果：

![img](https://pic2.zhimg.com/80/v2-0cf9e7a5006cd1c1a88e5b5e763b89a5_hd.jpg)

其中， $W^{(0)}\in R^{C*H}$为第一层的权值矩阵，用于将节点的特征表示映射为相应的隐层状态。 $W^{(1)}\in R^{H*F}$为第二层的权值矩阵，用于将节点的隐层表示映射为相应的输出（ $F$ 对应节点标签的数量)。最后将每个节点的表示通过一个softmax函数，即可得到每个节点的标签预测结果。

对于半监督分类问题，使用所有有标签节点上的期望交叉熵作为损失函数：

![img](https://pic3.zhimg.com/80/v2-b30e889063f00b9345010da0e96453de_hd.jpg)

其中， $y_L$表示有标签的节点集。



##### 二、代码复现





#### 5.论文：Learning Convolutional Neural Networks for Graphs

2.空间域卷积



##### 一、论文理解



##### 二、代码复现



**#################################################################**

### I3D

#### 论文：Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset,cvpr2017

##### 一、论文理解

基本思想：根据之前3D-ConvNets的缺点(1.参数多；2.无法利用在ImageNet上预训练过的2D网络)，提出了I3D方法，先用ImageNet数据集训练2D网络，再扩张成3D的卷积网络；同时结合双流网络，每个流使用I3D训练思路。

###### 1.摘要：

(1)文章提供了一个新的数据集kinetics，分析了目前的框架在该数据集上的表现，以及经过该数据集预训练之后，目前的框架在之前小规模数据集上的表现提升了多少；

(2)提出了新的方法I3D，I3D框架在该数据集上预训练之后在其余数据集上表现非常好。

###### 2.简介：

(1)在ImageNet数据集上预训练一个网络框架，可以将这个网络用于其他场景的任务(分类、分割等)，但是在视频领域还没有这样一个数据集；

(2)本文提出Kinetics数据集，旨在解决这个问题，Kinetics数据集有400个人类动作类，每个类超过400个样本；一些经典的网络框架在Kinetics数据集上预训练，再迁移到HMDB-51和UCF上微调，效果提升很多，并且提升的幅度和网络结构有关；

(3)I3D基于InceptionV1网络结构，将2D卷积和池化操作拓展为3D，拓展的过程称为I3D，所以I3D是一种训练方法，训练思路，并不是一个具体的框架；(https://blog.csdn.net/zhy455/article/details/83420904，链接里面介绍了Inception系列的一些知识)

(4)**这篇文章介绍各种视频分类框架，包括CNN+LSTM，Two-stream，3D卷积网络C3D，并推出Two-Stream Inflated 3D ConvNets即I3D(实际上就是双流网络，每个流都是InceptionV1结构，训练时将2D卷积和池化操作拓展为3D)；**



###### 3.动作分类网络

(1)CNN+LSTM

卷积网络不能提取帧之间的关系，因而加入LSTM时序模型，可以捕捉时序空间特性和长时间的相关性。InceptionV1最后一层pool，512个隐含节点之后接LSTM。

问题：

1)LSTM如何训练？输入和输出分别是什么？时序特征怎么体现？

2)文章中说的测试阶段只考虑最后一帧的输出是什么意思？每5帧保留1帧进行采样是指什么？



(2)C3D

C3D有一个非常重要的特征:它们直接创建时空数据的层次表示。直接将一个视频的所有帧输入到卷积网络中，但存在维度问题，参数更多难以训练，无法使用ImageNet预训练。

问题：

1)3D卷积和2D卷积差别在哪里？图片通道和帧数如何区分？



(3)Two_Stream Networks

最后一层卷积神经网络的LSTMs特性可以模拟高层变化，但可能无法捕获在许多情况下至关重要的细微的低层运动。

通过将来自单个RGB帧和10个外部计算的光流帧的预测平均，然后将它们传递给ImageNet预先训练的ConvNet的两个流网络，从而对视频的短时间快照建模。

该流具有一个自适应的输入卷积层，其输入通道数是流帧数的两倍(因为流有水平和垂直两个通道)，在测试时从视频中采样多个快照，并对动作预测进行averaged。

问题：

1)高低层变化指什么？

2)光流如何计算？为什么有垂直和水平两个方向？

3)光流和视频帧分别输入到两个流网络？视频帧流网络是2D卷积网络还是3D卷积网络？最后如何汇聚(如何平均预测)？



(4)3D Fused Two-Stream Network

向网络中输入的是5个连续的RGB帧，每帧间隔为10帧，以及相应的光流片段。

在最后一层卷积层之后，使用3D ConvNet把空间流和时间流**融合**（相比于传统双流是在softmax后才做fusion，把softmax输出的score进行平均）

问题：

1)和双流网络一样的问题，结构是怎样的？



(5)The New: Two-Stream Inflated 3D ConvNets

2D卷积拓展到3D卷积，给2D卷积核增加一个维度(时间维度)，比如将维度为$N*N$拓展为$N*N*N$。

观察到一个图像可以通过把它重复地复制成一个视频序列来转换成一个(乏味的)视频。然后，通过满足我们所说的无聊视频固定点，三维模型可以在ImageNet上进行隐式预训练:无聊视频上的合并激活应该与原始的单图像输入相同。

以每秒25帧来处理视频。我们使用64帧片段训练模型，并使用整个视频进行测试，对预测进行时间平均。

虽然3D卷积神经网络应该能够直接从RGB输入中学习运动特征，但它仍然是纯前馈计算，而光流算法在某种意义上是递归的(例如，它们对流场执行迭代优化)。也许由于这种缺乏递归性的原因，在实验中，我们仍然发现它能够有效地具有如图2所示的双流配置，即一个I3D网络训练RGB输入，另一个训练携带优化的、平滑的流信息的流输入。我们分别对这两个网络进行训练，并对它们在测试时的预测取平均值。

问题：

1)多通道的2D卷积是否称得上是3D卷积？

2)将ImageNet转换为视频数据集，在用3D卷积网络预训练，3D卷积核如何卷积？

3)64帧片段训练模型指什么？对预测进行时间平均指什么？

4)I3D的双流和原始双流区别在哪里(如何从2D卷积核扩展到3D卷积核)？

具体实现：**把2D模型中的核参数在时间维上不断复制，形成3D核的参数，同时除以N，保证输出和2D上一样；别的非线性层结构都与原来的2D模型一样**

5)figure3中的Rec.Field指什么？



(6)I3D实现细节

在训练过程中，我们在空间上和时间上都采用了随机裁剪——将较小的视频尺寸调整为256像素，然后随机裁剪224×224的补丁——同时，在足够早的时间内选择开始帧以保证所需的帧数。对于较短的视频，我们将视频循环多次以满足每个模型的输入接口。在训练过程中，我们对每个视频都坚持使用随机的左右翻转。

我们用TV-L1算法计算光流

问题：

1)TV-L1算法？如何计算光流？



###### 4.Kinetics数据集

包括Kinetics数据集和miniKinetics数据集



###### 5.实验部分

我们测试了UCF-101和HMDB-51的分离1个测试集，并测试了Kinetics的剩余测试集



问题：

1)split 1 test sets和held-out test set分别指什么？



###### 6.实验评估

为了研究在Kinetics数据集上训练的网络模型的泛化能力，考虑两种度量方式。一种是固定参数，然后改动最后一层的分类结构并在UCF-101/HMDB-51数据集上训练最后一层的参数；二是预训练，在Kinetics数据集训练结果参数上进行微调。这两种方法都提升了效果，都比从头训练好。





##### 二、代码复现

##### (https://github.com/deepmind/kinetics-i3d)





**#################################################################**

### Non-local机制

#### 论文：Non-local Neural Networks

个人理解是CNN卷积算子提取的特征是局部范围的特征，需要引入Non-local机制，使用非局部操作算子(基于非局部均值操作)来提取长范围依赖的特征。核心思想是在计算每个像素位置输出时候，不再只和邻域计算，而是和图像中所有位置计算相关性，然后将相关性作为一个权重表征其他位置和当前待计算位置的相似度。可以简单认为采用了一个和原图一样大的kernel进行卷积计算。

##### 一、论文理解(https://blog.csdn.net/elaine_bao/article/details/80821306)

1.摘要

convolution和recurrent都是对局部区域进行的操作，所以它们是典型的local operations。受计算机视觉中经典的非局部均值（non-local means）的启发，本文提出一种non-local operations用于捕获长距离依赖（long-range dependencies），即如何建立图像上两个有一定距离的像素之间的联系，如何建立视频里两帧的联系，如何建立一段话中不同词的联系等。

non-local operations在计算某个位置的响应时，是考虑所有位置features的加权——所有位置可以是空间的，时间的，时空的。这个结构可以被插入到很多计算机视觉结构中，在视频分类的任务上，non-local模型在Kinetics和Charades上都达到了最好的结果


2.简介

3.相关工作

4.Non-local Neural Networks

(1)定义：

公式：$y_i=\frac{1}{C(x)}\sum_jf(x_i,x_j)g(x_j)$

其中$x$表示输入信号（图片，序列，视频等，也可能是它们的features），$y$表示输出信号，其size和$x$相同。$f(x_i,x_j)$用来计算$i$和所有可能关联的位置$j$之间pairwise的关系，这个关系可以是比如$i$和$j$的位置距离越远，$f$值越小，表示$j$位置对$i$影响越小。**$g(x_j)$用于计算输入信号在$j$位置的特征值**。$C(x)$是归一化参数。

作为对比，non-local的操作也和fc层不同。公式(1)计算的输出值受到输入值之间的关系的影响（因为要计算pairwise function），而fc则使用学习到的权重计算输入到输出的映射，在fc中$x_j$和$x_i$的关系是不会影响到输出的，这一定程度上损失了位置的相关性。另外，non-local能够接受任意size的输入，并且得到的输出保持和输入size一致。而fc层则只能有固定大小的输入输出。

**non-local是一个很灵活的building block，它可以很容易地和conv、recurrent层一起使用，它可以被插入到dnn的浅层位置，不像fc通常要在网络的最后使用。这使得我们可以通过结合non-local以及local的信息构造出更丰富的结构。**

问题：

1)$g(x_j)$用于计算输入信号在$j$位置的特征值，如何理解？$g(x_j)$是原始输入信号值？？

$g$相当于一个transformation，是原始信号的一个特征表示。



(2).实现方式

定义中的$f,g$有不同的形式，就是non-local的不同表现方式。实验显示不同的表现形式其实对non-local的结果并没有太大影响，表明non-local这个行为才是主要的提升因素。

为了简化，论文只考虑$g$是线性的情况，即$g(x_j)=W_gx_j$，其中$W_g$是一个可学的权重矩阵，实际中是通过空间域的$1*1conv$或时空域的$1*1*1conv$实现的。

1)高斯形式

$f(x_i,x_j)=e^{x^T_ix_j}$

$x^T_ix_j$其中是点乘相似度（dot-product similarity）。也可以用欧式距离，但是点乘在深度学习平台上更好实现。此时归一化参数$C(x)=\sum_jf(x_i,x_j)$。
2)Embedded Gaussian

高斯函数的一个简单的变种就是在一个embedding space中去计算相似度，$f(x_i,x_j)=e^{\theta(x_i)^T\phi(x_j)}$.其中$\theta(x_i)=W_\theta x_i$和$\phi(x_j)=W_\phi x_j$.归一化参数与之前一致。

我们发现self-attention模块其实就是non-local的embedded Gaussian版本的一种特殊情况。对于给定的$i$，$\frac{1}{C(x)}f(x_i,x_j)$就变成了计算所有$j$的$softmax$，即$y=softmax(x^TW_\theta^TW_\phi x)g(x)$，这就是[47]中self-attention的表达形式。这样我们就将self-attention模型和传统的非局部均值联系在了一起，并且将sequential self-attention network推广到了更具一般性的space/spacetime non-local network，可以在图像、视频识别任务中使用。
另外和[47]的结论不同，文章发现attention的形式对于要研究的case来说并不是必须的。为了说明这一点，文章给出另外两种$f$的表达形式。

3)Dot product

$f$也可以定义成点乘相似度，即$f(x_i,x_j)=\theta(x_i)^T\phi(x_j)$.

在这里，归一化参数设为$C(x)=N$，其中$N$是$x$的位置的数目，而不是$f$的和，这样可以简化梯度的计算。这种形式的归一化是有必要的，因为输入的size是变化的，所以用x的size作为归一化参数有一定道理。

dot product和embeded gaussian的版本的主要区别在于是否做softmax，softmax在这里的作用相当于是一个激活函数。

4)Concatenation

$f(x_i,x_j)=ReLU(w_f^T[\theta(x_i),\phi(x_j)])$

这里的$[.,.]$表示的是concat，$w_f$是能够将concat的向量转换成一个标量的权重向量。这里设置$C(x)=N$.

问题：

1)Concatenation中的concat??



(3)Non-local Block

将上述的non-local操作变形为一个non-local block，以便于其可以插入到已有的结构中。定义一个non-local block为:

$z_i=W_zy_i+x_i$,

![non-local block](/Users/momo/Documents/video/non-local block.png)

**Non-local Blocks的高效策略**。设置$W_g$,$W_\theta$,$W_\phi$的channel的数目为$x$的channel数目的一半，这样就形成了一个bottleneck，能够减少一半的计算量。$W_z$再重新放大到$x$的channel数目，保证输入输出维度一致。
还有一个subsampling的trick可以进一步使用，就是将(1)式变为：$y_i=\frac{1}{C(\hat{x})}\sum_jf(x_i,\hat{x}_j)g(\hat{x}_j)$，其中$\hat{x}$是$x$下采样得到的（比如通过pooling），我们将这个方式在空间域上使用，可以减小1/4的pairwise function的计算量。**这个trick并不会改变non-local的行为，而是使计算更加稀疏了**。这个可以通过在图2中的$\phi$和$g$后面增加一个max pooling层实现。
作者在本文中的所有non-local模块中都使用了上述的高效策略。

问题：

1)figure2中non-local机制体现在哪一步？？如何体现长范围依赖？？

将输入的$x$看为一个整视频，有$T$帧，每一帧规模为$H*W$，1024个channel(已经经过一些卷积核)。

**权重部分**：

$T*H*W*1024$通过$1*1*1*512$卷积核($\theta$)变为$T*H*W*512$，将$T$帧的pixel平铺为一个向量($T*W*H$的feature拉伸为一个向量)，变为$THW*512$的矩阵，每一列都是所有的pixel，作为$\theta(x)$；

$T*H*W*1024$通过$1*1*1*512$卷积核($\phi$)变为$T*H*W*512$，将$T$帧的pixel平铺为一个向量($T*W*H$的feature拉伸为一个向量)，变为$THW*512$的矩阵，每一列都是所有的pixel，转置一下每一行都是所有的pixel，作为$\phi(x)$；

$\theta(x)\phi(x)$是$THW*THW$的矩阵，元素$a_{ij}$可以看作$f(x_i,x_j)$，再对每一行做归一化(每一个元素除以该行元素的和)，相当于$softmax$。元素$a_{ij}$是pixel中$x_i,x_j$之间的权重系数(0-1之间)，相当于$\frac{1}{C(x)}f(x_i,x_j)$。

**原始信号部分**：

$T*H*W*1024$通过$1*1*1*512$卷积核($g$)变为$T*H*W*512$，将$T$帧的pixel平铺为一个向量($T*W*H$的feature拉伸为一个向量)，变为$THW*512$的矩阵，每一列都是所有的pixel，作为$g(x)$。

**乘积**

将权重部分($THW*THW$)和原始信号部分($THW*512$)的结果相乘，结果为$THW*512$，权重部分的第$i$行乘以原始信号的第$j$列相当于$\frac{1}{C(x)}\sum_jf(x_i,x_j)g(x_j)$.

**残差块**

最后与原始信号相加即为残差块。





5.视频分类模型

将non-local block插入到C2D或I3D中，就得到了non-local nets

问题：

1)Table1里面ResNet-50 C2D中的res指什么？为何都对应3行？并且后面都乘以一个数字？

3行是resnet中的bottle neck结构(如下图右边)，一个bottle neck由三个卷积堆叠而成，*3是连续三个bottle neck。resnet-50就是由若干bottle neck块以及若干conv block构成。

![WechatIMG14](/Users/momo/Documents/video/WechatIMG14.jpeg)

#### 二、代码复现

https://github.com/Tushar-N/pytorch-resnet3d



**#################################################################**

### Fast RCNN

#### 论文：Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

##### 一、论文理解

1.摘要

文章引入了一个区域提出网络（RPN），该网络与检测网络(Fast R-CNN)共享全图像的卷积特征；region proposal network可以理解为生成box的网络，region就是box。

文章将RPN和Fast R-CNN通过共享卷积特征进一步合并为一个单一的网络——使用最近流行的具有“注意力”机制的神经网络术语，RPN组件告诉统一网络在哪里寻找目标。

2.简介

(1)忽略花费在区域提议上的时间，最新版本Fast R-CNN[2]利用非常深的网络[3]实现了接近实时的速率；

(2)我们引入了新的区域提议网络（RPN），它们共享最先进目标检测网络的卷积层。通过在测试时共享卷积，计算区域提议的边际成本很小（例如，每张图像10ms）；

问题：1)RPN和Fast R-CNN共享卷积层如何理解？



(3)在这些卷积特征之上，我们通过添加一些额外的卷积层来构建RPN，这些卷积层同时在规则网格上的每个位置上回归区域边界和目标分数。因此RPN是一种**全卷积网络（FCN）**，可以针对生成检测区域建议的任务进行端到端的训练；

问题：1)Faster RCNN这篇文章里面提出的RPN是在Fast RCNN的基础上加一些卷积层，然后整体称为RPN，还是说后面添加的一些卷积层称为RPN?添加的卷积层有啥特点?(如果是一般的卷积层不能加快计算速度)

后续添加的一些卷积层称为RPN，后续文章里面说明为了整合RPN和Fast RCNN。



(4)RPN旨在有效预测具有广泛尺度和长宽比的区域提议， 引入新的“锚”盒作为多种尺度和长宽比的参考。模型在使用单尺度图像进行训练和测试时运行良好，从而有利于运行速度。

问题：1)广泛尺度？长宽比？2)“锚”？3)单尺度图像？



(5)在PASCAL VOC检测基准数据集上[11]综合评估了方法，其中具有Fast R-CNN的RPN产生的检测精度优于使用选择性搜索的Fast R-CNN的强基准；

(6)RPN和Faster R-CNN的框架已经被采用并推广到其他方法，如3D目标检测[13]，基于部件的检测[14]，实例分割[15]和图像标题[16]；

(7)在ILSVRC和COCO 2015竞赛中，Faster R-CNN和RPN是ImageNet检测，ImageNet定位，COCO检测和COCO分割中几个第一名参赛者[18]的基础。



3.相关工作

(1)region proposal

(2)Deep Networks for Object Detection



4.Faster R_CNN

Faster R_CNN由两个模块组成。第一个模块是深度全卷积网络用来提出区域，第二个模块是使用提出的区域的Fast R-CNN检测器[2]。整个系统是一个单个的，统一的目标检测网络（图2）。使用最近流行的“注意力”[31]机制的神经网络术语，RPN模块告诉Fast R-CNN模块在哪里寻找.

![Faster R-CNN](/Users/momo/Documents/video/Faster R-CNN.png)



分析比较R-CNN，SPP Net，Fast R-CNN以及Faster R-CNN的区别和联系(https://blog.csdn.net/v_JULY_v/article/details/80170182?utm_source=distribute.pc_relevant.none-task)

R-CNN(Selective Search+CNN+SVM)

R-CNN的简要步骤如下
(1) 输入测试图像
(2) 利用选择性搜索Selective Search算法在图像中从下到上提取2000个左右的可能包含物体的候选区域Region Proposal
(3) 因为取出的区域大小各自不同，所以需要将每个Region Proposal缩放（warp）成统一的227x227的大小并输入到CNN，将CNN的fc层的输出作为特征
(4) 将每个Region Proposal提取到的CNN特征输入到SVM进行分类

SPP Net(ROI pooling)

在R-CNN中，“因为取出的区域大小各自不同，所以需要将每个Region Proposal缩放（warp）成统一的227x227的大小并输入到CNN”。

但warp/crop这种预处理，导致的问题要么被拉伸变形、要么物体不全，限制了识别精确度。例如一张16:9比例的图片硬是要Resize成1:1的图片，会导致图片失真。

SPP Net的作者Kaiming He等人逆向思考，既然由于全连接FC层的存在，普通的CNN需要通过固定输入图片的大小来使得全连接层的输入固定。那借鉴卷积层可以适应任何尺寸，为何不能在卷积层的最后加入某种结构，使得后面全连接层得到的输入变成固定的呢？

这个“化腐朽为神奇”的结构就是spatial pyramid pooling layer。下图便是R-CNN和SPP Net检测流程的比较：

![spp net](/Users/momo/Documents/video/spp net.png)

它的特点有两个:
1.结合空间金字塔方法实现CNNs的多尺度输入。
SPP Net的第一个贡献就是在最后一个卷积层后，接入了金字塔池化层，保证传到下一层全连接层的输入固定。换句话说，在普通的CNN机构中，输入图像的尺寸往往是固定的（比如224*224像素），输出则是一个固定维数的向量。SPP Net在普通的CNN结构中加入了ROI池化层（ROI Pooling），使得网络的输入图像可以是任意尺寸的，输出则不变，同样是一个固定维数的向量。

2.只对原图提取一次卷积特征
在R-CNN中，每个候选框先resize到统一大小，然后分别作为CNN的输入，这样是很低效的。而SPP Net根据这个缺点做了优化：只对原图进行一次卷积计算，便得到整张图的卷积特征feature map，然后找到每个候选框在feature map上的映射patch，将此patch作为每个候选框的卷积特征输入到SPP layer和之后的层，完成特征提取工作。



Fast R-CNN(Selective Search+CNN+ROI)

R-CNN的进阶版Fast R-CNN就是在R-CNN的基础上采纳了SPP Net方法，对R-CNN作了改进，使得性能进一步提高。

R-CNN和Fast R-CNN的区别：

先说R-CNN的缺点：即使使用了Selective Search等预处理步骤来提取潜在的bounding box作为输入，但是R-CNN仍会有严重的速度瓶颈，原因也很明显，就是计算机对所有region进行特征提取时会有重复计算，Fast-RCNN正是为了解决这个问题诞生的。

![fastr-cnn](/Users/momo/Documents/video/fastr-cnn.png)

与R-CNN框架图对比，可以发现主要有两处不同：一是最后一个卷积层后加了一个ROI pooling layer，二是损失函数使用了多任务损失函数(multi-task loss)，将边框回归Bounding Box Regression直接加入到CNN网络中训练。

(1) ROI pooling layer实际上是SPP-NET的一个精简版，SPP-NET对每个proposal使用了不同大小的金字塔映射，而ROI pooling layer只需要下采样到一个7x7的特征图。对于VGG16网络conv5_3有512个特征图，这样所有region proposal对应了一个7*7*512维度的特征向量作为全连接层的输入。

换言之，这个网络层可以把不同大小的输入映射到一个固定尺度的特征向量，而我们知道，conv、pooling、relu等操作都不需要固定size的输入，因此，在原始图片上执行这些操作后，虽然输入图片size不同导致得到的feature map尺寸也不同，不能直接接到一个全连接层进行分类，但是可以加入这个神奇的ROI Pooling层，对每个region都提取一个固定维度的特征表示，再通过正常的softmax进行类型识别。

(2) R-CNN训练过程分为了三个阶段，而Fast R-CNN直接使用softmax替代SVM分类，同时利用多任务损失函数边框回归也加入到了网络中，这样整个的训练过程是端到端的(除去Region Proposal提取阶段)。

也就是说，之前R-CNN的处理流程是先提proposal，然后CNN提取特征，之后用SVM分类器，最后再做bbox regression，而在Fast R-CNN中，作者巧妙的把bbox regression放进了神经网络内部，与region分类和并成为了一个multi-task模型，实际实验也证明，这两个任务能够共享卷积特征，并相互促进。

![Fast R-CNN](/Users/momo/Documents/video/Fast R-CNN.png)



Faster R-CNN(RPN+CNN+ROI)

Fast R-CNN存在的问题：存在瓶颈：选择性搜索，找出所有的候选框，这个也非常耗时。需要找出一个更加高效的方法来求出这些候选框。

解决：加入一个提取边缘的神经网络，也就说找到候选框的工作也交给神经网络来做。

在Fast R-CNN中引入Region Proposal Network(RPN)替代Selective Search，同时引入anchor box应对目标形状的变化问题（anchor就是位置和大小固定的box，可以理解成事先设置好的固定的proposal）。

具体做法：
1.对整张图片输进CNN，得到feature map
2.卷积特征输入到RPN，得到候选框的特征信息
3.对候选框中提取出的特征，使用分类器判别是否属于一个特定类 
4.对于属于某一类别的候选框，用回归器进一步调整其位置

RPN简介：
　　• 在feature map上滑动窗口
　　• 建一个神经网络用于物体分类+框位置的回归
　　• 滑动窗口的位置提供了物体的大体位置信息
　　• 框的回归提供了框更精确的位置

一种网络，四个损失函数;
　　• RPN calssification(anchor good.bad)
　　• RPN regression(anchor->propoasal)
　　• Fast R-CNN classification(over classes)
　　• Fast R-CNN regression(proposal ->box)

问题：

1)anchor(提前指定的region proposal)在RPN中如何使用？

2)RPN如何得到框的位置？

(可参考https://blog.csdn.net/gentelyang/article/details/80469553)

3)如何理解框的回归？四个值微调到四个值，还是多个值回归到四个值？



##### 二、代码复现

https://github.com/rbgirshick/py-faster-rcnn





**#################################################################**

### GCN+I3D+Faster R-CNN+Non-local应用到视频分类中

#### 论文：Videos as Space-Time Region Graphs

##### 一、论文理解

1.摘要

这篇文章将视频表示为时空区域图，图节点是由来自不同帧的目标区域定义的。这些节点由两种类型的关系连接:(i)捕捉相关对象之间的长期依赖关系的相似关系和(ii)捕捉附近对象之间的交互作用的时空关系。

问题：

1)节点是每一帧当中的目标区域？两种节点关系具体分别指什么？



2.简介

(1)动作识别需要两个关键因素，一是物体的形状以及形状如何变化；二是为动作识别建模‘人-对象’和‘对象-对象’交互。

(2)当前流行的网络并不能做到这两点因素。1)目前最先进的基于双流卷积神经网络的方法仍然在学习如何根据单个视频帧或局部运动向量对动作进行分类。局部运动显然不能模拟形状变化的动力学；2)RNN和C3D这些框架都关注于从整个场景中提取的特性，而未能捕获长期的时间依赖(转换)或基于区域的关系。

(3)我们将输入视频表示为空时区域图，其中图中的每个节点表示视频中感兴趣的区域。节点之间的关系有两种定义方式：1)具有相似外观或语义相关的区域被连接在一起；2)空间上重叠、时间上相近的物体被连接在一起。

(4)所提出的方法，在Charades和Something-Something这种只靠2D的方法是很难分类的数据集上显著提升。

(5)文章的贡献：1)提出了一种长视频中不同物体间关系的图像表示方法；2)多关系边推理的图卷积网络模型；3)在复杂的环境下，获得了动作识别中比当前最流行的方法还好的表现。

问题：

1)双流网络只考虑局部动作？(因为光流只在相邻帧之间计算？)RNN和C3D为何关注于整个场景？(因为卷积方式？)



3.相关工作

1)手动设计特征IDT；

2)神经网络RNN和C3D，然而是从整个场景中提取特征；

3)在非局部神经网络[58]中(Non-local Neural Networks)，对空间和时间的两两关系进行了建模。然而，**在特征空间中的每一个像素(从低层到高层)都应用了非局部算子**，而我们的推理是基于一个具有对象级特征的图。此外，**非局部操作符不处理任何时间顺序信息，而时序信息在我们的时空关系中显式建模的**；

问题：

1)Non-local Neural Network有什么用？(https://blog.csdn.net/elaine_bao/article/details/80821306，论文Non-local Neural Network)

2)非局部操作符不处理任何时间顺序信息，而时序信息在我们的时空关系中显式建模的，如何理解？



4)GCNs的输出是每个对象节点的更新之后的特征，可用于执行分类；

5)与这些工作不同的是，**我们的时空图不仅编码局部关系，而且编码跨越时空的任何一对物体之间的长范围依赖关系。通过使用具有长范围关系的图卷积，它支持在对象的起始状态和结束状态之间有效地传递消息。**这种全局图的推理框架提供了在最先进技术基础上的重大改进。



4.模型总览

![ECCV model](/Users/momo/Documents/video/ECCV model.png)

模型将一长段视频帧(超过5秒)作为输入，并将其转发给3D卷积神经网络；

步骤：

1)首先利用I3D网络提取视频的特征，最后一个卷积层的输出的视频特征是T×H×W×d.是视频帧数(时间维度)，H*W是feature map的空间维度，d是channel。在用I3D卷积网络提取特征之前，使用RPN提取框架(bounding box)；

2)给定每个T特征帧的边界框，我们应用RoIAlign来提取每个边界框的特征。注意，RoIAlign是独立地应用于每个特性帧上的。每个object的特征向量有d维(首先将每帧上RoIAlign作用后的region proposal对齐到7\*7\*d，然后maxpooling到1\*1\*d)。我们将object编号表示为N，因此RoIAlign之后的特征维度为N*d(N的值为T帧上所有object的数量个数值)；

3)我们现在构造一个包含N个节点的图，这些节点对应于在T帧上聚合的N个object proposals，为了简单起见，我们将这个大图分解为两个具有相同节点但有两个不同关系的子图:相似图和时空图。除了GCN特性外，我们还对整个视频表示进行平均池处理(T\*H\*W\*d)，以获得与全局特性相同的d维特性。然后将这两个特性连接起来进行视频级分类；

问题：

1)RPN和I3D如何结合使用？

2)RoIAlign是什么？如何对每一帧使用？

RoIAlign类似于RoIPooling，可以裁剪和规范object feature到相同的规模，Fast-RCNN中有介绍。



5.视频中图表示

(介绍特征提取和图的构建构成)

5.1视频表示

(1)3D ConvNet

给定一段较长的视频(约5秒)，我们从中抽取32个视频帧，每两帧之间的时间长度相同，用一个3D ConvNet(ResNet-50 I3D model)来提取这些视频帧的特征。这个模型的输入是32\*224\*224，32帧，每一帧的空间规模是224*224；输出的feature map规模是16\*14\*14；

(2)RPN

先用RPN提取每个视频帧中感兴趣区域的bounding box，再输入到ResNet-50+I3D中；为了在最后一个卷积层的顶层提取对象特征，我们**将边界框从16个输入RGB帧(从32个I3D输入帧中采样，采样率为每2帧1帧)投射到16个输出特征帧**；

(3)RoIAlign

然后将16个带有bounding box的特征帧输入到RoIAlign(RoIAlign类似于RoIPooling，可以裁剪和规范object feature到相同的规模，Fast-RCNN中有介绍)，提取每个object proposal里的特征；将feature proposal池化到$7*7*d$的feature，再池化到$1*1*d$feature输出；

问题：RoIAlign如何提取特征？卷积还是池化？



5.2图的相似性















##### 二、代码复现：(复现I3D和GCN组合，即文章中table 4中的最后一行，上面其余几行可以注释相应的代码复现)

1.something-something数据集

something数据集是一个中等规模的数据集，是为了识别视频中的动作的，其甚至都不在意具体是什么实施了这个动作，又有谁承载了这个动作。其标注形式就是**Moving something down，Moving something up**这种。这也是数据集名字的由来。从我目前的阅读来看，这个数据集相对小众了一些，目前在V1上的结果少有top1超过50%，在V2上少有超过60%。V1/V2均有174个类别，分别有108499/220847个的视频。处理数据的时候一定要看准数据数量是否准确。发布于2017年。((https://zhuanlan.zhihu.com/p/69064522))

