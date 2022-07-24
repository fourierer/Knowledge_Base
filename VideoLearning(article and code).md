

# 文献阅读以及代码解读

专业术语：

（1）frames：视频帧；

（2）clips：一个视频的片段会很长，需要sample很多个短的clips，最后的预测结果是所有clip平均的结果，可以理解为一个clip就是一次sample。比如一个视频有200帧，设定一个clip要sample32帧，此时不能uniform sample，这样帧的间隔会很大，一般每两帧sample一帧，所以一个clip覆盖了64帧。这样一个视频就可以sample很多个clips；

（3）crop：裁剪，在Image Recognition中，预处理都需要crop来做数据增强，video classification也是一样，可以对不同的clips进行crop，这样最终的crops=clips*crop，来得到更多的clips。



#################################################################

### 视频抽帧采样（sampling）

#### 论文：Temporal Segment Networks: Towards Good Practices for Deep Action Recognition

##### 一、论文理解

(该论文是最早将视频sample之后再分析的文章)

视频抽帧：1)减小计算量；2)视频中的帧会有很多冗余。现在的视频处理与分析都会先抽帧再分析。





##### 二、代码复现



**#################################################################**

### 光流-1

#### 论文：Dense trajectories and motion boundary descriptors for action recognition



问题：兴趣点如何选？（采样？）光流在兴趣点之间如何计算？











**#################################################################**

### 光流-2

#### 论文：Action Recognition with Improved Trajectories



问题：改进地方在哪里？









**#################################################################**



### 双流网络

#### 论文：Two_Stream Convolutional Networks for Action Recognition in video

##### 一、论文理解

1.视频分类要捕获静态帧的外观信息和帧之间的动作信息；

2.双流网络，空间流从静态帧学习，时间流以密集光流的形式来学习动作信息;

3.

问题：

1.双流网络当中空间流网络输入的形式是？（单帧图片还是经过采样的所有视频帧？）

2.时间流网络中，为何无法使用图像分类的权重进行预训练？（光流是单通道的？）



##### 二、代码复现：（https://github.com/jeffreyyihuang/two-stream-action-recognition）

1.spatial_cnn.py

(1)23行

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
```

(2)25-31行

argparse是命令行参数解析模块，使用时需要import argparse，使用该工具可以直接在linux命令行中进行调参，不需要在.py文件中修改参数。下面进行测试说明该模块的用法。

https://blog.csdn.net/huangfei711/article/details/80325946?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task， 链接中是各种情况，主要有：--和-都可以进行赋值，--是全称，-是缩写，不加-和--的表示必须要赋值参数；action=’store_true’，表示该选项不需要接收参数，只需指定，如python test.py --evaluate.即可设定该参数为 true；如果不写明，则为false。

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
print(y.lr)
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

测试命令3：$ python Test.py --epochs 40 --lr 0.2

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

### 双流改进

#### 论文：Temporal Segment Networks: Towards Good Practices for Deep Action Recognition

为了改进双流网络中时间流网络无法使用图像分类预训练的结果，将光流信息映射到0～255之间，形成灰度图；将预训练的权重（三维形式）平均到一个通道，与光流信息channel相对应。









**#################################################################**

### RNN用于视频分类

#### 论文：Beyond short snippets: Deep networks for video classification

效果不是特别好，应用不多。







**#################################################################**

### C3D，三维卷积网络



#### 论文：3D convolutional neural networks for human action recognition





#### 论文：Learning Spatiotemporal Features with 3D Convolutional Networks

使用3D卷积核，直觉上更适合视频内容，但是无法使用二维卷积网络预训练权重，并且参数较多。







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

### R(2+1)D

#### 论文：A Closer Look at Spatiotemporal Convolutions for Action Recognition

通过分解3D卷积来减少3D卷积网络的卷积核参数，网络深度比3D卷积网络更深，性能更好。

##### 一、论文理解

1.简介

（1）深度网络I3D目前在动作识别方面的效果最好，但与最佳手工方法iDT相比，改进的幅度并不像图像分类那样巨大；

（2）基于图像的2D CNN (ResNet-152)操作在视频的单独帧上实现的性能非常接近最先进的sport-1m基准。这一结果既令人惊讶又令人沮丧，因为2D CNNs无法模拟时间信息和运动模式，而后者被认为是视频分析的关键方面。基于这些结果，我们可以假设时间推理对于精确的动作识别来说并不是必须的，因为在一个序列的静态帧中已经包含了强大的动作类信息；

（3）我们证明，在同样深度的情况下，在具有挑战性的动作识别基准(如Sports- 1M和Kinetics)上进行训练和评估时，3D ResNets的表现明显优于2D ResNets；

（4）提出两种时空卷积的新形式，可以看作是介于二维(空间卷积)和三维卷积之间的一种卷积形式：

1）第一种形式被称为混合卷积(MC)，它只在网络的早期层使用3D卷积，而在顶层使用2D卷积，这种设计背后的基本原理是，运动建模是一种低/中级的操作，可以通过网络早期层的3D卷积来实现，对这些中级操作特性(通过顶层的2D卷积来实现)进行空间推理，从而实现精确的动作识别；这种方法的精度比2D卷积网络高3%-4%，并且与具有3倍多参数的3D卷积网络的性能相匹配；

2）第二种形式是(2+1)D卷积块，将3D卷积分解为两个独立的、连续的操作：2D空间卷积和1D时间卷积；



2.相关工作

（1）在本研究中，包括了帧上的2D卷积、clips上的2D卷积、3D卷积、交错(混合)3D-2D卷积，以及将3D卷积分解为2D空间卷积和1D时间卷积的过程，即(2+1)D卷积；

**注：帧上的2D卷积是逐帧（假设有L帧）进行卷积，得到L个feature map，k个卷积核就得到kL个feature map作为下一层的输入，在网络的最后层进行spatialtemporal pool融合时序信息；clips上的2D卷积相当于多通道卷积，把clips中的所有帧（假设有L帧）当成一个整体，共3L个通道（每帧有3通道），进行多通道的2D卷积。**

（在3中会有介绍）



3. Convolutional residual blocks for video

在这项工作中，只考虑“普通的”残差块(即每个block由两个convolutional layer组成，每个layer之后都有ReLU激活函数；如下图所示：

![ResNet3D_simple_block](/Users/momo/Documents/video/ResNet3D_simple_block.png)

3.1. R2D: 2D convolutions over the entire clip

（1）clips上的2D卷积忽视了时序，将L帧看成通道，把clips中的所有帧（假设有L帧）当成一个整体，共3L个通道（每帧有3通道），进行多通道的2D卷积；

3.2. f-R2D: 2D convolutions over frames

（1）帧上的2D卷积是逐帧（假设有L帧）进行卷积，得到L个feature map，k个卷积核就得到kL个feature map作为下一层的输入，在网络的最后层（top层）进行spatialtemporal pool融合时序信息；

3.3. R3D: 3D convolutions

（1）卷积核在时间上的尺寸设置为3，即一次卷3帧；

3.4. MCx and rMCx: mixed 3D-2D convolutions

在ResNet3D（simple block系列，每个block只有两层）的基础之上，将最后一层变为clip上的2D卷积得到的网络称为MC5，将最后两层变为clip上的2D卷积得到的网络称为MC4，依次类推，MC1实际上就是clip上的2D卷积对应的网络，如下图a)；MCx系列如下图b)；rMCx是将MCx的中间5个block逆过来，如下图c)。

![r(2+1)d_variants](/Users/momo/Documents/video/r(2+1)d_variants.png)

3.5. R(2+1)D: (2+1)D convolutions

（1）将3D卷积核$N_{i-1}\times t\times d\times d$分解为2D的空间卷积$N_{i-1}\times 1\times d\times d$和1D的时间卷积$M_i\times t\times 1\times 1$；

参数量如下：

1）3D卷积核：$N_{i-1}\times t\times d\times d\times N_i$；

2）(2+1)D卷积核：$N_{i-1}\times 1\times d\times d\times M_i+M_i\times t\times 1\times 1\times N_i$；

（2）R(2+1)D的优点有两个，一是通过选择$M_i$（$M_i=\frac{N_{i-1}\times t\times d\times d\times N_i}{N_{i-1}\times 1\times d\times d+\times t\times 1\times 1\times N_i}$）的值，让R(2+1)d的参数量与3D卷积模块一样，这样在保证参数一样的同时还增加了非线性激活个次数，增加模型拟合能力；二是2D卷积和1D卷积相对于3D卷积来说比较好优化；



4. Experiments

（1）基础网络使用ResNet3D-18,34，如下图所示：

![ResNet18_34](/Users/momo/Documents/video/ResNet18_34.png)

（2）将帧Resize到$128*171$，再随机裁减到$112*112$，选取连续的$L$帧作为输入；epoch为1000，batch_size为32；初始学习率为0.01，然后指数衰减，每10个epoch学习率除以10；







**#################################################################**

### S3D-G

#### 论文：Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification











### 下面的几篇论文跟视频分类无直接关系，但都是ECCV_wang这篇文章中所用到的工具，所以也需要了解

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

(1)$A$是邻接矩阵，除非图上有自环，否则对角线上的元素全都为0。所以对于$AH^l$，$A$的第$i$行乘以$H^l$的第$j$列时，$A$的第$i$行表示第$i$个节点与周围节点的邻接值，$H^l$的第$j$列所得值是节点$i$表示各个节点的第$j$维特征，所以$A$的第$i$行乘以$H^l$的第$j$列表示用$i$节点对其他节点的邻接值对各个节点的第$j$维特征做加权，得到第$i$个节点新特征的第$j$维，不包括节点自身的特征；(可以通过矩阵$A$加上单位阵来解决这个问题)

(2)矩阵$A$没有正则化，与$A$相乘之后会改变特征的范围。可以预处理对A做正则化如$D^{-1}A$，其中$D$是度矩阵，是矩阵$A$每行元素的和作为对角元素。此时对于$D^{-1}AH^l$，$D^{-1}A$的第$i$行乘以$H^l$的第$j$列相乘所得的值表示用$i$节点对其他节点的邻接值对各个节点的第$j$维特征做加权；有时候也使用归一化准则$D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$，此时不再是邻居节点的特征加权和的平均值。

(为了解决上述两个局限性，假设传播方式为$H^{l+1}=\sigma(\hat D^{-\frac{1}{2}}\hat{A}\hat D^{-\frac{1}{2}}H^{l}W^{l})$，其中：$\hat{A}=A+I$，这就是谱卷积的形式。后面会介绍)

举例说明：

对于骨架行为识别任务：假设数据集有17个关键点，矩阵$A_{17*17}$表示骨架关节点之间的邻接关系，特征$H_{17*3}$表示各个关节点的特征，假设特征包括两个坐标和一个置信因子，即$(x,y,c)$，$W_{3*64}$是要训练的参数，假设特征从3维到64维，$AH_{ij}$表示用第$i$个关节点的邻接权重对所有关节点的第$j$维特征做加权，结果作为第$i$个节点新特征的第$j$维，$W$起到更改特征维数的作用，相当于正常卷积中的通道个数。

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

1)figure2中block如何和公式（1）联系？即如何体现一个位置$x_i$和其他位置$x_j$的长范围依赖（Non-Local）关系？

将输入的$x$看为一个整视频，有$T$帧，每一帧空间尺寸为$H*W$，1024个channel(已经经过一些卷积核的结果)，则公式（1）中的$x_i$指的是$1024*1$的向量：要明确公式中位置的概念，一个$T$帧的视频，空间规模是$H*W$，则在时空上共有$THW$个位置，每个位置$x_i$都是一个$1024*1$的向量，下面解释Non-Local（长范围依赖关系）：

**权重部分**：

$T*H*W*1024$通过$1024*1*1*1*512$卷积核(即$W_{1024*512}$)变为$T*H*W*512$，这一步相当于$\theta(x)$，即$Wx$，将$1024*1$的$x_i$变为$512*1$的$x_i$，此时每个位置都是$\theta(x_i)$。再将$T$帧的pixel平铺为一个向量($T*W*H$的feature拉伸为一个向量)，变为$THW*512$的矩阵，每一列都是一个位置$x_i$；

$T*H*W*1024$通过$1*1*1*512$卷积核($\phi$)变为$T*H*W*512$，将$T$帧的pixel平铺为一个向量($T*W*H$的feature拉伸为一个向量)，变为$THW*512$的矩阵，每一列都是一个位置$x_j$，转置一下每一行都是一个位置$x_j$；

$\theta(x)\phi(x)$是$THW*THW$的矩阵，元素$a_{ij}$可以看作$\theta(x_i)\phi(x_j)$，再对每一行做归一化(每一个元素除以该行元素的和)，相当于$softmax$。元素$a_{ij}$是pixel中$x_i,x_j$之间的权重系数(0-1之间)，相当于$\frac{1}{C(x)}f(x_i,x_j)$。

**原始信号transformer**：

$T*H*W*1024$通过$1*1*1*512$卷积核($g$)变为$T*H*W*512$，将$T$帧的pixel平铺为一个向量($T*W*H$的feature拉伸为一个向量)，变为$THW*512$的矩阵，每一列都是一个位置$x_j$，作为$g(x)$。

**原始信号transformer和权重部分的加权乘积**：

将权重部分($THW*THW$)和原始信号transformer部分($THW*512$)的结果相乘，结果为$THW*512$，权重部分的第$i$行乘以原始信号transformer的第$j$列相当于$\frac{1}{C(x)}\sum_jf(x_i,x_j)g(x_j)$.

**残差块$z_i=W_zy_i+x_i$**：

加权乘积的结果为$y$，共$THW$个位置，每个位置是一个$512*1$向量$y_i$，还要经过一个$1*1*1$卷积$W_z$，最后和原始信号相加整体视为残差块。





5.视频分类模型

将non-local block插入到C2D或I3D中，就得到了non-local nets

问题：

1)Table1里面ResNet-50 C2D中的res指什么？为何都对应3行？并且后面都乘以一个数字？

3行是resnet中的bottle neck结构(如下图右边)，一个bottle neck由三个卷积堆叠而成，*3是连续三个bottle neck。resnet-50就是由若干bottle neck块以及若干conv block构成。

![WechatIMG14](/Users/momo/Documents/video/WechatIMG14.jpeg)

#### 二、代码复现

https://github.com/Tushar-N/pytorch-resnet3d



**#################################################################**

### Faster RCNN

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

这篇文章将视频表示为时空区域图，图节点是由来自不同帧的目标区域定义的。这些节点由两种类型的关系连接:(i)捕捉相关对象之间的长范围依赖的相似关系和(ii)捕捉附近对象之间的交互作用的时空关系。

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

**1.数据集下载**

下载something-something数据集压缩文件，网站：https://20bn.com/datasets/something-something/v1，需要注册才能下载。

something数据集是一个中等规模的数据集，是为了识别视频中的动作的，其甚至都不在意具体是什么实施了这个动作，又有谁承载了这个动作。其标注形式就是**Moving something down，Moving something up**这种。这也是数据集名字的由来。从我目前的阅读来看，这个数据集相对小众了一些，目前在V1上的结果少有top1超过50%，在V2上少有超过60%。V1/V2均有174个类别，分别有108499/220847个的视频。处理数据的时候一定要看准数据数量是否准确。发布于2017年。((https://zhuanlan.zhihu.com/p/69064522))

下载25个压缩文件(压缩数据集)以及4个csv文件(标签信息)，

(1)something-something-v1-label.csv中是174个类别的名称，如：

Holding something

Turning something upside down

......

(2)something-something-v1-train.csv和something-something-v1-validation.csv中存储训练集验证集的标签信息，如：

100218;Something falling like a feather or paper

24413;Unfolding something

.......

(3)something-something-v1-test.csv中只有视频类别的序号，没有标签

**数据集可以不用下载到video/classification/data/something/raw当中，只需将解压后的数据集跟该文件夹建立一个超链接即可，因为一般数据集会放在一个统一的文件夹里面(比如服务器中的data文件夹，代码放在home文件夹)，并且后续写其他算法用到这个数据集的时候，就可以省去复制的时间。**



**2.数据集的解压与整理**

(1)在数据集所在文件夹(something-something，自行设置)中解压数据

```shell
cat 20bn-something-something-v1-?? | tar zx
```

数据集解压之后something-something文件夹中出现20bn-something-something-v1的文件夹，里面共有108499个文件夹，共174个类别。每个类别文件夹中都是已经抽好帧的jpg图片，图片个数根据视频长短决定。

(2)创建超链接

```shell
ln -s .../20bn-something-something-v1 .../video_classification/data/something/raw
ln -s /data/sz/raw/something-something/20bn-something-something-v1 /home/sz/Video_Classification/eccv_wang/video_classification/data/something/raw
```

在something-something文件夹下输入指令ln -s .../20bn-something-something-v1 .../video_classification/data/something/raw(下面的实例是我自己的服务器上的路径)，注意在建立软链接时，源文件和目标文件一定要是完整的绝对路径。回车之后在raw文件夹下面生成一个快捷方式文件夹20bn-something-something-v1，这个快捷方式文件夹并不占内存，但是可以访问到原来的20bn-something-something-v1里的所有内容。

(3)整理数据集

video_classification/data/something/process.py

将/home/sz/Video_Classification/eccv_wang/video_classification/data/something/raw/20bn-something-something-v1中的数据集分为训练集和测试集两部分，此处的数据是从超链接的数据源中移动得到。

```python
import os
import tqdm
import pandas as pd
import subprocess

train = pd.read_csv('annotations/something-something-v1-train.csv', sep=';', header=None)
validation = pd.read_csv('annotations/something-something-v1-validation.csv', sep=';', header=None)
labels = pd.read_csv('annotations/something-something-v1-labels.csv', sep='\n', header=None)
#print(type(validation))  # <class 'pandas.core.frame.DataFrame'>
#print(labels)  # 174*1的DataFrame
#print(labels[0])  # DataFrame的第一列
#print(labels[0].to_dict().items())  # 将第一列的值和行号(类别)组成元组，元组再组成列表

labels = dict((v,k) for k,v in labels[0].to_dict().items())
#print(labels)  # 将第一列的值作为key，将行号(类别)作为value，组成字典

# 将train和validation中第二列的类别描述，换成数字形式(1-174)
train = train.replace({1: labels})
validation = validation.replace({1: labels})


# 将/home/sunzheng/Video_Classification/eccv_wang/video_classification/data/something/raw/20bn-something-something-v1中的数据集分为训练集和测试集两部分，此处的数据是从超链接的数据源中移动得到
for i in tqdm.tqdm(range(len(train))):
    if not os.path.isdir('frames/train/{}'.format(train.loc[i][1])):
        os.makedirs('frames/train/{}'.format(train.loc[i][1]))
    cmd = 'mv raw/20bn-something-something-v1/{} frames/train/{}'.format(train.loc[i][0], train.loc[i][1])
    subprocess.call(cmd, shell=True)


for i in tqdm.tqdm(range(len(validation))):
    if not os.path.isdir('frames/valid/{}'.format(train.loc[i][1])):
        os.makedirs('frames/valid/{}'.format(train.loc[i][1]))
        cmd = 'mv raw/20bn-something-something-v1/{} frames/valid/{}'.format(validation.loc[i][0], validation.loc[i][1])
    subprocess.call(cmd, shell=True)
```

移动之后，源文件就没有了训练和验证集，只有测试集。(可以通过解压再次得到所有数据集)



**3.环境配置**

(1)根据detectron2/INSTALL.md安装[detectron2](https://github.com/facebookresearch/detectron2)

安装之后可以conda  list查看安装列表，有detectron2库，结果如下：

![conda list](/Users/momo/Documents/video/conda list.png)



这里采用的detectron2是facebook写的一个用于提取bounding box的一个库，由于比较新，所以对服务器上的环境要求比较高，值得注意的有以下几点(不会写轮子只能跟着别人要求走！)：

1)Python >= 3.6，使用anaconda建立虚拟环境时可以解决；

2)Pytorch >= 1.3，搭建pytorch和相应版本的torchvision(通过pytorch官网的指令)。

3)GCC >= 5.0



(2)安装必要的package

```shell
pip install -r requirements.txt
```



**4.数据预处理**

1生成文件索引

```shell
python parse_annotations.py
```

parse_annotations.py:

```python
import os
import glob
import tqdm
import torch
import numpy as np

def parse_annotations(root):

    def parse(directory):

        data = []
        for cls in tqdm.tqdm(os.listdir(directory)):  # 对train或者test文件夹中的类别做循环，cls是类别名称
            cls = os.path.join(directory, cls)  #cls是类别路径
            for frame_dir in os.listdir(cls):  # 对类别中视频文件夹做循环，frame_dir是视频文件夹

                frame_dir = os.path.join(cls, frame_dir)  # frames_dir是视频文件夹路径

                frames = glob.glob('%s/*.jpg'%(frame_dir))  # 返回视频文件夹中所有匹配的文件路径列表
                if len(frames)<32:
                    continue
                frames = sorted(frames)
                frames = [f.replace(root, '') for f in frames]
                data.append({'frames':frames})
        return data

    train_data = parse('%s/frames/train'%root)  # train_data是一个列表，每一个值都是一个字典，这个字典只有一个键，对应的值是训练集中某个视频文件夹中所有帧图片的路径，且该视频文件夹超过32帧
    val_data = parse('%s/frames/valid'%root)  # 同理，test_data存储测试集中大于32帧的视频文件夹中所有帧图片的路径

    annotations = {'train_data':train_data, 'val_data':val_data}
    torch.save(annotations, 'data/something_data.pth')

# if not os.path.exists('data/something_data.pth'):
parse_annotations('data/something')
print ('Annotations created!')
```

该代码作用是将抽好帧的文件夹video_classification/data/frame中的训练集和测试集中的所有帧生成索引，保存在文件video_classification/data/something_data.pth中，可以查看something_data.pth文件中的内容(图中为一部分)：

![something_data_pth](/Users/momo/Documents/video/something_data_pth.png)



2.提取bounding box

提取的box将保存在data/something/bbox中，可以将demo/extract_box.py中73～77行注释取消，只提取部分frame的box，以加快速度：

```shell
cd detectron2
python demo/extract_box.py
```



3.提取i3d的inference result

提取的feature将保存在data/something/feats中，如果需要去掉non local block，可以将extract_feat.py第90行注释取消：

```shell
python extract_feat.py
```

当未下载i3d pertained weight时，应首先下载权重：

```shell
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl -P pretrained/
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_nonlocal_32x2_IN_pretrain_400k.pkl -P pretrained/

python -m utils.convert_weights pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_kinetics.pth
python -m utils.convert_weights pretrained/i3d_nonlocal_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_nl_kinetics.pth
```



**5.训练**

不同的GCN模型可从models/gcn_model.py第100, 101, 102行进行调整：

```python
python main.py

# 从保存的第5 epoch权重开始训练
python main.py --load_state 5
```







**#################################################################**

### ST-GCN

#### 论文：Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition

是一种基于骨骼点的动作识别方法, 通过人体姿态估计算法提取视频中人类关节位置的时间序列而对动态骨骼建模, 并将GCN扩展为时空图卷积网络而捕捉这种时空的变化关系来进行动作识别。

输入网络的$X$尺寸为$batch\_size*3*150*17$，$150$是每个样本包含的帧数，$17$代表每一帧包含的关节点个数，3表示每个关节点的特征维数。基于图卷积网络的公式$f(H^l)=\sigma(AH^lW^l)$，$A_{17*17}$是邻接矩阵，$H_{batch\_size*3*150*17}$是每层的特征，$W_{3*d}$是每一层的卷积核，用于更改每个关节点的特征维度。实际上的卷积操作流程（只简要介绍前几层）为：

输入网络的$X$尺寸为$batch\_size*3*150*17$：（1）经过gcn网络$3*192*1*1$卷积，输出$batch\_size*192*150*17$，即在空间维度做卷积变换每个关节点的特征维数，相当于图卷积网络公式中$W$矩阵的作用；（2）使用torch.enisum函数将邻接矩阵$A_{3*17*17}$乘到（1）输出的特征上，输出$batch\_size*64*150*17$（具体过程看代码），即用某个关节点与周边关节点的邻接关系来重新计算该点的特征，相当于图卷积网络公式中的矩阵$A$；（3）经过tcn网络$64*64*9*1$卷积，输出$batch\_size*64*150*17$，即在时间维度来卷积，加上gcn中对空间维度来做卷积，网络被称为时空图卷积网络。



**#################################################################**

### ResNet3D

#### 论文1：Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition

##### 一、论文理解

1.简介

将2DResNet拓展到3D，在Activity和Kinetics数据集上进行训练和测试。



2.相关工作

2.1. Action Recognition Database

（1）HMDB-51和UCF-101是不太大的数据集，在这两个数据集上很难训练出表现好的模型；

（2）Sports-1M和YouTube-8M是很大的数据集，但是标注存在噪声，即视频中有很多帧与标签的内容无关，会对训练造成影响；

（3）Kinetics数据集比Sports-1M和Youtube-8M小，但是标注质量非常高。



2.2. Action Recognition Approach

（1）双流网络、以及双流中每一流都使用2DResNet；

（2）C3D；

（3）I3D，使用了2D网络在ImageNet上的预训练结果；

（4）inception系列网络应用到3D中；

（5）这篇文章是将ResNet应用到3D中；



3.3D Residual Networks

3.1. Network Architecture



![3DResNet](/Users/momo/Documents/video/3DResNet.png)



3.2. Implementation

3.2.1 Training

使用带momentum的SGD训练，输入的视频是16帧，不够16帧的循环视频。

Training sample？？



3.2.2 Recognition



4.Experiments 

（1）ActivityNet![Result on ActivityNet](/Users/momo/Documents/video/Result on ActivityNet.png)

在小数据集上，3DResNet容易过拟合，在Sports-1M上预训练过的C3D效果更好。



（2）Kinetics

![Result on Kinetics](/Users/momo/Documents/video/Result on Kinetics.png)

3DResNet-34在没有预训练的情况下，在Kinetics数据集上的表现比在Sports-1M预训练的C3D要好。



#### 论文2：**Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?**

实验部分：

（1）各个网络结构在Kinetics数据集上的表现：

![3DResNet Kinetics](/Users/momo/Documents/video/3DResNet Kinetics.png)



（2）各个网络结构在Kinetics数据集上预训练，在UCF-101和HMDB-51上微调：

![3DResNet Kinetics pretrained](/Users/momo/Documents/video/3DResNet Kinetics pretrained.png)



##### 二、代码复现（https://github.com/kenshohara/3D-ResNets-PyTorch）





**#################################################################**

### CSN

#### 论文：Video Classification with Channel-Separated Convolutional Networks

##### 一、论文理解

1.简介

（1）受到轻量级的2D卷积神经网络启发，在本文中将3D卷积操作替换成pointwise $1*1*1$或者depthwise $3*3*3$；

（2）CSN网络中提出了两种分解，分别称为interaction-reduced和interaction-preserved；

（3）CSN网络有更高的训练误差，但是也有更好的泛化性能。

2.相关工作

2.1.Group convolution

各种轻量级卷积神经网络：Xception，MobileNet，ShuffleNet，ResNeXt...

2.2.Video classification

各种3D卷积分解方式：S3D，R(2+1)D...

CSN和Xception非常相似，Xception在通道和空间上分解2D卷积，CSN在通道和时空上分解3D卷积；

模型的变种ir-CSN与ResNeXt[39]及其3D版本[16]在使用瓶颈块进行组/深度卷积方面有相似之处。主要区别在于ResNext[39,16]在其3×3×3层中使用组卷积，组大小固定(如G = 32)，而我们的ir-CSN在所有3×3×3层中都使用深度卷积，这使得我们的架构完全是信道分离的；



3.Channel-Separated Convolutional Networks

3.1.background

Group convolution

![Group convolution](/Users/momo/Documents/video/Group convolution.png)

（1）传统卷积：输出的某个通道和输入的每一个通道有关，即卷积核与输入每一个通道都进行卷积运算；

（2）组卷积：输出的某个通道只和输入的固定几个通道有关，即卷积核与输入的固定的几个通道进行卷积运算；

（3）Depthwise convolution：输出的某个通道与输入的一个通道有关，即卷积核与输入的一个通道进行卷积运算；



Counting FLOPS, parameters, and interactions

给通道分组限制了通道之间的interaction，如果一个卷积层有$C_{in}$的通道数，分成$G$组，则每个核和$\frac{C_{in}}{G}$个通道相关，这些通道两两会有interaction，所以每个核涉及到$C_{\frac{C_{in}}{G}}^2$个interaction pairs。如下图所示：

![interaction](/Users/momo/Documents/video/interaction.jpeg)



计算各种量：

(1)parameters:$C_{out}$个卷积核，每个卷积核需要卷$k$帧，每一帧有$\frac{C_{in}}{G}$个channels，所以共$C_{out}\cdot\frac{C_{in}}{G}\cdot k^3$个参数；

(2)FLOPs:输出的每一个体素都通过$\frac{C_{in}}{G}\cdot k^3$次运算得到，共$C_{out}\cdot THW$个体素，共经过$C_{out}\cdot\frac{C_{in}}{G}\cdot k^3\cdot THW$；

(3)interactions:一共有$C_{out}$个3D卷积核（$k*k*k$），这里$k$指帧数，3D卷积核每次卷$k$帧。对于输入层来说，每一帧有$C_{in}$个channel，每一帧的内部channel都有$C_{\frac{C_{in}}{G}}^2$个interaction pairs，所以$C_{out}$个卷积核共$C_{out}\cdot C_{\frac{C_{in}}{G}}^2$。



3.2.Channel Separation

传统3D卷积操作包括了通道间的交互（channel interactions）和像素与周边元素的交互（local interaction）。但是在CSN网络中，把传统3D卷积操作分成$1*1*1$的传统卷积和$k*k*$的depthwise convolution，把这两种卷积方式放到不同的卷积层中实现不同的interaction：

（1）$1*1*1$传统卷积为了channel interaction，没有local interaction；

（2）$k*k*k$depthwise convolution为了local interaction，没有channel interaction；

实际上这部分对3D卷积操作的拆分（channel-separated）和MobileNet V1中对2D卷积

完全一样，在2D中叫depth-separated，在这篇文章中叫channel-separated。



3.3. Example: Channel-Separated Bottleneck Block

![Channel-Separated Bottleneck Block](/Users/momo/Documents/video/Channel-Separated Bottleneck Block.png)

Interaction-preserved channel-separated bottleneck block和Interaction-reduced channel-separated bottleneck block只是CSN中的两种结构，还有其他的block设计，3.4介绍。



3.4. Channel Interactions in Convolutional Blocks

Group convolution applied to ResNet blocks

![Group convolution applied to ResNet blocks1](/Users/momo/Documents/video/Group convolution applied to ResNet blocks1.png)

![Group convolution applied to ResNet blocks2](/Users/momo/Documents/video/Group convolution applied to ResNet blocks2.png)

4. Ablation Experiment

Two main findings:

(1)相似的网络深度和channel interaction的数量，会有相似的准确率，interaction-preserving blocks显著地减少了计算量，对于浅层网络只有轻微的精度损失，而对于更深层的网络则有精度的提高；

(2)传统的3D卷积一帧里面的各个channel之间都有interaction，对于深层网络，这会造成过拟合。



4.1. Experimental setup

1.Dataset：Kinetics是行为识别的一个标准的benchmark，240K的训练数据，20K的验证集；

2.Base architecture：以ResNet3D作为基础框架，模型输入的规模是$T*224*224$。

![ResNet3D architectures](/Users/momo/Documents/video/ResNet3D architectures.png)

结构中的ResNet3D-simple和ResNet3D-bottleneck中的单元分别是Figure-3a和Figure-4a中的simple和bottleneck。



4.2. Reducing FLOPs, preserving interactions

![Channel-Separated Networks vs ResNet3D](/Users/momo/Documents/video/Channel-Separated Networks vs ResNet3D.png)

比较不同深度下ResNet3D，ir-CSn和ip-CSN的表现，结果正如第4节开头描述的：相似的网络深度和channel interaction的数量，会有相似的准确率，interaction-preserving blocks显著地减少了计算量，对于浅层网络只有轻微的精度损失，而对于更深层的网络则有精度的提高。

实验结果：

1.对于浅层网络，ir-CSN和ip-CSN的表现都不如ResNet-3D。此时ResNet-3D效果好得益于参数比较多，并且ip-CSN表现比ir-CSN要好，因为保留了channel-interaction在一定程度上提高了准确率；

2.对于深层网络，ir-CSN和ip-CSN都比ResNet-3D要表现好。并且ir-CSN和ip-CSN的差距在缩小，因为channel interactions的数量几乎一样。



结论：channel interactions和模型的表现联系密切，但flops不是。所以模型应该减少flops，保留channel interaction。



4.3. What makes CSNs outperform ResNet3D?

与ResNet3D相比，ip-CSN具有更高的训练误差，但更低的测试误差(见表2)。这表明，CSN中的channel-separated使模型正则化，避免了过拟合。



4.4. The effects of different blocks in group convolutional networks

用sectionr3.4中的结构去替换ResNet3D中的卷积块做一些实验，实验图如下：![ResNet3D accuracy:computation tradeoff by transforming group convolutional blocks](/Users/momo/Documents/video/ResNet3D accuracy:computation tradeoff by transforming group convolutional blocks.png)

解读：

1.simple block只有2层，bottleneck block有3层，simple-X-8是用simple-X替换的resnet-18中的block，simple-X-16是用simple-X替换的resnet-34中的block；bottleneck-X-8是用bottleneck-X替换的resnet-26中的block，bottleneck-X-16是用bottleneck-X替换的resnet-50中的block；

2.图中不同的标志表示不同的结构，相同的标志表示相同的结构，并且相同的标志对应的结构中一定是有group-convolution。可以调节组卷积的参数G，所以才会有不同的GFLOPs；

3.从右往左看，对于浅层网络（resnet-18,24,36）由于浅层网络参数数量导致的能力限制，计算量减小（减小参数数量或者增加组卷积的组数），网络表现越差；对于深层网络（resnet-50），观察右边图的红色曲线：首先从标准卷积bottleneck出发，变为bottleneck-G，增加group的数量，此时参数减少，interaction减少，效果提升一直提升；到bottleneck-D，效果达到最好（green star对应的位置），变为bottleneck-DG，再增加group的数量，效果开始下降了；

4.从这个图中说明，选择网络结构需要一个accuracy和computation之间的tradeoff，作者认为bottleneck-D是最好的一个结构，这个结构恰好是ir-CSN。



5.Comparison with the State-of-the-Art

（1）数据集Sports1M

![Comparison with state-of-the-art architectures on Sports1M](/Users/momo/Documents/video/Comparison with state-of-the-art architectures on Sports1M.png)

（2）数据集Kinetics

![Comparison with state-of-the-art architectures on Ki- netics](/Users/momo/Documents/video/Comparison with state-of-the-art architectures on Ki- netics.png)

（3）数据集something-something(V1)

![Comparisons with state-of-the-art methods on Something2-V1](/Users/momo/Documents/video/Comparisons with state-of-the-art methods on Something2-V1.png)



##### 二、代码复现（https://github.com/facebookresearch/VMZ）





**#################################################################**

### X3D

#### 论文：X3D: Expanding Architectures for Efficient Video Recognition

##### 一、论文理解

1.简介

（1）将2D图像分类网络框架拓展到3D视频分类时，通常将网络的输入、提取的feature和卷积核拓展到时间维度。其他的网络设置，如深度（网络层数）、宽度（通道数）以及空间尺寸基本是从2D网络继承而来。保持深度、宽度等设置不变，同时将模型拓展到时间维度，是可以提升模型的表现的，但是如果考虑到实际应用（准确率和计算量之间的平衡），这是往往得不到最优解，一般的2D轻量级网络也是这样；

（2）将一个小的2D图像识别的网络框架扩展为一个3D的视频分类网络框架，可以调节一下参数：

1）输入视频的持续时间：$\gamma_t$；

2）帧率：$\gamma_T$；

3）输入视频的空间分辨率：$\gamma_s$；

4）网络宽度：$\gamma_w$；

5）bottleneck宽度：$\gamma_b$；

6）网络深度：$\gamma_d$；

调节以上参数得到X3D模型（Expand 3D）

（3）在不同的计算和精度范围内，X3D的性能达到了最先进水平，同时需要4.8×和5.5×更少的multiply-adds和参数，以达到与以前工作相同的精度；



2.相关工作

2.1.Spatiotemporal (3D) networks

（1）多数直接直接将2D的图像分类网络拓展为3D的视频分类网络：I3D，C3D（包括ResNet3D）；

（2）SlowFast框架在Slow和Fast pathway中探索了跨几个轴、不同时间、空间和通道分辨率的分辨率平衡；

2.2.Efficient 2D networks

（1）MobileNetV1,V2和ShuffleNetV1,V2主要用了channel-wise separable convolution（深度可分离卷积）、channel shuffle等网络设计方法；

（2）MobileNetV3在MobileNetV2的基础上增加了SE(Squeeze-Excitation) attention block；

（3）MnasNet中一系列通过神经网络框架自搜索的方法，得到一个网络框架，搜索空间中包括channel-wise separable模块，SE block以及MobileNetV3 Swish non-linearities等模块；

（4）在MnasNet搜索方法的基础上，将线性比例因子（linear scaling factor）应用于空间、宽度和深度轴，以创建一系列用于图像分类的EfficientNet；

**（5）EfficientNet采用了grid-search的方法，考虑$k$个维度，每个维度有$d$个值，共训练$d^k$个模型；MnasNet训练了8000左右个模型，对于视频来说，这些是不太现实的，因为最新版本的Kinetics数据集有195M帧图像，是ImageNet的162.5倍。X3D也是采用grid-search的方法，不同的是每次只拓展某个单一维度，一共6个维度，每个维度有5个step，共训练了30个模型。**



2.3.Efficient 3D networks

CSN(channel separated network)中将resnet拓展到3D，同时使用了MobileNetV1,V2中的channel-wise separated convolution技巧。



3.X3D Networks

（1）移动图像分类领域可以观察到类似的进展，其中收缩修改(较浅的网络，较低的分辨率，较薄的层，可分离的卷积)允许在较低的预算下运行。考虑到图像卷积网络设计的历史，视频架构没有类似的进展，因为视频领域的架构通常是基于图像模型的直接时间扩展；

（2）视频架构的扩展涉及到一下几个问题：

1）视频长时间输入和稀疏采样是否优于短时间输入稠密采样？

2）是否需要更高的空间分辨率？以前的工作已经使用低分辨率的视频分类来提高效率，视频类型通常比互联网图像的空间分辨率更粗糙。因此，是否存在性能达到饱和的最大空间分辨率?

3）是使用高帧率但较窄的通道的网络，还是使用较宽的模型来缓慢处理视频?

4）当增加网络宽度时，是在ResNet块中全局扩展网络宽度更好，还是像在移动图像分类网络中使用channel-wise的可分卷积那样扩展内部(bottleneck)宽度更好?



3.1中介绍基础baseline的X2D框架，3.2中使用3.3中的方法拓展X2D的时间维度，进而得到X3D。

3.1.Basis instantiation

（1）X2D作为baseline的网络架构，基本网络设计遵循ResNet结构和具有退化(单帧)时间输入的SlowFast网络的快速路径（Fast pathway）设计，下图展示了基础框架（参数$\gamma_T,\gamma_t,\gamma_s,\gamma_w,\gamma_b,\gamma_d$都设置为1）；

![X2D](/Users/momo/Documents/video/X2D.png)

（2）X2D基础网络的输入尺寸是$1\times112^2$，Conv1是MobileNet系列中的channel-wise操作；

（3）X2D网络只有1.63M参数和20.67M的FLOPs；



3.2. Expansion operations

经过6个维度的分别拓展，出现6个类型的网络，$X-Fast,X-Temporal,X=Spatial,X-Depth,X-Width,X-Bottleneck$；



3.3. Progressive Network Expansion

Forward expansion：

（1）$J(X)$代表当前的扩展因子$X$好的程度，对应模型的准确率，$C(X)$表示当前扩展因子的复杂度度量；

（2）网络扩展是为了找到最好的平衡扩展因子$X,X=argmax_{Z,C(Z)=c}$；

（3）文章中执行的扩展只改变a的一个膨胀因子，而保持其他因子不变，因此只有Z的不同子集需要计算，其中每一个都只在$X$的一个维度上改变。具有最佳计算/精度权衡的扩展因子保留到下一步。这是coordinate descent的一种形式在超参数空间由这些轴定义；



4.Experiments: Action Classification

扩展方法：每一步训练$a,a=6$个模型，每个模型扩展一个轴，共进行5步，共训练了30个模型，从中选择计算量和准确率平衡最好的模型即可。

和ip-CSN-152相比在Kinetics-400上从头开始训练：

ip-CSN-152：top-1准确率77.8%，GFLOPs：109，Param：32.8M

X3D-XXL：top-1准确率80.4%，GFLOPs：194，Param：20.3M



**总结：这篇文章参考了2D轻量级网络EfficientNet的思路，对网络中的各个参数进行扩展搜索，得到最合适的参数。**

##### 二、代码复现（https: //github.com/facebookresearch/SlowFast）









**#################################################################**

### 2D卷积，3D卷积，I3D，ResNet3D，S3D-G，R(2+1)D以及CSN中卷积的运作机制比较

#### 一、2D卷积

2D卷积是最基础的卷积操作，2D卷积的时候注意多通道卷积，卷积核的通道数和信号的通道数是一样的，然后一个卷积核作用在原信号之后产生一个通道，多个卷积核作用在原信号之后产生多个通道。（下图为单通道卷积）

![2D卷积](/Users/momo/Documents/video/2D卷积.gif)

#### 二、3D卷积

3D卷积多了一个**深度通道**，这个深度可能是**视频上的连续帧**，也可能是**立体图像中的不同切片**。3D卷积和多通道卷积有区别，多通道卷积在不同的通道上的卷积核参数是不同的，权重共享体现在空间上；而3D卷积本身卷积核就是3D的，所以在不同深度上滑动时用的是同一个卷积，权重共享主要体现在深度上。

![3D卷积](/Users/momo/Documents/video/3D卷积.gif)

如图所示，不考虑颜色的影响，假设输入是一个视频的9个帧，总体形成一个立方体信号，一个$3*3*3$的卷积核在立方体进行卷积，$3*3$是指空间上的卷积，第三维的$3$是指一次卷3帧，得到输出；如果是中间的层，仍然是卷3帧，只不过每一帧的channels数和上一层的卷积核数量一样，而不再是视频中的3个channels。

下图是文章（3D convolutional neural networks for human action recognition）中的3D卷积神经网络：

![3D卷积网络](/Users/momo/Documents/video/3D卷积网络.png)

网络很浅，只有3个卷积层和1个全连接层，2个池化层，这样的网络规模和LeNet5差不多。不过3D多了一个维度，计算量多了很多。这里有两个3D卷积层，卷积核大小分别是7x7x3，7x6x3，前两维是空间的卷积，后一维是时间的卷积（每次卷3帧或者卷3个feature map），看得出来，不需要保持一致，而且通常空间的卷积核大小和时间就不会一致，毕竟处理的“分辨率”不同。（第一层$7*7*3$是两个卷积核，导致C2层是$23*2@54*34$，这里的2个卷积核表示C2层中每一帧实际上是2个channel，即C2层是23帧，每一帧是2个channel，每个channel都是54*34的）



#### 三、I3D

I3D实际上是利用了ImageNet上预训练的2D卷积网络的预训练结果，将2D卷积核进行膨胀，在时间维度上进行扩充t倍（不是T帧），扩充的参数和2D是一致的，然后整体除以t，作为一个3维的卷积核，剩下卷积的步骤和3D卷积的步骤是一致的。



#### 四、ResNet3D

直接将ResNet2D推广到3D，卷积原理步骤和C3D一样。



#### 五、S3D-G

#### 六、R(2+1)D

将3D卷积核$N_{i-1}\times t\times d\times d$分解为2D的空间卷积$N_{i-1}\times 1\times d\times d$和1D的时间卷积$M_i\times t\times 1\times 1$；本质上还是两个3D卷积，只不过分别在时间维度和空间维度将值设置为1，并称之为空间卷积和时间卷积。



#### 七、CSN

将3DResNet中bottleneck中的$3*3*3$convolution替换成$1*1*1$的conventional convolution以及$3*3*3$的depthwise convolution。

![Channel-Separated Bottleneck Block](/Users/momo/Documents/video/Channel-Separated Bottleneck Block.png)

$1*1*1$的卷积实际上就是传统的3D卷积，$3*3*3$的组卷积在卷积时会不一样。假设输入通道是$C_{in}$，分为$G$组，每组$\frac{C_{in}}{G}$个通道，输出通道是$C_{out}$，传统的卷积过程在规定了输出通道为$C_{out}$之后，就确定了有$C_{out}$个卷积核，但是组卷积确定了有$\frac{C_{out}}{G}$个卷积核，每个卷积核尺寸为$3*3*C_{in}$，每个卷积核又分为$G$组，每个卷积核的$G$组都会和输入通道的$G$组分别卷积，生成$G$个feature map，相当于每个卷积核会生成$G$个feature map，共有$\frac{C_{out}}{G}$个卷积核，所以最终的输出通道还是$C_{out}$个。此过程涉及到的参数量是$k*k*C_{in}*\frac{C_{out}}{G}$。

一般卷积：

![普通卷积](G:\Documents\Knowledge_Base\source_image\普通卷积.webp)



组卷积：

![组卷积](G:\Documents\Knowledge_Base\source_image\组卷积.webp)



深度可分离卷积：

![深度可分离卷积](G:\Documents\Knowledge_Base\source_image\深度可分离卷积.webp)