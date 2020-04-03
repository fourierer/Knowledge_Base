# 课程笔记



#################################################################

# 黄凯奇《视频处理与分析》



### 第一章、绪论

#### 基本概念

1.视频就是利用人眼视觉暂留的原理，通过播放一系列图片，使人产生运动的感觉(实际上就是一系列图片)。

2.按照视频的处理和存储方式，视频分为模拟视频（Analog Video）和数字视频（Digital Video），数字视频是连续信号，模拟视频是采样和离散化的信号。以前的视频是模拟视频，现在的视频都是数字化的，本课程分析数字视频。

3.视频处理：视频增强、视频压缩等；视频分析：视频检索、视频监控等。

4.Frame rate：帧率

Fps：frame per second，现在认为24帧每秒很流畅

5.视频增强为了改善视频的质量；

视频压缩为了传输和存储，比如现在的视频直播，压缩技术使得过程非常流畅；

视频有冗余需要去除，存储设备会耗费大量财力物力。视频处理是不够的，海量的视频数据需要去分析解决。

视频分析关键技术：运动目标检测、运动目标跟踪（在哪？），目标分类识别（是谁？），异常行为分析（在干什么？）。



#### 视频技术的应用：

1.安全领域

家庭安全：智能家居，如独居老人在家摔倒，智能家居可以自动识别；

交通安全：辅助驾驶监控，防止危险发生；

公共安全：监控

个人安全：生物特征识别（虹膜识别），考勤通关等

2.娱乐领域

笑脸检测，人脸美颜（直播领域较多），电影特效，风格迁移，人机交互（一般对光照要求较高）

3.生活便捷

增强现实（AR），街景（图像拼接，3D建模）（VR），字符识别（OCR）（车牌识别），物体识别（无人超市，但是没有继续做下去，存在很多技术问题），Google glass，

4.其他领域

导购机器人，物流机器人，医学





### 第二章、视频分析系统及应用

#### 一、视频监控技术发展

1.监控技术发展的三个阶段

人力现场监控（人眼+人脑）--传统视频监控（机器眼+人脑）--智能视觉监控（机器眼+机器脑）

2.视频系统架构

（1）摄像机：视频传感器（CCD、CMOS）+处理器

（2）传输：电缆、光纤、网络传输、微波传输

（3）显示：

3.智能视频监控核心技术

微处理器（模式视频监控系统）、数字压缩编码技术（数字视频监控系统）、网络技术，高清技术（网络监控，高清监控）、计算机视觉技术和模式识别技术等。

（1）运动目标检测：

静态图片中的物体检测研究：

2012年以前，都在voc数据集（20个类别）上测评，而且用的都是比较传统的方法；

2012年开始在ImageNet上测评和比赛，开始使用deep learning的方法；

现在基本上使用微软的MS COCO数据集上测评；

主要算法有R-CNN,Fast-RCNN,Faster-RCNN等



动态图片（视频）中的物体检测研究：

UCF-50



（2）运动目标跟踪（在单帧目标检测的基础上加上多帧检测目标id不变）

单相机单目标跟踪、单相机多目标跟踪、多相机单目标跟踪、多相机多目标跟踪等。

难点：人密集的场合、球场上人的衣服非常相似等场合。





（3）目标分类和识别：

物体分类：AlexNet(2012),VGG(2014),GoogleNet(2014),ResNet(2015),DenceNet(2016),2019年已经取消了比赛，因为这个数据集已经起不到推动物体分类算法发展的作用了。

行人重识别（Person Re-identity）:

类似于目标跟踪，只不过是两张图片的matching,而不是目标跟踪的动态跟踪。

数据集：RAP(Richly Annotated Pedestrian) Dataset,室内属性



（4）异常行为分析：

异常行为是（行为识别，视频分类）的一个特例，要分辨出异常行为所在的那一类。

数据集：
UCF,HMDB,Kinetics,something，这些数据集类别的定义实际上在一定程度上会限制行为识别的发展。

#### 二、智能视频监控技术应用

1.检测类：攀爬检测、禁区检测

2.检测+分类：非法停车、丢包检测

3.检测+跟踪：徘徊检测，自动跟踪，人数统计

4.目标分类，行为分析：车牌识别，打架检测

5.其他：火警检测



发展阶段：

1.2004-2007:起步阶段，主要功能为入侵检测，行业集中在轨道交通；

2.2008-2009:人群密度检测，徘徊检测；

3.2010-：接力跟踪；

需求还在扩大，由单一应用转向多样。



### 第三章、目标物体识别

#### 一、总体介绍和发展历史

1.总体介绍：物体识别包括领域

物体分类：识别是否为飞机

物体检测：用bounding box将飞机框出来

物体分割：将飞机与背景图分割出来



2.发展历史

（1）1965-1980:配准（Alignment）

需要在图中找出目标物体，特征点匹配（刚体配准相对简单），通过基元（geon）来匹配识别。

（2）Eigenface（PCA）

（3）sliding windows

（4）1990-？local feature，基于特征

（5）2000-？Parts and Structure

（6）2003-present：bags of features models

（7）global feature

（8）2011-present：Deep Learning



物体识别任务中，背景对物体的影响：

1.如果识别的物体较简单，和背景差别较大，此时可以不需要背景；当要识别的物体有遮挡，很难直接识别，此时需要背景的辅助来推理；

2.attention机制实际上就是加入推理内容，比如加强背景和目标之间的关系（Non-local），视频分类中帧与帧之间的关系；



#### 二、典型方向和研究方法

**典型方向：**

1.RGB-D图像识别（RGBD image recognition）

和普通RGB图像相比，多了个D



2.视频分类/识别（Video Classification ）

和Image classification不一样的是，视频分类中最重要的是如何建模时序信息。



3.行人再识别（Fine-Gained image recognition）

行人重识别，相当于对同一个人进行识别，可以利用不同的属性进行细粒度识别（即识别一个具体类中，精细的小类，需要关注更细节的信息）





**研究方法：**

**1.词包模型（bags of feature）**

Image classification:

（1）特征提取：比如将图像分为很多不同的patch

（2）学习“视觉词典”：

一般是采用聚类方法，形成词典，又称为码本（code book）

（3）使用“视觉词典”来量化特征（coding）

Sparse coding SPM（一个特征和多个基相关），LLC（Locality-constrained Linear Coding，认为基与基之间有相关性）

（4）池化，获得更稳定的特征，最大汇聚和平均汇聚



**2.深度学习**

ImageNet和深度卷积神经网络相互成就。

1.1998年：LeCun提出LeNet，用来字符识别；

2.2012年：Alex提出AlexNet，用于ImageNet识别；（1.改进了优化方法，随机梯度下降方法的应用；2.使用非监督数据来预训练；3.使用了GPU加速）；

3.VGG，相比于AlexNet，使用小卷积核核小池化核，层数更深；

4.GoogleNet，在网络需要的计算不变的前提下，通过工艺改进提升网络的宽度和深度，即在增加网络深度和宽度的基础上，控制参数数量和计算量；

5.ResNet，解决了深度网络中无法有效传播梯度的关系，重新定向了深层神经网络中的信息流；

6.DenceNet，Xception以及一系列轻量级网络ShuffleNet，MobileNet(V1,V2,V3)，GhostNet(2020CVPR)；









### 第四章、视频编解码与目标跟踪

#### 一、视频编解码

1.视频压缩可以进行的原因

（1）视频中有很多冗余：空间冗余（空间中目标区域只有一小部分）、时间冗余（连续帧很相似）、结构冗余（图像中的像素存在明显的分布模式）、信息熵冗余、知识冗余（由很多先验知识可以得到）、视觉冗余（人的视觉系统的分辨能力为26灰度等级，多的级别无法察觉）

（2）人的视觉特征表现为对亮度信息很敏感，而对边缘的急剧变化不敏感



2.数据压缩算法的综合评价指标

（1）压缩倍数（压缩率）：由压缩前后的数据量之比或者压缩后的比特流中每个显示像素的平均比特数来表示

（2）图像质量：主观评价和客观评价（信噪比）

（3）压缩和解压缩的速度：对称压缩（要求解压缩和压缩都需要实时进行）、非对称压缩（只要求解压缩是实时的）、压缩的计算量（一般编码的计算量要比解码的计算量大）



3.数据压缩算法分类

（1）按照压缩方法是否产生失真分类：无损压缩（RLE/JPEG/MPEG/H264，一般无损压缩饿压缩率只有3倍左右，无法满足需求），有损压缩（主要是去除时空冗余，再用变换将信息集中到少数几个部分，主要保存模型参数和变换方法。方法有预测编码、变换编码（DCT变换，小波变换，KLT变换））

有损的$8\times8$DCT主要应用于JPEG，H.261，H.263等

无损的$4\times4/8\times8$整数DCT以及哈达码变换应用于H.264

小波变换被应用于JPEG2000以及H.265中



4.视频压缩的几个重要压缩标准

ISO的MPEG-2,MPEG-4、ITU的H.264、中国的AVS

（1）很多主流媒体采用MPEG-4压缩标准：如avi,mov,asf,mp4

（2）H.264只是MPEG-4的一部分

（3）H.263和AVS编码效率高于MPEG-2



#### 二、视觉目标跟踪

1.简单介绍

（1）概念：在连续的图像帧中跟踪指定目标，比较精确地估计出该目标在每一帧图像中的位置

（2）挑战：非刚体、光线变化、跟踪目标所占区域小、跟踪数目太多、再识别、实时性

（3）应用：基于运动的识别、人机交互



2.单相机单目标跟踪

（1）简介

特点：时间空间都连续

优点：对于固定相机，算法简单，计算量小

缺点：视野较小，遮挡严重

（2）数据库

TB-100，VOT，TrackingNet，LaSOT，GOT-10k(one-shot，训练和测试的类别不一样，比如测试集中有狗的跟踪，但是训练集中并没有狗的跟踪类别)

（3）算法分类

1.目标表达表达算法（形状表达、表观表达）

主要算法：根据目标区域的统计特性来区分背景和目标所在区域

2.时序预测模型

主要算法：

（1）Kalman滤波

（2）粒子滤波：准确率高，计算量大

（3）Mean Shift：计算量小，但准确率不如粒子滤波

（4）轮廓跟踪：不适用于轮廓变化非常大的跟踪

3.基于分类（检测）的跟踪：区分目标和背景的判别式模型

（1）TLD

（2）Struck

（3）CSK

4.基于深度学习的跟踪方法（对外观变化和物体旋转会更具有鲁棒性）

（1）SiamRPN：使用RPN

（2）SiamMask



3.单相机多目标跟踪



4.多相机单目标跟踪



5.多相机多目标跟踪









#################################################################

## 胡占义《计算机视觉》

### 第一章、绪论

#### 一、Marr计算机视觉简介

1.计算机视觉奠基人：David Marr

2.对于计算机视觉，输入是图像或者视频，输出根据任务确定。

3.当前人工智能两大主要途径：（1）大数据+深度学习；（2）通过模拟大脑：系统复杂到某个地步，很难从理论上解析。

4.深度学习里程碑式的研究：（1）1958年，Perceptron，单层神经网络无法实现异或运算；（2）1975年，BP算法；（3）2006年，Hinton以及2012年AlexNet；

5.在**物体视觉**方面，如物体识别和分类等，目前深度学习方法超过了其他所有方法，甚至超过了人类视觉系统；但是在**空间视觉**方面，如视觉定位，三维物体重建，深度学习方法仍无法与基于几何的方法相媲美。



#### 二、计算机视觉发展历史与现状

主要四大阶段。

三大会议：ICCV,ECCV(便理论),CVPR(偏理论)

两大刊物：PAMI,IJCV



1.马尔计算视觉理论（1981-）

马尔认为：类似于计算机对给定为题的计算过程，视觉感知是一个“对图像信息的逐层加工处理过程”。

马尔认为视觉的主要目标是重建物体的三维形状，从现在的生物视觉的研究进展看这是不正确的。物体识别是基于二维图像的，深度信息起的作用不大，即物体识别不必先进行三维形状重建。



2.主动视觉大辩论（1988-1994）

马尔的三维重建理论缺乏目的性以及与环境的主动交互性（主动视觉）



3.分层三维重建理论（1992-）

计算机视觉从工业应用到精度要求不太高的领域，如通讯，虚拟现实，考古，是计算机视觉第二次发展高潮的主要原因。

图像--射影空间--放射空间--欧式空间



4.基于学习的视觉（2001）

（1）子空间方法，如流形学习；

（2）深度学习；



图像物体识别代表性理论：马尔三维重建理论、巴乔的二维图像模型、低卡洛的分层去纠缠理论物体识别的逆生成模型





### 第二章、计算机视觉

#### 1.深度学习与卷积神经网络



#### 2.图像底层特征提取

特征提取：

能有效反映图像内容的信息就是特征，对于图像，边缘和轮廓能反映图像内容。

1.边缘提取

（1）边缘定义：

边缘是图像中亮度突然变化的区域，图像灰度构成的曲面上的陡峭区域，像素灰度存在阶梯状或屋脊状变化的区域。

（2）图像微分算子：

一维：一阶导数的极值点，二阶导数的过零点；

梯度向量：$\nabla I(x,y)=(\frac{\partial I}{\partial x},\frac{\partial I}{\partial y})^T$，每一个像素都可以计算出梯度向量

二维：拉普拉斯算子$\triangle I=\nabla^2I=\frac{\partial^2I}{\partial x^2}+\frac{\partial^2I}{\partial y^2}$

水平梯度算子（检测竖直边缘），竖直梯度算子（检测水平边缘）

1）拉普拉斯算子：只用一个模版便可计算得到，实际中几乎不独立使用拉普拉斯算子，二次求导数对噪声非常敏感，配合卷积模糊使用。

2）高斯拉普拉斯算子（LoG）：先用高斯核进行卷积模糊，再利用拉普拉斯算子检测边缘，等价于先对高斯核求拉普拉斯，再用拉普拉斯卷积核整体对图像做卷积。

（3）canny算子

1）首先计算图像与高斯核的卷积，即做模糊；

2）用一阶有限差分计算偏导数的两个阵列，即计算两个方向的梯度，此时每一个像素都有一个梯度向量；

3）计算每个像素的梯度向量的方向和辐值，非极大值抑制，在该点周围方向（8个方向）选取多个像素点再计算多个辐值，如果该像素的梯度向量的辐值是极大值，则作为候选边缘点；

4）采用双阈值方法，低阈值得到低阈值边缘图，高阈值得到高阈值边缘图，认为高阈值边缘点是可靠的，低阈值边缘点可以用于修正边缘，如果某低阈值边缘能够连接到高阈值边缘，则认为该低阈值边缘点可靠；



2.特征点提取

近几年的边缘检测的文章很少了，特征点还是有很多。

2.1.角点检测：

（1）Harris角点检测算法：

角点定义：以某个点为中心做窗口，窗口向**任意方向**的移动都导致窗口中图像灰度的明显变化，则该点称为角点。

（2）将窗口平移$[u,v]$，产生的灰度变化$E(u,v)$
$$
E(u,v)=\sum_{x,y}w(x,y)[I(x+u,y+v)-I(x,y)]^2
$$
对$M$的特征值进行分类：

1）如果两个特征值都比较大，则是角点；

2）如果一个特征值较大，一个特征值较小，则是边缘；

3）如果两个特征值都比较小，则是平坦区域；

引入响应函数$R$，来判断是否为角点。



（3）Harris角点性质：

1）旋转不变性，由于旋转之后仍然在该方向有变化，所以角点仍能被检测；

2）部分的仿射不变性（加性的放射不变性），对图像灰度做加减运算，仍能检测出角点；

3）不具有尺度不变性，比如图像缩小时，用于检测的窗口中灰度变化差与之前的灰度变化差不一样，判断结果会发生变化；



2.2.ORB特征

1.FAST：判断该点的灰度是否为周边的极大值，如果是则判断为特征点；

2.BRIEF描述

汉明距离（Hamming Distance）：两个字符串的之间的汉明距离是指两个字符串对应位置的不同字符的个数



3.SIFT（Scale Invariant Feature Transform）特征点检测

（1）SIFT是一种特征提取算法，DoG特征检测+SIFT描述子，是最佳的描述子，没有之一。

（2）不变性：旋转不变性，尺度不变性

（3）SIFT特征提取流程

1）图像的尺度空间

用不同的高斯卷积核与原图像做卷积，不同的核尺寸$\sigma$的卷积结果构成图像尺度空间；

2）高斯差分尺度空间

图像的尺度空间中的图作差，DoG和LoG比较相似；

3）高斯金字塔

用不同尺寸的高斯卷积核对原图做卷积，相邻尺寸的卷积结果相减；将图像缩放（下采样）再进行相同步骤；得到高斯金字塔。

4）DoG尺度空间极值点检测

将高斯金字塔建立三维尺度空间，搜索每个点的$2\sigma$领域（三维空间领域），若该点为局部极值点，则保存为候选关键点；

5）关键点的精确定位

在离散采样中搜索到的极值点不一定是真实空间的极值点，用一个函数去拟合DoG，计算真正的极值点（如何定位真正的极值点？真正的极值点不一定是整像素）；

6）去除不稳点的关键点

i）去除对比度低的点，设定阈值

i）去除边缘上的点

（4）特征点的描述

如何构造一个有区分性的特征点描述向量？

1）以特征点为中心，确定一个圆形邻域，邻域中的每个点都有方向和辐值。将方向分为多个区间（假设为8个，区间边界为方向），根据方向将这些点划分到各个区间里面，每个点对区间的边界方向都有贡献，根据公式来确定该点对边界方向的贡献辐值，将相同方向中的点的辐值相加，辐值最大的那个方向确定为主方向；主方向为了确定旋转不变性；

2）根据主方向来构造，在每个特征点周围取$16*16$邻域，每$4*4$像素作为一个块，共16个块。每个块里面有8个值，共$16*8=128$个值，形成一个$1*128$的向量；用该向量作为描述向量；

SIFT可以调节的参数：

提取阶段：1.高斯金字塔下采样次数；2.对比度阈值

描述阶段：1.方向划分；2.领域大小（描述向量维度）；3.光照阈值；



2.3.DNN提取特征描述子

在三维重建中，DNN提取到的特征子结果并没有直接使用SIFT精确。



2.4.ICP（Iterated Closest Point）问题

不是一个特征子提取的方法，而是一个两个点集匹配的方法。

如果两个集合的点集中的点相互对应的上，则可以通过求解能量函数来得到最优解。没有匹配关系，可以先用粗糙方法估计一个匹配，然后使用迭代的方法来优化匹配。



2.5.鲁棒估计：RNASAC

可以用来做匹配。









#################################################################

## 雷震《视频监控与应用》









