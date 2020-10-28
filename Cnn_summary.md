# CNN summary

卷积神经网络的各个特点和优势

卷积神经网络的三个基本概念：局部感受野（Local Receptive Fields）、共享权重（Share Weights）、池化（Pooling）。

单隐藏层非线性的神经网络就可以拟合任何连续函数。

这样做的成本会成倍的增加许多神经元，这样做在计算上是不可行的。通过增加网络的深度可以增加许多非线性的特征，能够更好的近似目标函数

> 概念背景详情：https://blog.csdn.net/weixin_42398658/article/details/84392845

## 核心CNN 发展历程

核心的发展历程：Lenet --> Alexnet --> VGG --> GoogLenet(Inception) --> Resnet

> 详细发展内容：https://www.sohu.com/a/156386201_717210

## 早期探索

### Hubel和Wesel的工作

卷积神经网络的结构设计灵感来自于Hubel和Wesel等人的工作，在很大程度上遵循了灵长类动物视觉皮层的基本结构，即从视网膜位的区域接受一个输入，然后通过外侧膝状核执行多尺度的高通滤波和归一化，然后通过V1到V4的视觉皮层不同区域进行检测，视觉皮层的V1和V2相当于卷积层和下采样层，而颞下区是更深的层，最终对图像进行一个推断。卷机神经网络本身就有提取不同特征的能力，高层特征是底层特征的组合，借助这种自动提取特征的功能，卷积网络逐步替代了传统人工设计的复杂的提取器。从像素中得到良好的特征表示。

![image-20200923145455023](https://github.com/Single-Wu/paper/blob/main/images/image-20200923145455023.png?raw=true)

> 详细内容参考：https://blog.csdn.net/weixin_40920183/article/details/106718232

### Lenet

麻雀虽小，五脏俱全，第一个经典的CNN网络，具备了一些现在基本的CNN网络的基本组件，在没有应用GPU的时候，能够保存参数和计算就成了一个关键优势。

大概有六万个参数

![网络解析（一）：LeNet-5详解](http://cuijiahua.com/wp-content/uploads/2018/01/dl_3_1.png)

主要特征：

- CNN主要用这3层的序列：convolution, pooling, no-linearity
- 用卷积提取空间特征
- 由空间平局得到子样本
- 同tanh和sigmoid得到非线性
- 用muliti-layer neual network(MLP)作为最终分类器
- 层层之间用系数的连接矩阵，避免大的计算成本

LeNet的设计较为简单，因此其处理复杂数据的能力有限；此外，在近年来的研究中许多学者已经发现全连接层的计算代价过大，而使用全部由卷积层组成的神经网络。

### Alexnet（王者归来）

由于受到计算机性能的影响，虽然LeNet在图像分类中取得了较好的成绩，但是并没有引起很多的关注。 直到2012年，Alex等人提出的**AlexNet**网络在ImageNet大赛上以远超第二名的成绩夺冠，卷积神经网络乃至深度学习重新引起了广泛的关注。

*具有六千万个参数*

![img](https://picb.zhimg.com/80/v2-29c8b75b2cf5248f025fdf12a246801e_1440w.jpg)

AlexNet的特点：

- 更深的网络结构
- 使用层叠的卷积层，即卷积层+卷积层+池化层来提取图像的特征
- 使用Dropout抑制过拟合
- 使用数据增强Data Augmentation抑制过拟合
- 使用Relu替换之前的sigmoid的作为激活函数
- 提出了**LRN**层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，**增强了模型的泛化能力**。（局部相应归一化）
- 多GPU训练

> 详细内容参考：https://www.cnblogs.com/wangguchangqing/p/10333370.html
>
> ​						   https://www.cnblogs.com/xiaoboge/p/10465534.html（图解网络结构）
>
> ​						   https://zhuanlan.zhihu.com/p/22659166（含代码）
>
> ​							https://www.jianshu.com/p/c014f81242e7（局部相应归一化）

### VGG

VGG主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能。VGG有两种结构，分别是VGG16和VGG19，两者并没有本质上的区别，只是网络深度不一样。

相比与Alexnet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中较大的卷积核（11x11, 7x7 ,5x5），这样在相同的感受野中提升网路的深度，进而提升神经网络的效果

![img](https://pic2.zhimg.com/80/v2-ea924e733676e0da534f677a97c98653_1440w.jpg)

VGG16包含16个隐藏层，VGG19包含19个隐藏层

VGG16大约有1.4亿个参数

VGGnet非常简洁

![img](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180217131751843-269987601.png)

**优点**

- VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。
- 几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好：
- 验证了通过不断加深网络结构可以提升性能。

**缺点**

- VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。VGG可是有3个全连接层啊！



> 详细内容参考：https://zhuanlan.zhihu.com/p/41423739
>
> ​							https://zhuanlan.zhihu.com/p/23518167（含代码）



## 深度化

### Resnet

随着深度的不断增加，将会出现以下问题：

1. 计算资源的消耗
2. 模型容易过拟合
3. 梯度消失/梯度爆炸问题的产生
4. 网络发生退化现象

网络发生了退化（degradation）的现象：随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，当你再增加网络深度的话，训练集loss反而会增大。

![image-20200925081836807](D:\Note\image\image-20200925081836807.png)



特点：

- 超深的网络结构
- 提出residual模块（残差模块）
- 使用Batch Normalization加速训练



### Densenet

resnet在训练深层网络中有一些层的贡献是很少的。

和resnet不同，Densenet是在做恒等映射后的特征图进行相加，而这里是进行了一个连接，这种级联的方式可以显示的区分不同深度的信息。

<img src="https://pic1.zhimg.com/v2-6130b6b0e45eae06e88f9f18bf814384_1440w.jpg?source=172ae18b" alt="读DenseNet" style="zoom:70%;" />

特点：

- 减轻了vanishing-gradient(梯度消失)
- 加强了feature的传递
- 更有效的利用了feature
- 一定程度上减少了参数数量



## 模块化

这里研究者就可以设计一个深层次的网路，但是设计一个深层次的网路是相当繁琐的，由此启发研究者对于卷积神经网络的结构实现一种模块化。

### GoogLenet（Inception V1）

用更多的卷积，更深的层次可以得到更好的结构。

<img src="D:\Note\image\141544_FfKB_876354.jpg" alt="img" style="zoom:60%;" />

特点：

- 引用了Inception结构（融合不同尺度的特征信息）
- 使用1x1的卷积核进行降维以及映射处理
- 添加两个辅助分类器帮助训练
- 丢弃全连接层，使用平均池化层（大大减少模型参数）

**辅助分类器**

GoogLenet中有两个辅助分类器，一个拥有5x5大小步长为3的average pooling，作为4a和4d的inception的输出结果。

$$
out_{size}=(in_{size} - F_{size}+2P)/S+1
$$

1. 由于4a输出结果是14x14x512的张量，平均池化完后输出（14-5）/3+1 = 4也就是 4x4x512,同理可得，4d的输出是4x4x528.
2. 然后是128个用于降低维度1x1的卷积，用RL激活
3. 采用节点个数为1024的全连接层，用RL激活
4. 两个全连接层之间用dropped out，以70%随机失活
5. 概率的输出，softmax激活输出类别概率。

<img src="D:\Note\image\image-20200921152601762.png" alt="image-20200921152601762" style="zoom:30%;" />

辅助分类器是为了防止梯度消失，它会按照一个最小的比例（0.3）融合到最终的分类结果中，相当于模型的融合。同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个网络的训练很有裨益。而在实际测试的时候，这两个额外的softmax会被去掉。

**网络结构解析**

原始输入图像为224x224x3，且都进行了零均值化的预处理操作（图像每个像素减去均值）

![这里写图片描述](D:\Note\image\20160102104027555)

详细内容参考：https://my.oschina.net/u/876354/blog/1637819

​                          https://blog.csdn.net/weixin_42764391/article/details/89713947（网络结构运行解析）

### Inception

大量的文献表明可以将稀疏矩阵聚类为较为密集的子矩阵来提高计算性能，就如人类的大脑是可以看做是神经元的重复堆积，因此，GoogLeNet团队提出了Inception网络结构，就是构造一种“基础神经元”结构，来搭建一个稀疏性、高计算性能的网络结构。

网络结构设计三步走：split->transform->merge

**Inception V1**

通过设计一个稀疏网络结构，但是能够产生稠密的数据，既能增加神经网络表现，又能保证计算资源的使用效率。谷歌提出了最原始Inception的基本结构：

![img](https://static.oschina.net/uploads/space/2018/0317/141510_fIWh_876354.png)

由于5x5的卷积核所需要的计算量太大，造成特这个图的厚度很大，为了避免这种情况，在3x3前、5x5前、max pooling后分别加上了1x1的卷积核，以起到了降低特征图厚度的作用，这也就形成了Inception v1的网络结构。

![img](D:\Note\image\141520_31TH_876354.png)

**Inception V2**

如何在不增加过多计算量的同时提高网络的表达能力就成为了一个问题。
Inception V2版本的解决方案就是修改Inception的内部计算逻辑，提出了比较特殊的“卷积”计算结构。

主要特点：

- 卷积分解
- 降低特征图大小



**Inception V3**
Inception V3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这样的好处，既可以加速计算，又可以将1个卷积拆成2个卷积，使得网络深度进一步增加，增加了网络的非线性（每增加一层都要进行ReLU）。
另外，网络输入从224x224变为了299x299。

**Inception V4**

Inception V4研究了Inception模块与残差连接的结合。

## 高效化

深度学习和工业的应用越来越紧密，需要更多的低功耗少内存的高速度的网路部署在移动设备中。

**在大幅降低模型精度的前提下，最大程度的提高运算速度**

提高运算所读有两个可以调整的方向：

1. 减少可学习参数的数量；
2. 减少整个网络的计算量。

这个方向带来的效果是非常明显的：

1. 减少模型训练和测试时候的计算量，单个step的速度更快；
2. 减小模型文件的大小，更利于模型的保存和传输；
3. 可学习参数更少，网络占用的显存更小。

### SqueezeNet

SqueezeNet能够在ImageNet数据集上达到AlexNet近似的效果，但是参数比AlexNet少50倍，结合他们的模型压缩技术 Deep Compression，模型文件可比AlexNet小510倍。

压缩模型使用了三个策略：

- 将3x3卷积替换成1x1卷积
- 减少3x3卷积的通道数
- 将采样后置：较大的Feature Map含有更多的信息，因此将降采样往分类层移动（注意这样的操作虽然会提升网络的精度，但是它有一个非常严重的缺点：即会增加网络的计算量。）

**Fire模块** 

SqueezeNet是由若干个Fire模块结合卷积网络中卷积层，降采样层，全连接等层组成的。

squeeze:仅包含一组1x1的卷积 (较小的参数数量对于特征的压缩)

expand:包含一组连续的1x1的卷积和3x3的same padding卷积concatnate组成



![image-20201021083426203](D:\Note\image\image-20201021083426203.png)

**网络架构**

图3是SqueezeNet的几个实现，左侧是不加short-cut的SqueezeNet，中间是加了short-cut的，右侧是short-cut跨有不同Feature Map个数的卷积的。还有一些细节图3中并没有体现出来：

1. 激活函数默认都使用ReLU；
2. fire9之后接了一个rate为0.5的dropout；
3. 使用same卷积。![image-20201021100509808](D:\Note\image\image-20201021100509808.png)



达到了和alexnet相同的效果，参数是alexnet的510分之一

> 详细资料参考：https://zhuanlan.zhihu.com/p/49465950

### MobileNet

#### MobileNet V1

核心是提出了**深度可分离卷积**：将传统的卷积因式分解成两个操作：depthwise convolution（深度卷积） 和 1x1的pointwise convolution（逐点卷积)。



<img src="D:\Note\image\image-20201021132606148.png" alt="image-20201021132606148" style="zoom:50%;" />



### ShuffleNet

组卷积：



### GhostNet





## 注意力机制

与上下文相关的特征在图像的任务中也是有重要的作用的，在人类系统那中，我们称这种现象叫做注意力机制。

### SENet

对于CNN网络来说，其核心计算是卷积算子，从本质上讲，卷积是对一个局部区域进行特征融合，这包括空间上（H和W维度）以及通道间（C维度）的特征融合。

### scSE



### CBAM







## 自动化

网络结构需要设置大量的超参数：网络结构，学习率，优化方法

先验知识可能限制了网络的性能，研究者提出了一种自动化学习网络。可以根据要求处理的问题，自动的去找出一个最优的网路结构。

### NASNet



### EfficientNet









## 卷积核

卷积核可以理解为用于特征提取

卷积后的计算公式：

- 输入图片大小 W*W
- Filter（卷积核）大小F*F
- 步长 Step
- padding（填充）的像素数P，P=1就相当于给图像填充后图像大小为W+1 *W+1
- 输出图片的大小为N * N

$$
N=\lfloor\frac{W-F+2P}{Step}\rfloor+1
$$





## 激活函数

