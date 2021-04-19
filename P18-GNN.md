## Intro

### Why do we need GNN?

- Classification
  
  <img src="P18-GNN.assets/image-20210409165612262.webp" alt="image-20210409165612262" style="zoom: 33%;" />

- Generation
  
  <img src="P18-GNN.assets/image-20210409165651960.webp" alt="image-20210409165651960" style="zoom:33%;" />

- 还有其他很多应用不一一列举。总之 GNN 就是可以 Model 有图关系的数据。

本节课我们会讨论三个问题：

1. 如何使用数据中的结构和关系来帮助模型？
2. 如果这个关系图很大，比如说有 20k 个节点，要怎么办？
3. 如果不是所有的节点都有标签怎么办？

<img src="P18-GNN.assets/image-20210409175100944.webp" alt="image-20210409175100944" style="zoom: 33%;" />

一个很常见的情况就是我们的无标签数据远多于有标签数据。但是图本身的结构有「近朱者赤近墨者黑」的概念，因此我们需要利用其邻接节点。这种情况下就是 **半监督学习（Semi-Supervised Learning）**。

我们再回想一下 CNN 的结构，我们用一组卷积核在图片上进行滑动，这样的操作也是考虑了邻接像素之间的联系。但是对于图来说，它的结构不像图像那么规整，是不容易用卷积核来处理的。那我们要如何把节点的特征嵌入一个特征空间，使其可以应用卷积操作呢？

1. 方法一：把卷积（相关）的思想扩展到图 —— 空间卷积（Spatial-Based Convolution），即在空域或定点域上直接对邻接点进行操作
2. 方法二：使用信号处理中的卷积 —— 频谱卷积（Spectral-Based Convolution），先把信号转到傅里叶域（Fourier Domain，也就是频域），在频域上对信号和滤波器相乘，就可以得到原信号经过滤波器卷积以后的信号，最后从频域转回时域。

## Roadmap

<img src="P18-GNN.assets/image-20210409175852292.webp" alt="image-20210409175852292" style="zoom: 33%;" />

## Tasks, Dataset and Benchmark

- Tasks
  - Semi-supervised node classification
  - regression
  - graph classification
  - graph representation learning
  - link prediction
- Common dataset
  - CORA: citation network. 2.7k nodes and 5.4k links
  - TU_MUTAG: 188 molecules with 18 nodes on average

## Spatial-based GNN

这里有两个术语：

- Aggregate：用邻接节点的 feature 更新当前节点在下一层的 hidden state
- Readout：把所有节点的 feature 集合起来代表整个 graph

<img src="P18-GNN.assets/image-20210409180441548.webp" alt="image-20210409180441548" style="zoom: 25%;" />

### NN4G（Neural Networks for Graph）

**Aggregate：**

NN4G 的做法是先累加邻接节点的 hidden state，经过一个线性变换，再和当前节点的经过另一个线性变换的 hidden state 相加。

<img src="P18-GNN.assets/image-20210409180744320.webp" alt="image-20210409180744320" style="zoom:33%;" />

**Readout：**

对每一层的所有节点的 hidden state 求平均，再算加权和。

<img src="P18-GNN.assets/image-20210409181012405.webp" alt="image-20210409181012405" style="zoom:25%;" />

### DCNN（Diffusion-Convolution Neural Network）

第一个隐层是对节点的 1-邻接节点求平均再做线性变换，第二个隐层是对节点的 2-邻接节点求平均再做线性变换，以此类推。

<img src="P18-GNN.assets/image-20210409181821456.webp" alt="image-20210409181821456" style="zoom:25%;" />

我们把每一层的 hidden state 叠起来就得到了某个节点 $H^0 \sim H^K$ 的 Feature Map. 然后再对它们做一次线性变换，就得到了某个节点的输出：

<img src="P18-GNN.assets/image-20210409182123635.webp" alt="image-20210409182123635" style="zoom:25%;" />

### DGC (Diffusion Graph Convolution）

和 DCNN 唯一的区别就是，DCNN 的 Node Features 是每一层 的 feature 堆叠在一起，而 DGC 是直接相加。

不同 Model 实现非线性的操作不同罢了。

### MoNET（Mixture Model Networks）

- 定义了一个测量节点距离的方法，其中 deg 表示度数（无向图）
  
  $$
  u(x, y) = (\frac{1}{\sqrt{deg(x)}}, \frac{1}{\sqrt{deg(y)}})^\top
  $$

- 使用加权平均而不是直接对所有邻接特征做平均
  
  <img src="P18-GNN.assets/image-20210409182721739.webp" alt="image-20210409182721739" style="zoom:25%;" />

### GraphSAGE

- **SA**mple and aggre**G**at**E**

- 可以在 Inductive 和 Transductive 的 setting 下使用
  
  > 两者的区别是：
  > 
  > |     | Inducive                              | Transducive                        |
  > | --- | ------------------------------------- | ---------------------------------- |
  > | 训练  | 只用到训练集                                | 训练集+无标签的测试集都有使用                    |
  > | 预测  | 只要样本属于同样的空间就可以预测（Specific to general） | 只能预测见过的无标签样本（Specifig to spesific） |
  > | 泛化  | 有新样本时无需重新训练                           | 有新样本时需要重新训练                        |
  > | 消耗  |                                       | 计算量大于 Inducive，但是可能效果更好            |
  > 
  > 一般 Inductive 叫做归纳式，Transductive 叫做直推式。正常情况下我们说的监督学习一般都是 Inductive 的，而半监督学习可以是 Transductive 的。

- GraphSAGE 从邻接节点中学习如何 embed 当前节点的 feature
  
  <img src="P18-GNN.assets/image-20210409183617008.webp" alt="image-20210409183617008" style="zoom: 33%;" />
  
  根据伪代码应该是（4）先累加邻接节点的 hidden state，（5）将累加的邻接节点的 hidden state 和当前节点的 hidden state 做 concat 然后做线性变换，sigmoid 激活，得到一个新的 hidden state. 对这个 hidden state 除以自身的二范式，得到当前节点的新的 hidden state. 上述步骤重复 K 次。  

**Aggregation：** mean, max pooling or LSTM

<img src="P18-GNN.assets/image-20210409184256469.webp" alt="image-20210409184256469" style="zoom:33%;" />

### GAT（Graph Attention Networks）

<img src="P18-GNN.assets/image-20210409184808757.webp" alt="image-20210409184808757" style="zoom: 33%;" />

其中的 enegry 是相邻节点之间的（like, 相似度？）权重。

<img src="P18-GNN.assets/image-20210409185532609.webp" alt="image-20210409185532609" style="zoom:33%;" />

### GIN（Graph Isomorphism Network）

> isomorphism n. 类质同象；同形

- A GNN can be at most as powerful as WL isomorphic test
- Theoretical proofs were provided

其结论是，在更新 hidden state 的时候，最好使用以下的方式：

$$
h^{(k)}_v = \mathrm{MLP}^{(k)}\big((1 + \varepsilon^{(k)}) \cdot h_v^{(k-1)} + \sum_{u\in \mathcal{N}(v)}h_u^{(k-1)}\big)
$$

也就是说，在做更新的时候，邻接节点的 hidden state 是直接累加，而不是求平均或最大值。

原因如下图，(a) 中，Mean 和 Max 不能区分两者，(b) 中 Max 不能区分两者，(c) 中 Max 和 Mean 不能区分两者。

<img src="P18-GNN.assets/image-20210409190150467.webp" alt="image-20210409190150467" style="zoom:33%;" />

## Graph signal Processing and Spectral-based GNN

之前说到 Graph 不能直接应用的卷积的原因是，节点的邻接节点不像图像有很清晰的二维结构。

那么有一个想法是，把这些节点看做信号，将节点和卷积核都应用傅里叶变换，再将两者相乘。之后再使用逆傅里叶变换投射回原来的时域。也就是，我们在顶点域上不好做卷积，所以学习信号处理的方式在频域上做卷积。

<img src="P18-GNN.assets/image-20210409190522169.webp" alt="image-20210409190522169" style="zoom:25%;" />

（Warning of 信号处理）

我们知道信号是一组向量空间里面的基底的线性组合。也就是公式 3（合成）

那如果我们想知道某个 component 的大小，就将这个信号和对应的基底做内积（因为 basis 都是正交的，所以其他 basis 对应的成分乘完都会变成 0）


$$
\begin{aligned}
\overrightarrow{A} = \sum_{k=1}^N a_k \hat{v}_k \\
 \\
a = \overrightarrow{A} \cdot \hat{v}_j \\
 \\
\hat{v}_i \cdot \hat{v}_j = \delta_{ij}
\end{aligned}
$$

在时域上我们常用的一组基底是 cos 和 sin，假设对于一组周期信号，我们可以把它展开成一个 Fourier Series。这里选用的基底就是 $e^{jk\omega_0t}$

<img src="P18-GNN.assets/image-20210409191828342.webp" alt="image-20210409191828342" style="zoom: 25%;" />

<img src="P18-GNN.assets/image-20210409192001449.webp" alt="image-20210409192001449" style="zoom:25%;" />

<img src="P18-GNN.assets/image-20210409193206821.webp" alt="image-20210409193206821" style="zoom:33%;" />

我们来看一个简单的例子。左图是一个 Graph，右图是各个节点在定点域上的信号。假设说这是一张城市路网图，那么这些信号值可以表达人口、气温等等。我们先假设它们是一些标量（可以是向量）

<img src="P18-GNN.assets/image-20210409193339759.webp" alt="image-20210409193339759" style="zoom:25%;" />

下面讲一个很重要的概念叫图拉普拉斯分解。

首先，一个 Graph Laplacian 就是其 Degree Matrix 减去邻接矩阵。那因为这两者都是实对称矩阵，因此它们的差也是实对称矩阵，并且可以证明 Graph Laplacian 是半正定的（所有的特征值都 $\ge$ 0）。因此我们可以对它做一个谱分解。

这里的 $U$ 就是 $L$ 的特征向量组成的矩阵，$\lambda$ 就是对应的特征值。这些特征向量是正交规范（长度为 1）的。

我们把这些特征值称为这个 graph 的 frequency，而对应的特征向量就是其对应的正交基。

<img src="P18-GNN.assets/image-20210409193605908.webp" alt="image-20210409193605908" style="zoom:25%;" />

来看一个简单的例子：

<img src="P18-GNN.assets/image-20210409195432407.webp" alt="image-20210409195432407" style="zoom:25%;" />

把它画出来：

<img src="P18-GNN.assets/image-20210409195544395.webp" alt="image-20210409195544395" style="zoom:25%;" />

那么为什么特征值表示频率，特征向量表示基底呢？

我们来看离散时域傅里叶基（Discrete time fourier basis）：

<img src="P18-GNN.assets/image-20210409195723099.webp" alt="image-20210409195723099" style="zoom:25%;" />

这里的重点就是 **频率越大，相邻两点之间的信号变化量就越大。**

下面来讲解如何解释顶点的频率：

<img src="P18-GNN.assets/image-20210409200044876.webp" alt="image-20210409200044876" style="zoom:25%;" />

我们把 $L$ 看成是 Graph 上的一个运算，那么已知一组信号 $f$，$Lf$ 的结果就是当前节点的信号强度减去邻接节点的信号强度，也就是当前节点和邻接节点的信号强度差。当然实际计算这个强度差是要取平方的，因此我们用 $f^TLf$：

<img src="P18-GNN.assets/image-20210409200310232.webp" alt="image-20210409200310232" style="zoom:25%;" />

其意义就是节点之间信号变化的平方，也就是 graph signal 的平滑度。回到上面说的 **频率越大，相邻两点之间的信号变化量就越大。**反过来，我们可以用这个信号变化量的大小来反映频率的大小。

我们现在把信号 $f$ 换成特征向量 $u$，由第一条公式可以看到，对这些信号做拉普拉斯变换的结果就是其对应的特征值。因此，「对应特征值小的特征向量就是低频的位置」。

<img src="P18-GNN.assets/image-20210409200646869.webp" alt="image-20210409200646869" style="zoom:25%;" />

来看一个更明显的例子：

<img src="P18-GNN.assets/image-20210409200912921.webp" alt="image-20210409200912921" style="zoom:25%;" />

所以，我们给定一段信号 $x$， 给定一组 Eigen Vector 的正交基，我们就可以做分析了：

<img src="P18-GNN.assets/image-20210409202025863.webp" alt="image-20210409202025863" style="zoom:25%;" />

注意到因为 $U$ 是一个正交矩阵，它的逆变换就是 $U$ 的转置。所以逆运算只要将 $\hat{x}$ 左乘 $U$ 就可以。这样就变成合成了。

<img src="P18-GNN.assets/image-20210409202502942.webp" alt="image-20210409202502942" style="zoom:25%;" />

到这里为止我们已经理清了要如何把 graph 从时域转换到频域，以及如何转换回去。接下来我们要做 Filter。

一个定理是，顶点域上的卷积等价于频域上的乘积。

<img src="P18-GNN.assets/image-20210409202853136.webp" alt="image-20210409202853136" style="zoom:25%;" />

<img src="P18-GNN.assets/image-20210409203030936.webp" alt="image-20210409203030936" style="zoom:25%;" />

上面得到了信号 $\hat{x}$ 在频域的值，下面要把它变换回时域：

<img src="P18-GNN.assets/image-20210409203331724.webp" alt="image-20210409203331724" style="zoom:25%;" />

因此我们最终要学习的目标就是一个 $g_\theta(\Lambda)$. 现在的问题在于，这个函数的复杂度取决于 Graph 的大小。 

那我们来看一个例子，假如 $g_\theta(L) = L$，可以看到在 $L$ 的第一行第三列的值是 0，也就是说顶点 3 对顶点 0 的值不会有影响。

但是如果 $g_\theta(L) = cos(L)$，用泰勒展开我们可以看到这个函数里包含了 $L$ 的高次项。我们知道拉普拉斯矩阵的幂次表示延长路径距离，这就会使得顶点 3 对顶点 0 产生影响。

<img src="P18-GNN.assets/image-20210409203947532.webp" alt="image-20210409203947532" style="zoom:25%;" />

<img src="P18-GNN.assets/image-20210409204009004.webp" alt="image-20210409204009004" style="zoom:25%;" />

<img src="P18-GNN.assets/image-20210409211516841.webp" alt="image-20210409211516841" style="zoom:25%;" />

现在的问题是，如果我们的 graph 有 N 个节点，把 $cos$ 展开到 N 次项，就会导致整张图像的所有节点都会对其他节点产生影响。这就违背了 CNN 的 Localize. 

我们现在就有了两个问题：

1. 学到的函数 $g_\theta(L)$ 复杂度是 O(N)
2. Localization

### Models

#### ChebNet

为了解决上述两个问题，ChebNet 提出的方法是把 $g_\theta(L)$ 限制为拉普拉斯的多项式，这就使得 $g_\theta(L)$ 的复杂度降低到 O(L)，并且是 K-localized.

但是这会引入另一个问题，就是在计算的时候需要算矩阵的 k 次幂，因此计算复杂度是 $O(N^2)$

<img src="P18-GNN.assets/image-20210409211815784.webp" alt="image-20210409211815784" style="zoom:25%;" />

解决这个问题的办法是使用切比雪夫多项式：（这里可以看一下秦九韶算法）

<img src="P18-GNN.assets/image-20210409212142537.webp" alt="image-20210409212142537" style="zoom:25%;" />

我们对拉普拉斯做一个变换 $\hat{\Lambda} = \frac{2\Lambda}{\lambda_{max}} - I$ 使其满足切比雪夫多项式的限制。这样一来，我们的搜寻目标就改变为找到一个切比雪夫多项式的线性变换：

<img src="P18-GNN.assets/image-20210409212510307.webp" alt="image-20210409212510307" style="zoom:25%;" />

这样做的意义是，把一个普通的多项式转换成拉普拉斯多项式以后，那个幂次会变得比较好算。

<img src="P18-GNN.assets/image-20210409212710970.webp" alt="image-20210409212710970" style="zoom:25%;" />

把它展开来，可以发现这个幂次变成了一个递归式：

<img src="P18-GNN.assets/image-20210409212939433.webp" alt="image-20210409212939433" style="zoom:25%;" />

于是我们的计算复杂度从 $O(N^2)$ 变成了 $O(KE)$，其中 E 是边的数目。

<img src="P18-GNN.assets/image-20210409213058353.webp" alt="image-20210409213058353" style="zoom:25%;" />

当然实作的时候是要学多个 $g_\theta$ 的：

<img src="P18-GNN.assets/image-20210409214116322.webp" alt="image-20210409214116322" style="zoom:25%;" />

上图是一个 channel 的处理，多通道就要做多次：

<img src="P18-GNN.assets/image-20210409214149214.webp" alt="image-20210409214149214" style="zoom:25%;" />

#### GCN（Graph Convolutional Network）

GCN 其实就是在 ChebNet 的基础上引入了三个条件 + renormalization trick.

第一个条件是限制 K = 1，于是多项式被限制到了只有 0 次和一次项。

第二个条件是，假设拉普拉斯是规范化的，那么它的最大特征值约等于 2，于是公式可以进一步化简。第三个 $\because$ 是拉普拉斯的性质，可以写成那种形式的展开。

第三个条件是要求 $\theta_0' = \theta_1'$ 那么多项式就可以进一步化简： 

<img src="P18-GNN.assets/image-20210409214442816.webp" alt="image-20210409214442816" style="zoom:25%;" />

所谓的 renormalization trick 其实就是给图加上自环（也即对角线变成全 1）。最后，我们可以把整个式子写成：


$$
h_v = f(\frac{1}{|\mathcal{N}(v)|}\sum_{u\in \mathcal{N}(v)}Wx_u+b), \forall v \in \mathcal{V}
$$

也就是下一层的 hidden state 等于上一层的 hidden state 经过线性变换，取平均以后再激活。

（后面是介绍上面所有的模型在公开数据集上的表现）

结论是深层的 GCN 存在 Information Loss，可以用 Drop Edge 的方式避免 over-smoothing.

## Graph Generation

（略）

## GNN for NLP

（略）