---
sidebar_label: "多头注意力"
sidebar_position: 2
---

# 注意力机制

Transformer的每一层都是由TransformerLayer组成

![](./imgs/multi-head-attention.png)

多头注意力的计算方法可以分如下三个步骤：

**（1）获取QKV向量**：假设输入序列为$\mathbf X = [x}_1,...,x}_L]\in \mathbb{R}^{L \times D_{}}$，经过线性变换得到$\mathbf x_{i}$对应的查询向量$\mathbf q_{i}\in \mathbb{R}^{D_{}}$、键向量$\mathbf k_{i}\in \mathbb{R}^{D_{}}$和值向量$\mathbf v_{i}\in \mathbb{R}^{D_{}}$。对于整个输入序列$\mathbf X$，线性变换的过程可以简写为

$$
\qquad \mathbf Q=\mathbf X\mathbf W^{Q} \in \mathbb{R}^{L \times D_{}},\\
\qquad \mathbf K=\mathbf X\mathbf W^{K} \in \mathbb{R}^{L \times D_{}},\\
\qquad \mathbf V=\mathbf X\mathbf W^{V} \in \mathbb{R}^{L \times D_{}},
$$

其中$\mathbf W^{Q} \in \mathbb{R}^{D_{}\times D_{}}$，$\mathbf W^{K} \in \mathbb{R}^{D_{}\times D_{}}$，$\mathbf W^{V} \in \mathbb{R}^{D_{}\times D_{}}$是可学习的映射矩阵。

接下来，可以沿着Q、K、V向量的最后一维度，按照头的数量平均拆分成H份，获得用于多头计算的$\{Q_i \in \mathbb{R}^{L \times D_h}, K_i \in \mathbb{R}^{L \times D_h}, V_i\in \mathbb{R}^{L \times D_h}| i=1,2,..., H\}$向量，其中$D_h = \frac{D_v}{H}$。接下来，需要根据每个头的QKV向量进行自注意力计算。


**（2）自注意力计算**：分别计算每个头的自注意力，以第$j$个头$head_{j}$为例，公式为
$$
head_{j}= \mathrm{attention}(\mathbf Q_j,\mathbf K_j,\mathbf V_j) \in \mathbb{R}^{L\times D_h},
$$

**（3）多头结果融合**：将多个头计算的自注意力结果进行融合，得到最终得输出向量$\mathbf Z$。

$$
\mathbf Z=\mathrm{MultiHeadAttention}(\mathbf X)
\triangleq
(head_{1} \oplus head_{2} \oplus...\oplus head_{H})\mathbf W
$$

其中$\oplus$表示沿着向量的最后一维进行拼接，拼接后向量维度可能并不等于原始向量的维度$D$，因此这里乘以矩阵$ \mathbf W\in \mathbb{R}^{(H*D_h) \times D}$，将向量映射为原始的输入维度$D$。

* **加与规范层**

如图13所示，每个的Transformer Encoder Layer包含2个加与规范层（Add&Norm），其作用是通过加入残差连接和层规范化两个组件，使得网络训练更加稳定，收敛性更好。

这里以第1个加与规范层为例，假设多头自注意力的输入和输出分别为$\boldsymbol{X}\in \mathbb{R}^{L \times D}$和$\boldsymbol{Z} \in \mathbb{R}^{L \times D}$，那么加与规范层可以表示为

$$
\boldsymbol{L} = \text{LayerNorm}(\boldsymbol{X} + \boldsymbol{Z})
$$

其中$\boldsymbol{L} \in \mathbb{R}^{L \times D}$，LayerNorm表示层规范化。接下来，向量$\boldsymbol{L}$将经过前馈层和第2个加与规范层的计算，获得本层Transformer Encoder的输出向量$\boldsymbol{O}\in \mathbb{R}^{L \times D}$。 


由于Feed Forward层比较简单，接下来不再赘述。