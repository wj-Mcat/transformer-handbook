---
sidebar_label: "自注意力模型"
sidebar_position: 2
---

# 自注意力模型

## 背景

自注意力模型（Self-Attention Model）的设计思想来源于解决循环神经网络在解决长序列数据时遇到的问题：

1. 如何计算**全局的**信息依赖，而不局限于距离的远近？
2. 如何使得计算可以**并行化**，而不是只能串行进行？

## Self Attention

为了解决如上两个问题，自注意力模型采用**查询-键-值**（Query-Key-Value，QKV）的计算方式。对于输入序列$\mathbf X=[\mathbf{x}_1,...,\mathbf{x}_L]\in \mathbb{R}^{L\times D_{}}$，每一个输入$\mathbf{x}_i$都有三个向量表示：查询向量$\mathbf{q}_i\in \mathbb{R}^{D_{k}}$、键向量$\mathbf{k}_i\in \mathbb{R}^{D_{k}}$和值向量$\mathbf{v}_i\in \mathbb{R}^{D_{v}}$。

:::warning 提醒
查询向量$\mathbf{q}_i\in \mathbb{R}^{D_{k}}$和键向量$\mathbf{k}_i\in \mathbb{R}^{D_{k}}$与值向量$\mathbf{v}_i\in \mathbb{R}^{D_{v}}$的维度大小是由两个不同的变量表示。也就是说可Query-Key与Value的维度可以不一致。

如果是一致：即为self-attention。

如果不一致：即为cross-attention。
:::

假设输入序列为$\mathbf X = [\mathbf{x}_1,...,\mathbf{x}_L]\in \mathbb{R}^{L\times D_{}}$，经过线性变换得到$\mathbf x_{i}$对应的查询向量$\mathbf q_{i}\in \mathbb{R}^{D_{k}}$、键向量$\mathbf k_{i}\in \mathbb{R}^{D_{k}}$和值向量$\mathbf v_{i}\in \mathbb{R}^{D_{v}}$。对于整个输入序列$\mathbf X$，线性变换的过程可以简写为：

$$
\qquad \mathbf Q=\mathbf X\mathbf W^{Q} \in \mathbb{R}^{L \times D_{k}},\\
\qquad \mathbf K=\mathbf X\mathbf W^{K} \in \mathbb{R}^{L \times D_{k}},\\
\qquad  \mathbf V=\mathbf X\mathbf W^{V} \in \mathbb{R}^{L \times D_{v}},
$$

其中$\mathbf W^{Q} \in \mathbb{R}^{D_{k}\times D_{}}$，$\mathbf W^{K} \in \mathbb{R}^{D_{k}\times D_{}}$，$\mathbf W^{V} \in \mathbb{R}^{D_{v}\times D_{}}$是可学习的映射矩阵。默认情况下，可以设置映射后的$\mathbf Q、\mathbf K、\mathbf V$的特征向量维度相同，都为$D$。

展示了输入为两个向量序列$\mathbf{x}_1$和$\mathbf{x}_2$的自注意力的计算过程，需要执行如下步骤：

![](https://ai-studio-static-online.cdn.bcebos.com/31a007d0d37d410195856d7bdea39225280b79dde17441748802539d91aba467)

:::tip
$a_{12}$表示原始文本中索引为1、2的两个token的相似度计算数值。

$A \in \mathbb{R}^{L\times L} = softmax( \{a_{00}, a_{01}, a_{02}, ..., a_{L-1, L-1} \})$

$X \in \mathbb{R}^{L \times D_v} = A * V$
:::

总结一下计算过程：

1）两个输入向量$\mathbf{x}_1$和$\mathbf{x}_2$经过线性变换，分别获得它们在查询向量$\mathbf Q$、键向量$\mathbf K$和值向量$\mathbf V$。

2）使用点积计算，先得出$\mathbf q_1$在$\mathbf k_1$和$\mathbf k_2$的分数$\alpha_{11}$和$\alpha_{12}$；再将分数缩放并使用Softmax进行归一化，获得$\mathbf x_1$的自注意力分布分数 $\hat\alpha_{11}$和$\hat\alpha_{12}$；最后根据该位置的自注意力分布对$\mathbf v_1$和$\mathbf v_2$进行加权平均，获得最终$\mathbf x_1$位置的输出向量$\mathbf z_1$ 。

3）同理，可以获得$\mathbf{x}_2$的自注意力向量$\mathbf z_2$，即每个输入向量的位置均对应一个自注意力输出的编码向量$\mathbf z_i \in \mathbb{R}^{1 \times D_{v}}$。

在上述过程中已经计算出$A$：注意力矩阵，用来在$V$上根据权重计算出隐藏层向量表示，这个也是最核心的向量表示过程。

为了加快计算效率，在实际应用时，可以使用矩阵计算的方式一次性计算出所有位置的自注意力输出向量，即

$$
\mathbf Z=\mathrm{attention}(\mathbf Q,\mathbf K,\mathbf V) =  softmax(\frac{\mathbf Q\mathbf K^T}{\sqrt{D}}) \mathbf V
$$

其中$\mathbf Z \in \mathbb{R}^{L\times D_{v}}$。

在自注意力计算过程中，为了防止注意力分布具有较大的方差，导致Softmax的梯度比较小，不利于模型的收敛，在计算过程中除以了一个$\sqrt{D}$，可以有效降低方差，加速模型收敛。

## Multi Head Attention

以上介绍的是计算单个Self-Attention，这也是组成多头注意力机制（Multi Head Attention）的子单元。计算示意图如下所示：

![](./imgs/multi-head-attention.png)

在Self Attention的$Q$、$K$、$V$其实是将最低纬度平均成多份之后来计算，然后计算结束之后需要将其拼接起来，此时就需要