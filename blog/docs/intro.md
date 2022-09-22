---
sidebar_label: "介绍"
sidebar_position: 1
---

# 介绍

[Transformer](https://arxiv.org/abs/1706.03762)在2017年被提出时，作为一种序列转换工具，旨在将一个符号序列转换为另一个符号序列，当时最流行的例子是机器翻译，在2018年随着[BERT](https://arxiv.org/abs/1810.04805)的问世正式让众多研究者意识到Transformer的重要性，于是随着近些年的发展和演变，其已迅速成为多个研究领域中不可或缺的工具。

在现在的学习和工作当中，需要对Transformer的深入掌握，在此我编写一个Transformer系列文章，将我在学习过程中所掌握的知识点、使用方法以及最新论文等内容分享出来给大家。如果有任何内容上的问题，欢迎各位大佬批评指正，可在[Repo](https://github.com/wj-Mcat/transformer)中发[issue](https://github.com/wj-Mcat/transformer/issues/new)、PR来调整。

废话不多说，咱们就开始Transformer之旅吧。

<!-- more -->

## 背景

在传统的序列生成任务当中，通常使用循环神经网络和卷积神经网络建模，可是存在两个问题：
* 长文本信息依赖比较差
* 并行计算，导致计算效率低

而Transformer旨在解决以上问题，完全摒弃了RNN和CNN，而是采用注意力机制能够很好的解决以上问题，并在两个机器翻译任务当中的训练时间和最终BLEU分值都有明显的提升。

## 模型结构

Transformer是一个基于多头自注意力的序列到序列的模型，网络结构如下所示（图片来源于《[Attention is all you need](https://dl.acm.org/doi/10.5555/3295222.3295349)》）。

<div style={{textAlign: 'center'}}>
  <img src="https://ai-studio-static-online.cdn.bcebos.com/60a2424a69004f08afc5d8c511cf11352d59ce61c19c42b8afc2fb1b3efca9e1" />
</div>

Transformer模型包括编码器（Encoder，左边）和解码器（Decoder，右边）两部分，且都是由多层TransformerLayer堆叠而成，如下图所示，假设编码器和解码器都为6层，将Transformer的核心部分进行展开，编码器的输出向量会分别输入至每层的Transformer Decoder Layer中，与多头自注意力模块进行信息融合。

![](https://ai-studio-static-online.cdn.bcebos.com/bb2f976d927a4635b6dbf5bc4592208a850e903e4f8b46fb8160ec2ab917af28)

:::tip 知识点预警
1. Decoder Layer的输入组成结构是什么？
2. Decoder Layer接受来自于Encoder Layer的输出数据有哪些？query、key、value到底是那些？
3. 是否可单独使用Encoder、Decoder来进行建模？如果可以，构建原理和应用场景有哪些？
:::

### 模型嵌入层

Encoder和Decoder的输入包含两部分：Input Embedding和Positional Encoding，然后将结果相加，得到TransformerLayer的输入。

示例代码可为：

```py showLineNumbers {4}
input_ids = [465, 263, 2163, 28736]
position_ids = [0, 1, 2, 3]

embedding = word_embedding(input_ids) + position_embedding(position_ids)
```

详细可阅读子章节[Transformer的输入](https://wj-mcat.github.io/transformer/docs/basic/input)

### 编码器

Encoder是由N层EncoderLayer堆叠而成，而每个EncoderLayer包含四个模块：多头注意力层（Multi-Head Attention）、残差网络（Add&Norm）、前馈层（Feed Forward）和正则化（Norm），模型结构如下所示：

![](/img/transformer-encoder-layer.png)

其中多头注意力层的核心计算逻辑是自注意力机制，详细原理介绍可看：[编码器和解码器](./basic/encoder-decoder.md)。

:::tip
Transformer编码器由N个EncoderLayer堆叠而成，嵌入层输出的字符级别的特征表示首先被传入第一层的Encoder Layer，经过自注意计算后，得到向量序列$[\mathbf o_1, \mathbf o_2,\mathbf o_3,\mathbf o_4]$。这个输出会被作为下一层的输入传递到下一层的Encoder Layer，每层的Encoder Layer的计算逻辑是相同的，依此类推，最后一层的输出被视为Transformer Encoder输出的特征编码向量。
:::

### 多头注意力

**多头自注意力**（Multi-Head Self-Attention）的本质是多组自注意力（Self Attention）计算的组合，每组注意力计算被称为一个"头"，不同组注意力的计算是相互独立的。

![](https://ai-studio-static-online.cdn.bcebos.com/68d77776a5314a6f890e3969a10ce0932f81c970084840fa986bf25b1cdc7e55)

如上图所示为多头自注意力计算示意图，是一个$H$（比如：Bert当中为12头）代表自注意力模块的数量。首先输入序列$\mathbf X\in \mathbb{R}^{ L\times D}$（例如 $\mathbf X\in \mathbb{R}^{ 128\times 768}$， $D$表示隐藏层向量维度，比如768，$L$表示输入文本向量长度，比如128）被映射到了不同的$Q$、$K$、$V$向量空间。

然后根据头数$H$将Q、K、V向量沿着最后一维等分为H份，其中$Q_i \in \mathbb{R}^{ L\times (D/H)}$（例如：$Q_i \in \mathbb{R}^{ 128\times 64}$）然后在每一份$Q$、$K$、$V$子空间中分别执行自注意力计算得到$[head_1,…,head_H]$，其中$head_i \in \mathbb{R}^{ L\times (D/H)}$，例如$head_i \in \mathbb{R}^{ 128\times 64}$，最后将输出结果根据最低纬度进行拼接，作为最终的输出向量$\mathbf Z$，其中$Z \in \mathbb{R}^{ L\times D}$，例如：$Z \in \mathbb{R}^{ 128\times 768}$。

通过以上过程得到该层EncoderLayer的输出，而这个输出会被作为下一层的输入传递到下一层的EncoderLayer。

详细原理介绍请看：[多头注意力](https://wj-mcat.github.io/transformer/docs/basic/multi-head-attention)

## 模型类型

在Transformer-Based模型上来看，可将模型分为三种类型：Encoder-Only、Decoder-Only以及Encoder-Decoder模型。

### Encoder Only

Encoder-Only的代表模型是BERT，在预训练时只采用Encoder，为Bi-Directional，旨在编码文本语义信息，然后用于下游任务，比如：自然语言理解，文本分类，目标词预测，问答预测，文本情感分析，序列标注等等，而此类任务对于抽取某一个领域的信息十分在行，一般通过fine-tune之后能够达到很好的效果。

其中经典代表模型有: BERT、RoBERTa、ALBERT等。

### Decoder only

Decoder模型是uni-directional，也就是单向的，为自回归模型，主要用于生成式的任务当中，比如机器翻译、序列文本生成等。

代表模型有：T5, AlphaCode, Switch, ST-MoE, RETRO

### Encoder-Decoder

什么时候该使用Encoder-Decoder模型？做序列到序列的任务时，文本摘要，机器翻译等，其中Encoder和Decoder之间的权重并没有共享，输入的分布和输出的分布不属于与同一个类别，如文本和图像的分布，但用于下游任务时也可只使用Encoder或Decoder的权重。

代表模型有：Dec-only: GPT-{1,2,3}, {🐭, 🐹}, PaLM

## 模型总结

从整体上来看，模型主要分为：输入、Encoder、Decoder、输出以及最终概率输出五个部分组成。

* 输入是指Encoder的输入，主要分为：Token Embedding、Position Embedding、Token Type Embedding。
* Encoder和Decoder在模型结构上一致，都是由多成TransformerLayer组成
    * 而TransformerLayer又是由MultiHead Attention + 残差网络组成
    * MultiHead Attention又是由 SelfAttention组成
* Decoder在输入上与Encoder不太一样，后者的QKV保持一致，所以是SelfAttention、前者Q来自于Encoder，KV来自于Decoder的输入
* 最终概率输出是在模型做预训练时才会使用到数值

## 术语

* Transformer: 指Transformer整体模型
* Encoder：Transfomer中的编码器模块
* Encoder Layer：编码器是由多层网络叠加而成，而EncoderLayer在网络结构等同于TransformerLayer
* Decoder：Transformer中的解码器
* Decoder Layer：解码器是由多层网络叠加而成，而DecoderLayer在网络结构等同于TransformerLayer
* TransformerLayer：此网络结构有多头注意力模块和前馈神经网络构建而成

## 参考链接

* [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* [Transformers from Scratch](https://e2eml.school/transformers.html)