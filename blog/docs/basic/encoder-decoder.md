---
title: "编码器和解码器"
sidebar_label: "编码器和解码器"
sidebar_position: 3
---


## 背景

Encoder和Decoder作为Transformer中的基础组件，其整体计算框架如下所示：

![](https://ai-studio-static-online.cdn.bcebos.com/bb2f976d927a4635b6dbf5bc4592208a850e903e4f8b46fb8160ec2ab917af28)

从上图可看出，N层的Encoder会计算最后的隐藏层输出HiddenStates，而Decoder的每一层TransformerLayer会接受来自于Encoder最后一层的输出作为一部分的输入。

编码器和解码器的组合是一个序列到序列组合模型，这种模型在语音和语言领域的任务中非常常见，例如：

* 机器翻译：输入“Deep learning”，输出“深度学习”。
* 语音识别：输入“一段音频声波信号”，输出一段文字，如“深度学习”。
* 信息摘要：输入“一篇报道飞桨Wave Summit峰会的200字新闻稿”，输出“50字的峰会内容和活动盛况的概要”。
* 序列标注：输入“飞桨是很强大的深度学习框架”，输出带有文法标注的结果“名词、动词、形容词、名词”。
* 问答与对话：输入“中国的首都是哪个城市？”，输出“北京”。

![](https://ai-studio-static-online.cdn.bcebos.com/ab81ed0b53bb4e599720dc689ba52a95de39a12a582848f7a7e49c861062e63d)

除了上述应用外，很多输入是序列，输出是类别的自然语言领域的任务也可以转化成Seq2Seq的问题，如常见的情感分类任务，通过加提示词的方式，也可以转化成Seq2Seq的任务，如所示。

![](https://ai-studio-static-online.cdn.bcebos.com/3f028d66deea4097a8e0c21b95dbdcee916a24113cce4c988e8ef74c709bec17)

Seq2Seq模型的输入包括两部分数据：需要分类的评论语句“这个行李箱不够结实，用几次轮子就坏了”和人工加入的提示问题“这个评论是正向还是负向？”模型的输出是分类的标签“负向”。由此可见，一个典型的文本分类问题，可以通过加提示词的方式编程一对问答，转化成了序列到序列的任务。

以上序列到序列的任务得益于编码器和解码器的结构，能够对两种不同领域的数据进行相互转化，那接下来我们来从整体层面来解释一下此类模型。

## 编码器

Encoder是由多层的TransformerLayer组成，至于其原理大家可看[自注意力机制](./self-attention.md)。

Encoder主要用作语义编码，是一个Bi-Directional的模型。

### Bi-Directional

为什么说是双向模型呢？

传统的RNN是单向的，即从左到右理解语义，不过也有BiRNN（如：BiLSTM）将从左到右和从右到左的结果拼接起来即可得到双向的语义信息。可是这类语义理解有点反人类，因为相当于你顺读一遍，再倒读一遍，然后就可以视作为你每个token都能够获得全局的信息。

可是Transformer中引入了SelfAttention，Q和K会计算一个$Attention \in \mathbb{R}^{ L\times L}$矩阵，其中$L$为文本的长度。在此矩阵当中，$Attention_{i,j}$代表了文本中第$i$个token和$j$个token的相似度，那$Attention_i$就代表了第$i$个token和所有其他token的相似度为多少。

注意，这里的第$i$个token已经是可以**看到**该行文本的所有其他token，不仅仅只是左边，或只是右边的，所以说：Encoder是双向的。

## 解码器

解码器相对于编码器在计算效率上而言要差，可是能够基于编码器的隐藏层向量来解码出对应领域的文本。

### 解码过程

* TransformerLayer的输入

在Encoder中TransformerLayer每一层的输入只有一个：X，然后转化为对应的QKV，可是在Decoder当中，QK和


解码器的过程是一步一步完成的，

```text
(1, 0, 0, 0, 0, …, 0) => (<SOS>)
(1, 1, 0, 0, 0, …, 0) => (<SOS>, ‘Bonjour’)
(1, 1, 1, 0, 0, …, 0) => (<SOS>, ‘Bonjour’, ‘le’)
(1, 1, 1, 1, 0, …, 0) => (<SOS>, ‘Bonjour’, ‘le’, ‘monde’)
(1, 1, 1, 1, 1, …, 0) => (<SOS>, ‘Bonjour’, ‘le’, ‘monde’, ‘!’)
```

推荐观看: [Transformer models: Decoders](https://www.youtube.com/watch?v=d_ixlCubqQw)。

### Uni-Directional

解码器
