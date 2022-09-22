---
title: "模型的输入"
sidebar_label: "Input"
sidebar_position: 1
---

# Embedding 源码解析

> 此处以bert模型为例讲解embedding模块代码。

虽然原始论文中的`position encoding`模块使用了对应正弦余弦公式来计算其位置信息，可是在BERT当中使用可学习的Embedding来作为position embedding，预训练模型的参数也是从权重文件中加载而来，源码如下所示：

```py title=BertEmbedding showLineNumbers {9-11,15-17,19}
class BertEmbeddings(Layer):
"""
Include embeddings from word, position and token_type embeddings
"""

def __init__(self, vocab_size, hidden_size=768, hidden_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16):
    super(BertEmbeddings, self).__init__()
    # 定义Embedding模块
    self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
    self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
    self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

def forward(self, input_ids, token_type_ids=None, position_ids=None, past_key_values_length=None):
    # 计算 word-embedding、position-embedding以及token-type-embedding
    input_embedings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = input_embedings + position_embeddings + token_type_embeddings
    return embeddings
```
