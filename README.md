# transformer-implementation-pytorch
Unofficial PyTorch implementation of transformer-like models.

implemented models:
- Transformer [[source](https://arxiv.org/abs/1706.03762?context=cs)]

**Roadmap**:

- [ ] BERT
- [ ] FlowFormer


## 1 Quick Start
### 1.1 Installation
1. python version = 3.7
2. `pip install git+https://github.com/leoxiang66/transformer-implementation-pytorch.git`

### 1.2 Usage
```python
import models
vocab_size = 5000
trf = models.Transformer(6,768,vocab_size)
```


## 2 Reference
1. https://www.bilibili.com/video/BV1jB4y1u7Gi?share_source=copy_web
2. https://www.bilibili.com/video/BV1mk4y1q7eK?p=2&share_source=copy_web

## 3 Citation
```
@software{Xiang_PyTorch_Implementation_of,
author = {Xiang, Tao},
title = {{PyTorch Implementation of Transformers}}
}
```
