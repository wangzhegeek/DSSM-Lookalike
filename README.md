# DSSM-Lookalike Pipeline

**Update 2025.3.15：** 从TensorFlow版本迁移到PyTorch版本，并丰富了模型实现、数据处理和示例代码。新增了MovieLens数据集的推荐示例，增强了模型的正则化和优化策略。

implemention of paper: [Learning deep structured semantic models for web search using clickthrough data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)

一个基于DSSM的Lookalike向量化召回pipeline简单实现，包括以下部分：
* 数据获取：使用[2018腾讯广告算法大赛](https://wx.jdcloud.com/market/jdata/list/17)比赛中的训练数据（100000条）(aid,uid,label)。
* 三种数据负采样方法：<1>随机负采样。<2>根据出现频次负采样。<3>RALM采样方法，参考[腾讯实时Look-alike](https://arxiv.org/abs/1906.05022)
* DSSM实现，与原论文在loss计算方式上略有不同。
* 输出user embedding和item embedding。

较为完整的复现了工业界常用的基于DSSM的向量化召回pipeline。

## 项目结构
本项目基于PyTorch实现，目录结构如下：
- `models/`: 模型实现代码
  - `dssm.py`: DSSM模型实现
  - `utils.py`: 工具函数，包括负采样和余弦相似度计算
  - `dataset.py`: 数据集和数据加载器
  - `trainer.py`: 训练和评估函数
- `examples/`: 示例代码
  - `run_dssm_neg_sample.py`: DSSM模型训练和评估示例
  - `run_dssm_pytorch.py`: 优化版DSSM模型实现
  - `run_dssm_optimized.py`: 带学习率调度器的DSSM模型
  - `run_dssm_balanced.py`: 平衡数据集版本的DSSM模型
  - `run_dssm_movielens.py`: 基于MovieLens-1M数据集的电影推荐示例
  - `run_dssm_simple.py`: 简化版DSSM模型，适用于快速实验
- `data/`: 数据目录
  - `ml-1m/`: MovieLens-1M数据集
- `saved_model/`: 保存的模型和嵌入

## 环境配置
* python 3.6+
* pytorch >= 1.7.0
* pandas >= 1.0.0
* scikit-learn >= 0.24.0
* numpy >= 1.19.0
* tqdm >= 4.50.0
* matplotlib >= 3.3.0

## 代码运行示例
```bash
# 运行基本DSSM模型
python examples/run_dssm_neg_sample.py

# 运行MovieLens电影推荐示例
python examples/run_dssm_movielens.py

# 运行优化版DSSM模型
python examples/run_dssm_optimized.py
```

## 模型说明
DSSM (Deep Structured Semantic Model) 是一种深度学习模型，用于学习用户和物品的低维表示。本项目实现了基于PyTorch的DSSM模型，主要特点包括：

1. 支持多种负采样方法
2. 使用余弦相似度计算用户和物品的相似性
3. 支持早停机制，避免过拟合
4. 提供完整的训练、评估和嵌入提取功能
5. 支持学习率调度器，优化训练过程
6. 提供MovieLens电影推荐示例，展示实际应用场景

## MovieLens推荐示例
本项目提供了基于MovieLens-1M数据集的电影推荐示例。该示例展示了如何：
1. 加载和处理MovieLens数据集
2. 构建用户和电影特征
3. 训练DSSM模型学习用户和电影的嵌入表示
4. 使用学习到的嵌入进行电影推荐

MovieLens-1M数据集包含约100万条评分记录，涉及6000多名用户对4000多部电影的评价。通过DSSM模型，我们可以学习用户和电影的低维表示，并基于余弦相似度进行个性化推荐。

## 引用
如果您使用了本项目的代码，请引用原论文：
```
@inproceedings{huang2013learning,
  title={Learning deep structured semantic models for web search using clickthrough data},
  author={Huang, Po-Sen and He, Xiaodong and Gao, Jianfeng and Deng, Li and Acero, Alex and Heck, Larry},
  booktitle={Proceedings of the 22nd ACM international conference on Information \& Knowledge Management},
  pages={2333--2338},
  year={2013}
}
```


