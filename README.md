# DSSM-Lookalike Pipeline
implemention of paper: [Learning deep structured semantic models for web search using clickthrough data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)

一个基于DSSM的Lookalike向量化召回pipeline简单实现，包括以下部分：
* 数据获取：使用[2018腾讯广告算法大赛](https://wx.jdcloud.com/market/jdata/list/17)比赛中的训练数据（100000条）(aid,uid,label)。
* 三种数据负采样方法：<1>随机负采样。<2>根据出现频次负采样。<3>RALM采样方法，参考[腾讯实时Look-alike](https://arxiv.org/abs/1906.05022)
* DSSM实现，与原论文在loss计算方式上略有不同。
* 输出user embedding和item embedding。

较为完整的复现了工业界常用的基于DSSM的向量化召回pipeline。

## 网络模块构建参考
* code: https://github.com/shenweichen/DeepCTR
* author: shenweichen

## 环境配置
* python 3.6.5
* tensorflow == 1.14.0
* pandas

## 代码运行示例：
```
python examples/run_dssm_neg_sample.py
```


