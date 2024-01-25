## Introduction - 介绍

如何高效、合理地进行投资组合已经成为一个热点问题。针对目前预测股价投资的不稳定性和投资比例难以确定的问题，提出了趋势峰价跟踪法。首先，由于股价异常的影响，TPPT策略设置了可调整的历史窗口宽度。利用斜率值判断预测方向跟踪价格变化，采用指数移动平均和峰值等权斜率值三态价格预测方法。其次，对累积财富目标进行细化，并加入基于梯度投影算法（BP）的快速误差反向传播算法。该算法求解投资比例，并将资产的增值能力反馈到投资比例上，以使积累的财富最大化。最后，通过五个典型数据对八种策略的实证分析和统计检验，表明TPPT策略在平衡风险和收益方面具有很大的优势，是一种稳健有效的在线投资组合策略。

## 复现论文

基于趋势促进价格跟踪集成学习算法的在线投资组合策略

An online portfolio strategy based on trend promote price tracing ensemble learning algorithm

来源：https://doi.org/10.1016/j.knosys.2021.107957

引用：

```
@article{10.1016/j.knosys.2021.107957,
author = {Dai, Hong-Liang and Liang, Chu-Xin and Dai, Hong-Ming and Huang, Cui-Yin and Adnan, Rana Muhammad},
title = {An Online Portfolio Strategy Based on Trend Promote Price Tracing Ensemble Learning Algorithm},
year = {2022},
issue_date = {Mar 2022},
publisher = {Elsevier Science Publishers B. V.},
address = {NLD},
volume = {239},
number = {C},
issn = {0950-7051},
url = {https://doi.org/10.1016/j.knosys.2021.107957},
doi = {10.1016/j.knosys.2021.107957},
journal = {Know.-Based Syst.},
month = {mar},
numpages = {10},
keywords = {Three-state price, Price anomaly, Gradient projection, Ensemble learning algorithm, Online portfolio investment, Investment ratio}
}
```

## Requirements - 必要条件

详见requirements.txt

## Configuration - 配置

python==3.8

## Usage - 用法

安装依赖库

```
pip install -r requirements.txt
```

运行文件

```
cd ./TPPT

python SimpleTest.py
```

## Contact - 联系

唐璟玥 2310274009@email.szu.edu.cn

深圳大学

