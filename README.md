# 胎心数据分析

## 数据分析
- `info` 表, 大多数为空或者全部为0或者1的字段: 
- `jiasutimes` == 0, `jiansutimes` == 0, `wanjiansuzhenfu` == NaN, `wanjiansuzhouqi` == NaN, `wanjiansutimes` == 0
- `info` 表，`id` 与 `data` 表对应

## Requirements
- python (2.7.10)
- scikit-learn (0.17.1)
- numpy (1.11.0+)
- pandas (0.18.1)
- tensorflow () / theano ()
- keras ()
- CUDA ()
- cuDNN ()

## 问题

- 数据缺失严重，存在很多连续的零值

## 思路

- [x] 将零值用整体平均值替代，效果不好
- [x] 发现很多零都是连续的，用零两端的平均值作为替代，效果不好
- [ ] 用朴素贝叶斯方法来处理零值
- [ ] 用KNN来对于缺失的点，找到与其类似的记录的点在缺失的时候的历史发生情况

## 代码说明

- `code/import.py` 将数据库中的数据导出成csv文件
- `code/svm.py` 基于所有的y以及info中某些字段的svm
- `run.sh` 执行输入导入、训练全过程

## 运行说明
1. 克隆代码

   ```
   git clone git@github.com:mengkeli/fetal_heart_monitoring.git
   ```

2. `data/info.csv`和`data/data.csv`已经是导出好的全部特征文件,这两个文件数据损失严重，弃用
`data/data_gzip.csv`是从数据库导出的文件，作为实验数据，24360行
`data/data_gzip.csv.filter`是`data/data_gzip.csv`过滤零值之后的数据
`data/data_gzip.csv.filter.mean`是`data/data_gzip.csv`将零值用整体均值填充之后的数据
`data_zero_filer_03_50.csv` 剔除零值超过30%的，连续零值超过50个，剩余21810条，bad_row = 2158， zero_row = 389


3. `code/svm.py`为所有特征的`svm`, 目前正确率不是很高, 期待大家的调参结果

## Evaluation
See `doc/evaluation.xlsx`

## Reference
- [MemN2N](https://github.com/priyank87/memn2n)
- [Python轻量级web框架Flask文档](http://flask.pocoo.org/)

## 代码目录说明
```
TODO: 待整理
```
