# 胎心数据分析

## 数据分析
- `info` 表, 大多数为空或者全部为0或者1的字段: 
- `jiasutimes` == 0, `jiansutimes` == 0, `wanjiansuzhenfu` == NaN, `wanjiansuzhouqi` == NaN, `wanjiansutimes` == 0
- `info` 表，`id` 与 `data` 表对应
- 统计得`nst_result` == 1，2，3，4 分别对应正常（18699/13309），可疑（13368/8867），异常（28/15），无法判读（85/6）

## Requirements
- python (2.7.10)
- scikit-learn (0.17.1)
- numpy (1.11.0+)
- pandas (0.18.1)
- tensorflow (1.1.0) / theano (v0.6.9)
- keras (2.0.9)
- CUDA (8.0)
- cuDNN (V6)

## 数据处理


## 问题

- 数据缺失严重，存在很多连续的零值

## 思路

- [x] 将零值用整体平均值替代，效果不好
- [ ] 发现很多零都是连续的，用零两端的平均值作为替代，效果不好
- [ ] 用朴素贝叶斯方法来处理零值
- [ ] 用KNN来对于缺失的点，找到与其类似的记录的点在缺失的时候的历史发生情况

## 神经网络
- `fetal_MKNet.py` 基于卷积神经网络的算法MKNet
- `fetal_lstm.py` 基于循环神经网络的算法MKRNN

## 代码说明

- `code/func.py` 数据读取、处理的基本函数
- `code/import.py` 将数据库中的数据导出成csv文件
- `code/data_transformation.py` 将csv文件中数据转化为特征
- `code/data_process.py` 数据预处理（数据清洗、去噪等）
- `code/data_operations.ipynb` 对数据进行的部分单独操作
- `code/svm.py` 基于所有的y以及info中某些字段的svm
- `code/rf.py` 基于所有的y以及info中某些字段的rf
- `code/SVM.ipynb` 用网格搜索寻找SVM最佳超参数，训练SVM，画ROC曲线
- `code/RF.ipynb` 用网格搜索寻找RF最佳超参数，训练RF，画ROC曲线
- `code/NN_ROC.ipynb` 画神经网络模型的ROC曲线
- `code/model_structure.ipynb` 画神经网络模型的结构图

## 运行说明
1. 克隆代码

   ```
   git clone git@github.com:mengkeli/fetal_heart_monitoring.git
   ```

2. `data/info.csv`和`data/data.csv`已经是导出好的全部特征文件,这两个文件数据损失严重，弃用
  `data/data_gzip.csv`是从数据库导出的文件，作为实验数据，24360行
  `data/data_gzip.csv.filter`是`data/data_gzip.csv`过滤零值之后的数据
  `data/data_gzip.csv.filter.mean`是`data/data_gzip.csv`将零值用整体均值填充之后的数据
  `data_zero_filer_03_50.csv` 剔除零值超过30%的，连续零值超过50个，剩余22199条，bad_row = 2158， zero_row = 3


3. `http://lark:8885/`用jupyter notebook运行代码

## Evaluation
See `doc/evaluation.xlsx`

