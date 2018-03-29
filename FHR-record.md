# FHR - record

##batch_size 设置

- batch_size = 1时，随机梯度下降，梯度容易抵消，算法在一定epoches内不收敛
- 随着 batch_size 增大，处理相同数据量的速度越快。
- 随着 batch_size 增大，达到相同精度所需要的 epoch 数量越来越多。
- 由于上述两种因素的矛盾， Batch_Size 增大到**某个**时候，达到**时间上**的最优。
- 由于最终收敛精度会陷入不同的局部极值，因此 Batch_Size 增大到**某些**时候，达到最终收敛**精度上**的最优。
- 一般设为2的倍数，这样每个batch的数据刚好能塞进内存里，便于并行计算
##epoch设置

一个epoch 就是指数据集里所有数据全训练一遍

epoch太大，缺点有两个…一个是过拟合（overfit）另一个是训练时间太长

## Convolution层





## Pooling层

kernel_size：



## 网络结构



```python
model.add(Conv2D(filters=3, kernel_size=(10, 20),
                 strides=(4, 8),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4)))


```

| 20180327 | filters | kernel_size | strides | pool_size | batch_size | epoch | Score  |
| -------- | ------- | ----------- | ------- | --------- | ---------- | ----- | ------ |
| 1        | 3       | (10,20)     | (1,2)   | (4,4)     | 128-时间短 | 10    | 0.6552 |
| 2        |         |             | (2,4)   |           |            |       | 0.6776 |
| 3        |         |             | (4,8)   |           |            |       | 0.6790 |
| 4        |         |             | (4,8)   |           | 256-时间长 |       | 0.6841 |
| 5        |         |             |         |           | 128        | 20    | 0.6951 |
| 6        |         |             |         |           | 64         | 10    | 0.6901 |
| 7        | 10      |             |         |           | 64         |       | 0.6763 |
| 8        | 10      |             |         |           | 128        |       | 0.6894 |

```python
model.add(Conv2D(filters=4, kernel_size=(10, 20),
                 strides=(4, 8),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(8, (10, 20), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(1,1), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) # 最后一层，输出结果是2个类别

```

| 20180328 | filters | kernel_size | strides | pool_size | Dense | drop_out | Score  |
| -------- | :------ | ----------- | ------- | --------- | ----- | -------- | ------ |
| 1        | 3       | (10,20)     | (4,8)   | (4,4)     | 0     | s0       |        |
|          | 5       | (10,10)     | (1,1)   |           |       |          | 0.7714 |
| 2        | 3       | (10,20)     | (4,8)   |           |       |          |        |
|          | 5       | (10,20)     | (1,1)   |           |       |          | 0.7918 |
| 3        | 3       |             | ()      |           |       |          |        |
|          | 5       |             | (2, 4)  | (4,4)     |       |          | 0.7815 |
| 4        | 4       | (10,20)     | (4,8)   |           |       |          |        |
|          | 8       | (10,20)     | (1,1)   | (4,4)     |       |          | 0.7921 |
| 5        |         |             |         |           |       | 0.25     | 0.7957 |
| 6        |         |             |         |           | 128   | 0.25     | 0.8025 |
| 7        |         |             |         |           |       | 0.25/0.2 | 0.8011 |
| 8        | 4       | (10,20)     |         |           |       |          |        |
|          | 8       |             |         |           |       |          | 0.7948 |

###fetal_mlp.py

```python
model.add(Dense(512, activation='relu', input_shape=(288240,)))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

```





| 20180328 | Dense1 | drop_out1 | Dense2 | drop_out2 |      | train_acc | Test_acc |
| -------- | ------ | --------- | ------ | --------- | ---- | --------- | -------- |
| 1        | 512    | 0.2       | 512    | 0.2       |      | 0.9942    | Test_acc |
| 2        | 128    | 0.4       | 128    | 0.4       |      | 0.9901    | 0.7289   |
| 3        |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |
|          |        |           |        |           |      |           |          |