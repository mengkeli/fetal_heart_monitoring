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

### fetal_cnn.py

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
| 1        | 3       | (10,20)     | (4,8)   | (4,4)     | 0     | 0        |        |
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



`加了一个卷积块`

```python
model.add(Conv2D(filters=4, kernel_size=(10, 20),
                 strides=(4, 8),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(4, (10, 20), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(1,1), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters=4, kernel_size=(10, 20),
                 strides=(4, 8),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(1,1), padding='same'))

model.add(Conv2D(filters=4, kernel_size=(10, 20),
                 strides=(4, 8),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(1,1), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten()) # 20180402_1
```



| 20180402 | filters   | kernel  | pool | strides | Drop_out | train_err | Test_err |
| -------- | --------- | ------- | ---- | ------- | -------- | --------- | -------- |
| 1        | 4         | 11-11-3 | 4    | 4-4-4   | 0.25     | 0.8324    | 0.7948   |
| 2        | 加dropout |         |      |         |          | 0.7888    | 0.7953   |

| 20180403 | filters   | kernel   | pool     | strides | Drop_out        | train_err | Test_err |
| -------- | --------- | -------- | -------- | ------- | --------------- | --------- | -------- |
| 1        | 8-8-8     | 11-11-3  | 4-4-4    | 4-2-2   | 0.25            | 0.8109    | 0.8097   |
| 2        |           |          |          | 2-2-2   |                 | 0.8011    | 0.7768   |
| 3        | 重跑1     |          |          |         |                 |           | 0.7842   |
| 4        | 1的epoch  | 加到20   |          |         |                 |           | 0.7790   |
| 5        | 4去除第一 | 个卷积块 | 的一个   | 卷积层  |                 | 0.8210    | 0.7839   |
| 6        |           |          |          |         | 加一个0.25      | 0.8100    | 0.7215   |
| 7        |           |          |          |         | 0.5-0.25-0.2    | 0.8120    | 0.7808   |
| 8        |           |          |          |         | 0.5-0.4-0.25    | 0.8060    | 0.7887   |
| 9        | 8-8-16    |          |          |         |                 | 0.8120    | 0.8029   |
| 10       | 8-8-16    | 第一卷积 | 块加一层 |         |                 | 0.8203    | 0.7878   |
| 11       |           | 第二卷积 | 块加一层 |         |                 | 0.8151    | 0.7968   |
| 12       |           |          |          | 2-2-2   |                 | 0.8278    | 0.7759   |
| 13       |           |          |          |         | 0.5-0.50.4-0.25 | 0.8037    | 0.7817   |
|          |           |          |          |         |                 |           |          |




###fetal_mlp.py

```python
model.add(Dense(512, activation='relu', input_shape=(288240,)))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

```

| 20180328 | Dense1 | drop_out1 | Dense2 | drop_out2 | train_acc | Test_acc | notes  |
| -------- | ------ | --------- | ------ | --------- | --------- | -------- | ------ |
| 1        | 512    | 0.2       | 512    | 0.2       | 0.9942    |          | 过拟合 |
| 2        | 128    | 0.4       | 128    | 0.4       | 0.9901    | 0.7289   | 过拟合 |



| 20180404 | Dense      | drop_out    | train_acc | Test_acc | notes  |
| -------- | ---------- | ----------- | --------- | -------- | ------ |
| 1        | 64-64-64   | 0.4-0.4-0.4 | 0.9799    | 0.7087   | 过拟合 |
| 2        |            | 0.5-0.5-0.5 | 0.9622    | 0.7362   | 过拟合 |
| 3        |            | 0.6-0.6-0.6 | 0.9003    | 0.6947   |        |
| 4        | 64-128-256 |             | 0.9380    | 0.7317   |        |
|          |            |             |           |          |        |
|          |            |             |           |          |        |


###fetal_cifar_cnn.py

| 20180329 | Batch | epoch | drop_out | Dense | opt  | aug   | train_acc | Test_acc |
| -------- | ----- | ----- | -------- | ----- | ---- | ----- | --------- | -------- |
| 1        | 16    | 10    | 0.5      | 128   | RMSp | True  | 0.7397    | 0.7720   |
| 2        | 8     | 10    | 0.5      | 128   | Adam | True  | 0.5995    | 0.6014   |
| 3        | 8     | 10    | 0.25     | 128   | Adam | False | 0.6014    | 0.5939   |
| 4        | 16    | 10    | 0.25     | 128   | Adam | True  | 0.7730    | 0.7808   |

| 20180330        | Batch     | epoch            | Pool    | Dense | Kernal    | aug  | train_acc | Test_acc |
| --------------- | --------- | ---------------- | ------- | ----- | --------- | ---- | --------- | -------- |
| 1               | 16        | 10               | (2,2)   | 128   | (5,5)     | True | 0.6017    | 0.5928   |
| 2               | 16        | 30               | (4,4)   | 128   | ==(3,3)== | True | 0.7813    | 0.7939   |
| 3               | 16        | 5                | (4,4)   | 128   | (3,3)     | True | 0.7642    | 0.7727   |
| 4减少一个卷积块 |           | 5                |         |       |           |      | 0.7603    | 0.7738   |
| 5padding-valid  |           | 5                |         |       |           |      | 0.7558    | 0.7709   |
| 6               | 3个卷积块 | 10               |         |       |           |      | 0.6008    | 0.5966   |
| 7               | 2个卷积块 | 图像增强0.2->0.4 |         |       |           |      | 0.7241    | 0.7709   |
| 8               | 3个卷积块 | 30epoch          | RMSprop |       |           |      |           |          |

###fetal_hierarchical_rnn.py

| 20180330 | batch |      |      |      |      |      | Train_err | test_err |
| -------- | ----- | ---- | ---- | ---- | ---- | ---- | --------- | -------- |
| 1        | 16    |      |      |      |      |      |           | 0.7398   |
| 2        |       |      |      |      |      |      |           |          |
| 3        |       |      |      |      |      |      |           |          |
| 4        |       |      |      |      |      |      |           |          |




## data_process.py

- 20180329

图像可视化，曲线平滑处理

- 20180330

用python画出平滑的曲线（插值法）

numpy、scipy、matplotlib

插值：

1. nearest 最近邻插值法
2. zero： 阶梯插值
3.  slinear：线性插值
4. quadratic、cubic：2、3阶B样条曲线插值

拟合和插值的区别：

简单来说，插值就是根据原有数据进行填充，最后生成的曲线一定过原有点。

拟合是通过原有数据，调整曲线系数，使得曲线与已知点集的差别（最小二乘）最小，最后生成的曲线不一定经过原有点。

- 20180422

（1）对胎心率信号曲线进行缺失值统计，比例大于阈值的样本被剔除；

（2）对胎心率信号曲线进行断点检测，连续断点数超过阈值的样本被剔除；50

（3）对胎心率信号曲线进行线性插值缺失值修补；

（4）对胎心率信号曲线进行降噪（任何连续5次低于10 bpm的心率都被视为稳定心率。然后，每当相邻心率之间的差异高于25 bpm时，样本将通过之前心率与新稳定心率之间的线性插值进行取代。），将数据值控制在合理范围内。

展示用图的标准尺寸为300*30

训练用图的尺寸为10*0.5，在能看清图像的前提下，使得图像最小



zero_rate = 0.100000
zero_length = 20
origin_row = 24360.
total_row = 18237.
bad_row = 6109.
zero_row = 14.

zero_rate = 0.100000
zero_length = 30
origin_row = 24360.
total_row = 20123.
bad_row = 4143.
zero_row = 94.

zero_rate = 0.150000
zero_length = 30

total_row = 20202.

bad_row = 4143.
zero_row = 15.



zero_rate = 0.150000
zero_length = 20
origin_row = 24360.
total_row = 18248.
bad_row = 6109.
zero_row = 3.