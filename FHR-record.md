# FHR - record

## SVM

```
Best parameters set found on development set:
()
{'kernel': 'rbf', 'C': 1000, 'gamma': 0.0001}
()
Grid scores on development set:
()
0.381 (+/-0.107) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.001}
0.382 (+/-0.107) for {'kernel': 'rbf', 'C': 1, 'gamma': 0.0001}
0.393 (+/-0.103) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.001}
0.396 (+/-0.090) for {'kernel': 'rbf', 'C': 10, 'gamma': 0.0001}
0.398 (+/-0.135) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.001}
0.410 (+/-0.106) for {'kernel': 'rbf', 'C': 100, 'gamma': 0.0001}
0.353 (+/-0.114) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}
0.419 (+/-0.097) for {'kernel': 'rbf', 'C': 1000, 'gamma': 0.0001}
()
Detailed classification report:
()
The model is trained on the full development set.
The scores are computed on the full evaluation set.
()
             precision    recall  f1-score   support

          1       0.83      0.89      0.86      5612
          2       0.83      0.75      0.79      4002
          3       0.00      0.00      0.00         9
          4       0.33      0.10      0.15        31
          5       0.00      0.00      0.00         2

avg / total       0.83      0.83      0.83      9656

```



## RF

## CNN

###batch_size 设置

- batch_size = 1时，随机梯度下降，梯度容易抵消，算法在一定epoches内不收敛
- 随着 batch_size 增大，处理相同数据量的速度越快。
- 随着 batch_size 增大，达到相同精度所需要的 epoch 数量越来越多。
- 由于上述两种因素的矛盾， Batch_Size 增大到**某个**时候，达到**时间上**的最优。
- 由于最终收敛精度会陷入不同的局部极值，因此 Batch_Size 增大到**某些**时候，达到最终收敛**精度上**的最优。
- 一般设为2的倍数，这样每个batch的数据刚好能塞进内存里，便于并行计算
###epoch设置

一个epoch 就是指数据集里所有数据全训练一遍

epoch太大，缺点有两个…一个是过拟合（overfit）另一个是训练时间太长

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

| 20180504 | filters           | kernel  | pool | strides | Drop_out | train_err | Test_err |
| :------- | ----------------- | ------- | ---- | ------- | -------- | --------- | -------- |
| 1        | 4-8               | 3–3     | 4    | 2-1     | 0.25-0.2 | 0.6035    | 0.6021   |
| 2        | 4-8               | 3–3     | 4    | 1-1     | 0.25-0.2 | 0.6035    | 0.6021   |
| 3        | 32-32-32-32       | 3-3-3-3 | 2    | 2       | 0.25-0.2 | 0.9556    | 0.6608   |
| 4        |                   |         |      |         |          | 0.8596    | 0.6622   |
| 5        | 32-32-32-32-32-32 |         |      |         |          | 0.7416    | 0.6888   |

| 20180505 | filters               | kernel               | pool               | strides | Drop_out            | train_err | Test_err |
| -------- | --------------------- | -------------------- | ------------------ | ------- | ------------------- | --------- | -------- |
| 1        | 4-8                   | (10,20)strides=(4,8) | (4,4)strides=(1,1) |         | 0.25-0.2            | 0.8599    | 0.7962   |
| 2        | 8-8-8                 |                      |                    |         | 0.25-0.2            | 0.8232    | 0.8009   |
| 3        | 8-8-8-8               |                      |                    |         | 0.25-0.25-0.2       | 0.8380    | 0.7852   |
| 4        |                       |                      |                    |         | 0.3-0.3-0.3         | 0.8399    | 0.7793   |
| 5        |                       |                      |                    |         | 0.3-0.3-0.3-0.3-0.3 | 0.8616    | 0.7935   |
| 6        |                       |                      |                    |         |                     | 0.9221    | 0.7711   |
| 7        |                       |                      |                    |         | 在加两个0.1dropout  | 0.8718    | 0.7852   |
| 8        | padding=same          |                      |                    |         |                     | 0.8840    | 0.7927   |
| 9        | 8-8-8-8-16-16         |                      |                    | (4,4)   |                     | 0.8974    | 0.7840   |
| 10       | 4-4-8-8               | (10,20)strides=(4,4) | (4,4)strides=(2,2) |         |                     | 0.8704    | 0.7910   |
| 11       | 4-4-8-8-16-16         |                      |                    |         |                     | 0.8661    | 0.7980   |
| 12       | 4-4-8-8-16-16         | (10,10)strides=(4,4) |                    |         |                     | 0.8516    | 0.8054   |
| 13       | 8-8-16-16-32-32       | (11,11)strides=(4,4) | (2,2)strides=(2,2) |         |                     |           |          |
| 14       | 8-8-16-16-32-32       |                      |                    |         |                     | 0.9369    | 0.7763   |
| 15       | 8-8-16-16-32-32-64-64 |                      |                    |         |                     | 0.9146    | 0.7952   |

kernel=(11,11),strides=(2,2)的时候时间为76s一个epoch，（4,4）的时候是29s

kernel=(3,3)*5,strides=(4,4)时间为28s。节省了1s的时间

14在13的基础上加了1*1卷积核



```python
model.add(Conv2D(8, (11, 11), strides=(4, 4),input_shape=input_shape, activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(Conv2D(8, (11, 11), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(16, (11, 11), activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(Conv2D(16, (11, 11), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(32, (11, 11), activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(Conv2D(32, (11, 11), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(64, (11, 11), activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(Conv2D(64, (11, 11), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax')) 
```



| 20180507 | filters                     | kernel               | pool               | strides | Drop_out | train_err | Test_err |
| -------- | --------------------------- | -------------------- | ------------------ | ------- | -------- | --------- | -------- |
| 1        | 8-8-16-16-32-32-64-64       | (11,11)strides=(4,4) | (2,2)strides=(2,2) |         | 0.5      | 0.8648    | 0.8072   |
| 2        |                             |                      | (4,4)              |         | 0.3      | 0.8987    | 0.8009   |
| 3        | 8-8-16-16-32-32-64-64       |                      |                    |         | 0.25-0.2 | 0.8741    | 0.8034   |
| 4        | 8-8-16-16-32-32-32-32-64-64 |                      |                    |         |          | 0.8342    | 0.8168   |
| 5        |                             | (13,13)              |                    |         |          | 0.8720    | 0.7965   |
| 6        |                             | (7,7)                |                    |         |          | 0.8371    | 0.8152   |
| 7        |                             | (5,5)                |                    |         |          | 0.8166    | 0.8197   |
| 8        |                             | 加了1*1卷积层        |                    |         |          | 0.8291    | 0.8189   |

5、6、7作为一组对比，显示小的卷积核在同样的准确率下能够降低过拟合。

下一步考虑增加卷积块，来提高准确率。

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

| 20180505 | filters | kernel | pool | strides | Drop_out | train_err | Test_err |
| -------- | ------- | ------ | ---- | ------- | -------- | --------- | -------- |
| 1        | 4-8     | 3–3    | 4    | 2-1     | 0.25-0.2 | 0.6035    | 0.6953   |

### fetal_vgg.py

```python
model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

1. val_acc = 0.6593， train_acc = 0.7997


2. Dense层256改成512之后：val_acc = 0.6789， train_acc = 0.7784
3. Dense层512改成1024之后：val_acc = 0.6699， train_acc = 0.8158

**取Dense层512**

4.drop_out由0.25改为0.3：val_acc = 0.6874， train_acc = 0.8201

5.取消第一层strides：val_acc = 0.6647， train_acc = 0.8261



## LSTM

```python
model = Sequential()
model.add(LSTM(1024, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True, activation='relu'))
# model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(units=output_size))
# model.add(Activation('tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
```

1. acc：0.6118

```
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
```





## data_process.py

- 20180329

图像可视化，曲线平滑处理

- 20180330

用python画出平滑的曲线（插值法）

numpy、scipy、matplotlib



展示用图的标准尺寸为300*30

训练用图的尺寸为10*0.5，在能看清图像的前提下，使得图像最小

join_data_label	之后01_30还有20118条数据



## 实验参数调整

ROC_rf 图

![ROC_rf](data/ROC_rf.png)

ROC_cnn 图

![ROC_cnn](data/ROC_cnn.png)

###batch_size 对模型稳定性的影响

batch_size = 512, data_aug = False

![cnn_90_8-16-32-64_nodataaug_acc](data/cnn_90_8-16-32-64_nodataaug_acc.png)

### 防止模型过拟合

early stopping和增加fc层，之前：

![cnn_90_8-16-32-64_nodataaug_512_acc](data/cnn_90_8-16-32-64_nodataaug_512_acc.png)

![cnn_90_8-16-32-64_nodataaug_512_loss](data/cnn_90_8-16-32-64_nodataaug_512_loss.png)

这时，我们就说模型的方差较大（可以形象理解为模型波动较大）。

所以，需要在偏置和方差之间加以平衡。当模型过于简单，参数较少，则偏差较大（但是方差较小），当模型过于复杂，参数很多，则方差较大（但是偏差较小）。

之后：



将模型8-16-32-64的结构变为8-16-32，模型复杂度降低，准确率反而上升

8-16-32-64：

![cnn_90_8-16-32-64_nodataaug_512_acc](data/cnn_90_8-16-32-64_nodataaug_512_acc.png)

8-16-32：

![cnn_90_8-16-32_nodataaug_512_acc](data/cnn_90_8-16-32_nodataaug_512_acc.png)

batch_size=1024，lr=0.003: （batch_size增大之后，曲线变得平滑）

![cnn_90_8-16-32_nodataaug_1024_lr0003_acc](data/cnn_90_8-16-32_nodataaug_1024_lr0003_acc.png)

batch_size=1024，lr=0.0005: 

![cnn_90_8-16-32_nodataaug_1024_lr00005_acc](data/cnn_90_8-16-32_nodataaug_1024_lr00005_acc_1.png)











