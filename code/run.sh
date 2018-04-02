#!/bin/bash

nohup python fetal_cifar10_cnn.py > log/cifar10_cnn_20180330_8.log &
nohup python fetal_hierarchical_rnn.py >  log/hrnn_20180330_1.log &
nohup python fetal_resnet.py > log/resnet_20180330_1.log &
