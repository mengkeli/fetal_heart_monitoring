# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data_file):
    '''
    plot the data distribution of a specific data_file
    '''
    data = np.loadtxt(data_file, delimiter=',' skiprows = 1)
    x= data[:, 2]
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9,6))

#第二个参数是柱子宽一些还是窄一些，越大越窄越密  
ax0.hist(x,40,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  

##pdf概率分布图，一万个数落在某个区间内的数有多少个  
ax0.set_title('pdf')  
ax1.hist(x,20,normed=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)  

#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率  
ax1.set_title("cdf")  
fig.subplots_adjust(hspace=0.4)  
plt.show()  

