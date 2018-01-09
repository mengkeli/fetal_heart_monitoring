# -*- coding:utf-8 -*-

from sklearn import cross_validation, svm
from sklearn.externals import joblib
import pandas as pd

datapath = '../data/'
data = pd.read_csv(datapath + 'data.csv')
info = pd.read_csv(datapath + 'info.csv')
merge = pd.merge(data, info, on='id')

# 提取特征和y
labels = merge['nst_result']
merge.drop(
    ['id', 'userid', 'super_id', 'service_id', 'hospitalid', 'doctorid', 'index_num', 'status', 'flag', 'nst_result'],
    axis=1, inplace=True)

train = merge.fillna(0).as_matrix()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.1)

# svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

# save model
# model_file = '../model/svm_train_model.m'
# joblib.dump(clf, model_file)

# # predict
# clf = joblib.load(model_file)
# clf.predict(X_test)
