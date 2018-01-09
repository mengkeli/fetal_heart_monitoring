# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation, svm
from sklearn.externals import joblib
import pandas as pd

datapath = '../data/'
data = pd.read_csv(datapath + 'data_gzip.csv.neighbor_mean.mean')
info = pd.read_csv(datapath + 'info.csv')
merge = pd.merge(data, info, on='id')

# 提取特征和y
labels = merge['nst_result']
merge.drop(['id', 'nst_result'], axis=1, inplace=True)

train = merge.fillna(0).as_matrix()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.2)

# RF
srf = RF(n_estimators=500, n_jobs=-1)
srf.fit(X_train, y_train)
print srf.score(X_test, y_test)

# save model
# model_file = '../model/svm_train_model.m'
# joblib.dump(clf, model_file)

# # predict
# clf = joblib.load(model_file)
# clf.predict(X_test)