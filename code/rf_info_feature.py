# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation, svm
from sklearn.externals import joblib
import pandas as pd

datapath = '../data/'
info = pd.read_csv(datapath + 'info.csv')

# 提取特征和y
labels = info['nst_result']
features = ['jixian', 'jixianbianyi', 'taidongjiasutime', 'taidongjiasufudu', 'taidongcishu', 'bianyizhouqi', 'jiasu', 'jiasuzhenfu', 'jiansu', 'jiansuzhenfu', 'jiansuzhouqi']

train = info[features].fillna(0).as_matrix()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.3)

# RF
srf = RF(n_estimators=1000, n_jobs=-1)
srf.fit(X_train, y_train)
print srf.score(X_test, y_test)

# save model
model_file = '../model/rf_info_feature.model'
joblib.dump(srf, model_file)

# # predict
# clf = joblib.load(model_file)
# clf.predict(X_test)
