# -*- coding:utf-8 -*-

from sklearn import cross_validation, svm
from sklearn.externals import joblib
import pandas as pd

datapath = '../data/'
info = pd.read_csv(datapath + 'info.csv')

# 提取特征和y
labels = info['nst_result']
features = ['jixian', 'jixianbianyi', 'taidongjiasutime', 'taidongjiasufudu', 'taidongcishu', 'bianyizhouqi', 'jiasu', 'jiasuzhenfu', 'jiansu', 'jiansuzhenfu', 'jiansuzhouqi']

train = info[features].fillna(0).as_matrix()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.2)
from sklearn.cross_validation import StratifiedKFold
cv = StratifiedKFold(y, n_folds=6)
# svm
clf = svm.SVC(kernel='rbf', C=1000, gamma=1e-4, verbose=0)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

# save model
model_file = '../model/svm_info_feature.model'
joblib.dump(clf, model_file)

# # predict
# clf = joblib.load(model_file)
# clf.predict(X_test)