# -*- coding:utf-8 -*-

from sklearn import cross_validation, svm
from sklearn.externals import joblib
import pandas as pd
import func

datapath = '../data/'
data = pd.read_csv(datapath + 'data_gzip.csv.filter.mean.subtrain')
info = pd.read_csv(datapath + 'info.csv')
merge = pd.merge(data, info, on='id')
with_pca = True

# 提取特征和y
labels = merge['nst_result']
merge.drop(['id', 'nst_result'], axis=1, inplace=True)

# 去掉info带来的特征
info_features = ['jianceshichang', 'jixian', 'jixianbianyi', 'taidongjiasutime', 'taidongjiasufudu', 'taidongcishu',
                 'bianyizhouqi', 'jiasu', 'jiasuzhenfu', 'jiasutimes', 'jiansu', 'jiansuzhenfu', 'jiansuzhouqi',
                 'jiansutimes', 'wanjiansu', 'wanjiansuzhenfu', 'wanjiansuzhouqi', 'wanjiansutimes', 'jixian',
                 'jixianbianyi', 'taidongjiasutime', 'taidongjiasufudu', 'taidongcishu']

merge.drop(info_features, axis=1, inplace=True)
train = merge.fillna(0).as_matrix()

lowDMat, reconMat = func.pca(train, 100)
trainMat = lowDMat if with_pca else train
X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainMat, labels, test_size=0.3)

# svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

# save model
model_file = '../model/svm_data_feature_subtrain_with_pca.model'
joblib.dump(clf, model_file)

# # predict
# clf = joblib.load(model_file)
# clf.predict(X_test)
