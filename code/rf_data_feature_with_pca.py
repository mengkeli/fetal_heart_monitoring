# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation, svm
from sklearn.externals import joblib
import pandas as pd
import func

datapath = '../data/'
data = pd.read_csv(datapath + 'data_gzip.csv.filter.mean')
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

lowDMat, reconMat = func.pca(train, 1000)
trainMat = lowDMat if with_pca else train
X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainMat, labels, test_size=0.1)

# RF
srf = RF(n_estimators=1000, n_jobs=-1)
srf.fit(X_train, y_train)
print srf.score(X_test, y_test)

# save model
model_file = '../model/rf_data_feature_with_pca.model'
joblib.dump(srf, model_file)

# # predict
# clf = joblib.load(model_file)
# clf.predict(X_test)