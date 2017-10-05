# -*- coding: utf-8 -*-
'''
create time: 2017/9/29 19:02
author: duxin
site: 
email: duxin_be@outlook.com
'''
import pandas as pd, numpy as np
import tensorflow as tf
import copy, time, string
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
import xgboost as xgb
from Tools import *

# /********************* 读取数据 ********************/
dfTrainOri = pd.read_csv(filepath_or_buffer = 'data/train.csv')
dfTestOri = pd.read_csv(filepath_or_buffer = 'data/test.csv')
dfTestYOri = pd.read_csv(filepath_or_buffer = 'data/gender_submission.csv')

# /******************* 数据预处理/包装 ***************/
dfWhole = pd.concat([dfTrainOri, dfTestOri])
dfWhole.index = dfWhole['PassengerId']
dfWhole = dfWhole.drop('PassengerId', axis = 1)
dfWhole

dfWholeY = dfWhole['Survived']
dfWholeX = copy.copy(dfWhole)
dfWholeX['SEmbarked'] = dfWholeX['Embarked'] == 'S'
dfWholeX['CEmbarked'] = dfWholeX['Embarked'] == 'C'
dfWholeX['QEmbarked'] = dfWholeX['Embarked'] == 'Q'
dfWholeX['nameLength'] = dfWholeX['Name'].apply(len)
dfWholeX.loc[dfWholeX['Sex'] == 'male', 'Sex'] = 1
dfWholeX.loc[dfWholeX['Sex'] == 'female', 'Sex'] = 0
dfWholeX = dfWholeX.drop(['Survived','Name','Ticket','Cabin','Embarked'], axis = 1)
dfWholeX.loc[np.isnan(dfWholeX['Age']), 'Age'] = np.mean(dfWholeX['Age'])
dfWholeX.loc[np.isnan(dfWholeX['Fare']), 'Fare'] = np.mean(dfWholeX['Fare'])
arWholeX = np.array(dfWholeX).T
arWholeY = np.array(dfWholeY).T.reshape([1, -1])

ratioTrainDev = 8  # Train集合size与Dev集合size之比
trainSetSize = np.sum(~np.isnan(arWholeY))
devSetSize = int(1/(1 + ratioTrainDev)*trainSetSize)
trainSetSize += -devSetSize
arTestX = arWholeX[:, np.isnan(arWholeY).reshape([-1])]
arTrainX = arWholeX[:, :trainSetSize]
arTrainY = arWholeY[:, :trainSetSize]
arDevX = arWholeX[:, trainSetSize:(trainSetSize+devSetSize)]
arDevY = arWholeY[:, trainSetSize:(trainSetSize+devSetSize)]

dpTrainX = dataPool(data = arTrainX, axis = 1)
dpTrainY = dataPool(data = arTrainY.reshape([1,-1]), axis = 1)
dpDevX = dataPool(data = arDevX, axis = 1)
dpDevY = dataPool(data = arDevY, axis = 1)
dpTestX = dataPool(data = arTestX, axis = 1)

# /********************* 归一化 **********************/
Xrange = [np.min(arTrainX, axis = 1).reshape([-1, 1]), np.max(arTrainX, axis = 1).reshape([-1, 1])]
arTrainX =  (arTrainX - Xrange[0])/ (Xrange[1] - Xrange[0]) * 2 - 1
arTestX = (arTestX - Xrange[0]) / (Xrange[1] - Xrange[0]) * 2 - 1

# /********************** PCA ************************/
arTrainX, arDevX, arTestX = pcaProcess(Xtrain = arTrainX.T, Xdev = arDevX.T, Xtest = arTestX.T, n_components = 5)
arTrainX = arTrainX.T
arDevX = arDevX.T
arTestX = arTestX.T

# /*********************** SVM ***********************/
clf = svm.SVC(kernel = 'rbf', gamma = 800, C = 5)
clf.fit(arTrainX.T, arTrainY.reshape([-1]))
predTrain = clf.predict(arTrainX.T)
precisionTrain = sum(predTrain == arTrainY.reshape([-1])) / predTrain.shape[0]
print(precisionTrain)
predDev = clf.predict(arDevX.T)
precisionDev = sum(predDev == arDevY.reshape([-1])) / predDev.shape[0]
print(precisionDev)

predsTest = clf.predict(arTestX.T)
print("训练集准确率: ", np.mean(predTrain == arTrainY))
print("开发集准确率: ", np.mean(predDev == arDevY))

# /**************** 准确率变化曲线 ******************/
# plt.figure()
# plt.plot(iterPath, crTrainPath)
# plt.plot(iterPath, crTestPath)
# plt.show()

# /**************** 输出预测结果 ********************/
result = pd.DataFrame({'PassengerId':dfTestYOri['PassengerId']} )
result['Survived'] = list(map(lambda x:int(x), predsTest.reshape([-1])))
result
result.to_csv(path_or_buf='data\\result%s.csv'%(time.strftime('%Y%m%d')), header=True, index=None)