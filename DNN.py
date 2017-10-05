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

# /**************** 神经网络设置 *********************/
layersize = [200, 100, 20, 1]
layercount = len(layersize)
seed = time.time()
X = tf.placeholder(shape = [arTrainX.shape[0], None], dtype = 'float')
Y_ = tf.placeholder(shape = [layersize[-1], None], dtype = 'float')
W = dict()
b = dict()
Y = dict()
Y[0] = X
layersizeTemp = [arTrainX.shape[0]] + layersize
for i in range(layercount):
    W[i+1] = tf.Variable(tf.random_normal(seed = np.random.randint(seed), shape = [layersizeTemp[i+1], layersizeTemp[i]]) * 0.1)
    b[i+1] = tf.Variable(tf.zeros(shape = [layersizeTemp[i+1], 1]))
    if i != layercount - 1:
        Y[i+1] = tf.nn.relu(tf.matmul(W[i+1], tf.nn.dropout(Y[i], keep_prob = 0.9)) + b[i+1])
Y[layercount] = tf.nn.sigmoid(tf.matmul(W[layercount], Y[layercount-1]) + b[layercount])
Youtput = Y[layercount]

correct_prediction = tf.equal(tf.round(Youtput), Y_)
correct_rate = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

cost = tf.reduce_mean(tf.square(Youtput - Y_))
for w in W:
    cost += tf.reduce_mean(tf.square(W[w]))
#cost = - tf.reduce_mean(Y_*tf.log(Youtput) + (1 - Y_)*tf.log(1 - Youtput))

# /****************** 参数初始化 *********************/
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# /********************** 训练 **********************/
train_step = tf.train.GradientDescentOptimizer(1).minimize(cost)
iterPath = []
crTrainPath = []
crTestPath = []
for i in range(200001):
    #sess.run(train_step, feed_dict = {X:dpTrainX.getData(size=50), Y_:dpTrainY.getData(size=50)})
    sess.run(train_step, feed_dict = {X:arTrainX, Y_:arTrainY})
    if i % 1000 == 0:
        iterPath.append(i)
        crTrain = sess.run(correct_rate, feed_dict={X:arTrainX, Y_: arTrainY})
        crTest = sess.run(correct_rate, feed_dict={X:arDevX, Y_: arDevY})
        crTrainPath.append(crTrain)
        crTestPath.append(crTest)
        print(crTrain, crTest)

# /********************** 预测 **********************/
predsTrain = sess.run(tf.round(Youtput), feed_dict = {X:arTrainX})
predsDev = sess.run(tf.round(Youtput), feed_dict = {X:arDevX})
predsTest = sess.run(tf.round(Youtput), feed_dict = {X:arTestX})
print("训练集准确率: ", np.mean(predsTrain == arTrainY))
print("开发集准确率: ", np.mean(predsDev == arDevY))

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