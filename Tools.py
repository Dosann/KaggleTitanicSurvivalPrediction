# -*- coding: utf-8 -*-
'''
create time: 2017/9/29 19:02
author: duxin
site: 
email: duxin_be@outlook.com
'''

from sklearn.decomposition import PCA

class dataPool():
    def __init__(self, data, axis = 1):
        self.dataAxis = axis
        if self.dataAxis == 1:
            self.dataset = data
        else:
            self.dataset = data.T
        self.currentPoint = 0
        self.lastpoint = 9
        self.datasetSize = self.dataset.shape[1]

    def getData(self, size = None):
        if size == None:
            size = int(self.datasetSize / 10)
        self.lastPoint = self.currentPoint
        self.currentPoint += size
        block1 = self.dataset[:, self.lastPoint:self.currentPoint]
        block2 = self.dataset[:, 0:max([self.currentPoint-self.datasetSize, 0])]
        datablock = np.hstack([block1, block2])
        self.currentPoint = self.currentPoint % self.datasetSize
        return datablock

def calculatePrecision(inputX, inputY, correctFunction, sess):
    return

def pcaProcess(Xtrain, Xdev = None, Xtest = None, n_components = 5):
    # 输入矩阵的一行为一条数据
    pca = PCA(n_components = n_components)
    pca.fit(X = Xtrain)
    dataset = []
    Xtrain = pca.transform(Xtrain)
    dataset.append(Xtrain)
    if Xdev != None:
        Xdev = pca.transform(Xdev)
        dataset.append(Xdev)
    if Xtest != None:
        Xtest = pca.transform(Xtest)
        dataset.append(Xtest)
    return dataset