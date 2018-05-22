import random
import numpy as np
import scipy.linalg as la 
import pandas as pd

from abc import ABCMeta
from abc import abstractmethod

class mixtureModel:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.X = 0
        self.n, self.p = 0,0
        self.k = 0
        self.centroids = 0
       
    def setUp(self,X,numberOfClusters,centroidChoiceRandom = True):
        X_train = X.copy()
        self.X = X_train
        self.n,self.p = self.X.shape
        self.k = numberOfClusters
        self.centroid_init(centroidChoiceRandom)
        
    @abstractmethod
    def centroid_init(self):
        return
    
class kmeans(mixtureModel):
    def __init__(self):
        mixtureModel.__init__(self)
        self.dataSplit = dict()
        
    def centroid_init(self,centroidChoiceRandom = True):
        if centroidChoiceRandom.all() == True:
            self.centroids = np.random.uniform(0,1,(self.p,self.k))
        else:
            self.centroids = centroidChoiceRandom
        for i in range(0,self.k):
            temp = dict()
            temp['centroid'] = self.centroids[i]
            temp['datapoints'] = []
            self.dataSplit[i] = temp
    

    def _magnitudeDifference(self,A,B):
        Z = np.array(A) - np.array(B)
        s = 0
        for z in Z:
            s+=z**2
        return s
    
    def dataInCentroid(self,datapoint):
        smallest = 0
        idx = None
        for i in range(0,self.k):
            diff = self._magnitudeDifference(self.dataSplit[i]['centroid'],datapoint)
            if i == 0:
                smallest = diff
                idx = i
            elif diff == min(smallest,diff):
                idx = i
        self.dataSplit[idx]['datapoints'].append(datapoint)
        
    def dataDivisor(self):
        for x in self.X:
            self.dataInCentroid(x)
        
        for i in range(0,self.k):
            self.centroids[i] = self.dataSplit[i]['centroid']
        
    def updateMeans(self):
        for i in range(0,self.k):
            self.dataSplit[i]['centroid'] = np.array(self.dataSplit[i]['datapoints']).mean(axis = 0)
            self.dataSplit[i]['datapoints'] = []
            
    def _diffChecker(self,tol = 0.00000001):
        result = True
        for i in range(0,self.k):
            if (self.centroids[i] - self.dataSplit[i]['centroid']).all() > tol :
                result = False
        return result
            
    def fit(self, tol = 0.00000001):
        self.dataDivisor()
        self.updateMeans()
        while self._diffChecker():
            self.dataDivisor()
            self.updateMeans()
        self.dataDivisor()
        
class GaussianMixtureModel(mixtureModel):
    pass