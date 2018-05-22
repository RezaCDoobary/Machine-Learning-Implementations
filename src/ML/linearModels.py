import random
import numpy as np
import scipy.linalg as la 
from ML.optimiser import *
import pandas as pd

from abc import ABCMeta
from abc import abstractmethod

class LinearModel:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.X = 0
        self.y = 0
        self.n, self.p = 0,0
        self.beta = 0
        self.sample = None
        self.hack = 0
        
    def setUp(self,X,y):
        
        X_train = X.copy()
        y_train = y.copy()
        self.X = X_train
        self.y = y_train
        
        
        eps = 0.03
        
        self.hack = eps
        
        if isinstance(self.X,pd.DataFrame):
            self.X.insert(0,'ones',self.hack)
        else:
            dd = np.array(np.random.normal(self.hack,0,len(self.X)))
            self.X = np.hstack((dd,self.X))
        
        self.n,self.p = self.X.shape
        self.beta_init()
    
    def setBeta(self,beta):
        self.beta = beta
        
    def getBeta(self):
        return self.beta
    
    @abstractmethod
    def beta_init(self):
        return
        
    @abstractmethod
    def cost(self, randomSelection = None):
        return
    
    @abstractmethod
    def dcost(self, randomSelection = None):
        return
    
    def _fixBeta(self):
        self.beta[0] = self.beta[0]*self.hack
        
    def predict(self,X):
        alpha = self.beta[0]
        beta = self.beta[1:]
        res = alpha + np.dot(X,beta)
        return res

class LinearRegression(LinearModel):
    def __init__(self):
        LinearModel.__init__(self)

    def beta_init(self):
        self.beta = np.random.uniform(0.5,0.5,self.p)
    
    def diff(self, randomSelection = None):
        if randomSelection is not None:
            self.sample = random.sample(range(self.n), randomSelection)
            x = self.X.iloc[self.sample]
            y = self.y.iloc[self.sample]
        else:
            x = self.X
            y = self.y
        
        yhat = np.dot(x,self.beta)
        yhat = yhat.reshape(y.shape)
        diff = y - yhat
        if isinstance(diff,pd.DataFrame)!= True:
            diff = (y - yhat).ravel()
        
        return diff
        
    def cost(self, randomSelection = None):
        diff = self.diff(randomSelection)
        cost = (1/(2*self.n))*np.dot(diff.T,diff)
        return cost
    
    def dcost(self, randomSelection = None):
        diff = self.diff(randomSelection)
        if randomSelection is not None:
            x = self.X.iloc[self.sample]
        else:
            x = self.X
        dcost = -(1/self.n)*np.dot(diff.T,x).ravel()
        return dcost
    
    def fit(self,optimiser,max_iter = 100000,tol = 10e-8):
        if optimiser  == 'OLS':
            self._OLS()
        else:
            optimiser.compute(max_iter, tol)
        
        self._fixBeta()
        return self.beta
    
    def _OLS(self):
        tempX = self.X
        
        first = la.inv(np.dot(tempX.T,tempX))
        second = np.dot(tempX.T,self.y)
        
        beta = np.dot(first,second)
        self.beta = beta.ravel()

class TikhonovModel(LinearModel):
    def __init__(self):
        LinearModel.__init__(self)

    def setUp(self,X,y,gamma_matrix):

        X_train = X.copy()
        y_train = y.copy()
        
        self.X = X_train
        self.y = y_train
        
        
        self.gamma_matrix = gamma_matrix
        eps = 0.03
        
        self.hack = eps
        
        if isinstance(self.X,pd.DataFrame):
            self.X.insert(0,'ones',eps)
        else:
            le = self.X.shape[0]
            dd = np.array([0.03]*le)
            dd = dd.reshape((len(dd),1))
            self.X = np.hstack((dd,self.X))
        
        self.n,self.p = self.X.shape
        self.beta_init()
        
    def beta_init(self):
        self.beta = np.random.uniform(0.5,0.5,self.p)
    
    def diff(self, randomSelection = None):
        if randomSelection is not None:
            self.sample = random.sample(range(self.n), randomSelection)
            x = self.X.iloc[self.sample]
            y = self.y.iloc[self.sample]
        else:
            x = self.X
            y = self.y
        
        yhat = np.dot(x,self.beta)
        yhat = yhat.reshape(y.shape)
        diff = y - yhat
        if isinstance(diff,pd.DataFrame)!= True:
            diff = (y - yhat).ravel()
        return diff
        
    def cost(self, randomSelection = None):
        diff = self.diff(randomSelection)
        gamma_part = np.dot(self.gamma_matrix,self.beta)
        gamma = np.dot(gamma_part.T,gamma_part)
        cost = np.dot(diff.T,diff) + gamma
        return cost
    
    def dcost(self, randomSelection = None):
        diff = self.diff(randomSelection)
        if randomSelection is not None:
            x = self.X.iloc[self.sample]
        else:
            x = self.X
        dcost = -2*np.dot(diff.T,x) + 2*np.dot(np.dot(self.gamma_matrix.T,self.gamma_matrix),self.beta).ravel()
        return dcost
    
    def fit(self,optimiser,max_iter = 100000,tol = 10e-8):
        if optimiser  == 'OLS':
            self._OLS()
        else:
            optimiser.compute(max_iter, tol)
        
        self._fixBeta()
        return self.beta
    
    def _OLS(self):
        tempX = self.X
        tempG = self.gamma_matrix
        
        first_inside = np.dot(tempX.T,tempX) + np.dot(tempG.T,tempG)
        first = la.inv(first_inside)
        second = np.dot(tempX.T,self.y)
        
        beta = np.dot(first,second)
        self.beta = beta.ravel()        
    
class RidgeRegression(TikhonovModel):
    def __init__(self):
        TikhonovModel.__init__(self)
        
    def setUp(self,X_train,y_train, labda):
        g_mat = labda*  np.eye(X_train.shape[1]+1)
        g_mat[0,0] = 0
        self.gamma_matrix = g_mat
        super(RidgeRegression,self).setUp(X_train,y_train,self.gamma_matrix)
        
        