import random
import numpy as np
import scipy.linalg as la 
import pandas as pd
from tqdm import tqdm, tqdm_notebook


from abc import ABCMeta
from abc import abstractmethod

class LinearModel:
    __metaclass__ = ABCMeta

    def __init__(self, method = None):
        self.X = 0
        self.y = 0
        self.n, self.p = None, None
        self.weights = None
        self.sample = None
        self.method = method
        
    def setUp(self,X,y):
        
        X_train = X.copy()
        y_train = y.copy()
        self.X = X_train
        self.y = y_train

        self.X = np.insert(self.X, 0, values=1, axis=1)
        
        self.n,self.p = self.X.shape
       
    @abstractmethod
    def weights_init(self):
        return
        
    @abstractmethod
    def loss(self):
        return

    @abstractmethod
    def fit(self, X, y):
        return
        
    def predict(self,X):
        alpha = self.weights[0]
        beta = self.weights[1:]
        res = alpha + np.dot(X,beta)
        
        return res

class LinearRegression(LinearModel):
    def __init__(self, method = None):
        LinearModel.__init__(self, method)

    def weights_init(self, X):
        self.weights = X.mean(0) + np.random.random()
    
    def loss(self, weights):
        diff = np.matmul(self.X, weights)-self.y
        return (1/(2*self.n))*np.dot(diff,diff)
    
    def fit(self, X, y):
        if self.method is None:
            self.setUp(X, y)
            self._OLS()
        elif self.method is not None:
            self.setUp(X, y)
            self.weights_init(self.X)
            self.method._weights_init(self.weights)
            self.method._loss_function(self.loss)
            self.method.optimise()
            self.weights = self.method.get_weights()
    
    
    def _OLS(self):
        tempX = self.X
        
        first = la.inv(np.dot(tempX.T,tempX))
        second = np.dot(tempX.T,self.y)
        
        weights = np.dot(first,second)
        self.weights = weights.ravel()

class Ridge(LinearModel):
    """
    data needs to be centered.
    """
    def __init__(self,lagriangian_constant, method = None):
        LinearModel.__init__(self, method)
        self.lagriangian_constant = lagriangian_constant

    def weights_init(self, X):
        self.weights = X.mean(0) + np.random.random()
    
    def loss(self, weights):
        diff = np.matmul(self.X, weights)-self.y
        return (1/(2*self.n))*np.dot(diff,diff) + self.lagriangian_constant*np.dot(weights,weights)
    
    def fit(self, X, y):
        if self.method is None:
            X_train = X.copy()
            y_train = y.copy()

            self.X = X_train
            self.y = y_train
            self.n,self.p = self.X.shape

            self._OLS()
        elif self.method is not None:
            X_train  = X.copy()
            y_train = y.copy()

            self.X = X_train
            self.y = y_train
            self.n,self.p = self.X.shape

            self.weights_init(self.X)
            self.method._weights_init(self.weights)
            self.method._loss_function(self.loss)
            self.method.optimise()
            self.weights = self.method.get_weights()
    
    
    def _OLS(self):
        tempX = self.X

        identity = np.eye(self.p,self.p)
        
        first = la.inv(np.dot(tempX.T,tempX) + identity*self.lagriangian_constant)
        second = np.dot(tempX.T,self.y)
        
        weights = np.dot(first,second)
        self.weights = weights.ravel()

    def predict(self,X):
        beta = self.weights
        res = np.dot(X,beta)
        
        return res


class Lasso(LinearModel):
    """
    data needs to be centered.
    """
    def __init__(self,lagriangian_constant, method = None):
        LinearModel.__init__(self, method)
        self.lagriangian_constant = lagriangian_constant

    def weights_init(self, X):
        self.weights = X.mean(0) + np.random.random()
    
    def loss(self, weights):
        diff = np.matmul(self.X, weights)-self.y

        
        second_part = self.lagriangian_constant*np.sum(np.abs(weights))

        return (1/(2*self.n))*np.dot(diff,diff) + second_part
    
    def fit(self, X, y):
        if self.method is None:
            self.setUp(X,y)
            self._OLS()
        elif self.method is not None:
            self.setUp(X,y)
            self.weights_init(self.X)
            self.method._weights_init(self.weights)
            self.method._loss_function(self.loss)
            self.method.optimise()
            self.weights = self.method.get_weights()
    
    
    def _OLS(self):
        raise TypeError('The Lasso Regression method cannot be solved analytically - please choose an optimiser.')
