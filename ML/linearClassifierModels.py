import random
import autograd.numpy as np
import scipy.linalg as la 
import pandas as pd
from tqdm import tqdm, tqdm_notebook


from abc import ABCMeta
from abc import abstractmethod

class ClassifierModel:
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
        
        self.k = len(np.unique(y_train))
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


class LogisticRegression(ClassifierModel):
    """
    data needs to be centered.
    """
    def __init__(self,lagriangian_constant, method):
        ClassifierModel.__init__(self, method)
        self.lagriangian_constant = lagriangian_constant
        self.method = method
        self.binary = False

    def weights_init(self, X):
        self.weights = X.mean(0) + np.random.random()
        
    def weights_multi_init(self, X):
        self.weights = np.random.multivariate_normal(X.mean(0),np.eye(len(X.mean(0))),(self.k))
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def _softmax(self, array):
        num = np.exp(array)
        den = sum(num)
        return num/den
    
    def _create_one_hot(self, Y, num_categories):
        Y = Y.reshape(1,len(Y)).ravel()

        def create_arr(number, num_cat):
            res = np.zeros(num_categories)
            res[number] = 1
            return res

        return np.array(list(map(lambda x: create_arr(x, num_categories), Y)))
    
    def loss_binary(self, weights):
        probabilities = self._sigmoid(np.matmul(self.X, weights)).reshape(-1,1)

        loss_i = np.multiply(self.y,np.log(probabilities)) + np.multiply((1-self.y),np.log(1-probabilities))
        loss = -np.mean(loss_i) + self.lagriangian_constant*np.dot(weights,weights)
        return loss
    
    
    
    def loss_multiple(self, weights):
        #print(weights.shape)
        #print(self.X.shape)
        log_softmax = np.log(np.array(list(map(lambda x: self._softmax(x),np.matmul(self.X, weights.T)))))
        ys_one_hot = self._create_one_hot(self.y,self.k)
        loss = -np.mean(np.multiply(ys_one_hot, log_softmax))
        loss += self.lagriangian_constant*np.dot(weights.T,weights).sum()
        return loss
    
    
    def fit(self, X, y):
        self.setUp(X,y)
        
        if len(np.unique(self.y)) == 2:
            self.binary = True
            self.weights_init(self.X)
            self.method._weights_init(self.weights)
            self.method._loss_function(self.loss_binary)
            self.method.optimise()
            self.weights = self.method.get_weights()
        else:
            self.weights_multi_init(self.X)
            self.method._weights_init(self.weights)
            self.method._loss_function(self.loss_multiple)
            self.method.optimise()
            self.weights = self.method.get_weights()
            
        
    def predict_proba(self, X):
        if self.binary:
            probabilities = self._sigmoid(np.matmul(self.X, self.weights)).reshape(-1,1)
            return probabilities
        else:
            probabilities = np.array(list(map(lambda x: self._softmax(x),np.matmul(self.X, self.weights.T))))
            return probabilities
            
    
    def predict(self, X):
        if self.binary:
            probs = self.predict_proba(X)
            return np.array(self.predict_proba(X)>0.5,int)
        else:
            probs = self.predict_proba(X)
            return np.argmax(self.predict_proba(X),1)