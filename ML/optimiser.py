import random
import numpy as np
import scipy.linalg as la 
import pandas as pd
from tqdm import tqdm, tqdm_notebook

from autograd import grad
class optimiser:
    def __init__(self):
        pass
    
    def optimise(self):
        pass

class gradientDescent(optimiser):
    def __init__(self, learning_rate, max_iteration = 100, tolerance = 0.0001):
        
        self.lr = learning_rate
        self.history = []
        self.loss_function = None
        self.weights_init = None
        self.max_iteration = max_iteration
        self.tolerance = tolerance
    
    def _weights_init(self,weights):
        self.weights_init = weights

    def _loss_function(self, loss_function):
        self.loss_function = loss_function

    def _create_dloss(self):
        self.dloss_function = grad(self.loss_function)

    def optimise(self):
        self._create_dloss()
        current = self.loss_function(self.weights_init)

        t = tqdm(range(0,self.max_iteration))

        for i in t:
            self.history.append(current)
            self.weights_init -= self.lr*self.dloss_function(self.weights_init)

            if abs(self.loss_function(self.weights_init) - current) < self.tolerance:
                print('Tolerance reached')
                break
            current = self.loss_function(self.weights_init)
            if i%10 == 0:
                t.set_description('train loss: {:.6f}'.format(current))

    def get_weights(self):
        return self.weights_init
