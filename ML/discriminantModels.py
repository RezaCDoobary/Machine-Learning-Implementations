import cvxopt
from cvxopt import matrix, solvers 
import numpy as np
import kernels as kl

class DiscriminantClassifier:
    pass


class SVM(DiscriminantClassifier):
    def __init__(self, kernel, C = None):
        self.kernel = kernel
        self.C = C 
        
    def setUp(self,X,y):
        
        X_train = X.copy()
        y_train = y.copy()
        self.X = X_train
        self.y = y_train

        #self.X = np.insert(self.X, 0, values=1, axis=1)
        
        self.N,self.p = self.X.shape
        
    def _perform_convex_optimisation(self, verbose = False):
        # to do with convex opt part there are three main components:
        
        # step 1 : min xPx + Qx:
        K = self.kernel.covariance(self.X,self.X)
        T = self.y[:,np.newaxis]*self.y
        P = matrix(T*K,tc = 'd') 
        
        Q = -np.ones(self.N)
        Q = matrix(Q,tc = 'd')
        
        # step 2 : such that Gx < h:
        if self.C is None:
            G = -np.eye(self.N)
            G = matrix(G,tc='d')
            h = np.zeros(self.N)
            h = matrix(h, tc = 'd')
        else:
            G_top = np.eye(self.N)
            G_bottom = -np.eye(self.N)
            G = np.concatenate([G_top, G_bottom ],0)
            G = matrix(G,tc='d')
            h_top = np.ones(self.N)*self.C
            h_bottom = np.zeros(self.N)
            h = np.concatenate([h_top, h_bottom ],0)
            h = matrix(h, tc = 'd')
        
        # step 3 : and Ax = b:
        A = matrix(self.y.reshape(1,-1), tc = 'd')
        b = matrix(0, tc = 'd')
        
        if verbose:
            cvxopt.solvers.options['show_progress'] = True
        else:
            cvxopt.solvers.options['show_progress'] = False
            
        minima = cvxopt.solvers.qp(P, Q, G, h, A, b)
        return minima
    
    
    def _create_support_data(self, minima, eps = 10e-11):
        self.all_lagrangian_multipliers = np.array(minima['x'])
        self.support_indices = np.array(self.all_lagrangian_multipliers) > eps
        self.support_indices = self.support_indices.reshape(1,-1).ravel()


        self.support_multipliers = self.all_lagrangian_multipliers[self.support_indices].ravel()
        self.support_vectors = self.X[self.support_indices]
        self.support_output = self.y[self.support_indices]
        
        self.K_support = self.kernel.covariance(self.support_vectors, self.support_vectors)
        
        
    def get_support_data(self):
        
        return {'lagrangian_multipliers': self.support_multipliers,\
                'support_vectors':self.support_vectors, 'support_output':self.support_output,\
                'support_kernel_matrix':  self.K_support, 'intercept':self.intercept}
        
        
        
    def fit(self, X, y, verbose = False, eps = 10e-11):
        self.setUp(X,y)
        minima = self._perform_convex_optimisation(verbose)
        self._create_support_data(minima)
        
        amtm = np.multiply(self.support_multipliers, self.support_output)
        self.intercept = np.mean(self.support_output - np.matmul(amtm, self.K_support))
        
    
    def predict_value(self, X):
        res = 0
        for i in range(0,len(self.support_multipliers)):
            res += self.support_multipliers[i]*self.support_output[i]\
            *self.kernel.covariance(X,self.support_vectors[i])
        res+=self.intercept
        
        return res
    
    def predict(self, X):
        values = self.predict_value(X)
        return np.array([np.sign(val) for val in values])
        
    