import cvxopt
from cvxopt import matrix, solvers 
import numpy as np
import kernels as kl
from numpy.linalg import inv

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
        
class LeastSquares(DiscriminantClassifier):
    def __init__(self):
        pass
        
    def setUp(self, X, y):
        self.classes = np.unique(y)
        self.K = len(self.classes)
        
        
        idx = 0
        self.N,self.p = X.shape
        new_col = np.array([1]*self.N)
        self.X = np.insert(X, idx, new_col, axis=1)
        
        self.T = np.zeros((self.K,len(y)))
        for i,c in enumerate(self.classes):
            self.T[i] = np.array(y== self.classes[i],int)
        
    def fit(self, X, y):
        self.setUp(X,y)
        
        first = inv(np.matmul(self.X.T,self.X))
        second = np.matmul(self.X.T, self.T.T)
        self.W = np.matmul(first, second).T
        
    def loss(self):
        diff = np.matmul(self.W,self.X.T) - T
        E = np.trace(np.matmul(diff, diff.T))/2
        return E
    
    
    def predict_discriminant_function(self, X):
        
        n_w, p_w  = self.W.shape
        n_x, p_x = X.shape
        if p_x != p_w:
            idx = 0
            new_col = np.array([1]*n_x)
            X = np.insert(X, idx, new_col, axis=1)
        return np.matmul(self.W,X.T).T
        
    def predict(self,X):

        disc = self.predict_discriminant_function(X)
        return np.argmax(disc,1)
    
class Perceptron(DiscriminantClassifier):
    """
    Binary case
    """
    def __init__(self):
        pass
    
    def setUp(self, X, y):
        self.classes = np.unique(y)
        self.K = len(self.classes)
        
        self.y = y
        idx = 0
        self.N,self.p = X.shape
        new_col = np.array([1]*self.N)
        self.X = np.insert(X, idx, new_col, axis=1)
        self.T = self.y - np.array(self.y == 0,int)
        
        self.W = self.X.mean(0)
        
    def _find_misclassifications(self):
        res = np.array(np.matmul(self.W,self.X.T) >= 0,int) - np.array(np.matmul(self.W,self.X.T) < 0,int)
        misclassified_indices = np.where(res != self.T) 
        return misclassified_indices
        
    def loss(self):
        misclassified_indices = self._find_misclassifications()
        X_part = np.matmul(self.W,self.X[misclassified_indices].T)
        T_part = self.T[misclassified_indices]

        return -np.sum(np.multiply(X_part, T_part))
    
    def fit(self, X, y, learning_rate = 0.001, max_iterations = 10000, print_every = 1000, tolerance = 10e-8,\
           verbose = True):
        self.setUp(X,y)
        current_W = self.W
        for _ in range(max_iterations):

            misclassified_indices = self._find_misclassifications()
            X_part = self.X[misclassified_indices]
            T_part = self.T[misclassified_indices]
            self.W = self.W + learning_rate*np.matmul(X_part.T, T_part)
            loss = self.loss()
            if _%print_every==0:
                if verbose:
                    print(loss)
            if abs(np.sum(self.W - current_W)) < tolerance:
                if verbose:
                    print('Tolerance reached at {} with inner product difference {}'.\
                          format(self.W, abs(np.sum(self.W - current_W))))
                break
            
            if loss == 0:
                if verbose:
                    print('Loss = 0 reached')
                break
                
    def predict_discriminant(self, X):
        n_w  = self.W.shape
        n_x, p_x = X.shape
        if p_x != n_w:
            idx = 0
            new_col = np.array([1]*n_x)
            X = np.insert(X, idx, new_col, axis=1)
        return np.matmul(self.W,X.T)    

        
    def predict(self,X):
        disc = self.predict_discriminant(X)
        return np.array(disc > 0,int)
    
    
class DiscriminantAnalysis(DiscriminantClassifier):
    def __init__(self, alpha = 1):
        # alpha = 1 : full QDA
        # alpha = 0 : LDA (pooled covariance matrices)
        self.alpha = alpha
        #alpha preset to one so that we have quadratic discriminant
            
    def setUp(self, X, y):
        self.y = y
        self.X = X
        self.classes = np.unique(self.y)
        
        if (np.sort(self.classes) != np.arange(len(self.classes))).all():
            raise ValueError('Please make class labels increasing integers')
        
        self.K = len(self.classes)
        self.n, self.p = self.X.shape
        
    def _compute_k_data(self):
        self.prior = {}
        self.means = {}
        self.covariances = {}
        self.Nk = {}

        for k in self.classes:
            idx = np.where(self.y == k)
            X_temp = self.X[idx]
            self.Nk[k] = X_temp.shape[0]
            self.prior[k] = self.Nk[k]/self.n
            self.means[k] = X_temp.mean(0)
            self.covariances[k] = np.cov(X_temp.T)
            
    def _compute_pooled_covariance(self):
        self.pooled_covariance = 0
        for k,v in self.covariances.items():
            self.pooled_covariance+=v*self.n*self.prior[k]
        self.pooled_covariance*=(1/(self.n-self.K))
        
    def _compute_reg_covariance(self):
        for k in self.covariances.keys():
            self.covariances[k] = self.alpha*self.covariances[k] +  (1-self.alpha)*self.pooled_covariance
        
    def fit(self, X, y):
        self.setUp(X,y)
        self._compute_k_data()
        self._compute_pooled_covariance()
        self._compute_reg_covariance()
        
    def _delta(self,k,x):
        first = 0.5*np.log(np.linalg.det(self.covariances[k]))
        diff = (x - self.means[k])
        second = 0.5*np.sum(diff*np.matmul(diff, np.linalg.inv(self.covariances[k])),1)
        third = np.log(self.prior[k])

        return -first - second + third
        
    def predict_discriminant(self, X):
        n,p = X.shape
        delta = np.zeros((n,self.K))
        
        for k in self.classes:
            delta[:,k] = self._delta(k,X)
        
        return delta
    
    def predict(self, X):
        return np.argmax(self.predict_discriminant(X),1)