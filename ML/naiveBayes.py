from scipy.stats import multivariate_normal
import numpy as np
from collections import Counter


class NaiveBayes:
    def __init__(self):
        pass
class GaussianNaiveBayes(NaiveBayes):
    """
    This is just the continous case for now, still need to do the categorical case.
    """
    def __init__(self):
        pass
    
    def setUp(self, X, y):
        self.classes = np.unique(y)
        self.K = len(self.classes)
        
        self.X = X
        self.y = y
        
        self.N, self.p = self.X.shape
    
    def _find_distribution_data(self):

        self.data = {}
        self.PCk = {}

        for p_i in np.arange(self.p):
            self.data[p_i] = {}
            for k in self.classes:
                self.data[p_i][k] = {'means':None,'var':None}
                
        for k in self.classes:
            idx = np.where(self.y == k)
            X_temp = self.X[idx]
            n_k,p_k = X_temp.shape
            means = X_temp.mean(0)

            for i,m in enumerate(means):
                self.data[i][k]['means'] = means[i]

            var = X_temp.var(0)

            for i,m in enumerate(var):
                self.data[i][k]['var'] = var[i]

            self.PCk[k] = n_k/self.N
    
    def fit(self, X, y):
        self.setUp(X, y)
        self._find_distribution_data()
        
    def predict_proba(self, X):
        n,p = X.shape
        result = np.zeros((n,self.K))
        for k in self.classes:
            prob = self.PCk[k]


            res = np.zeros((n,self.p))

            for feature in range(self.p):
                res[:,feature] = multivariate_normal.pdf(X[:,feature], self.data[feature][k]['means'],\
                                                         self.data[feature][k]['var'])
            result[:,k] = np.prod(res,1)
            result[:,k]*=prob
        
        return result
    
    def predict(self, X):
        result = self.predict_proba(X)
        prediction = np.argmax(result, 1)
        return prediction
    
    def get_gaussian_data_per_feature(self, i):
        return self.data[i]

class MultiNomialNaiveBayes(NaiveBayes):
    """
    This is just the categorical NB.
    """
    def __init__(self):
        pass
    
    def setUp(self, X, y):
        self.classes = np.unique(y)
        self.K = len(self.classes)
        
        self.X = X
        self.y = y
        
        self.N, self.p = self.X.shape
    
    def _find_distribution_data(self):
        
        self.classes = np.sort(np.unique(self.y.T[0]))

        self.probs = {}
        self.PCk = {}
        for k in self.classes:
            self.classkdata = self.X[np.where(self.y.T[0] == k)].T
            self.probs[k] = {i:Counter(x) for i,x in enumerate(self.classkdata)}
            nk = self.classkdata.shape[1]
            self.PCk[k] = nk
            for key in self.probs[k]:
                for counter in self.probs[k][key]:
                    self.probs[k][key][counter]/=nk
                    
    
    def fit(self, X, y):
        self.setUp(X, y)
        self._find_distribution_data()
        
    def predict_proba(self, X):
        N,p = X.shape
        res = []
        for k in self.classes:
            pro = np.prod(np.array([list(map(lambda x: self.probs[k][i][x], X[:,i])) for  i in range(0,self.p)]).T,1)
            pro*=self.PCk[k]
            res.append(pro)
        res=np.array(res)
        res = res.T
        predict_proba = res/res.sum(1).reshape(N,1)
        return predict_proba
    
    def predict(self, X):
        result = self.predict_proba(X)
        prediction = np.argmax(result, 1)
        return prediction
    
    def get_multinomial_data_per_class(self, k):
        return self.probs[k]