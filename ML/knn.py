from scipy.stats import mode
import numpy as np

class KNN:
    def __init__(self, neighbours, difference_metric):
        self.neighbours = neighbours
        if difference_metric == 'euclidean':
            self.difference_metric = self._euclidean
        elif difference_metric == 'manhattan':
            self.difference_matric = self._manhattan
        else:
            raise NotImplementedError("The difference metric must be in ['euclidean','manhattan']")
        
    def _euclidean(self,x,y):
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            return np.sqrt(((x - y)**2).sum(1))
        else:
            raise TypeError('Wrong types, should be numpy arrays')

    def _manhattan(self,x,y):
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            return np.array([abs(xx - yy) for xx,yy in zip(x,y)]).sum()
        else:
            raise TypeError('Wrong types, should be numpy arrays')
        
    def setUp(self, X, y):
        self.classes = np.unique(y)
        self.K = len(self.classes)
        
        self.X = X
        self.y = y
        
        self.N, self.p = self.X.shape
        
    def fit(self, X, y):
        self.setUp(X,y)
        # There is nothing to really fit here
        

class KNNClassifer(KNN):
    def __init__(self,neighbours, difference_metric):
        super().__init__(neighbours, difference_metric)
    
    def predict(self, X):      
        res = []
        for points in X:
            distances = self.difference_metric(np.array(points),self.X)
            res.append(np.argsort(distances)[:self.neighbours])
        res = np.array(res)
        new_labels = self.y[res]
        return np.array([mode(np.array(nl))[0][0] for nl in new_labels])
    
class KNNRegression(KNN):
    def __init__(self, neighbours, difference_metric):
        super().__init__(neighbours, difference_metric)
        
    def predict(self, X):
        res = []
        for points in X:
            distances = self.difference_metric(np.array(points),self.X)
            res.append(np.argsort(distances)[:self.neighbours])
        res = np.array(res)
        results = self.y[res]
        return np.array([nl.mean() for nl in results])