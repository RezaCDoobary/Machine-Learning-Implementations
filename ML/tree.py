from collections import Counter
from scipy.stats import mode
import numpy as np

class TreeNode(object):
    def __init__(self, feature, value, left_indices, right_indices, measure_value, terminal_left_value = None,\
                terminal_right_value = None):
        self.feature = feature
        self.value = value
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.measure_value = measure_value
        self.terminal_left_value = terminal_left_value
        self.terminal_right_value = terminal_right_value
        
    def __str__(self):
        firstline = 'Feature :' + str(self.feature) 
        secondline = 'Value :' + str(self.value)
        condition = 'LHS if x > '+ str(self.value) + ' else RHS'
        measure = 'measure_value : ' + str(self.measure_value)
        result = firstline +'\n' + secondline +'\n'+condition +'\n'+measure
        return result

class DecisionTree:
    def __init__(self, measure = 'gini'):
        self.root = None
        if measure not in ['gini', 'entropy']:
            raise NotImplementedError("mesure must be in ['gini', 'entropy']")
        self.measure = measure
        if self.measure == 'gini':
            self.measure_function = self.gini_impurity
        elif self.measure == 'entropy':
            self.measure_function = self.entropy
                                    
    
    def setUp(self, X, y):
        self.classes = np.unique(y)
        self.K = len(self.classes)
        
        self.X = X
        self.y = y
        
        self.N, self.p = self.X.shape
        
    def gini_impurity(self, array):
        N = len(array)
        c = Counter(array)
        p = {k:v/N for k,v in c.items()}
        return 1-np.sum([v*v for v in p.values()])


    def entropy(self, array):
        N = len(array)
        c = Counter(array)
        p = {k:v/N for k,v in c.items()}
        return np.sum([-v*np.log2(v) for v in p.values()])
    
    def find_optimal_split(self,indices, measure):
        X = self.X[indices]
        Y = self.y[indices]

        N,p = X.shape
        current_index_set = np.arange(N)

        min_measure = np.inf
        best_value = None
        best_feature = None
        best_lhs = None
        best_rhs = None


        for i in range(0,p):
            res = (X[:,np.newaxis,i] > X[:,i])
            for j in range(0,N):
                lhs = np.where(res[:,j])[0]
                rhs = np.array(list(set(current_index_set) - set(lhs)))
                lhs_length, rhs_length = len(lhs), len(rhs)


                measure_index = (lhs_length/N)*measure(Y[lhs]) + (rhs_length/N)*measure(Y[rhs])

                if measure_index < min_measure:
                    min_measure = measure_index
                    best_value = X[j,i]
                    best_feature = i
                    best_lhs = lhs
                    best_rhs = rhs

        return min_measure, best_value, best_feature, indices[best_lhs], indices[best_rhs]
    
    def create_node(self, X, Y, indices):
        """
        >> X[root.left_indices][:,root.feature] > root.value
        >> array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True ...]
        """
        min_measure, best_value, best_feature, best_lhs, best_rhs = \
        self.find_optimal_split(indices, self.measure_function)
        return TreeNode(best_feature, best_value, best_lhs, best_rhs, min_measure)
    
    def build_dt(self,root, X, Y, depth, depth_limit):
        if depth == depth_limit:
            left_mode = mode(Y[root.left_indices])
            right_mode = mode(Y[root.right_indices])

            root.terminal_left_value = left_mode[0]
            root.terminal_right_value = right_mode[0]

        elif len(root.left_indices) == 1 and len(root.right_indices)!=1:
            root.terminal_left_value = np.array([Y[root.left_indices[0]]])
            root.right = self.create_node(X, Y, root.right_indices)
            root.right = self.build_dt(root.right, X, Y, depth, depth_limit)

        elif len(root.right_indices) == 1 and len(root.left_indices)!=1:
            root.terminal_right_value = np.array([Y[root.right_indices[0]]])
            root.left = self.create_node(X, Y, root.left_indices)
            root.left = self.build_dt(root.left, X, Y, depth, depth_limit)

        elif len(root.right_indices) == 1 and len(root.left_indices) ==1 :
            root.terminal_right_value = np.array([Y[root.right_indices[0]]])
            root.terminal_left_value = np.array([Y[root.left_indices[0]]])


        elif depth < depth_limit:
            depth +=1 
            root.left = self.create_node(X, Y, root.left_indices)
            root.right = self.create_node(X, Y, root.right_indices)


            root.left = self.build_dt(root.left, X, Y, depth, depth_limit)
            root.right = self.build_dt(root.right, X, Y, depth, depth_limit)
        return root
    
    def fit(self, X, y, depth_limit = 5):
        self.setUp(X,y)
        indices = np.arange(self.N)
        root = self.create_node(self.X, self.y, indices)
        depth = 0
        depth_limit = depth_limit
        root = self.build_dt(root, self.X, self.y, depth, depth_limit)
        self.root = root
    
    def classify_point(self, x_new):
        temp = self.root
        while True:
            if x_new[temp.feature] > temp.value:
                if temp.terminal_left_value is not None:
                    return temp.terminal_left_value
                    break
                else:
                    temp = temp.left
            else:
                if temp.terminal_right_value is not None:
                    return temp.terminal_right_value
                    break
                else:
                    temp = temp.right
                    
    def predict(self, X):
        return np.array([self.classify_point(x) for x in X]).ravel()