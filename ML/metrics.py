import numpy as np
import pandas as pd 


class classification_metrics:
    def __init__(self):
        self.data = pd.DataFrame()
        
    def _check(self, y_observed, y_predicted):
        N1,p1 = y_observed.shape
        N2,p2 = y_predicted.shape
        assert(N1 == N2), 'The length of the observed and predicted target needs to \
        be the same.'
        assert(p1 == p2), 'The number of targets need to be the same.'
        
        self.N = N1
        self.p = p1

        
    def _get_classification_data(self, y_observed, y_predicted):
        for i in range(0,self.p):
            obs = y_observed[:,i]
            pred = y_predicted[:,i]

            TP = np.sum((obs == 1) & (pred == 1))
            FP = np.sum((obs == 0) & (pred == 1))
            FN = np.sum((obs == 1) & (pred == 0))
            TN = np.sum((obs == 0) & (pred == 0))

            self.data.at[i,'TP'] = TP
            self.data.at[i,'FP'] = FP
            self.data.at[i,'FN'] = FN
            self.data.at[i,'TN'] = TN
            
    def get_count(self, y_observed, y_predicted):
        self._check(y_observed, y_predicted)
        self._get_classification_data(y_observed, y_predicted)
    
    def precision(self, y_observed, y_predicted, merge = 'average'):
        self.get_count(y_observed, y_predicted)
        self.data['Precision'] = self.data['TP']/(self.data['TP'] + self.data['FP'])
        
        if merge == 'average':
            return np.mean(self.data['Precision'])
        
    def recall(self, y_observed, y_predicted, merge = 'average'):
        self.get_count(y_observed, y_predicted)
        self.data['Recall'] = self.data['TP']/(self.data['TP'] + self.data['FN'])
        
        if merge == 'average':
            return np.mean(self.data['Recall'])
        
    def specifity(self, y_observed, y_predicted, merge = 'average'):
        self.get_count(y_observed, y_predicted)
        self.data['Specifity'] = self.data['TN']/(self.data['TN'] + self.data['FP'])
        
        if merge == 'average':
            return np.mean(self.data['Specifity'])
        
    def F_score(self, y_observed, y_predicted, beta = 1):
        rec = self.recall(y_observed, y_predicted)
        prec = self.precision(y_observed, y_predicted)
        
        return (1+beta**2)*(prec*rec)/(((beta**2)*prec)+rec)
    
    def balanced_accuracy(self, y_observed, y_predicted):
        rec = self.recall(y_observed, y_predicted)
        spec = self.specifity(y_observed, y_predicted)
        
        return 0.5*(rec+spec)
        
    

def precision_recall_curve(y_observed, y_scores, n_threshold = 10):
    if n_threshold >= len(y_scores.ravel()):
        raise ValueError('n_threshold <= len(y_scores)')
    N = len(y_scores)
    threshold = np.array([y_scores[i] for i in range(0,N,int(N/n_threshold))]).ravel()
    
    pr = []
    for thr in threshold:
        y_predicted = np.array(y_scores > thr,int).reshape(-1,1)
        
        cm = classification_metrics()
        prec = cm.precision(y_observed, y_predicted)
        rec = cm.recall(y_observed, y_predicted)
        pr.append([prec,rec])
        
    
    pr = np.array(pr)
    arg = np.argsort(pr[:,0])
    precision_ = pr[:,0][arg]
    recall_ = pr[:,1][arg]
    
    return precision_, recall_, threshold

# ROC_AUC

def ROC(y_observed, y_scores, n_threshold = 10):
    if n_threshold >= len(y_scores.ravel()):
        raise ValueError('n_threshold <= len(y_scores)')
    N = len(y_scores)
    threshold = np.array([y_scores[i] for i in range(0,N,int(N/n_threshold))]).ravel()
    
    curve_points = []
    for thr in threshold:
        y_predicted = np.array(y_scores > thr,int).reshape(-1,1)
        
        cm = classification_metrics()
        TPR = cm.recall(y_observed, y_predicted)
        FPR = 1 - cm.specifity(y_observed, y_predicted)
        curve_points.append([TPR,FPR])
        
    curve_points = np.array(curve_points)
    arg = np.argsort(curve_points[:,0])
    TPR_ = curve_points[:,0][arg]
    FPR_ = curve_points[:,1][arg]

    return TPR_, FPR_, threshold


def MSE(y_obs, y_pred):
    y_pred = np.array(y_pred)
    y_obs = np.array(y_obs)
    
    
    if (len(y_pred)>1) and (len(y_obs) >1) and (len(y_pred) != len(y_obs)):
        
        raise ValueError('len(y_pred)!= len(y_obs), they must be equal')
    
    diff = y_pred - y_obs
    N = len(diff)
    
    SE = diff**2
    MSE = np.sum(SE)/N
    return MSE


def R2_score(y_obs, y_pred):
    """
    In statistics, the coefficient of determination, denoted R2 or r2 and pronounced "R squared", 
    is the proportion of the variance in the dependent variable that is predictable from the 
    independent variable(s).
    """
    SS_tot = MSE(np.array([y_obs.mean()]), y_obs)
    if not SS_tot == y_obs.var():
        raise ValueError('SS_tot should just be the variance, something went wrong!')
        
    SS_model = MSE(y_pred, y_obs)
    Rsquared = 1 - SS_model/SS_tot
    return Rsquared