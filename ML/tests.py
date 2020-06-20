from scipy.stats import chi2
from scipy.stats import norm,t
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstwobign


class freqeuency_based_tests:
    def __init__(self):
        self.data = None
        self.tables = {}
        self.X = None
        self.y = None
        
    def _data_prep(self, X, y):
        self.X = X
        N,p = self.X.shape
        self.y = y
        temp = pd.DataFrame(np.concatenate([X,y.reshape(-1,1)],1))
        temp.columns = ['f'+str(i) for i in range(0,p)] + ['T']
        self.data = temp
        del temp
        
    def _compute_table_data(self, feature_label, target_label):
        frequency_table = pd.crosstab(index = self.data[feature_label], columns = self.data[target_label])
        total = frequency_table.sum(0).sum()
        expected = np.outer(frequency_table.sum(1),frequency_table.sum(0)/total)
        
        self.tables[feature_label] = [frequency_table, expected]



class chi2_test(freqeuency_based_tests):
    """
    Helps us understand any associations between two multinomial datasets, i.e categorial to categorical.
    
    The null hypothesis of the test is that the two multinomial distributions are unrealted.
    """
    def __init__(self):
        super(chi2_test, self).__init__()
        self.result = {}

    def _compute_chi2_stats_and_pval(self, frequency_table, expected, alpha):
        dof = (len(frequency_table.columns)-1)*(len(frequency_table.index)-1)
        stat = np.sum(((frequency_table.values - expected)**2)/expected)
        p_val = 1 - chi2.cdf(stat, dof, loc=0, scale=1)
        if p_val > alpha:
            reject = False
        else:
            reject = True
            
        return [stat, dof, p_val, reject]
    
    def compute(self, X, y, alpha = 0.05):
        self._data_prep(X,y)
        features = self.data.columns[:-1]
        target = self.data.columns[-1]
        
        for feat in features:
            self._compute_table_data(feat, target)
            f_table, expected = self.tables[feat]
            self.result[feat] = self._compute_chi2_stats_and_pval(f_table, expected, alpha)
        report = pd.DataFrame(self.result, index = ['chi_2 statistic','dof','p-value','Reject H_0?'])
        return report

class mutual_information(freqeuency_based_tests):
    """
    Mutual information : https://en.wikipedia.org/wiki/Mutual_information
    """
    def __init__(self):
        super(mutual_information, self).__init__()
        self.result = {}
        
    def _compute_mi(self, pxy, pxpy):
        return np.sum(pxy*np.log(pxy/pxpy))
    
    def compute(self, X, y):
        self._data_prep(X,y)
        features = self.data.columns[:-1]
        target = self.data.columns[-1]
        
        for feat in features:
            self._compute_table_data(feat, target)
            f_table, expected = self.tables[feat]
            
            pxy = (f_table/len(self.data)).values
            pxpy = expected/len(self.data)
            
            self.result[feat] = self._compute_mi(pxy,pxpy)
        report = pd.DataFrame(self.result, index = ['MI'])
        return report
    
    
def kendall_correlation(x,y):
    """
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    
    TO DO : at statistical significance measure, i.e. p-value. Use Mann-Kendall test.
    """
    N = len(x)
    numerator = 0
    non_tied_count = 0
    ties_in_x = 0 # ties in x but not in y
    ties_in_y = 0 # ties in y but not in x
    for i in range(0,N):
        for j in range(i+1,N):
            if x[i] - x[j] == 0 and y[i] - y[j] !=0:
                ties_in_x+=1
            if y[i] - y[j] == 0 and x[i] - x[j] != 0:
                ties_in_y+=1
            term = np.sign(x[i] - x[j])*np.sign(y[i] - y[j])
            numerator += term
            if term < 0 or term > 0:
                non_tied_count+=1

    return numerator/np.sqrt((non_tied_count + ties_in_x)*(non_tied_count+ties_in_y))

def pearson_correlation(X):
    size = len(X)
    N = len(X[0])
    find_var = lambda x,N: np.sum((x - x.mean())**2)/N
    find_corr = lambda x,y,N :np.matmul(x - x.mean(0),y - y.mean(0))/np.sqrt(find_var(x,N)*find_var(y,N))/N
    
    result = np.eye(size)


    for i in range(0,size):
        for j in range(i,size):
            result[i][j] = find_corr(X[i],X[j],N)
            result[j][i] = result[i][j]
            
    return result

class KolmogorobSmirnov:
    def __init__(self):
        pass
    
    
    def one_sample(self,X, distribution:str, alpha = 0.05):
        min_ = min(X) - 1
        max_ = max(X) + 1        
        points = np.linspace(min_,max_,10*len(X))
        f_cds = lambda X, xx : (X < xx).mean()
        
        if distribution == 'norm':
            F = lambda xx : norm.cdf(xx, loc = 0, scale = 1)
        else:
            raise NotImplementedError(distribution+' not implemented.')
            
        Dn = np.max([np.abs(f_cds(X,x) - F(x)) for x in points])
        
        KS_stat = np.sqrt(len(X))*Dn
        Kalpha = kstwobign.ppf(1-alpha)
        
        reject = KS_stat > Kalpha
            
        return KS_stat, reject
    
    def two_sample(self, X1, X2, alpha = 0.05):

        m,n = len(X1),len(X2)

        min_ = min(min(X1),min(X2)) - 1
        max_ = max(max(X1),max(X2)) + 1

        points = np.linspace(min_,max_,10*max(m,n))
        f_cds = lambda X, xx : (X < xx).mean()

        Dn = np.max([np.abs(f_cds(X1,x) - f_cds(X2,x)) for x in points])

        c = np.sqrt(-np.log(alpha/2)*0.5)
        postfactor = np.sqrt((n+m)/(n*m))

        reject = Dn > c*postfactor
        
        return Dn, reject

class TTest:
    def __init__(self):
        pass
    
    def one_sample(self, X, proposed_mean):
        X = np.array(X)
        x_bar = X.mean()
        n = len(X)
        
        sample_var = np.var(X, ddof = 1)
        denominator = np.sqrt(sample_var/n)
        numerator = x_bar - proposed_mean
        
        zscore_statistic = numerator/denominator
        
        p_value = stats.t.sf(np.abs(zscore_statistic),df = n-1)*2
        
        return zscore_statistic, p_value
    
    
    def _two_sample_same_var(self, X1, X2):
        n1,n2 = len(X1), len(X2)
        sp = np.sqrt(((n1 - 1)*np.var(X1, ddof = 1) + (n2 - 1)*np.var(X2, ddof= 1))/(n1+n2-2))
        den = sp*np.sqrt(1/n1 + 1/n2)
        num = np.mean(X1) - np.mean(X2)

        zscore_statistic = num/den
        p_value = stats.t.sf(np.abs(zscore_statistic),df = n1+n2-2)*2
        
        return zscore_statistic, p_value
    
    def _two_sample_diff_var(self, X1, X2):
        n1,n2 = len(X1), len(X2)
        num = np.mean(X1) - np.mean(X2)
        den = np.sqrt(np.var(X1, ddof = 1)/n1 + np.var(X2, ddof =1)/n2)

        zscore_statistic = num/den
        
        s1 ,s2 = np.var(X1, ddof = 1), np.var(X2, ddof = 1)

        df = ((s1/n1) + (s2/n2))**2/(((s1/n1)**2)/(n1-1) + ((s2/n2)**2)/(n2-1))


        p_value = stats.t.sf(np.abs(zscore_statistic),df = df)*2
        
        return zscore_statistic, p_value

    
    def two_sample(self, X1, X2, same_var = False):
        if same_var:
            return self._two_sample_same_var(X1, X2)
        else:
            return self._two_sample_diff_var(X1, X2)
        