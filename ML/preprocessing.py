import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def split(dataframe, decimal, report = True, randomly = True):
    N = dataframe.shape[0]
    numberOfSamples = int(1/decimal)
    
    x = int(decimal*N)
    train_samples = {}
    
    vec = random.sample(range(0, N), N)
    for i in range(0,numberOfSamples):
        set1 = {}
        if randomly:
            df_val_test = dataframe.iloc[vec[x*i:x*(i+1)]]
        else:
            df_val_test = dataframe[i*x:(i+1)*x]
        set1['test'] = df_val_test
        idx_com = np.array(list(set(dataframe.index) - set(df_val_test.index)))
        df_val_train = dataframe.loc[idx_com]
        set1['train'] = df_val_train
    
        train_samples[i+1] = set1
    
    if report == True:
        print('train/test split with',decimal*100,'resampled as test data')
        print('There are',numberOfSamples,'sets')
    return train_samples


def fromCatergoricalToOneHot(categorical_columns, dataframes_array):
    cat = dict()
    df_arr = []
    for x in dataframes_array:
        df_arr.append(x[categorical_columns])
    
    temp = pd.concat(df_arr,axis = 0)
    for cols in categorical_columns:
        x = temp[cols].unique()
        x = np.array(list(map(lambda y: str(y), x)))
        cat[cols]  = x
    
    CATEGORICALONE = []
    new_dfs = []
    for df in dataframes_array:
        for c in categorical_columns:
            ar = []
            for subcols in cat[c]:
                tmp = c+'_'+str(subcols)
                ar.append(tmp)
                CATEGORICALONE.append(tmp)
            for i in range(0,len(ar)):
                df[ar[i]] = pd.DataFrame(pd.DataFrame(df[c],dtype = 'str') == cat[c][i],dtype = 'int')

            del df[c]
        
        new_dfs.append(df)
        
    return CATEGORICALONE, new_dfs

def normalise(numerical_columns, dataframe):
    dataframe = dataframe[numerical_columns]
    return (dataframe - dataframe.mean())/dataframe.std()
    
import scipy.linalg as la

def pcaMat(matrix,components,onWhat = 'covariance', doChecks = False):
    matrix = np.array(matrix)
    
    if onWhat == 'covariance':
        COV = np.cov(matrix.T)
        mat = COV
    elif onWhat == 'correlation':
        CORR = np.coefcorr(matrix.T)
        mat = CORR
    elif onWhat == 'matrix':
        mat = matrix
        
    u,d,v = la.svd(mat)
    reconMat = np.dot(u,np.dot(np.diag(d),v))
    
    if doChecks:
        assert(np.isclose(reconMat,mat).all()),'The SVD decomposition did not work!'
    
        for i in range(0,len(v)):
            assert (np.isclose(np.dot(mat,v[i])/v[i] - d[i],0).all()),'The eigenspace decomposition is not consistent'
        
    srt = d.argsort()
    
    d,v = d[srt],v[srt]
    
    
    if doChecks:
        for i in range(0,len(v)):
            assert (np.isclose(np.dot(mat,v[i])/v[i] - d[i],0).all()),'The eigenspace decomposition is not consistent'
    
    assert(components <= len(d)),'Number of components specified is larger than the number of eigenvectors\
    (available eigenspace directions)'
    
    newdata = np.dot(matrix - matrix.mean(axis = 0),v.T)
    
    matrix_object = mat #e.g. covariance, correlation
    components = v #eigenvectors of covariance
    weights = d #eigenvalues of covariance
    transformed_data  = newdata #

    return matrix_object, components, weights, transformed_data


def pcaDf(numerical_column, dataframe, components):
    NEWCOLUMNS = []

    X = dataframe[numerical_column].as_matrix()
    p = components
    matrix_object, components, weights, transformed_data = pcaMat(X,p)
    for i in range(0,p):
        x = 'col_'+str(i)
        dataframe[x] = X[:,i]
        NEWCOLUMNS.append(x)
    for cols in numerical_column:
        del dataframe[cols]
    
    return NEWCOLUMNS, dataframe




