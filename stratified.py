import os, pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns, missingno as msno, statsmodels.api as sm
import random
from itertools import chain

def stratifiedCV(df,amountofFolds,treatTrain,target):
    seed = 100
    np.random.seed(seed)
    df.index = range(0,len(df))
    indices00 = df.index[(df[treatTrain] == 0) & (df[target] == 0)].tolist()
    indices01 = df.index[(df[treatTrain] == 0) & (df[target] == 1)].tolist()
    indices10 = df.index[(df[treatTrain] == 1) & (df[target] == 0)].tolist()
    indices11 = df.index[(df[treatTrain] == 1) & (df[target] == 1)].tolist()
    proportion00 = df.loc[indices00,:].shape[0]/df.shape[0]
    proportion01 = df.loc[indices01,:].shape[0]/df.shape[0]
    proportion10 = df.loc[indices10,:].shape[0]/df.shape[0]
    proportion11 = df.loc[indices11,:].shape[0]/df.shape[0]
    size00 = len(indices00)//amountofFolds
    remainder00 = len(indices00)%amountofFolds
    size01 = len(indices01)//amountofFolds
    remainder01 = len(indices01)%amountofFolds
    size10 = len(indices10)//amountofFolds
    remainder10 = len(indices10)%amountofFolds
    size11 = len(indices11)//amountofFolds
    remainder11 = len(indices11)%amountofFolds
    np.random.seed(seed)
    listExtra00 = list(np.random.choice(range(0,amountofFolds),remainder00,replace = False, p = None))
    np.random.seed(seed)               
    listExtra01 = list(np.random.choice(range(0,amountofFolds),remainder01,replace = False, p = None))
    np.random.seed(seed)
    listExtra10 = list(np.random.choice(range(0,amountofFolds),remainder10,replace = False, p = None))
    np.random.seed(seed)
    listExtra11 = list(np.random.choice(range(0,amountofFolds),remainder11,replace = False, p = None))
    
    foldList = []
    
    for i in list(range(0,amountofFolds)):
        if i in listExtra00:
            sampSize00 = size00 + 1
        else:
            sampSize00 = size00
        
        np.random.seed(seed)
        sample00 = np.random.choice(indices00,sampSize00,replace = False, p = None)
        indices00 = list(set(indices00)-(set(sample00)))

        if i in listExtra01:
            sampSize01 = size01 + 1
        else:
            sampSize01 = size01
        
        np.random.seed(seed)
        sample01 = np.random.choice(indices01,sampSize01,replace = False, p = None)
        indices01 = list(set(indices01)-(set(sample01)))

        if i in listExtra10:
            sampSize10 = size10 + 1
        else:
            sampSize10 = size10

        np.random.seed(seed)
        sample10 = np.random.choice(indices10,sampSize10,replace = False, p = None)
        indices10 = list(set(indices10)-(set(sample10)))

        if i in listExtra11:
            sampSize11 = size11 + 1
        else:
            sampSize11 = size11

        np.random.seed(seed)
        sample11 = np.random.choice(indices11,sampSize11,replace = False, p = None)
        indices11 = list(set(indices11)-(set(sample11)))

        foldList.append(np.concatenate((sample00,sample01,sample10,sample11)))
    
    return foldList;

def create_folds(df,amountofFolds,col_name_treatment,col_name_target,col_name_predictors):
    random.seed(100)
    folds = stratifiedCV(df,amountofFolds,col_name_treatment,col_name_target)
    get = []
    for i in list(range(0,amountofFolds)):
        a = np.arange(len(folds))
        get.append(a[np.arange(len(a))!= i])  
    trainIndexes = []
    testIndexes = []
    random.seed(100)
    for i in list(range(0,amountofFolds)):
        trainIndexes.append(list(chain.from_iterable([folds[i] for i in get[i]])))
        testIndexes.append(folds[i])
        
    train_folds = []
    test_folds = []
    
    for i in list(range(0,amountofFolds)):
        random.seed(100)
        train_folds.append(pd.concat([df.loc[trainIndexes[i],col_name_predictors],df.loc[trainIndexes[i],col_name_treatment],df.loc[trainIndexes[i],col_name_target]],axis = 1))
        test_folds.append(pd.concat([df.loc[testIndexes[i],col_name_predictors],df.loc[testIndexes[i],col_name_treatment],df.loc[testIndexes[i],col_name_target]], axis = 1))
    return [train_folds,test_folds];