import os, pandas as pd
import numpy as np
from stratified import create_folds
from joblib import dump, load
from fun_synthetic_data import *

def load_data(name_dataset,path_datasets):
    if name_dataset == 'Hillstrom':
        df = pd.read_csv("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
        df["segment"] = df["segment"].astype("category")
        df["history_segment"] = df["history_segment"].astype("category")
        df["zip_code"] = df["zip_code"].astype("category")
        df["channel"] = df["channel"].astype("category")
        
        """
        Model elements
        """
        
        T = 'segment'
        Y = 'visit'
        nonpredictors = [T,Y]
        X = [variable for variable in list(df.columns) if variable not in nonpredictors]
        
        """
        Preprocessing
        > We ignored the men e-mail campaign
        > Dummies are created
        """
        df = df[df[T] != "Mens E-Mail"]#Mens E-Mail
        df[T] = np.where(df[T] == 'Womens E-Mail',1,0)
        df = df.reset_index(drop=True)  
        predictors_4_dummy = ["history_segment","zip_code","channel"]
        df_predictors_4_dummy = pd.get_dummies(df[predictors_4_dummy], prefix_sep='_', drop_first = True)
        X_dummies = list(df_predictors_4_dummy.columns)
        X = [variable for variable in X if variable not in predictors_4_dummy]
        df_predictors = pd.concat([df_predictors_4_dummy,df[X]],axis =1)
        X = list(df_predictors.columns)
        raw_df = pd.concat([df[Y],df[T],df_predictors[X]],axis=1)
        
        """
        Subset df
        """
        Amount_folds = 3
        folds = create_folds(raw_df,Amount_folds,T,Y,X)   
        df = folds[1][1]
        df = df.reset_index(drop=True)
        
        """
        5 folds stratified CV
        """
        Amount_folds = 5
        folds = create_folds(df,Amount_folds,T,Y,X)
        
    elif name_dataset == 'Bank':
        path_data = path_datasets + 'bank.csv'
        raw_df = pd.read_csv(path_data, sep = ';',low_memory = False, header = 0,decimal=",")
        
        """
        Model elements
        """
        T = 'comm_after6'
        Y = 'churn_6mnths'
        nonpredictors = [T,Y]
        X = [variable for variable in list(raw_df.columns) if variable not in nonpredictors] 
        
        """
        Subset df 
        """
        Amount_folds = 20
        # The outcome labels are modified, so that Y = 1 refers to non-churner
        raw_df[Y] = np.where(raw_df[Y] == 1,0,1)
        # CV
        folds = create_folds(raw_df,Amount_folds,T,Y,X)
        # Select a random subset of the total dataset
        raw_df = folds[1][0]
        raw_df = raw_df.reset_index(drop=True)
        
        """
        5 folds stratified CV
        """
        Amount_folds = 5
        folds = create_folds(raw_df,Amount_folds,T,Y,X)   
    
    elif name_dataset == 'Criteo':
        path_data_original = path_datasets + 'criteo/' + 'criteo-uplift-v2.csv'
        path_data_subset = path_datasets + 'criteo/' + 'folds.sav'
        # Original dataset
        # df = pd.read_csv(path_data, sep = ',',low_memory = False, header = 0)
        
        """
        I select a subset of the full Criteo dataset
        """
        folds = load(path_data_subset)
        raw_df = folds[1][0]
        raw_df = raw_df.reset_index(drop=True)
        """
        Treatment:
        """
        T = 'exposure'
        """
        Outcome:
        """
        Y = 'visit'
        raw_df[Y] = raw_df[Y].astype('category')
        """
        Predictors:
        """
        nonpredictors = [T,Y,'treatment','conversion']
        X = [variable for variable in list(raw_df.columns) if variable not in nonpredictors]
        
        """
        5 folds stratified CV
        """
        Amount_folds = 5
        #dump(folds,'folds.sav')
        # CV
        folds = create_folds(raw_df,Amount_folds,T,Y,X) 
        
    elif name_dataset == 'Synthetic':
        get_syn = make_uplift_classification(n_samples = 5000,
                                     treatment_name=['control', 'treatment1'],
                                     n_uplift_increase_dict = {'treatment1': 10},
                                     n_classification_features = 5,
                                     n_classification_informative = 3,
                                     positive_class_proportion=0.5,
                                     #n_uplift_increase_mix_informative_dict={'treatment1': 2},
                                     delta_uplift_increase_dict={'treatment1': 0.07},
                                     y_name = ['Y'],
                                     random_seed = 100)

        """
        Elements
        """
        raw_df = get_syn[0]
        T = "treatment_group_key"
        raw_df[T] = np.where(raw_df[T] == "treatment1",1,0)
        Y = "Y"
        X = get_syn[1]
        
        """
        5 folds stratified CV
        """
        Amount_folds = 5
        folds = create_folds(raw_df,Amount_folds,T,Y,X)   
            
    else:
        print('Dataset not found')
        
    """
    Descriptive
    """
    desc_table = pd.crosstab(index = raw_df[Y], columns = raw_df[T], margins = False)
    
    descriptive = {'table':desc_table,
                   'Total number of observations': raw_df.shape[0],
                   'Total number of variables':len(X),
                   'Number of control group observations': raw_df[T].value_counts()[0],
                   'Number of treatment group observations':raw_df[T].value_counts()[1],
                   'Visit rate treatment':desc_table[1][1]/(desc_table[1][1] + desc_table[1][0]),
                   'Visit rate control':desc_table[0][1]/(desc_table[0][1] + desc_table[0][0]),
                   'Overall effect':desc_table[1][1]/(desc_table[1][1] + desc_table[1][0])-desc_table[0][1]/(desc_table[0][1] +desc_table[0][0])
                  }
        
    return {'name':name_dataset,'folds': folds,'T':T,'X':X,'Y':Y,'descriptive':descriptive};
        
        
    