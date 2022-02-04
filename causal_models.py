import os, pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, missingno as msno, statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from econml.dml import ForestDMLCateEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from econml.dml import NonParamDMLCateEstimator
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load

def train_causal(path_models, name_dataset,folds,T,Y,X,algorithm,load_model):
    """
    3 causal models: CF,T_learner,S_learner
    """
    random_state = 100
    model_names = ['CF','T_learner','S_learner']
    
    if load_model == False and algorithm == 'LR':
        algorithm_1 = LogisticRegression(solver='saga',max_iter=3000, random_state = random_state)
        algorithm_2 = LogisticRegression(solver='saga',max_iter=3000, random_state = random_state)
        algorithm_3 = LogisticRegression(solver='saga',max_iter=3000, random_state = random_state)
        algorithm_4 = LogisticRegression(solver='saga',max_iter=3000, random_state = random_state)
        
        effect_estimates = []
        for model in model_names:
            effect_estimates.append(fit_causal(folds,T,Y,X,model,algorithm_1,algorithm_2,algorithm_3,algorithm_4))
            
        """
        Save estimates:
        """
        path_model = path_models + f'effect_estimates_{algorithm}.sav'
        #dump(effect_estimates,path_model)
        
    elif load_model == False and algorithm == 'XGB':
        #xgb.XGBClassifier(objective = "binary:logistic",n_estimators = 100, random_state = random_state)
        algorithm_1 = xgb.XGBClassifier(objective = "binary:logistic",n_estimators = 500, random_state = random_state)
        algorithm_2 = xgb.XGBClassifier(objective = "binary:logistic",n_estimators = 500, random_state = random_state)
        algorithm_3 = xgb.XGBClassifier(objective = "binary:logistic",n_estimators = 500, random_state = random_state)
        algorithm_4 = xgb.XGBClassifier(objective = "binary:logistic",n_estimators = 500, random_state = random_state)
        
        effect_estimates = []
        for model in model_names:
            effect_estimates.append(fit_causal(folds,T,Y,X,model,algorithm_1,algorithm_2,algorithm_3,algorithm_4))
            
        """
        Save estimates:
        """
        path_model = path_models + f'effect_estimates_{algorithm}.sav'
        #dump(effect_estimates,path_model)
            
    elif load_model == True and algorithm == 'LR':
        path_model = path_models + f'{name_dataset}/' + f'effect_estimates_{algorithm}.sav'
        effect_estimates = load(path_model)
        
    elif load_model == True and algorithm == 'XGB':
        path_model = path_models + f'{name_dataset}/' + f'effect_estimates_{algorithm}.sav'
        effect_estimates = load(path_model)
        
    return effect_estimates;

def fit_causal(folds,T,Y,X,model,algorithm_1,algorithm_2,algorithm_3,algorithm_4):
    random_state = 100
    amountofFolds = len(folds[0])
    fitted_model = []
    
    if model == 'CF':
        for i in list(range(0,amountofFolds)):
            def CF(Y_train,T_train,X_train,df_test):
                np.random.seed(random_state)
                CF_model= ForestDMLCateEstimator(model_y = algorithm_1,
                                                 model_t = algorithm_2,
                                                 discrete_treatment=True,
                                                 n_crossfit_splits = 10,
                                                 n_estimators= 100,
                                                 n_jobs = -1,
                                                 random_state = random_state)
                np.random.seed(random_state)
                CF_model.fit(Y_train.to_numpy(), T_train.to_numpy(),X_train.to_numpy(), inference = 'blb')
                s = CF_model.effect(df_test[X].to_numpy())
                testIndices = df_test.index.values
                s = pd.DataFrame(s,testIndices, columns = {"s"})
                # Estimate P(Y(1)|X)
                df_train = pd.concat([Y_train,T_train,X_train], axis =1)
                df_treated = df_train[df_train[T] == 1]
                np.random.seed(random_state)
                model_treated = xgb.XGBClassifier(objective = "binary:logistic",n_estimators = 100, random_state = random_state)
                model_treated.fit(df_treated[X],df_treated[Y])
                testIndices = df_test.index.values
                prob_treatment = model_treated.predict_proba(df_test[X])[:,1]
                df_prob_treatment = pd.DataFrame(prob_treatment,testIndices, columns = {"Prob_treat"})
                # Summary results
                predictions = pd.concat([df_test,df_prob_treatment,s],axis = 1)
                return predictions;
            fitted_model.append(CF(folds[0][i][Y],folds[0][i][T],folds[0][i][X],folds[1][i]))
            
    elif model == 'NonParamDMLCateEstimator':
        for i in list(range(0,amountofFolds)):
            def NPDML(Y_train,T_train,X_train,df_test):
                np.random.seed(random_state)
                NPDML_model= NonParamDMLCateEstimator(model_y = algorithm_1,
                                                 model_t = algorithm_2,
                                                 model_final = algorithm_3,
                                                 discrete_treatment=True,
                                                 n_splits = 10,
                                                 random_state = random_state)
                np.random.seed(random_state)
                NPDML_model.fit(Y_train.to_numpy(), T_train.to_numpy(),X_train.to_numpy(), inference = 'bootstrap')
                s = NPDML_model.effect(df_test[X].to_numpy())
                testIndices = df_test.index.values
                s = pd.DataFrame(s,testIndices, columns = {"s"})
                # Estimate P(Y(1)|X)
                df_train = pd.concat([Y_train,T_train,X_train], axis =1)
                df_treated = df_train[df_train[T] == 1]
                np.random.seed(random_state)
                model_treated = xgb.XGBClassifier(objective = "binary:logistic",n_estimators = 100, random_state = random_state)
                model_treated.fit(df_treated[X],df_treated[Y])
                testIndices = df_test.index.values
                prob_treatment = model_treated.predict_proba(df_test[X])[:,1]
                df_prob_treatment = pd.DataFrame(prob_treatment,testIndices, columns = {"Prob_treat"})
                # Summary results
                predictions = pd.concat([df_test,df_prob_treatment,s],axis = 1)
                return predictions;
            fitted_model.append(NPDML(folds[0][i][Y],folds[0][i][T],folds[0][i][X],folds[1][i]))
    
    elif model == 'T_learner':
        for i in list(range(0,amountofFolds)):
            def T_learner(df_train,df_test):
                np.random.seed(random_state)
                # Split treated and control instances
                df_treated = df_train[df_train[T] == 1]
                df_control = df_train[df_train[T] == 0]
                # Fit a model for each subset
                model_treated = algorithm_1
                model_treated.fit(df_treated[X],df_treated[Y])
                model_control = algorithm_2
                model_control.fit(df_control[X],df_control[Y])
                # Predict
                testIndices = df_test.index.values
                prob_treatment = model_treated.predict_proba(df_test[X])[:,1]
                prob_control = model_control.predict_proba(df_test[X])[:,1]
                df_prob_treatment = pd.DataFrame(prob_treatment,testIndices, columns = {"Prob_treat"})
                df_prob_control = pd.DataFrame(prob_control,testIndices, columns = {"Prob_ctrl"})
                predictions = pd.concat([df_test,df_prob_treatment,df_prob_control],axis = 1)
                predictions["s"] = predictions["Prob_treat"] - predictions["Prob_ctrl"]
                return predictions;
            fitted_model.append(T_learner(folds[0][i],folds[1][i]))
    
    elif model == 'S_learner':
        for i in list(range(0,amountofFolds)):
            def S_learner(df_train,df_test):
                np.random.seed(random_state)
                # List of predictors (predictors + treatment flag)
                X_T = list(pd.concat([df_train[T],df_train[X]], axis = 1).columns)
                # Fit model
                model = algorithm_1
                model.fit(df_train[X_T],df_train[Y])
                # Predictions
                ## Prob. treatment
                testIndices = df_test.index.values
                df_test_treated = pd.concat([df_test[X],df_test[Y]], axis = 1) 
                df_test_treated[T] = 1
                prob_treatment = model.predict_proba(df_test_treated[X_T])[:,1]
                ## Prob. control
                df_test_control = pd.concat([df_test[X],df_test[Y]], axis = 1) 
                df_test_control[T] = 0
                prob_control = model.predict_proba(df_test_control[X_T])[:,1]
                # Summary df
                df_prob_treatment = pd.DataFrame(prob_treatment,testIndices, columns = {"Prob_treat"})
                df_prob_control = pd.DataFrame(prob_control,testIndices, columns = {"Prob_ctrl"})
                predictions = pd.concat([df_test,df_prob_treatment,df_prob_control],axis = 1)
                predictions["s"] = predictions["Prob_treat"] - predictions["Prob_ctrl"] 
                return predictions;
            fitted_model.append(S_learner(folds[0][i],folds[1][i]))
     
    elif model == 'X_learner':
        for i in list(range(0,amountofFolds)):
            def X_learner(df_train,df_test):
                np.random.seed(random_state)
                # Model 1 (conventional T-learner)
                df_treated = df_train[df_train[T] == 1]
                df_control = df_train[df_train[T] == 0]
                model_1 = algorithm_1
                model_1.fit(df_control[X],df_control[Y])
                model_2 = algorithm_2
                model_2.fit(df_treated[X],df_treated[Y])
                # Impute the ITE
                df_treated_indices = df_treated.index.values
                model_1_pred = model_1.predict(df_treated[X])
                pred_effect_control = pd.DataFrame(model_1_pred,df_treated_indices, columns = {"Pred_ctrl_effect"})
                df_treated = pd.concat([df_treated,pred_effect_control],axis = 1)
                df_treated['D'] = df_treated[Y] - df_treated['Pred_ctrl_effect']
                df_control_indices = df_control.index.values
                model_2_pred = model_2.predict(df_control[X])
                pred_effect_treatment = pd.DataFrame(model_2_pred,df_control_indices, columns = {"Pred_tr_effect"})
                df_control = pd.concat([df_control,pred_effect_treatment],axis = 1)
                df_control['D'] = df_control['Pred_tr_effect'] - df_control[Y]
                # Model 2 (T-learner with the imputed ITE as the new outcome variable)
                model_3 = algorithm_3
                model_3.fit(df_treated[X],df_treated['D'])
                model_4 = algorithm_4
                model_4.fit(df_control[X],df_control['D'])
                # Estimate the propensity score (PS)
                testIndices = df_test.index.values
                model_ps = xgb.XGBClassifier(objective = "binary:logistic",n_estimators = 100, random_state = random_state)
                model_ps.fit(df_test[X],df_test[T])
                ps = model_ps.predict_proba(df_test[X])[:,1]
                df_ps = pd.DataFrame(ps,testIndices, columns = {"ps"})
                # Predict
                Prob_ctrl = model_4.predict_proba(df_test[X])[:,1]
                Prob_treat = model_3.predict_proba(df_test[X])[:,1]
                df_prob_effect_control = pd.DataFrame(Prob_ctrl,testIndices, columns = {"Prob_ctrl"})
                df_prob_effect_treatment = pd.DataFrame(Prob_treat,testIndices, columns = {"Prob_treat"})
                predictions = pd.concat([df_test,df_ps,df_prob_effect_control,df_prob_effect_treatment],axis = 1)
                predictions["s"] = (predictions["Prob_ctrl"]*predictions["ps"]) + ((1-predictions["ps"])*predictions["Prob_treat"])
                return predictions;
            fitted_model.append(X_learner(folds[0][i],folds[1][i]))
    
    elif model == 'MOA':
        for i in list(range(0,amountofFolds)):
            def MOA(df_train,df_test):
                np.random.seed(random_state)
                # Create new outcome variable
                df_train["tr"] = np.where((df_train[T] == 1) & (df_train[Y] == 1),1,0)
                df_train["cn"] = np.where((df_train[T] == 0) & (df_train[Y] == 0),1,0)
                df_train["cr"] = np.where((df_train[T] == 0) & (df_train[Y] == 1),1,0)
                df_train["tn"] = np.where((df_train[T] == 1) & (df_train[Y] == 0),1,0)
                df_train["new_target"] = np.where((df_train["tr"] == 1) | (df_train["cn"] == 1),1,0)
                # Fit model with new outcome variable
                model = algorithm_1
                model.fit(df_train[X],df_train["new_target"])
                # Predict
                testIndices = df_test[X].index.values
                prob_t_1_c_0 = model.predict_proba(df_test[X])[:,1]
                prob_t_0_c_1 = model.predict_proba(df_test[X])[:,0]
                df_prob_t_1_c_0 = pd.DataFrame(prob_t_1_c_0,testIndices, columns = {"Prob_treat"})
                df_prob_t_0_c_1 = pd.DataFrame(prob_t_0_c_1,testIndices, columns = {"Prob_ctrl"})
                #predictions["s_v1"] = 2*predictions["Prob_treat"] - 1
                # Summary results
                predictions = pd.concat([df_test,df_prob_t_1_c_0,df_prob_t_0_c_1],axis = 1)
                predictions["s"] = predictions["Prob_treat"] - predictions["Prob_ctrl"]   
                return predictions;
            fitted_model.append(MOA(folds[0][i],folds[1][i]))
            
    return fitted_model;   