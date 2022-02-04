import statistics 
import os, pandas as pd
import numpy as np
import math

"""
Estimates the Qini and the profit
"""

def performance_profit(model_predictions,gamma,delta,frontier,metric,approach_name,T,Y,OB,TC):
    
    """
    Lenght of the bins: here is 100 but can be any number
    """
    len_fold = []
    for fold in list(range(len(model_predictions))):
        len_fold.append(len(model_predictions[fold]))
    min_lenght_folds = min(len_fold)
    min_lenght_folds = 100
    
    """
    Obtain the performance table:
    """
    def performance_table(df,min_lenght_folds,frontier,metric,approach_name,T,Y):
        if frontier == True:
            df['tau'] = gamma + delta*df['Prob_treat']
            df['target'] = np.where(df['s'] >= df['tau'],1,-1)
            #df['d'] = abs(delta*df.loc[:,'Prob_treat'] - df.loc[:,'s']+gamma)/(math.sqrt((delta**2)+1))#*df['target']
            df['d'] = (df.loc[:,'s'] - delta*df.loc[:,'Prob_treat'] - gamma)/(math.sqrt((delta**2)+1))
            df_sorted = df.sort_values(by = ["d"], ascending = False).reset_index(drop=True)
        else:
            df_sorted = df.sort_values(by = ["s"], ascending = False).reset_index(drop=True)
        
        df_sorted["ranking"] =  list(range(1,len(df_sorted)+1))
        labels = list(range(1,min_lenght_folds+1))
        df_sorted["bin"] = pd.cut(df_sorted['ranking'], len(labels), labels = labels).astype(int)
        df_sorted.index = df_sorted.index + 1
        df_sorted["ct"] = np.where(df_sorted[T]==0,1,0)
        df_sorted["y_tr"] = np.where((df_sorted[T]==1) & (df_sorted[Y]==1) ,1,0)
        df_sorted["y_ct"] = np.where((df_sorted[T]==0) & (df_sorted[Y]==1) ,1,0)
        n_t = df_sorted.pivot_table(index='bin', values= T, aggfunc='sum')
        n_c = df_sorted.pivot_table(index='bin', values= "ct", aggfunc='sum')
        n_y1_t = df_sorted.pivot_table(index='bin', values= 'y_tr', aggfunc='sum')
        n_y1_c = df_sorted.pivot_table(index='bin', values= "y_ct", aggfunc='sum')
        s_t = pd.concat([n_c,n_t,n_y1_c,n_y1_t], axis = 1)
        s_t.columns = ["n_c","n_t","n_y1_c","n_y1_t"]
        s_t["n_y0_t"] = s_t["n_t"] - s_t["n_y1_t"]
        s_t["n_y0_c"] = s_t["n_c"] - s_t["n_y1_c"]
        s_t["bin"] = s_t.index
        s_t["cumsum_n_t"] = s_t["n_t"].cumsum()
        s_t["cumsum_n_c"] = s_t["n_c"].cumsum()
        s_t["cumsum_n_y1_t"] = s_t["n_y1_t"].cumsum()
        s_t["cumsum_n_y1_c"] = s_t["n_y1_c"].cumsum()
        s_t["cumsum_n_y0_t"] = s_t["n_y0_t"].cumsum()
        s_t["cumsum_n_y0_c"] = s_t["n_y0_c"].cumsum()
        
        s_t["e11"] = s_t["cumsum_n_y1_t"]/s_t["cumsum_n_t"].iloc[-1]
        s_t["e10"] = s_t["cumsum_n_y1_c"]/s_t["cumsum_n_c"].iloc[-1]
        s_t["e01"] = s_t["cumsum_n_y0_t"]/s_t["cumsum_n_t"].iloc[-1]
        s_t["e00"] = s_t["cumsum_n_y0_c"]/s_t["cumsum_n_c"].iloc[-1]
        s_t["e11-e10"] = s_t["e11"] - s_t["e10"]
        
        """
        Calculate the Qini at each threshold
        """
        s_t["qini"] = s_t["cumsum_n_y1_t"] - (s_t["cumsum_n_y1_c"]*s_t["cumsum_n_t"])/s_t["cumsum_n_c"]
        s_t = s_t.replace(to_replace = np.nan, value = 0) 
        overall_inc_gains = s_t["qini"].iloc[-1]
        random_inc_gains = np.append([0],np.cumsum(np.repeat(overall_inc_gains / (s_t.loc[s_t.index[-1], "bin"]),(s_t.loc[s_t.index[-1], "bin"]))))
        
        """
        Calculate the profit at each threshold:
        """
        net_ben_T_1 = OB[1][1] - TC[1][1]
        net_ben_C_1 = OB[1][0] - TC[1][0]
        net_ben_T_0 = OB[0][1] - TC[0][1]
        net_ben_C_0 = OB[0][0] - TC[0][0]
        s_t['profit'] = s_t["e11"]*net_ben_T_1 - s_t["e10"]*net_ben_C_1 + s_t["e01"]*net_ben_T_0 - s_t["e00"]*net_ben_C_0
        #s_t['profit'] = s_t["e11-e10"]*OB[1][1] - s_t["e11"]*TC[1][1] - s_t["e01"]*TC[0][1]
        s_t = s_t.replace(to_replace = np.nan, value = 0) 

        # Qini
        qini = s_t["qini"]
        qini = pd.Series(qini).values
        qini = np.append([0],qini)

        # Profit
        profit = s_t['profit']
        profit = pd.Series(profit).values
        profit = np.append([0],profit)

        # X axis
        x = s_t["bin"]/(s_t.loc[s_t.index[-1], "bin"])
        x = pd.Series(x).values
        x = np.append([0],x)
        
        if metric == "qini":
            performance_table = pd.DataFrame(list(zip(x,qini, random_inc_gains)),columns =['x',approach_name, 'Random'])
        else:
            performance_table = pd.DataFrame(list(zip(x,profit)),columns =['x',approach_name])
            
        """
        Testing:
        """
        cum_nt = s_t["cumsum_n_y1_c"]
        cum_nt = pd.Series(cum_nt).values
        cum_nt = np.append([0],cum_nt)
        perf_testing = pd.DataFrame(list(zip(x,cum_nt)),columns =['x',approach_name])
        perf_testing[approach_name] = perf_testing[approach_name]/perf_testing[approach_name].iloc[-1]

        
        p11_ITEs = pd.concat([df[Y],df[T],df['Prob_treat'],df['s']], axis = 1)
        
        return performance_table;
    
    performance_tables_fold = []
    for fold in list(range(len(model_predictions))):
        performance_tables_fold.append(performance_table(model_predictions[fold],min_lenght_folds,frontier,metric,approach_name,T,Y))
    return performance_tables_fold;

"""
Calculate the qini metric (scalar value)
"""

def qini_metric(perf_object):
    x = list(perf_object.iloc[:,0]/100)
    y_inc = list(perf_object.iloc[:,1])
    y_ran = list(perf_object.iloc[:,2])
    
    def auc(x,y):
        auc = 0
        for i in list(range(1,len(x))):
            auc = auc + 0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1])
        return auc
 
    auc_inc = auc(x,y_inc)
    auc_ran = auc(x,y_ran)
    
    qini = auc_inc - auc_ran
    
    return qini;

def performance_metrics(effect_estimates,name_causal_model,T,Y,b,c):
    
    if name_causal_model == 'CF':
        model_predictions = effect_estimates[0]
    elif name_causal_model == 'T_learner':
        model_predictions = effect_estimates[1]
    elif name_causal_model == 'S_learner':
        model_predictions = effect_estimates[2]

    """
    The OB and TC matrices
    """
    def create_OB(b_11,b_10,b_01,b_00):
        OB = [[b_00,b_01],[b_10,b_11]]
        return OB;

    def create_TC(c_11,c_10,c_01,c_00):
        TC = [[c_00,c_01],[c_10,c_11]]
        return TC;

    av_qini_insensitive = []
    av_qini_sensitive = []
    std_qini_insensitive = []
    std_qini_sensitive = []
    av_profit_insensitive = []
    av_profit_sensitive = []
    std_profit_insensitive = []
    std_profit_sensitive = []
    max_profit_sensitive = []
    max_profit_insensitive = []
    MP_sen_index = []
    MP_sen_val = []
    MP_in_index = []
    MP_in_val = []

    for benefit in b:

        """
        Equal benefits
        """
        benefit_pos = benefit
        benefit_neg = benefit*1.2
        cost_pos = benefit_pos*c
        cost_neg = benefit_pos*c

        """
        OB and TC matrices
        """
        OB = create_OB(benefit_pos,benefit_neg,0,0)
        TC = create_TC(cost_pos,0,cost_neg,0)

        """
        Gamma and delta:
        """
        gamma = (OB[0][0] - TC[0][0] - OB[0][1] + TC[0][1])/(OB[1][0] - TC[1][0] - OB[0][0] + TC[0][0])
        delta = (OB[1][0] - TC[1][0] + OB[0][1] - TC[0][1] - OB[1][1] + TC[1][1] - OB[0][0] + TC[0][0])/(OB[1][0] - TC[1][0] - OB[0][0] + TC[0][0])

        qini_sensitive = []
        qini_insensitive = []
        profit_sensitive = []
        profit_insensitive = []
        list_sen_prof = []
        list_ins_prof = []
        
        for fold in list(range(len(model_predictions))):
            """
            Qini
            """
            perf_table_sensitive = performance_profit(model_predictions,gamma,delta,True,"qini","Profit-sensitive",T,Y,OB,TC)
            perf_table_insensitive = performance_profit(model_predictions,gamma,delta,False,"qini","Profit-insensitive",T,Y,OB,TC)
            qini_sensitive.append(qini_metric(perf_table_sensitive[fold]))
            qini_insensitive.append(qini_metric(perf_table_insensitive[fold]))
            """
            Profit
            """
            prof_table_sensitive = performance_profit(model_predictions,gamma,delta,True,"Profit","Profit-sensitive",T,Y,OB,TC)
            prof_table_insensitive = performance_profit(model_predictions,gamma,delta,False,"Profit","Profit-insensitive",T,Y,OB,TC)
            """
            Average profit:
            """
            profit_sensitive.append(prof_table_sensitive[fold].iloc[:,1].mean())
            profit_insensitive.append(prof_table_insensitive[fold].iloc[:,1].mean())
            """
            Max. profit and fractions
            """
            list_sen_prof.append(prof_table_sensitive[fold].iloc[:,1])
            list_ins_prof.append(prof_table_insensitive[fold].iloc[:,1])
        
        max_profit_sensitive = pd.concat(list_sen_prof, axis = 1)
        max_profit_sensitive['mean'] = max_profit_sensitive.mean(axis=1)
        MP_sen_index.append(max_profit_sensitive['mean'].idxmax())
        MP_sen_val.append(max(max_profit_sensitive['mean']))

        max_profit_insensitive = pd.concat(list_ins_prof, axis = 1)
        max_profit_insensitive['mean'] = max_profit_insensitive.mean(axis=1)
        MP_in_index.append(max_profit_insensitive['mean'].idxmax())
        MP_in_val.append(max(max_profit_insensitive['mean']))

        # Qini results
        av_qini_insensitive.append(statistics.mean(qini_insensitive))
        av_qini_sensitive.append(statistics.mean(qini_sensitive))
        std_qini_insensitive.append(statistics.stdev(qini_insensitive))
        std_qini_sensitive.append(statistics.stdev(qini_sensitive))

        # Profit results
        av_profit_insensitive.append(statistics.mean(profit_insensitive))
        av_profit_sensitive.append(statistics.mean(profit_sensitive))
        std_profit_insensitive.append(statistics.stdev(profit_insensitive))
        std_profit_sensitive.append(statistics.stdev(profit_sensitive))
    
    """
    Plot Qini curves
    """
    q_performance_to_plot_sensitive = performance_profit(model_predictions,gamma,delta,True,"qini","Profit-sensitive",T,Y,OB,TC)
    q_performance_to_plot_insensitive = performance_profit(model_predictions,gamma,delta,False,"qini","Profit-insensitive",T,Y,OB,TC)
    q_performance_to_plot_sensitive = pd.melt(pd.concat(q_performance_to_plot_sensitive), ['x'])
    q_performance_to_plot_insensitive = pd.melt(pd.concat(q_performance_to_plot_insensitive),['x'])
    q_performance_to_plot = pd.concat([q_performance_to_plot_sensitive,q_performance_to_plot_insensitive])
    q_performance_to_plot.columns = ['x','Approach',"Value"]
    
    """
    Plot profit 
    """
    
    p_performance_to_plot_sensitive = performance_profit(model_predictions,gamma,delta,True,"profit","Profit-sensitive",T,Y,OB,TC)
    p_performance_to_plot_insensitive = performance_profit(model_predictions,gamma,delta,False,"profit","Profit-insensitive",T,Y,OB,TC)
    p_performance_to_plot_sensitive = pd.melt(pd.concat(p_performance_to_plot_sensitive), ['x'])
    p_performance_to_plot_insensitive = pd.melt(pd.concat(p_performance_to_plot_insensitive),['x'])
    p_performance_to_plot = pd.concat([p_performance_to_plot_sensitive,p_performance_to_plot_insensitive])
    p_performance_to_plot.columns = ['x','Approach',"Value"]
    
    """
    Plots of cumulative positive instances in T and C
    """
    c_performance_to_plot_sensitive = performance_profit(model_predictions,gamma,delta,True,"qini","sensitive",T,Y,OB,TC)
    c_performance_to_plot_insensitive = performance_profit(model_predictions,gamma,delta,False,"qini","insensitive",T,Y,OB,TC)
    c_performance_to_plot_sensitive = pd.melt(pd.concat(c_performance_to_plot_sensitive), ['x'])
    c_performance_to_plot_insensitive = pd.melt(pd.concat(c_performance_to_plot_insensitive),['x'])
    c_performance_to_plot = pd.concat([c_performance_to_plot_sensitive,c_performance_to_plot_insensitive])
    c_performance_to_plot.columns = ['x','Approach',"Value"]
    
    """
    Distribution P11
    """

    #perf_dist = performance_profit(model_predictions,gamma,delta,True,"profit","Profit-sensitive",T,Y,OB,TC)
    #perf_dist = pd.concat(perf_dist, axis = 0)
    #perf_dist_t = perf_dist.loc[(perf_dist[T] == 1) & (perf_dist[Y] == 1)]
    #perf_dist_c = perf_dist.loc[(perf_dist[T] == 0) & (perf_dist[Y] == 1)]
    
    
    results = {"Qini_in":av_qini_insensitive,
               "Qini_sen":av_qini_sensitive,
               "Prof_in":av_profit_insensitive,
               "Prof_se":av_profit_sensitive,
               "MP_in_val":MP_in_val,
               "MP_sen_val":MP_sen_val,
               "MP_in_index":MP_in_index,
               "MP_sen_index":MP_sen_index,
               "plot_qinis":q_performance_to_plot,
               "plot_profit":p_performance_to_plot,
               "plot_cum_t":c_performance_to_plot,
              }
               
    return results;