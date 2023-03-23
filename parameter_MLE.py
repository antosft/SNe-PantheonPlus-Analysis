# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:16:56 2022

@author: zgl12
"""
# %% 
import numpy as np
import os
import sys
        
def Path(model,sample,file_folder):
    path = 'S:/Documents/Zac_Final_Build/output/'
    path = 'Pantheon/output'
    if model == 1:
        path_end = 'Timescape/'
    elif model == 2:
        path_end = 'LCDM/'
    elif model == 3:
        path_end = 'Milne/'
    
    path = path + sample + path_end + file_folder
    return path

def Params(stats,n):
    # MAP or MLE Parameters
    param = stats[n][1:]
    param = param.split(' ')
    param = [x for x in param if x]
    param = param[0]
    return param

def Parameter_Strip(prefix,model,evidence,sample,file_folder,m):
    
    evidence = str(evidence)
    path = Path(model,sample + '/',file_folder)
    string = "_0_1_2_1000_"
    string = string + evidence
    model = prefix + str(model) + '_'
    file_type = '_stats.dat'
    
    lin = np.linspace(0,0.1,21)
    
    logZ = []
    imp_logZ = []
    alpha = []
    beta = []
    colour = []
    x1 = []
    Q = []
    Q_u = []

    for l in lin:
        model = str(model)
        l = str(l)
        file_name = path + model + l + string + file_type
        
        try:
            stats = np.genfromtxt(file_name, delimiter = '/t', dtype = 'str')
        except:
            break

        nested_evidence = stats[0].replace('Nested Sampling Global Log-Evidence','')
        nested_evidence = nested_evidence.replace(':','')
        nested_evidence = nested_evidence.replace('+/-',',')
        nested_evidence = nested_evidence.replace(' ','')
        nested_evidence = nested_evidence.split(',')
        nested_sampling_global_log_evidence = nested_evidence[0]
        nested_evidence_error = nested_evidence[1]
        logZ.append(nested_sampling_global_log_evidence)

        nested_evidence = stats[1].replace('Nested Importance Sampling Global Log-Evidence','')
        nested_evidence = nested_evidence.replace(':','')
        nested_evidence = nested_evidence.replace('+/-',',')
        nested_evidence = nested_evidence.replace(' ','')
        nested_evidence = nested_evidence.split(',')
        nested_importance_sampling_global_log_evidence = nested_evidence[0]
        nested_importance_evidence_error = nested_evidence[1]
        imp_logZ.append(nested_importance_sampling_global_log_evidence)

        MLE_Q = Params(stats,1+m)
        Q.append(MLE_Q)
        MLE_alpha = Params(stats,2+m)
        alpha.append(MLE_alpha)
        MLE_x1  = Params(stats,3+m)
        x1.append(MLE_x1)
        MLE_sigma2_x1 = Params(stats,4+m)
        MLE_beta = Params(stats,5+m)
        beta.append(MLE_beta)
        MLE_colour = Params(stats,6+m)
        colour.append(MLE_colour)
        MLE_sigma2_c = Params(stats,7+m)
        MLE_M = Params(stats,8+m)
        MLE_sigma2_M = Params(stats,9+m)

        Q_uncert = stats[3]
        Q_u.append(Q_uncert)

    return Q, logZ, imp_logZ, alpha, beta, colour, x1, Q_u


# model = 1
# sample = 'Full_JLA/'
# file_folder = 'LinearSpace/'
# Q, logZ, imp_logZ, alpha, beta, colour, x1 = Parameter_Strip(model,sample,file_folder)

def ParamsSigma(stats):
    # MAP or MLE Parameters
    param = stats[2][1:]
    param = param.split(' ')
    param = [x for x in param if x]
    param = param[0]
    return param

# %%
