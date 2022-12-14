import sys
import pandas as pd
import numpy as np

###################### DEFINITIONS ###################################################
# -------------------- generate 3N x 3N covariance matrix ----------------------------
def PPcov(fulldf):
    fulldf = fulldf.sort_index()
    covdiag = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                         np.diagflat(np.array(fulldf.iloc[s].loc[['mBERR', 'x1ERR', 'cERR']])**2), # diagonal of cov
                         np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covlow = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                        np.dot(np.dot(np.diagflat([-2.5 / np.log(10) / fulldf.iloc[s].loc['x0'], 1, 1]), # Jacobian dotproduct with ...
                                      np.array([[0]*3, [fulldf.iloc[s].loc['COV_x1_x0']] + [0]*2, [fulldf.iloc[s].loc['COV_c_x0'], fulldf.iloc[s].loc['COV_x1_c'], 0]])), # ... cov matrix from PP for (x0, x1, c) ...
                               np.transpose(np.diagflat([-2.5 / np.log(10) / fulldf.iloc[s].loc['x0'], 1, 1]))), # ... dotproduct with transposed Jacobian
                        np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covdf = pd.DataFrame(covlow + covdiag + np.transpose(covlow)) # cov matrix for all SNe events
    
    idxcov = np.sort(list(fulldf.index + '_mB') + list(fulldf.index + '_x1') + list(fulldf.index + '_zc'))
    covdf.columns = idxcov
    covdf.index = idxcov
    covdf = covdf.groupby(level=0, axis=0).sum().groupby(level=0, axis=1).sum() # combine duplicated SNe
    
    return covdf

# -------------------- select columns as in BuildJLACases ----------------------------
def PPinput(fulldf):
    myinput = fulldf.loc[:, ['zCMB', 'mB', 'x1', 'c', 'HOST_LOGMASS', 'IDSURVEY', 'zHEL', 'RA', 'DEC']]
    myinput.loc[:, 'IDSURVEY'] = fulldf.index.value_counts()
    myinput = myinput.groupby(level=0).mean() # combine duplicated SNe
    return myinput.sort_index()

###################### RUN ###########################################################
# -------------------- read scaling of files -----------------------------------------
summary = pd.read_csv('Pantheon/fitopts_summary.csv', index_col=0)
summary.loc[:, 'weights'] = summary.loc[:, 'scale'] * summary.loc[:, 'vpecTo0']

# -------------------- read files ----------------------------------------------------
print('read files')
dfs = {fo: pd.read_table('Pantheon/calibration_files/' + fo + '_MUOPT000.FITRES', sep=' ', comment='#', skipinitialspace=True, index_col=[0, 1]).dropna(how='all', axis=0).dropna(how='all', axis=1).droplevel('VARNAMES:', axis=0) for fo in summary.index}

# -------------------- calculate PPcov and PPinput for individual files --------------
print('input calculation for individual files')
inp = [summary.loc[fo, 'weights'] * PPinput(dfs[fo]) for fo in summary.index]
print('start covariance calculation for individual files ...')
cov = [summary.loc[fo, 'weights'] * PPcov(dfs[fo]) for fo in summary.index]

# -------------------- combine PPcov and PPinput -------------------------------------
print('combine')
colinp = list(inp[0].columns)
idxinp = list(inp[0].index)
allinp = (pd.concat(inp, axis=1).groupby(level=0, axis=1).sum()) / summary.weights.sum()
allinp = allinp.loc[idxinp, colinp]
colcov = list(cov[0].columns)
idxcov = list(cov[0].index)
allcov = pd.concat(cov, axis=1).groupby(level=0, axis=1).sum()
allcov = allcov.loc[idxcov, colcov]

# -------------------- save ----------------------------------------------------------
print('save')
np.savetxt('Pantheon/Build/PP_full_COVd.txt', np.array(allcov))
np.savetxt('Pantheon/Build/PP_full_input.txt', np.array(allinp))
