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
                               np.transpose(np.diagflat([-2.5 / np.log(10) / fulldf.iloc[s].loc['x0'], 1, 1]))), # ... doproduct with transposed Jacobian
                        np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covdf = pd.DataFrame(covlow + covdiag + np.transpose(covlow)) # cov matrix for all SNe events
    
    idxcov = np.sort(list(fulldf.index + '_mB') + list(fulldf.index + '_x1') + list(fulldf.index + '_zc'))
    covdf.columns = idxcov
    covdf.index = idxcov
    covdf = covdf.groupby(level=0, axis=0).sum().groupby(level=0, axis=1).sum() # combine duplicated SNe
    
    return np.array(covdf)

# -------------------- select columns as in BuildJLACases ----------------------------
def PPinput(fulldf):
    myinput = fulldf.loc[:, ['zCMB', 'mB', 'x1', 'c', 'HOST_LOGMASS', 'IDSURVEY', 'zHEL', 'RA', 'DEC']]
    myinput.loc[:, 'IDSURVEY'] = fulldf.index.value_counts()
    myinput = myinput.groupby(level=0).mean() # combine duplicated SNe
    return np.array(myinput.sort_index())

###################### RUN ###########################################################
# -------------------- choose file to load from input arguments ----------------------
fitopt = int(sys.argv[1])
muopt = int(sys.argv[2])
namefitopt = str(1000 + fitopt)[1:]
namemuopt = str(1000 + muopt)[1:]

# -------------------- read file -----------------------------------------------------
mydf = pd.read_table('Pantheon/calibration_files/FITOPT' + namefitopt + '_MUOPT' + namemuopt + '.FITRES', sep=' ', comment='#', skipinitialspace=True, index_col=[0, 1]).dropna(how='all', axis=0).dropna(how='all', axis=1).droplevel('VARNAMES:', axis=0)
# mydf = mydf.sort_values(by='zCMB')

# -------------------- calculate and save PPcov and PPinput --------------------------
np.savetxt('Pantheon/Build/PP_full_F' + namefitopt + '_M' + namemuopt + '_COVd.txt', PPcov(mydf))
np.savetxt('Pantheon/Build/PP_full_F' + namefitopt + '_M' + namemuopt + '_input.txt', PPinput(mydf))
