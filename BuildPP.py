import sys
import pandas as pd
import numpy as np

###################### DEFINITIONS ###################################################
# -------------------- generate 3N x 3N covariance matrix ----------------------------
def PPcov(fulldf):
    covdiag = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                         np.diagflat(np.array(fulldf.iloc[s].loc[['mBERR', 'x1ERR', 'cERR']])**2), # diagonal of cov
                         np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    
    covlow = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                        np.dot(np.dot(np.diagflat([-2.5 / np.log(10) / fulldf.iloc[s].loc['x0'], 1, 1]), # Jacobian dotproduct with ...
                                      np.array([[0]*3, [fulldf.iloc[s].loc['COV_x1_x0']] + [0]*2, [fulldf.iloc[s].loc['COV_c_x0'], fulldf.iloc[s].loc['COV_x1_c'], 0]])), # ... cov matrix from PP for (x0, x1, c) ...
                               np.transpose(np.diagflat([-2.5 / np.log(10) / fulldf.iloc[s].loc['x0'], 1, 1]))), # ... dotproduct with transposed Jacobian
                        np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    
    return covlow + covdiag + np.transpose(covlow)

# -------------------- select columns as in BuildJLACases ----------------------------
def PPinput(fulldf):
    return np.array(fulldf.loc[:, ['zCMB', 'mB', 'x1', 'c', 'HOST_LOGMASS', 'IDSURVEY', 'zHEL', 'RA', 'DEC']])

###################### RUN ###########################################################
# -------------------- choose file to load from input arguments ----------------------
fitopt = int(sys.argv[1])
muopt = int(sys.argv[2])
namefitopt = str(1000 + fitopt)[1:]
namemuopt = str(1000 + muopt)[1:]

# -------------------- read file -----------------------------------------------------
mydf = pd.read_table('calibration_files/FITOPT' + namefitopt + '_MUOPT' + namemuopt + '.FITRES', sep=' ', comment='#', skipinitialspace=True, index_col=[0, 1]).dropna(how='all', axis=0).dropna(how='all', axis=1).droplevel('VARNAMES:', axis=0)

# -------------------- calculate and save PPcov and PPinput --------------------------
np.savetxt('Build/PP_full_F' + namefitopt + '_M' + namemuopt + '_COVd.txt', PPcov(mydf))
np.savetxt('Build/PP_full_F' + namefitopt + '_M' + namemuopt + '_input.txt', PPinput(mydf))
