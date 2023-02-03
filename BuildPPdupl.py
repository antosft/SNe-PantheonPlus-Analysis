import sys
import pandas as pd
import numpy as np
allnegew = {}
mkposdef = True # force the covariance matrix to be positive definite by dropping the bad SNe. These SNe latter are printed at the end
nominal = 'FITOPT000' # file used to get the statistical covariance and the reference values

###################### DEFINITIONS ###################################################
def blockidx(indices): # indices of covariance matrix for given SNe
    return np.sort(list(pd.Series(indices) + '_mB') + list(pd.Series(indices) + '_x1') + list(pd.Series(indices) + '_zc'))

def jacentry(s, fulldf): # only non-trivial entry of Jacobian block matrices
    return -2.5 / np.log(10) / fulldf.iloc[s].loc['x0']

# -------------------- generate 3N x 3N covariance matrix ----------------------------
def PPcovFIT(fulldf, fitopt):
    fulldf = fulldf.sort_index()
    jac = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                     np.diagflat([jacentry(s, fulldf), 1, 1]), # Jacobi matrix
                     np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covdiag = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                         np.diagflat(np.array(fulldf.iloc[s].loc[['x0ERR', 'x1ERR', 'cERR']])**2), # diagonal of cov
                         np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covlow = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                        np.array([[0]*3, [fulldf.iloc[s].loc['COV_x1_x0']] + [0]*2, [fulldf.iloc[s].loc['COV_c_x0'], fulldf.iloc[s].loc['COV_x1_c'], 0]]), # cov matrix from PP for (x0, x1, c)
                        np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covx0 = covlow + covdiag.astype(float) + np.transpose(covlow) # cov matrix for all SNe events
    covdf = pd.DataFrame(np.dot(jac, np.dot(covx0, np.transpose(jac))))
    
    idxcov = blockidx(fulldf.index)
    covdf.columns = idxcov
    covdf.index = idxcov
    normalize = (pd.DataFrame(covdf.index.value_counts()) @ pd.DataFrame(covdf.index.value_counts()).T) 
    covdf = covdf.groupby(level=0, axis=0).sum().groupby(level=0, axis=1).sum() / normalize # combine duplicated SNe
    
    ew = np.linalg.eigvals(covdf)
    print(fitopt, ':', len(ew[ew < 0]), 'out of', len(ew), 'are less than 0, ', len(ew[ew == 0]), 'are equal to zero')
    
    sneew = {sne: np.linalg.eigvals(covdf.loc[blockidx([sne]), blockidx([sne])]) for sne in fulldf.index}
    negew = [sne for sne in sneew.keys() if not np.all(sneew[sne] > 0)]
    allnegew[fitopt] = negew
    
    return covdf

def PPcovDUPL(fulldf):
    fulldf = fulldf.loc[:, ['mB', 'x1', 'c']].sort_index()
    val = pd.DataFrame(fulldf.sort_index().stack())
    val.index = blockidx(fulldf.index)
    mean = fulldf.groupby(level=0).mean()
    ref = pd.DataFrame(mean.sort_index().stack())
    ref.index = blockidx(mean.index)
    orgdelta = val - ref
    sigmae = []
    for sn in np.unique(fulldf.index):
        delta = pd.DataFrame(orgdelta).loc[blockidx([sn]), :]
        sigma = delta @ delta.T
        sigmae.append(sigma.groupby(level=0, axis=0).sum().groupby(level=0, axis=1).sum() / (len(delta) / 3)**2)
    return pd.concat(sigmae).fillna(0)

def PPcovSYST(fulldf, comparedf, scale): # systematic covariance according to Eq. (7) of Brout et al. 2022 (arXiv:2202.04077)
    fulldf = fulldf.sort_index() # mu
    val = pd.DataFrame(fulldf.loc[:, ['mB', 'x1', 'c']].stack())
    val.index = blockidx(fulldf.index)
    comparedf = comparedf.sort_index() # reference value of delta mu
    ref = pd.DataFrame(comparedf.loc[:, ['mB', 'x1', 'c']].stack())
    ref.index = blockidx(comparedf.index)
    delta = (val - ref) * scale # d delta mu / d S * sigma
    systdf = delta @ delta.T
    systdf = systdf.groupby(level=0, axis=0).sum().groupby(level=0, axis=1).sum() # combine duplicated SNe
    return systdf # return covariance contribution of a single psi

# -------------------- correct zCMB --------------------------------------------------
def boostz(z, RAdeg, DECdeg, vel=371.0, RA0=168.0118667, DEC0=-6.98303424):
    '''
    Tully et al 2008. Tully et al reference everything in the local group frame and because we are transforming to the local group frame, therefore we use this instead of PLANCK2013 data
    
    vcmb = 371.0 # km/s #Velocity boost of CMB
    l_cmb = 264.14 # CMB multipole direction (degrees)
    b_cmb = 48.26 # CMB multipole direction (degrees)
    # converts to
    ra_cmb = 168.0118667 #Right Ascension of CMB
    dec_cmb = -6.98303424 #Declination of CMB
    '''
    # Angular coords should be in degrees and velocity in km/s
    C = 2.99792458e5 # km/s #Light
    RA = np.radians(RAdeg)
    DEC = np.radians(DECdeg)
    RA0 = np.radians(RA0)
    DEC0 = np.radians(DEC0)
    costheta = np.sin(DEC)*np.sin(DEC0) + np.cos(DEC)*np.cos(DEC0)*np.cos(RA-RA0)
    return z + (vel/C)*costheta*(1+z)

# -------------------- select columns as in BuildJLACases ----------------------------
def PPinput(fulldf):
    myinput = fulldf.loc[:, ['zCMB', 'mB', 'x1', 'c', 'HOST_LOGMASS', 'IDSURVEY', 'zHEL', 'RA', 'DEC']]
    myinput.zCMB = boostz(fulldf.zHEL, fulldf.RA, fulldf.DEC)
    #myinput.loc[:, 'IDSURVEY'] = fulldf.index.value_counts()
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
print('input calculation for ' + nominal)
inp = PPinput(dfs[nominal])
print('start covariance calculation of fit for ' + nominal)
covfit = PPcovFIT(dfs[nominal], nominal)
print('start covariance calculation for duplicated SNe')
covdupl = PPcovDUPL(dfs[nominal])
print('start systematic covariance calculation')
covsyst = [PPcovSYST(dfs[fo], inp, summary.loc[fo, 'weights']) for fo in np.setdiff1d(summary.index, [nominal])]

# -------------------- combine PPcov and PPinput -------------------------------------
print('combine')
colinp = list(inp.columns)
idxinp = list(inp.index)
if mkposdef:
    idxinp = np.setdiff1d(idxinp, np.unique(np.concatenate([allnegew[k] for k in allnegew.keys()])))
allinp = inp
allinp = allinp.loc[idxinp, colinp]
idxcov = blockidx(idxinp)
colcov = idxcov
#colcov = list(cov[0].columns)
#idxcov = list(cov[0].index)
allcov = pd.concat(covsyst + [covfit, covdupl], axis=1).groupby(level=0, axis=1).sum() # sum over all psi of Eq. (7) to get full covariance, add statistical & duplication covariances
allcov = allcov.loc[idxcov, colcov]

# -------------------- save ----------------------------------------------------------
print('save')
np.savetxt('Pantheon/Build/PP_duplFO_COVd.txt', np.array(allcov))
np.savetxt('Pantheon/Build/PP_duplFO_input.txt', np.array(allinp))
print('resulting shape:', allcov.shape, allinp.shape)
print(allnegew)