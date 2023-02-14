import sys
import pandas as pd
import numpy as np
allnegew = {}
ewByFile = True # whether to calculate the eigenvalues within PPcovFIT (True) or at the end (False). Only if True all non-positive definite eigenvalues are dropped from FITOPT
mkposdef = True # force the covariance matrix to be positive definite by dropping the bad SNe. These SNe latter are printed at the end
nominal = 'FITOPT000' # file used to get the statistical covariance and the reference values
nominalmu = 'MUOPT000'

###################### DEFINITIONS ###################################################
def blockidx(indices): # indices of covariance matrix for given SNe
    form = pd.DataFrame(np.zeros((len(indices), 3)), index = indices, columns = ['mB', 'x1', 'zc'])
    return np.array(['_'.join(col) for col in form.stack().index])
    
def blocksne(indices): # reverse of blockidx
    return [s.rsplit('_', maxsplit=1)[0] for s in indices[::3]]

def jacentry(s, fulldf): # only non-trivial entry of Jacobian block matrices
    return -2.5 / np.log(10) / fulldf.iloc[s].loc['x0']

def eigenvalues(covdf, fitopt, allsne, returnsne=True):
    ew = np.linalg.eigvals(covdf)
    print(fitopt, ':', len(ew[ew < 0]), 'out of', len(ew), 'are less than 0, ', len(ew[ew == 0]), 'are equal to zero - saved:', returnsne)
    
    if returnsne:
        sneew = {sne: np.linalg.eigvals(covdf.loc[blockidx([sne]), blockidx([sne])]) for sne in allsne}
        negew = [sne for sne in sneew.keys() if not np.all(sneew[sne] >= 0)]
        allnegew[fitopt] = negew

def magmagmatrix(magdf): # return matrix with only the (mB, mB) entries of the diagonal being non-zero
    idxcov = blockidx(magdf.index)
    empty = pd.DataFrame(np.zeros(len(magdf.index)), index=magdf.index) # DataFrame with all zero entries (for x1 and c diagonal components of the covariance)
    table = pd.concat([magdf] + [empty]*2, axis=1)
    matrix = pd.DataFrame(np.diagflat(np.array(table.stack())))
    matrix.columns = idxcov
    matrix.index = idxcov
    return matrix

def offdiagduplicates(diagdf):
    offdiagdf = diagdf.copy()
    for i in range(int(offdiagdf.shape[0] / 3)):
        for j in range(i):
            if np.all(offdiagdf.index[(3*i):(3*i+3)] == offdiagdf.index[(3*j):(3*j+3)]):
                subdfi = offdiagdf.iloc[(3*i):(3*i+3), (3*i):(3*i+3)]
                subdfj = offdiagdf.iloc[(3*j):(3*j+3), (3*j):(3*j+3)]
                offdiagdf.iloc[(3*j):(3*j+3), (3*i):(3*i+3)] = (subdfi + subdfj) / 2
                offdiagdf.iloc[(3*i):(3*i+3), (3*j):(3*j+3)] = (subdfi + subdfj) / 2
    return offdiagdf

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
    
    eigenvalues(covdf, fitopt, fulldf.index, ewByFile)
    
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
        snN = fulldf.index.value_counts()[sn]
        sigma = pd.DataFrame(np.zeros((3, 3)), index=blockidx([sn]), columns=blockidx([sn]))
        if snN > 1:
            sigma = pd.DataFrame({snx: {sny: (np.array(pd.DataFrame(orgdelta).loc[snx, :].T) @ np.array(pd.DataFrame(orgdelta).loc[sny, :]))[0, 0] 
                                        for sny in blockidx([sn])} for snx in blockidx([sn])})
        sigmae.append(sigma / (snN - 1))
    sigmadupl = pd.concat(sigmae).fillna(0)
    const = magmagmatrix(pd.DataFrame(np.full(int(len(sigmadupl) / 3), 0.102), index=blocksne(sigmadupl.index)))
    return sigmadupl# + const

def PPcovSYST(fulldf, comparedf, scale): # systematic covariance according to Eq. (7) of Brout et al. 2022 (arXiv:2202.04077)
    usedf = fulldf.sort_index().groupby(level=0, axis=0).mean() # mu
    val = pd.DataFrame(usedf.loc[:, ['mB', 'x1', 'c']].stack())
    val.index = blockidx(usedf.index)
    usecomparedf = comparedf.sort_index().groupby(level=0, axis=0).mean() # reference value of delta mu
    ref = pd.DataFrame(usecomparedf.loc[:, ['mB', 'x1', 'c']].stack())
    ref.index = blockidx(usecomparedf.index)
    delta = (val - ref) * scale # d delta mu / d S * sigma
    systdf = delta @ delta.T
    return systdf # return covariance contribution of a single psi

def PPcovSTAT(fulldf):
    fulldf = fulldf.sort_index()
    # sigma_lens^2 = (0.055 * boostz)^2
    sigma2lens = pd.DataFrame((0.055 * boostz(fulldf.zHEL, fulldf.RA, fulldf.DEC))**2)
    
    # sigma_z^2 = D_boostz**2
    # D_boostz = d/dz boostz(z) * D_z = (1 + (vel/C)*costheta) * D_z = boostz(D_z)
    sigma2z = pd.DataFrame(boostz(fulldf.zHELERR, fulldf.RA, fulldf.DEC)**2)
    # sigma_z^2 - JLA approach # comment this line to use the boostz uncertainty (line above)
#     sigma2z = pd.DataFrame(sigmaz_JLA(fulldf.zCMB)**2)
    
    # 3N x 3N matrix
    sigma2 = offdiagduplicates(magmagmatrix(sigma2z + sigma2lens)) # populate entries with same SNe (with and without same survey)
    normalize = (pd.DataFrame(sigma2.index.value_counts()) @ pd.DataFrame(sigma2.index.value_counts()).T) 
    sigma2 = sigma2.groupby(level=0, axis=0).sum().groupby(level=0, axis=1).sum() / normalize # combine duplicated SNe
    return sigma2

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

def sigmaz_JLA(z):
    z.name = 0
    c_speed = 299792458/1000 # km/s
    return ((5 * 150 / c_speed) / np.log(10**z))

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
summarymu = pd.read_csv('Pantheon/muopts_summary.csv', index_col=0)

# -------------------- read files ----------------------------------------------------
print('read files')
dfs = {fo: pd.read_table('Pantheon/calibration_files/' + fo + '_MUOPT000.FITRES', sep=' ', comment='#', skipinitialspace=True, index_col=[0, 1]).dropna(how='all', axis=0).dropna(how='all', axis=1).droplevel('VARNAMES:', axis=0) for fo in summary.index}
dms = {mo: pd.read_table('Pantheon/calibration_files/FITOPT000_' + mo + '.FITRES', sep=' ', comment='#', skipinitialspace=True, index_col=[0, 1]).dropna(how='all', axis=0).dropna(how='all', axis=1).droplevel('VARNAMES:', axis=0) for mo in summarymu.index}

# -------------------- calculate PPcov and PPinput for individual files --------------
print('input calculation for ' + nominal)
inp = PPinput(dfs[nominal])
print('start covariance calculation of fit for ' + nominal)
covfit = PPcovFIT(dfs[nominal], nominal)
print('start covariance calculation for duplicated SNe')
covdupl = PPcovDUPL(dfs[nominal])
print('start statistical covariance calculation')
covstat = PPcovSTAT(dfs[nominal])
print('start systematic covariance calculation from FITOPTs')
covsyst = [PPcovSYST(dfs[fo], inp, summary.loc[fo, 'weights']) for fo in np.setdiff1d(summary.index, [nominal])]
print('start systematic covariance calculation from MUOPTs')
covmu = [PPcovSYST(dms[mo], inp, summarymu.loc[mo, 'scale']) for mo in np.setdiff1d(summarymu.index, [nominalmu])]

# -------------------- combine PPcov and PPinput -------------------------------------
print('combine')
allinp = inp
allcov = pd.concat(covsyst + covmu + [covfit, covdupl, covstat], axis=1).groupby(level=0, axis=1).sum() # sum over all psi of Eq. (7) to get full covariance, add statistical & duplication covariances
colinp = list(inp.columns)
idxinp = list(inp.index)
eigenvalues(allcov, 'combined', allinp.index, returnsne = not ewByFile)
if mkposdef:
    idxinp = np.setdiff1d(idxinp, np.unique(np.concatenate([allnegew[k] for k in allnegew.keys()])))
allinp = allinp.loc[idxinp, colinp]
idxcov = blockidx(idxinp)
colcov = idxcov
allcov = allcov.loc[idxcov, colcov]
eigenvalues(allcov, 'finalcov', allinp.index, returnsne=False) # check if resulting matrix has negative eigenvalues

# -------------------- save ----------------------------------------------------------
versionname = '1537'
print('save', versionname)
np.savetxt('Pantheon/Build/PP_' + versionname + '_COVd.txt', np.array(allcov))
np.savetxt('Pantheon/Build/PP_' + versionname + '_input.txt', np.array(allinp))
print('resulting shape:', allcov.shape, allinp.shape)
print(allnegew)