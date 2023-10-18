import numpy as np
import pandas as pd
import time

def boostz(z,RAdeg,DECdeg):
    """
    Determines The boosted z_{CMB} from the heliocentric redshift, RA and DEC

    outputs: boosted CMB redshift
    """
    # Angular coords should be in degrees and velocity in km/s
    
    vcmb = 371.0 # km/s
    # l_cmb = 264.14
    # b_cmb = 48.26
    # converts to
    ra_cmb = 168.0118667
    dec_cmb = -6.98303424
    
    C = 2.99792458e5 # km/s
    RA = np.radians(RAdeg)
    DEC = np.radians(DECdeg)
    RA0 = np.radians(ra_cmb)
    DEC0 = np.radians(dec_cmb)
    costheta = np.sin(DEC)*np.sin(DEC0) \
        + np.cos(DEC)*np.cos(DEC0)*np.cos(RA-RA0)
    return z + (vcmb/C)*costheta*(1.+z)

def Covariance():
    """
    Calculates the base covariance

    outputs: Full covariance
    """
    COVd = np.load('JLA_data/stat.npy')
    for i in ['cal', 'model', 'bias', 'dust', 'nonia', 'sigmaz', 'sigmalens']:
        COVd += np.load('JLA_data/'+i+'.npy')
    # COVd += sigma_lens
    return COVd

def JLA_data():
    
    # Tully et al 2008

    ndtypes = [('SNIa','S12'), \
               ('zcmb',float), \
               ('zhel',float), \
               ('e_z',float), \
               ('mb',float), \
               ('e_mb',float), \
               ('x1',float), \
               ('e_x1',float), \
               ('c',float), \
               ('e_c',float), \
               ('logMst',float), \
               ('e_logMst',float), \
               ('tmax',float), \
               ('e_tmax',float), \
               ('cov(mb,s)',float), \
               ('cov(mb,c)',float), \
               ('cov(s,c)',float), \
               ('set',int), \
               ('RAdeg',float), \
               ('DEdeg',float), \
               ('bias',float)]
    
    # width of each column
    delim = (12, 9, 9, 1, 10, 9, 10, 9, 10, 9, 10, 10, 13, 9, 10, 10, 10, 1, 11, 11, 10)
    
    # load the data
    file = 'S:\\Documents\\Zac_Final_Build\\General_480_Files\\tablef3.dat'
    data = np.genfromtxt(file, delimiter=delim, dtype=ndtypes, autostrip=True)
    
    zcmb = data['zcmb']
    mb = data['mb']
    x1 = data['x1']
    c = data['c']
    logMass = data['logMst'] # log_10_ host stellar mass (in units=M_sun)
    survey = data['set']
    zhel = data['zhel']
    ra = data['RAdeg']
    dec = data['DEdeg']
    
    # Survey values key:
    #   1 = SNLS (Supernova Legacy Survey)
    #   2 = SDSS (Sloan Digital Sky Survey: SDSS-II SN Ia sample)
    #   3 = lowz (from CfA; Hicken et al. 2009, J/ApJ/700/331
    #   4 = Riess HST (2007ApJ...659...98R)
    
    zcmb1 = boostz(zhel,ra,dec)

    COV = Covariance()
    JLA = np.column_stack((zcmb1,mb,x1,c,logMass,survey,zhel,ra,dec))
    return JLA, COV

def common_JLA(jla, cov):
    """
    Determines the common input and covariance from the common ID's in terms of JLA
    Lane (2022) compared these subsamples extensively

    outputs: Common covariance and input
    """
    # string = 'JLA/Build/'

    ids = np.genfromtxt('JLA_data/commonID.csv', dtype= int)

    common_input = jla[ids]

    cov_ids = np.vstack((3*ids, 3*ids+1, 3*ids+2))
    cov_ids = cov_ids.T
    cov_ids = cov_ids.ravel()    

    common_cov = cov[cov_ids,:] #sort rows
    common_cov = common_cov[:,cov_ids] #sort columns in the same manner  
    return common_input, common_cov

def random_JLA(input_sne, input_cov, string, Nseeds):
    l = len(input_sne)
    idx = np.arange(l)
    p = np.full(l, 1/l)

    m = 580 # size of the random subsamples

    versionname = '740random' + str(m)
    for seed in range(Nseeds):
        np.random.seed()
        choice = np.unique(np.random.choice(idx, size = m, replace=False, p=p))
        input_data = input_sne[choice]
        
        ind = np.column_stack((3*choice, 3*choice+1, 3*choice+2)).ravel()
        covmatrix = input_cov[ind,:]
        covmatrix = covmatrix[:,ind]

        # -------------------- save ----------------------------------------------------------
        print('save', versionname, seed)
        np.savetxt(string + 'JLA_' + versionname  + '_' + str(seed) + '_COVd.txt', covmatrix)
        np.savetxt(string + 'JLA_' + versionname  + '_' + str(seed) + '_input.txt', input_data)


start = time.time()
string = 'Pantheon/Build/PP_JLA_Common/'

jla, cov = JLA_data()
np.savetxt(string + 'JLA_input.txt', jla, delimiter='\t', \
            fmt=('%10.7f','%10.7f','%10.7f','%10.7f','%10.7f','%i','%9.7f','%11.7f','%11.7f'))
np.savetxt(string + 'JLA_COVd.txt', cov, delimiter='\t')
# jla = np.genfromtxt(string + 'JLA_input.txt')
# cov = np.genfromtxt(string + 'JLA_COVd.txt')


common_input, common_cov = common_JLA(jla, cov)
np.savetxt(string + 'JLA_comm_input.txt', common_input, delimiter='\t', \
        fmt=('%10.7f','%10.7f','%10.7f','%10.7f','%10.7f','%i','%9.7f','%11.7f','%11.7f'))
np.savetxt(string + 'JLA_comm_COVd.txt', common_cov, delimiter='\t')

random_JLA(jla, cov, string, Nseeds=1)

print("This took {0:.01f} seconds to calculate".format((time.time() - start)))
