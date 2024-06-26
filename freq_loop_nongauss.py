# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 03:50:28 2022

@author: zgl12
"""

# ================================ DESCRIPTION ================================ 
# This is a code for doing frequentist analysis on supernovae data, following the likelihood construction of arXiv:1506.01354.  
# Supernovae data is that of the JLA sample. 
# The code is a modified version of that of J. T. Nielsen, A. Guffanti, S. Sarkar: arXiv:1506.01354
#
# In the present analysis best fit parameters and statistical measures are calculated for the standard model flat and empty universe and for the timescape model. 
#
# This code reproduces the frequentists results of arXiv:1706.07236v2 for a single redshift cut.
# To redo the redshift dependence analysis, simply loop over a range of redshift cuts zmin. 
#
# The code can easily be modified to include differently constrained standard models. 
# The interpolation table given for the standard model includes the full OM, OL grid. 
#
# The code can be modified to include the analysis of any model with an average distance-redshift relation. 
#This requires constructing an interpolation table equivalent to that of 'InterpolationTS_myboost_sorted.npy' and 'InterpolationLCDM_myboost_sorted.npy' from the JLA dataset redshifts. 
# =============================================================================

# %% ------------------------------------------------------------------------

import numpy as np
from scipy import interpolate, linalg, optimize
import matplotlib.pyplot as plt
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

versionname = '1690'
zcuts = np.linspace(0,0.25,201)
start = time.time()

# ===============================================================================================

    
string = 'Pantheon/Build/'
name = versionname

# start_abs = time.time()

start = time.time()

interpolation_ts = string + 'PP_' + name + '_tabledL_ts.npy'	 # Grid range timescape: OM in [0.001,0.99] 
interpolation_standardmodel = string + 'PP_' + name + '_tabledL_lcdm.npy'	 # Grid range standard model: OM in [0,1.5], OL in [-.5,1.5]

# ================== CHOOSING CONDITIONS FOR OPTIMISATION ==================

ts = []
lcdm = []
milne = []
aic = []
logl_ts = []
logl_lcdm = []
zC = zcuts

print('================= FREQUENTIST ANALYSIS FOR P+' + name + ' Extended =================')






for i in range(len(zC)):
#zmin = 0.0  # Redshift cut

    trans_dL = True  # Transforming luminosity distance to heliocentric frame?
    
    tlrc = 1e-12  # Tolerance for maximization
    # ==========================================================================
    
    zmin = zC[i]
    # zmin = 0.033
    
    if i % 4 == 0:
        print('redshift cutoff: ' , zmin)
    # print('tolerance in optimization: ' , tlrc) 
    
    # ======================================== READING DATA  ========================================
    
    # Z = np.genfromtxt(string + 'jla_input.txt')
    df = pd.read_csv(string + 'PP_' + name + '_input.csv')
    
    df = df.drop(columns='Unnamed: 0')

    zCMB = df['zCMB']
    # zCMB = Z[:,0]
    COVd = np.genfromtxt(string +'PP_' + name + '_COVd.txt')
    
    # =============================================================================================== 
    
    
    def GetMaxLikelihood(model, ip, H0, zmin = zmin, Z = df.to_numpy(), COVd = COVd, tolerance = tlrc):
        
        c = 299792.458 # Speed of light in units km/s

        # === READING INTERPOLATION TABLE ===
        interp = np.load( ip )    
        # ===================================  

        # === ORDERING DATA AND INTERPOLATION TABLE AFTER INCREASING CMB REDSHIFT ===
        zCMB = Z[:,0]
        # print(zCMB)
        # zCMB = zCMB.to_numpy()
        N = len(zCMB) ; # Number of SNe
        orderCMB = np.argsort(zCMB)   
        Z = Z[orderCMB,:]
        if model == 'Standardmodel':
            interp = interp[orderCMB,:,:]
        elif model == 'Timescape':
            interp = interp[orderCMB,:]
        orderCMB3 = np.vstack((3*orderCMB, 3*orderCMB+1, 3*orderCMB+2))
        orderCMB3 = orderCMB3.T
        orderCMB3 = orderCMB3.ravel()    
        COVd = COVd[orderCMB3,:] #sort rows
        COVd = COVd[:,orderCMB3] #sort columns in the same manner    
        # =========================================================================== 


        # ===================== APPLY REDSHIFT CUT IN CMB FRAME =====================
        imin = np.argmax(Z[:,0] >= zmin) # returns first index with True ie z>=zmin
        Z = Z[imin:,]                   # remove data below zmin
        # print(Z.shape)
        # print(type(Z))
        if model == 'Standardmodel':
            interp = interp[imin:,:,:]
        elif model == 'Timescape':
            interp = interp[imin:,:]  
        COVd = COVd[3*imin:,3*imin:]     # extracts cov matrix of the larger matrix
        N = N - imin                     # number of SNe left in sample after redshift cut
        
        # print('Number of SNe left in sample after redshift cut:', N)
        # print('\nNumber of SNe removed in redshift cut:  ', imin) 
        # ===========================================================================     
        
        # ============================= INTERPOLATION TABLES ARE SPLINED =============================
        # Note that the interpolation is two-dimensional in the standard model case and one-dimensional in the timescape case
        tempInt = [] ;
        if model == 'Standardmodel': 
        # Spline interpolation of luminosity distance. 
            for i in range(N):
                tempInt.append(interpolate.RectBivariateSpline( np.linspace(0,1.0,101), np.linspace(0,1.0,101) , interp[i]))
        	    
            def dL( OM, OL ): # Units of Mpc
                return np.hstack( [tempdL(OM,OL) for tempdL in tempInt] );   

            cov_params = [-1]	# Index of parameters for covariance matrix propagation 
            res_params = [2, 3, 4, 0, 1]	# Index of parameters for residual calculation 
    	
        elif model == 'Timescape': 
            array_input = np.linspace(0.00,0.99,100)
            array_input[0] = 0.001	
            # Spline interpolation of luminosity distance
            for i in range(N):
                tempInt.append(interpolate.InterpolatedUnivariateSpline( array_input , interp[i]))	
    	    
            def dL( OM ): # Units of Mpc
                return np.hstack( [tempdL(OM)*H0/c for tempdL in tempInt] );
    	       
            cov_params = [-1]	# Index of parameters for covariance matrix propergation 
            res_params = [1, 2, 3, 0]	# Index of parameters for residual calculation 
        # ============================================================================================
    	   
        def MU( *cosm_params ):  # Distance modulus
            dl_ = dL(*[ cosm_params[i] for i in range(len(cosm_params)) ])
            if np.min(dl_) <= 0:
                absval_dL = np.abs(dl_)
                mu_val = 10.**10  # Punishment for unphysical values of parameter space
            else:
                if trans_dL:
                    correction_term_helio = 5*np.log10( (1 + Z[:,6]) / (1 + Z[:,0]) )  # Transforming to the heliocentric frame
                else:
                    correction_term_helio = 0
                mu_val =  5*np.log10( c/H0 *  dl_ ) + 25  +  correction_term_helio
            return mu_val
        
        def COV(VM): # Total covariance matrix
            ATCOVlA = linalg.block_diag( *[ np.array([[VM, 0, 0], [0, 0, 0], [0, 0, 0]])  for i in range(N) ] )
            return np.array(COVd + ATCOVlA)
    	
        
        def RES( A , B , M0, *cosm_params ): # Residual
            #print(*cosm_params)
            Y0A = np.array([[ M0-A*Z[i, 2]+B*Z[i, 3], Z[i,2], Z[i,3]] for i in range(N) ]) 
            mu = MU(*[ cosm_params[i] for i in range(len(cosm_params)) ]) ;
            if model == 'Timescape':
                mu = MU(cosm_params[0] )
            if model == 'Standardmodel':
                mu = mu[0]
                
            zhat = np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) ) for i in range(N) ] )
            Y0A = np.hstack(Y0A)
            return zhat - Y0A
        
        def m2loglike(pars , cov_params_ = cov_params, res_params_ = res_params): # -2 * Ln(Likelihood)
            cov = COV( *[ pars[i] for i in cov_params_ ] )
            # cov = COVd
            try:
                chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True ) 
            except np.linalg.linalg.LinAlgError: # If not positive definite
                return +13993*10.**20 
            except ValueError: # If contains infinity
                return 13995*10.**20
            res = RES( *[ pars[i] for i in res_params_ ] )
            part_log = 3*N*np.log(2*np.pi) + np.nansum( np.log( np.diag( chol_fac[0] ) ) ) * 2
            part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )
    	
            # Boolean variables. True if parameters are outside valid region.
            # if model == 'Standardmodel':
            #     bool_pars = pars[0]<0 or pars[0]>1.5 or pars[1]<-.50 or pars[1]>1.5
            # elif model == 'Timescape':    
            #     bool_pars = pars[0]< 0.001 or pars[0]> 0.99
    	
            # if bool_pars:
            #     # if outside valid region, give penalty	
            #     part_exp += 10000* np.nansum(np.array([ _**2 for _ in pars ]))	
            m2loglike = part_log + part_exp
            return m2loglike 	 
        
        def m2CONSflat( pars ):  # Constraint flat universe
            return pars[0] + pars[1] - 1

        def m2CONSempt( pars ):  # Constraint empty universe
            return pars[0]**2 + pars[1]**2
    	
        
        # ========================== INITIAL GUESSES FOR OPTIMISATION ==========================
        # order of parameters: OM, OL, alpha, x, V_x, beta, c, V_c, M, V_M
        #if zmin > 0.04:
        pre_found_flat = np.array([0.38, 0.62, 0.118, 
                                    2.43, -19.36])
        # order of parameters: OM, OL, alpha, x, V_x, beta, c, V_c, M, V_M
        pre_found_empty = np.array([5.16659541e-11, 5.18656056e-06, 0.118, 
                                    2.43, -19])  
        # order of parameters: OM, alpha, x, V_x, beta, c, V_c, M, V_M
        pre_found_timescape = np.array([0.38, 1.18e-01, 2.43, 
                                        -19.33])
        # else:
        #     pre_found_flat = np.array(res_lambdaCDM[0].x)
        #     pre_found_empty = np.array(res_lambdaCDM[1].x)
        #     pre_found_timescape = np.array(res_timescape.x)
        # ======================================================================================
    
        # ==================================== OPTIMISATION ====================================
        
        bounds_lcdm = [(0, 1.5), (-0.5, 1.5), (0, 1), (0, 7), (-20.3, -18.3)]
        bounds_ts = [(0.001, 0.99), (0, 1), (0, 7), (-20.3, -18.3)]
        
        if model == 'Standardmodel': 
            MCEflat = optimize.minimize(m2loglike, pre_found_flat, 
                                        method = 'SLSQP', 
                                        constraints = ({'type':'eq', 'fun':m2CONSflat}, ), 
                                        bounds = bounds_lcdm,
                                        tol= tolerance)
            MCEempty = optimize.minimize(m2loglike, pre_found_empty, 
                                         method = 'SLSQP', 
                                         constraints = ({'type':'eq', 'fun':m2CONSempt}, ),
                                         tol= tolerance) 
            return np.array([MCEflat,MCEempty])	
        elif model == 'Timescape':
            MLE = optimize.minimize(m2loglike, pre_found_timescape, method = 'SLSQP', bounds = bounds_ts, tol= tolerance)  
            return MLE	
        # ======================================================================================
    
    def AIC(m2loglike, k): # AIC measure for computing AIC ratio
        return 2*k + m2loglike 
    
    res_lambdaCDM = GetMaxLikelihood(model = 'Standardmodel', ip = interpolation_standardmodel, H0 = 66.7)
    res_timescape = GetMaxLikelihood( model = 'Timescape', ip = interpolation_ts, H0 = 61.7)
    
    prob_AIC_Flat = np.exp(-0.5 * AIC(res_lambdaCDM[0].fun, 9)  )
    prob_AIC_Empt = np.exp(-0.5 * AIC(res_lambdaCDM[1].fun, 8)  )
    prob_AIC_TS = np.exp(-0.5 * AIC(res_timescape.fun, 9)  )
    
    probflat = np.exp(-0.5 * res_lambdaCDM[0].fun)
    probempty = np.exp(-0.5 * res_lambdaCDM[1].fun)
    probtimescape = np.exp(-0.5 * res_timescape.fun)

    AIC = prob_AIC_TS/prob_AIC_Flat
    logl_lcdm.append(res_lambdaCDM[0].fun)
    logl_ts.append(res_timescape.fun)
    aic.append(AIC)
    ts.append(res_timescape.x)
    lcdm.append(res_lambdaCDM[0].x)
    milne.append(res_lambdaCDM[1].x)
    
ts = np.column_stack(ts)
lcdm = np.column_stack(lcdm)
milne = np.column_stack(milne)
aic = np.column_stack(aic)
logl_ts = np.column_stack(logl_ts)
logl_lcdm = np.column_stack(logl_lcdm)

freq_string = 'Pantheon/Build/'
string2 = 'PP_' + name + '_tripp_ext_'
np.savetxt(freq_string + string2 + 'TS.txt', ts, delimiter = '\t')
np.savetxt(freq_string + string2 + 'LCDM.txt', lcdm, delimiter = '\t')
np.savetxt(freq_string + string2 + 'Milne.txt', milne, delimiter = '\t')
np.savetxt(freq_string + string2 + 'aic.txt', aic, delimiter = '\t')
np.savetxt(freq_string + string2 + 'logl_lcdm.txt', logl_lcdm, delimiter = '\t')
np.savetxt(freq_string + string2 + 'logl_ts.txt', logl_ts, delimiter = '\t')

# print("This took {0:.02f} hours to calculate".format((time.time() - start)/3600))

print()
print("This took {0:.02f} hours to calculate".format((time.time() - start)/3600))
print()

# %%
