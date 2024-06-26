
from __future__ import division
# import pymultinest
# print('got pymultinest')

from scipy import linalg
import numpy as np
import pandas as pd
import sys
import time
import multiprocessing
from mpi4py import MPI
from spline_pipe import spl
import os



class loglike(object):

    def __init__(self,model,ipars,zcmb,zhel, tri, COVd, tempInt):
        self.model = model
        self.ipars = ipars
        self.zcmb = zcmb
        self.zhel = zhel
        self.tri = tri

        self.X = tri[:,1]
        self.C = tri[:,2]

        self.dims = len(ipars)

        self.COVd = COVd                
        self.tempInt = tempInt
        self.N = zcmb.shape[0]

    def __call__(self,cube,ndim,nparams):

        cube_aug = np.zeros(self.dims)
        for icube,i in enumerate(self.ipars):
            cube_aug[i] = cube[icube]

        # broadcast to all parameters. if model=ts then Q=fv0 otherwise Q=Omega_M0
        (self.Q, self.A, self.B, self.M, self.VM) = cube_aug

        return -0.5 * self.m2loglike()
        
    def dL(self):
        if self.model == 1:
            fv0 = self.Q
            OM = 0.5*(1.-fv0)*(2.+fv0)
            dist = (61.7/66.7) * np.hstack([tempdL(OM) for tempdL in self.tempInt])
        elif self.model == 2:
            OM = self.Q
            OL = 1.0 - OM
            H0 = 66.7 # km/s/Mpc
            c = 299792.458
            dist = c/H0 * np.hstack([tempdL(OM,OL) for tempdL in self.tempInt])
        if (dist<0).any():
            print('OM, z: ', OM, np.argwhere(dist<0))
        return dist

    def MU(self):
        mu_ = 5*np.log10(self.dL() * (1+self.zhel)/(1+self.zcmb)) + 25
        return mu_.flatten()

    def COV(self): # Total covariance matrix
        ATCOVlA = linalg.block_diag( *[ np.array([[self.VM, 0, 0], [0, 0, 0], [0, 0, 0]])  for i in range(self.N) ] )
        return np.array(self.COVd + ATCOVlA)

    def RES(self): 
        """
        Total residual, \hat Z - Y_0*A
        """

        Y0A = np.array([[ self.M-self.A*self.X[i]+self.B*self.C[i], self.X[i], self.C[i]] for i in range(self.N) ]) 

        mu = self.MU()

        zhat = np.hstack( [ (self.tri[i] -np.array([mu[i],0,0]) ) for i in range(self.N) ] )
        Y0A = np.hstack(Y0A)
        return zhat - Y0A

    def m2loglike(self):
        cov = self.COV()
        # cov = self.COVd
        try:
            chol_fac = linalg.cho_factor(cov, overwrite_a=True, lower=True) 
        except linalg.LinAlgError: # when not positive definite
            return 13993.*1e20
        except ValueError:
            return 13995.*1e20
        res = self.RES()

        part_log = 3*self.N*np.log(2*np.pi) + np.sum(np.log(np.diag(chol_fac[0])))*2
        part_exp = np.dot(res, linalg.cho_solve(chol_fac, res))

        return part_log + part_exp
	
def sort_and_cut(zmin,Zdata,covmatrix,splines):
	"""
	Sort by increasing redshift then cut all sn below zmin

	"""
	N0 = Zdata.shape[0]

	# Sort data in order of increasing redshift
	ind = np.argsort(Zdata[:,0])
	Zdata = Zdata[ind,:]
	splines = [splines[i] for i in ind]
	ind = np.column_stack((3*ind, 3*ind+1, 3*ind+2)).ravel()
	covmatrix = covmatrix[ind,:]
	covmatrix = covmatrix[:,ind]

	# Redshift cut
	imin = np.argmax(Zdata[:,0] >= zmin)    # returns first index with True ie z>=zmin
	Zdata = Zdata[imin:,:]                  # remove data below zmin
	covmatrix = covmatrix[3*imin:,3*imin:]  # extracts cov matrix of the larger matrix
	N = N0 - imin                           # number of SNe in cut sample
	splines = splines[imin:]                # keep splines of remaining snia only

	return Zdata, covmatrix, splines


def process_z_cut(z_cut, model, isigma, nlive, tol, folder, Z, COVd, Ntotal, prior_lims, host_corr, start, name=''):
        # output files (in output folder) : Pantheon_model_redshiftcut_0_1_2_1000_tolerance

        if model == 1:
            basename = 'Timescape/PP_' + name + '_TS_' + str(z_cut) + '_' + str(isigma) + '_' + str(nlive) + '_' + str(tol) + '_'
            p1prior = [(0.588,0.765), (0.500,0.799), (0.378,0.826), (0.001,0.999)] #fv0
            tempInt = spl(name, Ntotal).ts
        elif model == 2 or model == 3:
            basename = 'LCDM/PP_' + name + '_LCDM_' + str(z_cut) + '_' + str(isigma) + '_' + str(nlive) + '_' + str(tol) + '_'
            p1prior = [(0.162,0.392), (0.143,0.487), (0.124,0.665), (0.001,0.999)] #om
            tempInt = spl(name, Ntotal).lcdm
        else:
            raise ValueError('command line arguments allowed: 1 = Timescape, 2 = spatially flat LCDM')


        Z, COVd, tempInt = sort_and_cut(z_cut,Z,COVd,tempInt)

        prior_lims = [p1prior[isigma-1],] + prior_lims # add fv/om prior
        ipars = list(range(len(prior_lims)))

        # print(prior_lims)
        ndim = len(ipars)

        def prior(cube,ndim,nparams):
            """
            Transform parameters in the ndimensional 
            cube to the physical parameter space
            (see arXiv:0809.3437)

            """
            for icube, i in enumerate(ipars):
                lo, hi = prior_lims[i]
                cube[icube] = (hi-lo)*cube[icube] + lo
                
            cube[4] = 10.**cube[4]


        zcmb = Z[:,0]
        zhel = Z[:,6]
        tri = Z[:,1:4] 
        # snset = Z[:,5]

        vx = Z[:,-3]
        vc = Z[:,-2]
        cov_term = Z[:,-1]

        llike = loglike(model,ipars,zcmb,zhel, tri,COVd,tempInt)
        def llikenargs(cube, ndim, nparams): # needed for pymultinest to get len(inspect.getfullargspec(LogLikelihood).args) correctly without counting self in the __call__ function
            return llike(cube, ndim, nparams)


	# compute evidence 

        try:
            import pymultinest
        except :
            raise ImportError('pymultinest not found')

        pymultinest.run(llikenargs, prior, ndim, 
                        outputfiles_basename = folder+basename, 
                        multimodal = False, 
                        sampling_efficiency = 'model', 
                        n_live_points = nlive,
                        evidence_tolerance = tol,
                        const_efficiency_mode = False,
                        # resume = False,
                        n_iter_before_update = 20000,
                        verbose = False)
        
        # send_email(basename, start)
        print(f"Finished running {basename} in {(time.time()-start)/3600:.2f} hours")


if __name__ == '__main__':
    
    name = sys.argv[1]   # input file: 'Pantheon/Build/PP_' + versionname + '_input.txt' (see spline_pipe.py)
    
    start = time.time()
    c = 299792.458 # km/s

    # Prior limits
    prior_lims = [ # fv: 0.5*(np.sqrt(9.-8.*om)-1)
                (0.0, 1.0),     # alpha
                (0.0, 7.0),     # beta
                (-20.3, -18.3),  # M
                (-10.0, 4.0)]   # Variance M

    # cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    models = [1, 2]              # 1=Timescape, 2=Flat
    # models = [1]
    z_cuts = np.linspace(0, 0.1, 21)  # redshift cut e.g. 0.033
    # z_cuts = [0.25]
    isigma = int(2)             # 1, 2 or 3 sigma omega/fv prior
    nlive = int(800)           # number of live points used in sampling
    tol = float(1e-3)           # stop evidence integral when next contribution less than tol
    # tol = float(1)
    folder = 'Pantheon/Build/'
    save_folder = 'Pantheon/Build/tripp/Bayes/'
    save_folder = 'outputpipe'
    try:
        save_folder = save_folder + '/' + sys.argv[2]
    except:
        save_folder = save_folder

    df = pd.read_csv(folder + 'PP_' + name + '_input.csv') # ZAC CHANGED THIS
    df = df.drop(columns = 'Unnamed: 0')
    
    host = df['HOST_LOGMASS'].to_numpy()

    host_mask = host > 10

    host_corr = np.zeros_like(host)
    host_corr[host_mask] = 1/2
    host_corr[~host_mask] = -1/2


    Z = df.to_numpy()
    COVd = np.loadtxt(folder + 'PP_' + name + '_COVd.txt') # ZAC CHANGED THIS
    Ntotal = Z.shape[0]
    
    pool = multiprocessing.Pool(processes = 42)

    args_list = [(z_cut, model, isigma, nlive, tol, save_folder, Z, COVd, 
                  Ntotal, prior_lims, host_corr, start, name) for z_cut in z_cuts for model in models]

    # Map the function to process each z_cut in parallel
    pool.starmap(process_z_cut, args_list)

    # Close the pool to free resources
    pool.close()
    pool.join()