from scipy import interpolate, linalg, optimize, stats, integrate
import numpy as np
import sys
import time

from loadsplines import spl


class loglike(object):

    def __init__(self,model,zdep,case,ipars,zcmb,zhel,tri,snset,COVd,tempInt):
        self.model = model
        self.zdep = zdep		
        self.case = case
        self.ipars = ipars
        self.zcmb = zcmb
        self.zhel = zhel
        self.tri = tri
        self.snset = snset
        self.COVd = COVd                
        self.tempInt = tempInt
        self.N = zcmb.shape[0]
        self.snid = {'lowz':50, 'SDSS':1, 'SNLS':4, 'HST':100}

    def __call__(self,cube,ndim,nparams):

        cube_aug = np.zeros(21)
        for icube,i in enumerate(self.ipars):
            cube_aug[i] = cube[icube]

        # broadcast to all parameters. if model=ts then Q=fv0 otherwise Q=Omega_M0
        (self.Q, self.A, self.X0, self.VX, self.B, self.C0, self.VC, self.M0, self.VM,
        self.X0_2, self.C0_2, self.X0_3, self.C0_3, self.X0_4, self.C0_4,
        self.X1, self.C1, self.X2, self.C2, self.X3, self.C3) = cube_aug

        self.X4 = 0.0
        self.C4 = 0.0 # set to zero unless doing global linear

        if self.case == 2:
            """
            global linear x, global const c
            """
            self.X0_2 = self.X0_3 = self.X0_4 = self.X0
            self.X4 = self.X3 = self.X2 = self.X1
            self.C0_2 = self.C0_3 = self.C0_4 = self.C0	
        elif self.case == 3:
            """
            split linear x, global const c
            """
            self.C0_2 = self.C0_3 = self.C0_4 = self.C0
        elif self.case == 4:
            """
            global const x, global linear c
            """
            self.X0_2 = self.X0_3 = self.X0_4 = self.X0
            self.C0_2 = self.C0_3 = self.C0_4 = self.C0
            self.C4 = self.C3 = self.C2 = self.C1
        elif self.case == 5:
            """
            global const x, split linear c
            """
            self.X0_2 = self.X0_3 = self.X0_4 = self.X0
        elif self.case == 6:
            """ 
            global linear x and c
            """
            self.X0_2 = self.X0_3 = self.X0_4 = self.X0
            self.X4 = self.X3 = self.X2 = self.X1
            self.C0_2 = self.C0_3 = self.C0_4 = self.C0
            self.C4 = self.C3 = self.C2 = self.C1
        elif self.case == 8:
            """
            global linear x, split linear c
            """
            self.X0_2 = self.X0_3 = self.X0_4 = self.X0
            self.X4 = self.X3 = self.X2 = self.X1

        return -0.5 * self.m2loglike()

    def dL(self):
        if self.model == 1:
            fv0 = self.Q
            OM = 0.5*(1.-fv0)*(2.+fv0)
            dist = (61.7/66.7) * np.hstack([tempdL(OM) for tempdL in self.tempInt])
        elif self.model == 2 or self.model == 3:
            OM = self.Q
            OL = 1.0 - OM
            H0 = 66.7 # km/s/Mpc # ZAC CHANGED
            c = 299792.458
            dist = c/H0 * np.hstack([tempdL(OM,OL) for tempdL in self.tempInt])
        if (dist<0).any():
            print('OM, z: ', OM, np.argwhere(dist<0))
        return dist

    def MU(self):
        mu_ = 5*np.log10(self.dL() * (1+self.zhel)/(1+self.zcmb)) + 25
        return mu_.flatten()

    def COV(self): # Total covariance matrix
        block3 = np.array([[self.VM + self.VX*self.A**2 + self.VC*self.B**2,-self.VX*self.A, self.VC*self.B],
                   [-self.VX*self.A, self.VX, 0.],
                   [ self.VC*self.B, 0., self.VC]])
        ATCOVlA = linalg.block_diag(*[ block3 for i in range(self.N)])
        return self.COVd + ATCOVlA

    def RES(self): 
        """
        Total residual, \hat Z - Y_0*A
        """
        if self.zdep == True:
            X0_red = np.where(self.snset==self.snid['lowz'], self.X0 + self.X1*self.zhel, self.zhel)
            X0_red = np.where(self.snset==self.snid['SDSS'], self.X0_2 + self.X2*X0_red, X0_red)
            X0_red = np.where(self.snset==self.snid['SNLS'], self.X0_3 + self.X3*X0_red, X0_red)
            X0_red = np.where(self.snset==self.snid['HST'],  self.X0_4 + self.X4*X0_red, X0_red)

            C0_red = np.where(self.snset==self.snid['lowz'], self.C0 + self.C1*self.zhel, self.zhel)
            C0_red = np.where(self.snset==self.snid['SDSS'], self.C0_2 + self.C2*C0_red, C0_red)
            C0_red = np.where(self.snset==self.snid['SNLS'], self.C0_3 + self.C3*C0_red, C0_red)
            C0_red = np.where(self.snset==self.snid['HST'],  self.C0_4 + self.C4*C0_red, C0_red)

            Y0A = np.column_stack((self.M0-self.A*X0_red+self.B*C0_red, X0_red, C0_red)) # 2D-array
        elif self.zdep == False:
            Y0A = np.array([self.M0-self.A*self.X0+self.B*self.C0, self.X0, self.C0]) # 1D-array
            Y0A = np.tile(Y0A, (self.N,1)) # form Nx3 2D-array

        mu = self.MU()
        return np.hstack([(self.tri[i] - np.array([mu[i],0,0]) - Y0A[i]) for i in range(self.N)])

    def m2loglike(self):
        cov = self.COV()
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


def prior(cube,ndim,nparams):
    """
    Transform parameters in the ndimensional 
    cube to the physical parameter space
    (see arXiv:0809.3437)

    """
    for icube, i in enumerate(ipars):
        lo, hi = prior_lims[i]
        cube[icube] = (hi-lo)*cube[icube] + lo

    cube[3] = 10.**cube[3]
    cube[6] = 10.**cube[6]
    cube[8] = 10.**cube[8]


def getindices(zdep,case):
    """
    Assign indices to non-zero light curve parameters
    """
    ip = list(range(9))
    if zdep == 1:
        if case == 2:
            iadd = [15]
        elif case == 3:
            iadd = [9,11,13,15,17,19]
        elif case == 4:
            iadd = [16]
        elif case == 5:
            iadd = [10,12,14,16,18,20]
        elif case == 6:
            iadd = [15,16]
        elif case == 7:
            iadd = range(9,21)
        elif case == 8:
            iadd = [10,12,14,15,16,18,20]
    elif zdep == 0:
        iadd = []
    else:
        raise ValueError('third argument must be either 1 (z dependence) or 0 (no z dependence)')
    return ip + iadd

def sort_and_cut(zmin,Zdata,covmatrix,splines):
    """
    Sort by increasing redshift then cut all sn below zmin

    """
    N0 = Zdata.shape[0]

    # Sort data in order of increasing redshift
    ind = np.argsort(Zdata[:,0])
    ind = np.intersect1d(ind, np.arange(len(splines))) # ANTOCHANGE
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


def runbayes(runinput):

    c = 299792.458 # km/s

    # Prior limits
    prior_lims = [ # fv: 0.5*(np.sqrt(9.-8.*om)-1)
               (0.0, 1.0),     #alpha
               (-20.0, 20.0),  #X0
               (-10.0, 4.0),   #lVX
               (0.0, 4.0),     #beta
               (-20.0, 20.0),  #C0
               (-10.0, 4.0),   #lVC
               (-20.3, -18.3), #M0
               (-10.0, 4.0),   #lVM
               (-20.0, 20.0),  #X0_2
               (-20.0, 20.0),  #C0_2
               (-20.0, 20.0),  #X0_3
               (-20.0, 20.0),  #C0_3
               (-20.0, 20.0),  #X0_4
               (-20.0, 20.0),  #C0_4
               (-20.0, 20.0),  #X1
               (-20.0, 20.0),  #C1
               (-20.0, 20.0),  #X2
               (-20.0, 20.0),  #C2
               (-20.0, 20.0),  #X3
               (-20.0, 20.0) ] #C3

    model = int(runinput[1])    # 1=Timescape, 2=Empty, 3=Flat
    z_cut = float(runinput[2])  # redshift cut e.g. 0.033
    zdep = int(runinput[3])     # redshift dependence (0 or 1)
    case = int(runinput[4])     # redshift light curve model (1-8)
    isigma = int(runinput[5])   # 1, 2 or 3 sigma omega/fv prior
    nlive = int(runinput[6])    # number of live points used in sampling
    tol = float(runinput[7])    # stop evidence integral when next contribution less than tol
    versionname = runinput[8]

    # load data and covariances
    Z = np.loadtxt('Pantheon/Build/PP_' + versionname + '_input.txt') 
    COVd = np.loadtxt('Pantheon/Build/PP_' + versionname + '_COVd.txt')
    Ntotal = Z.shape[0]   # Number of SNe
    print(versionname, Ntotal)
    
    basename = 'Pantheon_' + str(model) + '_' + str(z_cut) + '_' + str(zdep) + '_' + str(case) + '_' + str(isigma) + '_' \
        + str(nlive) + '_' + str(tol) + '_'
    folder = ''

    if model == 1:
        p1prior = [(0.588,0.765), (0.500,0.799), (0.378,0.826), (0.001,0.999)] #fv0
        tempInt = spl(versionname, Ntotal).ts
        folder = 'Timescape/'
    elif model == 2 or model == 3:
        p1prior = [(0.162,0.392), (0.143,0.487), (0.124,0.665), (0.001,0.999)] #om
        tempInt = spl(versionname, Ntotal).lcdm
        folder = 'LCDM/'
        if model == 3:
            folder = 'Milne/'
    else:
        raise ValueError('command line arguments allowed: 1=Timescape, 2=empty, 3=Flat LCDM')

    Z, COVd, tempInt = sort_and_cut(z_cut,Z,COVd,tempInt)
    ipars = getindices(zdep,case)
    prior_lims = [p1prior[isigma-1],] + prior_lims # add fv/om prior
    ndim = len(ipars)

    zcmb = Z[:,0]
    zhel = Z[:,6]
    tri = Z[:,1:4] 
    snset = Z[:,5]

    llike = loglike(model,zdep,case,ipars,zcmb,zhel,tri,snset,COVd,tempInt)
    def llikenargs(cube, ndim, nparams): # needed for pymultinest to get len(inspect.getfullargspec(LogLikelihood).args) correctly without counting self in the __call__ function
        return llike(cube, ndim, nparams)


    # compute evidence 

    try:
        import pymultinest
    except ImportError:
        raise
    multistart = time.time()
    print('start multinest after {0:.02f}min'.format((multistart - start)/60))
    pymultinest.run(llikenargs, prior, ndim, 
            outputfiles_basename = 'Pantheon/output'+versionname+'/'+folder+basename,
            multimodal = False,
            sampling_efficiency = 'model',
            n_live_points = nlive,
            evidence_tolerance = tol,
            const_efficiency_mode = False,
            # resume = False,
            n_iter_before_update = 20000,
            verbose = False)

    print('took {0:.02f}min for multinest'.format((time.time() - multistart)/60))

if __name__ == '__main__':
    start = time.time()

    runbayes(sys.argv)

    print('finish after {0:.02f}min'.format((time.time() - start)/60))
