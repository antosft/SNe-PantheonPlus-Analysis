from scipy import interpolate
import numpy as np

class spl(object):
    def __init__(self, name, Ntotal):
        self.name = name
        self.Ntotal = Ntotal

        interp = np.load('Pantheon/Build/PP_' + self.name + '_tabledL_ts.npy')
        oms = np.linspace(0.00,0.99,100)
        oms[0] = 0.001
        self.ts = []
        for i in range(self.Ntotal):
            self.ts.append(interpolate.InterpolatedUnivariateSpline(oms, interp[i]))

        interpu = np.load('Pantheon/Build/PP_' + self.name + '_tabledL_lcdm.npy')
        oms = np.linspace(0,1.0,101)
        ols = np.linspace(0,1.0,101)
        self.lcdm = []
        for i in range(self.Ntotal):
            self.lcdm.append(interpolate.RectBivariateSpline(oms, ols, interpu[i]))
