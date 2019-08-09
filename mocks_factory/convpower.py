'''
   utility to compute the convoluted power spectrum from a cut-sky log-normal mock 
   with Nbodykit (c) Nick Hand, Yu Feng
   see http://nbodykit.readthedocs.io/en/stable/catalogs/mock-data.html#Log-normal-Mocks
'''

# 
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import numpy as np
from time import time
import fitsio as ft
from argparse import ArgumentParser
np.random.seed(1234)
import sys
sys.path.append('/global/homes/m/mehdi/github/DESILSS')
from syslss import powerspectrum

class CAT(object):
    def __init__(self, ra, dec, z):        
        self.RA  = ra
        self.DEC = dec
        self.Z   = z
        self.Weight = 1.0
        
        
        
ap = ArgumentParser(description='Power Spectrum Calculator using Nbodykit')
ap.add_argument('--data', default='3dbox_nmesh1024_L5274.0_bias1.5_seed1.cat')
ap.add_argument('--output', default='pkcut_3dbox_nmesh1024_L5274.0_bias1.5_seed1.npy')
# ap.add_argument('--random', default='/global/cscratch1/sd/mehdi/mocks/3dbox_1024_5274_p85cat_random.dat')
ap.add_argument('--random', default='/global/cscratch1/sd/mehdi/mocks/mock.ELGRAN.fits')
ap.add_argument('--nzfile', default='/global/cscratch1/sd/mehdi/mocks/eboss-ngc-anand2017.zsel')
ap.add_argument('--boxsize', type=float, default=5274)
ap.add_argument('--nmesh', type=int, default=256)
ap.add_argument('--P0fkp', type=float, default=6000.)
ap.add_argument('--zbin', nargs='*', type=float, default=[0.55,1.5])
ns = ap.parse_args()
#



t1    = time()
nz = np.loadtxt(ns.nzfile)
NZ = IUS(nz[:,0], 1.e-4*nz[:,1])

#
data   = np.loadtxt(ns.data)
random = ft.read(ns.random)
#
DATA   = CAT(data[:,0], data[:,1], data[:,2])
RANDOM = CAT(random['ra'], random['dec'], np.random.choice(data[:,2], size=random['ra'].size))

t2 = time()
print('time to read the file', t2-t1)
cutskypk = powerspectrum(DATA, RANDOM, NZ, universe_params={'Om0':.29, 'H0':71, 'flat':True})
pkcutsky = cutskypk.run(ns.zbin, ns.nmesh, P0fkp=ns.P0fkp)
np.save(ns.output, pkcutsky)
print('finished in {} s'.format(time()-t2))
