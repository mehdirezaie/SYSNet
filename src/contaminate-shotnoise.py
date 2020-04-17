

import fitsio as ft
import numpy  as np
import healpy as hp
import os
import sys

class mock(object):
    def __init__(self, featsfile, paramsfile, func='lin', sf=1207432.7901):
        # read inputs
        feats       = ft.read(featsfile)
        params      = np.load(paramsfile).item()
        # attrs
        self.hpix   = feats['hpind']
        self.feats  = feats['features']
        self.axfit     = params['axfit']
        self.xstats = params['xstats']
        #print('Will scale the covariance by %.4f'%sf)
        bfp_raw     = params['params'][func]
        self.bfp    = (bfp_raw[0], sf*bfp_raw[1])

        #
        # prepare
        self.n   = self.feats.shape[0]
        x        = (self.feats - self.xstats[0])/self.xstats[1] # select axis
        x_scaled = x[:, self.axfit]
        if func == 'lin':
            x_vector = np.column_stack([np.ones(self.n), x_scaled])
        elif func == 'quad':
            x_vector = np.column_stack([np.ones(self.n), x_scaled, x_scaled*x_scaled])
        else:
            exit(f"func:{func} is not defined")
        #
        # 
        self.x_vector = x_vector

    def simulate(self, kind='truth', seed=12345):
        if kind not in ['fixed', 'random', 'truth']:
            exit(f"kind : {kind} is not defined")
        np.random.seed(seed) # set the seed

        if kind == 'truth':
            thetas = self.bfp[0]
        elif kind == 'fixed':
            thetas = np.random.multivariate_normal(*self.bfp)
        elif kind == 'random':
            thetas = np.random.multivariate_normal(*self.bfp, size=self.n)
        else:
            exit(f"kind : {kind} is not defined")

        tx       = (thetas * self.x_vector)
        self.txs = np.sum(tx, axis=1)

    def project(self, hpin, tag):
        hpmin = hp.read_map(hpin, verbose=False)
        fpath = '/'.join((hpin.split('/')[:-1] + [tag]))
        mname = '_'.join((tag, 'mask',hpin.split('/')[-1]))
        fname = '_'.join((tag, hpin.split('/')[-1]))
        if not os.path.exists(fpath):
            os.makedirs(fpath)
         
        ngalcont = self.txs * hpmin[self.hpix]  
        fou = '/'.join((fpath, fname))
        mou = '/'.join((fpath, mname))
        
        ngal_neg   = ngalcont < 0.0
        hpix_neg   = self.hpix[ngal_neg]
        hpix_noneg = self.hpix[~ngal_neg]
        ngal_noneg = ngalcont[~ngal_neg]
        #
        #
        ngalm      = np.zeros_like(hpmin)
        ngalm[hpix_noneg] = np.random.poisson(ngal_noneg)
        #
        #
        negm       = np.zeros_like(hpmin)
        negm[hpix_neg]  = 1.0
        hp.write_map(mou, negm,  fits_IDL=False, overwrite=True, dtype=np.float64)
        hp.write_map(fou, ngalm, fits_IDL=False, overwrite=True, dtype=np.float64)
        print('%s is written'%fou) 

        
if __name__ == '__main__':    
    np.random.seed(123456) # set the global seed        
    seeds = np.random.randint(0, 4294967295, size=1000)        
    feats = sys.argv[1]
    regp  = sys.argv[2]    
    files = sys.argv[3:]
    
    print('feats', feats)
    print('regp',  regp)
    print('files[:2]', files[:2])
    
    for i,mock_i in enumerate(files):
        mymock  = mock(feats, 
                       regp,
                       func='lin', sf=23765.2929*0.05) # 0.1XtotalfracXvarngal = 2376.52929
        mymock.simulate(kind='random', seed=seeds[i])
        mymock.project(mock_i, 'cp2p')
