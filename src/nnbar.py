import numpy as np
from healpy import get_nside, nside2pixarea 


class NNBAR(object):
    """
    INPUTS:
    galmap, ranmap, mask,
    sysmap, bins, selection=None
    """

    def __init__(self, galmap, ranmap, mask, 
                       sysmap, bins, selection=None):
        #
        # inputs
        self.nside  = 256#get_nside(galmap)
        self.galmap = galmap[mask]
        self.ranmap = ranmap[mask]
        self.sysmap = sysmap[mask]
        #
        # selection on galaxy map
        if selection is not None:
            self.galmap /= selection[mask]
        #    
        # digitize
        self.sysl   = [0 for k in range(2*bins.size)]
        inds = np.digitize(self.sysmap, bins)
        for i in range(1,bins.size): # what if there is nothing on the last bin? FIXME
            self.sysl[2*i-2] = self.galmap[np.where(inds == i)].tolist()
            self.sysl[2*i-1] = self.ranmap[np.where(inds == i)].tolist()    
        self.avnden = np.sum([np.sum(self.sysl[i]) for i in np.arange(0,2*bins.size, 2)])\
                      /np.sum([np.sum(self.sysl[i]) for i in np.arange(1,2*bins.size, 2)])
        self.bins = bins
        
    def run(self, njack=20):
        sl = []
        ml = []
        nl = []
        for i in range(0, 2*self.bins.size-2, 2):
            ng   = 0.0            
            std  = 0.0
            npix = 0.0
            for j in range(0,len(self.sysl[i])):
                ng   += self.sysl[i][j]
                npix += self.sysl[i+1][j]
            if npix == 0.0:
                ml.append(np.nan)
                nl.append(np.nan)
                sl.append(np.nan)
                continue
            mean = ng/npix/self.avnden
            ml.append(mean)
            nl.append(npix)
            if len(self.sysl[i]) < njack:
                for k in range(0,len(self.sysl[i])):
                    std += (self.sysl[i][k]/self.sysl[i+1][k]/self.avnden-mean)**2.
                std = np.sqrt(std)/(len(self.sysl[i])-1.)
            else:
                jkf = len(self.sysl[i])//njack
                for k in range(0,njack):
                    ng   = 0
                    npix = 0
                    minj = jkf*k
                    maxj = jkf*(k+1)
                    for j in range(0,len(self.sysl[i])):
                        if j < minj or j >= maxj:
                            ng   += self.sysl[i][j]
                            npix += self.sysl[i+1][j]
                    mj = ng/npix/self.avnden
                    std += (mj-mean)**2.
                std = np.sqrt((njack-1.)/float(njack)*std)
            sl.append(std)
        #
        # area
        npixtot   = self.ranmap.size
        nrantot   = self.ranmap.sum()
        area1pix  = nside2pixarea(self.nside, degrees=True)
        npix2area = npixtot*area1pix/nrantot
        #
        # prepare output
        output   = {}
        output['nnbar']      = np.array(ml)
        output['area']       = np.array(nl) * npix2area
        output['nnbar_err']  = np.array(sl)
        output['bin_edges']  = self.bins
        attrs = {}
        attrs['njack']     = njack
        attrs['nbar']      = self.avnden
        attrs['nside']     = self.nside
        attrs['npix2area'] = npix2area

        output['attr'] = attrs
        self.output = output
    def save(self, path4output):
        print('writing the output in {}'.format(path4output))
        np.save(path4output, self.output)
