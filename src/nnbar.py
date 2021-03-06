import numpy as np
from healpy import get_nside, nside2pixarea 


class NNBAR(object):
    """
    INPUTS:
    galmap, ranmap, mask,
    sysmap, bins, selection=None
    """

    def __init__(self, galmap, ranmap, mask, 
                       sysmap, nbins=20, selection=None, binning='equi-area'):
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
        if binning == 'simple':
            bins = np.linspace(self.sysmap.min(), self.sysmap.max(), nbins+1) 
            self.sysl   = [0 for k in range(2*nbins)]
            inds = np.digitize(self.sysmap, bins)
            for i in range(1,bins.size): # what if there is nothing on the last bin? FIXME
                self.sysl[2*i-2] = self.galmap[np.where(inds == i)].tolist()
                self.sysl[2*i-1] = self.ranmap[np.where(inds == i)].tolist()    
        elif binning == 'equi-area':
            npts  = self.ranmap.size
            swtt  = self.ranmap.sum()/nbins  # num of randoms in each bin
            datat = np.zeros(self.sysmap.size, dtype=np.dtype([('ss', 'f8'), ('gs', 'f8'), ('ws', 'f8'), ('rid', 'i8')]))
            datat['ss'] = self.sysmap
            datat['gs'] = self.galmap
            datat['ws'] = self.ranmap
            #datat['rid'] = np.random.choice(np.arange(self.sysmap.size), size=self.sysmap.size, replace=False)
            datat['rid'] = np.random.permutation(np.arange(self.sysmap.size)) # 2x faster
            datas = np.sort(datat, order=['ss', 'rid'])
            ss, gs, ws = datas['ss'], datas['gs'], datas['ws']
            #ss, gs, ws = zip(*sorted(zip(self.sysmap, self.galmap, self.ranmap)))
            swti = 0.0
            i   = 0
            self.sysl = [0 for k in range(2*nbins)] 
            listg = []
            listr = []
            bins  = [ss[0]] # first edge is the lowest systematic
            j     =  0
            for wsi in ws:
                swti += wsi
                listg.append(gs[i])
                listr.append(ws[i])
                if (swti >= swtt) or (i == npts-1):
                    swti  = 0.0
                    bins.append(ss[i])
                    self.sysl[2*j]   = listg
                    self.sysl[2*j+1] = listr
                    listg = []
                    listr = []
                    j += 1
                i += 1
            bins = np.array(bins)
            print('min sys : %.2f  max sys : %.2f'%(ss[0], ss[npts-1]))
            print('num of pts : %d, num of bins : %d'%(i, j))
        self.avnden = np.sum([np.sum(self.sysl[i]) for i in np.arange(0,2*nbins, 2)])\
                      /np.sum([np.sum(self.sysl[i]) for i in np.arange(1,2*nbins, 2)])
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
            if (len(self.sysl[i]) < njack) or (njack == 0):  # use the typical std if njack is 0
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
