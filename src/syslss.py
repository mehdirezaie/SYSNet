#
#   Module to do Clustering Cell and Omega(theta)
# April 29, 2020 -- this code is pretty old, checkout utils from LSSutils 
# DEC 29, 2017 -- fsky is estimated within powersepctrum
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import fitsio as ft
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline as iusp
from tools import nzhist, write
from counter import paircount


#try:
#    from counter import paircount
#except:
#    raise Warning("You need to compile the counter module")
#    paircount = lambda x : x


# nbodykit stuff
from nbodykit.transform import SkyToCartesian
from nbodykit.cosmology import Cosmology
import nbodykit.lab as nb
#from nbodykit.algorithms.survey_paircount import AngularPairCount, SurveyDataPairCount

# nbodykit computing class
# parameters used in data challenge DESI
# https://desi.lbl.gov/trac/wiki/ClusteringWG/DataChallengeTasks/AnalysisQuickSurvey
universeparams = {'Om0': 0.31,'H0':69.0,'flat':True}


# see https://stackoverflow.com/questions/29702010/healpy-pix2ang-convert-from-healpix-index-to-ra-dec-or-glong-glat
def radec2hpix(nside, ra, dec):
    pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    return pix

# def IndexToDeclRa(index):
#     theta,phi=hp.pixelfunc.pix2ang(NSIDE,index)
#     return -np.degrees(theta-pi/2.),np.degrees(pi*2.-phi) # phi?


def hpixsum(nside, ra, dec, value=None, nest=False):
    '''
        make a healpix map from ra-dec
        hpixsum(nside, ra, dec, value=None, nest=False)
    '''
    pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra), nest=nest)
    npix = hp.nside2npix(nside)
    w = np.bincount(pix, weights=value, minlength=npix)
    return w


# Gauss-Legendre (interval a:b)
def gauleg(ndeg, a=-1.0, b=1.0):
    '''
       Gauss-Legendre (default interval is [-1, 1])
    '''
    x, w = np.polynomial.legendre.leggauss(ndeg)
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5*(x + 1)*(b - a) + a
    w *= 0.5*(b - a)
    return t, w


def maps2pcinput(delta, ranmap, mask=None):
    '''
        function to extract theta, phi, delta, fpix from ngal, nran maps
        maps2pcinput(galmap, ranmap, fo, returnw=False)
    '''
    nside = hp.get_nside(delta)
    if mask is None:
       mask  = np.argwhere(ranmap != 0.0).flatten()
    else:
       mask  = np.argwhere(mask).flatten()
    teta, phi = hp.pixelfunc.pix2ang(nside, mask, nest=False)
    return teta, phi, delta[mask], ranmap[mask]



def xi2cl(x, w, xi, nlmax):
    '''
        calculates Cell from omega
    '''
    cl  = np.zeros(nlmax+1)
    m   = np.arange(nlmax+1)
    for i in m:
        Pl    = np.polynomial.Legendre.basis(i)(x)
        cl[i] = (xi * Pl * w).sum()
    cl *= 2.*np.pi
    return cl

def cl2xi(cl, costheta):
    '''
        calculates omega from Cell at Cos(theta)
    '''
    x     = np.array(costheta)
    l     = np.arange(cl.size)
    coef  = (2*l+1) * cl
    coef *= 0.25/(np.pi)
    y     = np.polynomial.legendre.legval(x, c=coef, tensor=True)
    return y


def catit(catobj):
    n      = catobj.RA.size
    catlog = np.zeros(n, dtype=[('RA','f8'),('DEC','f8'),('Z','f8'),('Weight','f8')])
    catlog['RA'] = catobj.RA
    catlog['DEC'] = catobj.DEC
    catlog['Z'] = catobj.Z
    catlog['Weight'] = catobj.Weight
    return catlog

class paircounting(object):
    #
    def __init__(self, datar, randomr):
        data   = catit(datar) 
        random = catit(randomr)
        self.data = nb.ArrayCatalog(data)
        self.random = nb.ArrayCatalog(random)
    
    def angular(self, edges=np.linspace(0.1, 10., 20+1)):
        dd = nb.AngularPairCount(self.data, edges, ra='RA', dec='DEC', weight='Weight')
        dr = nb.AngularPairCount(self.data, edges, ra='RA', dec='DEC', weight='Weight', second=self.random)
        rr = nb.AngularPairCount(self.random, edges, ra='RA', dec='DEC', weight='Weight', second=self.random)
        dd = dd.result
        dr = dr.result
        rr = rr.result
        f  = self.data.csize/self.random.csize
        return (dd, rr, dr, f, edges)

    def surveydata(self, edges=np.linspace(0.1, 10., 20+1),  universe_params=None):
        # cosmology
        if universe_params is None:
            universe_params = universeparams
        cosmo = Cosmology(Om0=universe_params['Om0'], 
                          H0=universe_params['H0'], 
                          flat=universe_params['flat'])
        dd = nb.SurveyDataPairCount({'1d'}, self.data, edges, cosmo, ra='RA', 
                                dec='DEC', weight='Weight', redshift='Z')
        dr = nb.SurveyDataPairCount({'1d'}, self.data, edges, cosmo, ra='RA', 
                                dec='DEC', weight='Weight', redshift='Z',second=self.random)
        rr = nb.SurveyDataPairCount({'1d'}, self.random, edges, cosmo, ra='RA', 
                                dec='DEC', weight='Weight', redshift='Z',second=self.random)
        dd = dd.result
        dr = dr.result
        rr = rr.result
        f  = self.data.csize/self.random.csize
        return (dd, rr, dr, f, edges)

class powerspectrum(object):
    #
    def __init__(self, datar, randomr, nofz, universe_params=None):            
        #
        # make sure catalogs have the desired format
        #
        # cosmology
        if universe_params is None:
            universe_params = universeparams
        cosmo = Cosmology(Om0=universe_params['Om0'], 
                          H0=universe_params['H0'], 
                          flat=universe_params['flat'])
        data   = catit(datar)  
        randoms = catit(randomr)
        data = nb.ArrayCatalog(data)
        randoms = nb.ArrayCatalog(randoms)
        data['Position']    = SkyToCartesian(data['RA'], data['DEC'], data['Z'], cosmo=cosmo)
        randoms['Position'] = SkyToCartesian(randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo)
        #
        self.data    = data
        self.randoms = randoms
        self.nofz    = nofz
        
    def run(self, zlim, nmesh, Poles=[0,2,4], dK=None, kMin=0.0, use_fkp=True, P0fkp=6e3):
        zmin, zmax = zlim
        self.randoms['Selection'] = (self.randoms['Z'] >= zmin)&(self.randoms['Z'] < zmax)
        self.data['Selection']    = (self.data['Z']    >= zmin)&(self.data['Z']    < zmax)
        #
        # combine the data and randoms into a single catalog
        # add the n(z) columns to the FKPCatalog
        #
        fkp               = nb.FKPCatalog(self.data, self.randoms)
        fkp['randoms/NZ'] = self.nofz(self.randoms['Z'])
        fkp['data/NZ']    = self.nofz(self.data['Z'])
        mesh              = fkp.to_mesh(Nmesh=nmesh, nbar='NZ')
        #
        #
        r = nb.ConvolvedFFTPower(mesh, poles=Poles, dk=dK, kmin=kMin, use_fkp_weights=use_fkp, P0_FKP=P0fkp)
        output = {}
        output['attrs'] = r.attrs
        output['attrs']['zbin'] = zlim
        output['poles_data'] = r.poles
        output['nz'] = self.nofz
        return output

class AngularClustering2D(object):
    """
    Clustering Statistics Cell and Omega
    """
    
    def __init__(self, data, random, selection_function=None, mask=None, nside=256, hpmap=False):
        if hpmap:
            datamap = np.copy(data)
            randmap = np.copy(random)
        else:
            datamap = hpixsum(nside, data.RA, data.DEC, value=data.Weight)
            randmap = hpixsum(nside, random.RA, random.DEC, value=random.Weight)
        if selection_function is None:
            selection_function = np.ones(randmap.shape)
        delta   = np.zeros(datamap.shape)
        if mask is None:
            mask    = (randmap != 0.0) & (selection_function !=0.0)
        randsel = randmap * selection_function
        sf      = datamap[mask].sum() / randsel[mask].sum()
        delta[mask] = datamap[mask]/ (sf * randsel[mask]) - 1.0
        self.delta  = delta
        self.weight = randmap
        self.nside = nside
        self.mask  = mask
    
    def run(self, LMAX=None):
        if LMAX is None:
            LMAX           = 3*self.nside-1
        else:
            assert LMAX <=  3*self.nside-1
        x, w = np.polynomial.legendre.leggauss(LMAX)
        #map
        mapmasked       = hp.ma(self.delta * self.weight)
        mapmasked.mask  = np.logical_not(self.mask)
        clmap           = hp.anafast(mapmasked.filled(), lmax=LMAX)
        ximap           = cl2xi(clmap, x)
        #mask
        maskmasked      = hp.ma(self.weight)
        maskmasked.mask = np.logical_not(self.mask)
        clmask          = hp.anafast(maskmasked.filled(), lmax=LMAX)
        ximask          = cl2xi(clmask, x)
        # correct for the mask
        xifinal         = ximap / ximask
        clfinal         = xi2cl(x, w, xifinal, LMAX)
        # update
        output = {}
        output['attr'] = {'nside':self.nside, 'lmax':LMAX}
        output['cl'] = (np.arange(clfinal.size), clfinal)
        output['xi'] = (np.rad2deg(np.arccos(x)), xifinal)
        return output

    def run_paircount(self, maxang=10):
        bw = 3.*hp.nside2resol(self.nside)*180./3.1416  # 3x resol.
        bins = np.arange(bw, maxang, bw)
        #delta_i, rani = hp.ud_grade(delta, res), hp.ud_grade(weight, res)
        theta, phi, deltam, fpixm = maps2pcinput(self.delta, self.weight, self.mask)
        w = paircount(theta, phi, deltam, deltam, fpixm, np.deg2rad(bins))
        binc = 0.5*(bins[1:]+bins[:-1])
        return [binc, w[0]/w[1]]









class ngalsys(object):
    """
        Class to get overdensity (delta) vs. systematic
    """
    
    def __init__(self, galaxy, nside, hpmap=False):
        """
            galaxy catalog; galaxy.RA, galaxy.DEC and galaxy.Weight
            nside: healpix resolution
        """
        if hpmap:
            self.galorg = galaxy
        else:
            self.galorg = hpixsum(nside, galaxy.RA, galaxy.DEC, value=galaxy.Weight)
        self.nside  = nside
        
    def prepare_inputs(self, systematic, selection_function=None,
                       random=None, mask=None, fracdet_c=0.2):
        """
            prepares the systematic map, Ngalaxy map and the Fracdet map
            systematic: systematic.HPIX and systematic.SIGNAL
            selection_function: selection function (if None, random should be given)
            random: random catalog; random.RA, random.DEC, random.Weight
            mask: additional mask a boolean
        """
        self.sysmap = np.zeros(12*self.nside**2)
        self.sysmap[systematic.HPIX] = systematic.SIGNAL
        if selection_function is not None:
            self.ranmap = selection_function
        elif random is not None:
            self.ranmap = hpixsum(self.nside, random.RA, random.DEC, value=random.Weight)
        else:
            raise RuntimeError("either enter the selection function/fracdet or the randoms catalog")
        mskpix = (self.ranmap !=0.) & (self.ranmap >= fracdet_c)
        if mask is not None:
            mskpix &= mask
        self.ihpix   = np.argwhere(mskpix).flatten()
        self.fracdet = self.ranmap[mskpix]
        self.sysmap  = self.sysmap[self.ihpix]
        self.galmap  = self.galorg[self.ihpix] 
        
    def digitize_ngalsys(self, bins, t=0.2):
        '''
            the galaxies & randoms into different systematic values bins
        '''
        self.bins   = bins
        self.sysl    = [0 for k in range(2*self.bins.size)]
        inds = np.digitize(self.sysmap, self.bins)
        for i in range(1,bins.size): # what if there is nothing on the last bin? FIXME
            self.sysl[2*i-2] = self.galmap[np.where(inds == i)].tolist()
            self.sysl[2*i-1] = self.fracdet[np.where(inds == i)].tolist()    
        self.avnden = np.sum([np.sum(self.sysl[i]) for i in np.arange(0,2*bins.size, 2)])\
                      /np.sum([np.sum(self.sysl[i]) for i in np.arange(1,2*bins.size, 2)])
        
    def processjack(self, njack=20, return_outputs=False):
        '''
            the average and jackknife std in each bin
        '''        
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
        area1pix = hp.nside2pixarea(self.nside, degrees=True)
        npixtot  = self.fracdet.size
        nrantot  = self.fracdet.sum()
        npix2area = npixtot*area1pix/nrantot
        output   = {}
        output['delta'] = np.array(ml)
        output['area']  = np.array(nl) * npix2area
        output['delta_err'] = np.array(sl)
        output['bin_edges']  = self.bins
        attrs = {}
        attrs['nside'] = self.nside
        attrs['npix2area'] = npix2area
        attrs['njack']  = njack
        output['attr'] = attrs
        self.output = output
        if return_outputs:
            return output
#     def quickplot(self, xlab='X', ylabs=[r'$\delta$',r'area[deg$^{2}$]']):
#         x = self.output['bin_edges'][:-1]
#         y = self.output['delta']
#         ye = self.output['delta_err']
#         s = self.output['area']
#         plt.rc('axes.spines', top=False)
#         f,a = plt.subplots()
#         a.set_xscale('log')
#         a.errorbar(x, y-1.0, yerr=ye, c='b')
#         a2 = a.twinx()
#         a2.fill_between(x, s, step='pre', color='r', alpha=0.5)
#         a2.set_ylabel(ylabs[1], color='r')
#         a.set_ylabel(ylabs[0], color='b')
#         a.set_xlabel(xlab)

"""
class clustering(object):

    
    def __init__(self, mapfile, maskfile=None):
        self.mapfile  = mapfile
        self.maskfile = maskfile
        self.cl       = None
        self.xi       = None
    
    def get_stats(self, lmax=None):
        if lmax is None:
            nside          = hp.pixelfunc.get_nside(self.mapfile)
            LMAX           = 3*nside-1
        x, w = np.polynomial.legendre.leggauss(LMAX)
        if self.maskfile is not None:
            #map
            mapmasked       = hp.ma(self.mapfile)
            mapmasked.mask  = np.logical_not(self.maskfile)
            clmap           = hp.anafast(mapmasked.filled(), lmax=LMAX)
            ximap           = cl2xi(clmap, x)
            #mask
            maskmasked      = hp.ma(self.maskfile)
            maskmasked.mask = np.logical_not(self.maskfile)
            clmask          = hp.anafast(maskmasked.filled(), lmax=LMAX)
            ximask          = cl2xi(clmask, x)
            # correct for the mask
            xifinal         = ximap / ximask
            clfinal         = xi2cl(x, w, xifinal, LMAX)
        else:
            clfinal         = hp.anafast(self.mapfile, lmax=LMAX)
            xifinal         = cl2xi(clmap, x)
        # update
        self.cl = (np.arange(clfinal.size), clfinal)
        self.xi = (np.rad2deg(np.arccos(x)), xifinal)
    
    def plot_maps(self, ttl='mollview'):
        mp = hp.ma(self.mapfile)
        if self.maskfile is not None:
            mp.mask = np.logical_not(self.maskfile)
            hp.mollview(mp.filled(), title=ttl)
        else:
            hp.mollview(mp, title=ttl)

    def plot_stats(self, xlab=['l',r'$\theta$[deg]'],
                   ylab=[r'l(l+1)C$_{l}$',r'$\omega$']):
        el, cel = self.cl
        t, om   = self.xi
        plt.figure(figsize=(16,10))
        plt.suptitle('Cell & $\omega$')
        plt.subplot(121)
        plt.xlabel(xlab[0]);plt.ylabel(ylab[0])
        plt.plot(el, el*(el+1)*cel, 'b.', alpha=0.1)
        plt.xscale('log')
        plt.subplot(122)
        plt.xlabel(xlab[1]);plt.ylabel(ylab[1])
        plt.plot(t, om, 'b.');plt.loglog()


#def maps2pcinput(galmap, ranmap, fo, returnw=False):
#    '''
#        function to extract theta, phi, delta, fpix from ngal, nran maps
#        maps2pcinput(galmap, ranmap, fo, returnw=False)
#    '''
#    nside = hp.get_nside(galmap)
#    delta, fpix = n2deln(galmap, ranmap, returnw=returnw)
#    fofile = open(fo, 'w+')
#    for i in np.argwhere(fpix != 0.0).flatten():
#        teta, phi = hp.pixelfunc.pix2ang(nside, i, nest=False)
#        fofile.write("%f %f %f %f \n"%(teta, phi, delta[i], fpix[i]))
def get_xi_pc(delta, weight, res=512, maxang=180.):
    bw = hp.nside2resol(res)*180./3.1416
    bins = np.arange(bw, maxang, bw)
    delta_i, rani = hp.ud_grade(delta, res), hp.ud_grade(weight, res)
    theta, phi, deltam, fpixm = maps2pcinput(delta_i, rani)
    w = paircount(theta, phi, deltam, fpixm, np.deg2rad(bins))
    binc = 0.5*(bins[1:]+bins[:-1])
    return [binc, w]

def get_xi(delta, weight):
    obj = clustering(delta*weight, weight)
    obj.get_stats()
    xistats = np.copy(obj.xi)
    elcell  = np.copy(obj.cl)
    del obj
    return xistats, elcell



def decra2hpix(nside, dec, ra):
    pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    return pix

def binstat(nside, ra, dec, value, func='count'):
    pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    npix = hp.nside2npix(nside)
    bins = [i for i in range(npix+1)]
    w,_,_ = stats.binned_statistic(pix, value, statistic=func, bins=bins)
    return w


def pcxi2cl(x, xi, ndeg, nlmax):
    '''
        pair-counting omega to Cell
    '''
    xgau, wgau = gauleg(ndeg)
    spl = iusp(x, xi)
    ygau = spl(np.arccos(xgau))
    cl = xi2cl(xgau, wgau, ygau, nlmax)
    return cl

def paircountingx2cl(d4, nside):
    '''
        
    '''
    x = np.deg2rad(d4[:,0])
    xi = d4[:,1]/d4[:,2]
    ndeg = 3*nside - 1
    nlmax = ndeg
    cl = pcxi2cl(x, xi, ndeg, nlmax)
    l = np.arange(0, cl.size)
    return l, cl


def map2cl(mapname,lmax=None,mask=None):
    if lmax is None:
        nside = hp.pixelfunc.get_nside(mapname)
        LMAX = 3*nside-1
    if mask is not None:
        mapmasked = hp.ma(mapname)
        mapmasked.mask = np.logical_not(mask)
        cl = hp.anafast(mapmasked.filled(), lmax=LMAX)
        return cl
    else:
        cl = hp.anafast(mapname, lmax=LMAX)
        return cl

def map2clwcorrect(inmap, inmask=None, res=256):
    ndeg = 3*res
    costheta, weights = gauleg(ndeg)
    mask = None
    if inmask != None:
        mask = inmask.astype(np.bool)
        clmask = map2cl(mask.astype(np.float), mask=mask)
        omegamask = cl2xi(clmask, costheta)
    #
    #
    #
    clmap = map2cl(inmap,mask=mask)
    omegamap = cl2xi(clmap, costheta)
    #
    #
    if mask != None:
        omegafinal = omegamap/omegamask
        clfinal = xi2cl(costheta, weights, omegafinal, ndeg-1)
        tw = [np.arccos(costheta), omegafinal]
        lcl = [np.arange(len(clfinal)).astype('<f8'), clfinal]
        return lcl, tw
    else:
        tw = [np.arccos(costheta), omegamap]
        lcl = [np.arange(len(clmap)).astype('<f8'), clmap]
        return lcl, tw

def cat2map(fnmap, res=256, normalized=False):
    dmap = fitsio.read(fnmap)
    # if normalized:
    #     omap = hpixsum(res, dmap['RA'], dmap['DEC'], value=dmap['COMP'])
    # else:
    #     omap = hpixsum(res, dmap['RA'], dmap['DEC'])
    omap = hpixsum(res, dmap[:,0], dmap[:,1]) # in general RA-DEC
    return omap.astype('<f8')


def makedelta(galcat, rancat):
    galmap = hpixsum(nside, galcat['ra'], galcat['dec'])
    delta  = np.zeros(hp.nside2npix(nside))
    mask   = rancat != 0.0
    av     = galmap.sum()/rancat.sum()
    delta[mask] = galmap[mask]/(rancat[mask]*av) - 1.0 
    return delta

def n2deln(galmap, ranmap, returnw=False):
    ave = galmap.sum()/ranmap.sum()
    delta = np.zeros(galmap.size)
    arg = ranmap != 0.0
    delta[arg] = galmap[arg]/(ave*ranmap[arg]) - 1.0
    if returnw:
        return delta, ranmap
    maskmap = np.zeros(len(galmap))
    maskmap[arg] = 1.0
    return delta, maskmap

def map2clus(galn, rann, res=256, oudir='./'):
    galmap = cat2map(galn, res=res, normalized=False)
    ranmap = cat2map(rann, res=res, normalized=True)
    deltamap, mask = n2deln(galmap, ranmap)
    lcl, tw = map2clwcorrect(deltamap, inmask=mask, res=res)
    return lcl, tw

def binner(X, Y, Xbins, statmode = 'mean'):
    bin_stats,bin_edges,binnumber = stats.binned_statistic(X,Y,statistic=statmode,bins=Xbins)
    bin_std,bin_edges,binnumber = stats.binned_statistic(X,Y,statistic=np.std,bins=Xbins)
    bin_count,bin_edges,binnumber = stats.binned_statistic(X,Y,statistic='count',bins=Xbins)
    errorbar = bin_std/np.sqrt(bin_count)
    errorbarm = np.array(errorbar)
    errorbarm[errorbar>=bin_stats] = bin_stats[errorbar>=bin_stats]*.9999999
    bin_center = 0.5*(bin_edges[1:]+bin_edges[:-1])
    return bin_center,bin_stats,[errorbarm,errorbar]


def plotclw(lcl, tw, ou='figure.png'):
    font = {'family' : 'monospace',
            'weight' : 'normal',   #bold
            'size'   : 15}
    matplotlib.rc('font', **font)  # pass in the font dict as kwargs
    ncol = 2
    figsize = (8*ncol, 10)
    fig, axes = plt.subplots(nrows=1, ncols=ncol, figsize=figsize, dpi=None, sharey=False)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.2, hspace=None)
    axes[0].set_yscale("log", nonposy='clip')
    axes[0].set_ylabel(r'$C_{l}$')
    axes[0].set_xscale("log", nonposy='clip')
    axes[0].set_xlabel(r'$l$')
    axes[0].set_xlim([1, 1e3])
    axes[0].set_ylim([1e-7, 1e-1])
    bn, bs, be = binner(lcl[0], lcl[1], 15)
    axes[0].plot(lcl[0], lcl[1], 'b+', alpha=0.5)
    axes[0].errorbar(bn, bs, yerr=be, color='g', marker='+', ls='None')
    axes[1].set_yscale("log", nonposy='clip')
    axes[1].set_ylabel(r'$w$')
    axes[1].set_xscale("log", nonposy='clip')
    axes[1].set_xlabel(r'$\theta$')
    axes[1].set_xlim([0.1, 10.])
    axes[1].set_ylim([1e-3, 1.])
    axes[1].plot(np.rad2deg(tw[0]), tw[1], 'b+', alpha=0.5)
    bn, bs, be = binner(np.rad2deg(tw[0]), tw[1], 1.0)
    axes[1].errorbar(bn, bs, yerr=be, color='g', marker='+', ls='None')
    plt.savefig(ou, bbox_inches='tight')
    plt.show()
"""
#
# redundant functions will be removed -- DEC,17,2017
#
