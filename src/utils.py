"""
   tools for handing pixels
   a bunch of useful functions & classes for calculating
   cosmological quantities

   (c) Mehdi Rezaie medirz90@icloud.com
   Last update: Jun 23, 2019

"""
import os
import sys
import numpy as np
import numpy.lib.recfunctions as rfn
import scipy.special as scs
from   scipy import integrate
from   scipy.constants import c as clight


try:
    import camb
    from camb import model, initialpower
except:
    print("camb is not installed!")
try:
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KDTree
except:
    print('sklearn is not installed!')
try:
    import fitsio as ft
except:
    print('fitsio is not installed!')
try:
    import healpy as hp
except:
    print('healpy is not installed')
    
from scipy.stats import binned_statistic



def G_to_C(mapi, res_in=1024, res_out=256):
    '''
        Rotate the HI column density from G to C
        to avoid its negative pixels
    '''
    thph  = hp.pix2ang(res_out, np.arange(12*res_out*res_out))
    r     = hp.Rotator(coord=['C', 'G'])
    thphg = r(thph[0], thph[1])
    hpix  = hp.ang2pix(res_in, thphg[0], thphg[1])
    return mapi[hpix]

def fixHI(hifile='/Volumes/TimeMachine/data/NHI_HPX.fits', nside_out=256):
    '''
        read HI column density
        rotate it from G to C ("decrease" nside if needed)
        fill some negative pixels
        by assigning the mean of neighbors
    '''
    assert (nside_out < 1024), 'nside_out should be < 1024'
    # H II map
    hii = ft.FITS(hifile, lower=True)
    Hii = hii[1].read()
    Hiic = G_to_C(Hii['nhi'], res_out=nside_out)
    Hineg = np.argwhere(Hiic<=0.0).flatten()
    neighbors = hp.get_all_neighbours(nside_out, Hineg)
    Hiic[Hineg] = np.mean(Hiic[neighbors], axis=0) # fill in negative pixels
    return Hiic

def binit(el, cel, bins=None):
    '''
        bin the C_ell measurements
    '''
    if bins is None:
        bins = np.logspace(0, 2.71, 10)
    kw  = dict(bins=bins, statistic='sum')
    lb  = 0.5*(bins[1:]+bins[:-1])
    a2l = 2*el + 1
    clwt,_,_ = binned_statistic(el, a2l*cel, **kw)
    wt,_,_   = binned_statistic(el, a2l, **kw)
    #print(clwt, wt)
    return lb, clwt/wt

def moderr(el, cel, bins=np.logspace(0, 2.71, 10), fsky=1.0):
    '''
        get the mode counting error estimate
    '''
    kw  = dict(bins=bins, statistic='sum')
    lb  = 0.5*(bins[1:]+bins[:-1])
    a2l = 2*el + 1
    clwt,_,_ = binned_statistic(el, a2l*cel, **kw)
    wt,_,_   = binned_statistic(el, a2l, **kw)
    #print(clwt, wt)
    return lb, (clwt/wt)/(np.sqrt(0.5*fsky*wt))

def binit_jac(cljks, bins=None, njacks=20):
    '''
        Bin jackknife C_ell measurements and get the error estimate
    '''
    el = np.arange(cljks[0].size)
    cbljks = []
    for i in range(njacks):
        elb, clb = binit(el, cljks[i], bins=bins)
        cbljks.append(clb)
    elb, clm = binit(el, cljks[-1], bins=bins)
    clvar = np.zeros(clm.size)
    for i in range(njacks):
        clvar += (clm - cbljks[i])*(clm - cbljks[i])
    clvar *= (njacks-1)/njacks
    return elb, np.sqrt(clvar)
    
def histedges_equalN(x, nbin=10, kind='size', weight=None):
    '''
        https://stackoverflow.com/questions/39418380/
        histogram-with-equal-number-of-points-in-each-bin
        (c) farenorth
    '''
    if kind == 'size':
        npt = len(x)
        xp  = np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
    elif kind == 'area':
        sys.exit('FIX this routine for a repetitave x')
        npt1  = len(x)-1
        sumw = np.sum(weight) / nbin
        i    = 0
        wst  = 0.0
        xp   = [x.min()]  # lowest bin is the minimum
        #
        #
        datat        = np.zeros(x.size, dtype=np.dtype([('x', 'f8'), ('w', 'f8'), ('rid', 'i8')]))
        datat['x']   = x
        datat['w']   = weight
        datat['rid'] = np.random.choice(np.arange(x.size), size=x.size, replace=False)
        datas  = np.sort(datat, order=['x', 'rid'])
        xs, ws = datas['x'], datas['w'] #zip(*sorted(zip(x, weight)))
        for wsi in ws:
            wst += wsi
            i   += 1
            if (wst > sumw) or (i == npt1):
                xp.append(xs[i])
                wst = 0.0
        xp = np.array(xp)
    return xp

def radec2hpix(nside, ra, dec):
    pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    return pix

def hpix2radec(nside, pix):
    theta,phi = hp.pixelfunc.pix2ang(nside, pix)
    return np.degrees(phi), 90-np.degrees(theta)


def projectradec2hp(nside, ra, dec, value, statistic='mean'):
    '''
        project a quantity (value) onto RA-DEC, and then healpix
        with a given nside
        default is 'mean', but can work with 'min', 'max', etc
    '''
    hpix = radec2hpix(nside, ra, dec)
    nmax = 12*nside*nside
    result = binned_statistic(hpix, value, statistic=statistic, 
                                 bins=nmax, range=(0, nmax))[0]
    return result

def hpixsum(nside, ra, dec, value=None): 
    '''
        make a healpix map from ra-dec
        default is RING format
        hpixsum(nside, ra, dec, value=None)
    '''
    pix  = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
    npix = hp.nside2npix(nside)
    w    = np.bincount(pix, weights=value, minlength=npix)
    return w

def makedelta(map1, weight1, mask, select_fun=None, is_sys=False):
    delta = np.zeros_like(map1)
    if select_fun is not None:
        gmap = map1 / select_fun
    else:
        gmap = map1#.copy()

    #assert((randc[mask]==0).sum() == 0) # make sure there is no empty pixel
    if (weight1[mask]==0).sum() != 0:
        print('there are empty weights')
        m = weight1 == 0
        weight1[m] = 1.0 # enforece one
       
    if is_sys:
        sf = (gmap[mask]*weight1[mask]).sum() / weight1[mask].sum()
        delta[mask] = gmap[mask] / sf - 1
    else:
        sf = gmap[mask].sum()/weight1[mask].sum()
        delta[mask] = gmap[mask]/(weight1[mask]*sf)  - 1   
    return delta



def clerr_jack(delta, mask, weight, njack=20, lmax=512):
    '''
       
    '''
    npix = delta.size 
    hpix = np.argwhere(mask).flatten()
    dummy = np.ones(mask.sum())
    hpixl, wl, deltal,_ = split_jackknife(hpix, weight[mask], 
                                          delta[mask], dummy, njack=njack)
    print('# of jackknifes %d, input : %d'%(len(hpixl), njack))
    cljks = {}
    # get the cl of the jackknife mask
    wlt = wl.copy()
    hpixt   = hpixl.copy()
    wlt.pop(0)
    hpixt.pop(0)
    wlc = np.concatenate(wlt)
    hpixc  = np.concatenate(hpixt)
    maski  = np.zeros(npix, '?')
    maski[hpixc] = True 
    map_i  = hp.ma(maski.astype('f8'))
    map_i.mask = np.logical_not(maski)
    clmaskj = hp.anafast(map_i.filled(), lmax=lmax)
    sfj = ((2*np.arange(clmaskj.size)+1)*clmaskj).sum()/(4.*np.pi) 

    for i in range(njack):
        hpixt   = hpixl.copy()
        wlt     = wl.copy()
        deltalt = deltal.copy()
        #
        hpixt.pop(i)
        wlt.pop(i)
        deltalt.pop(i)
        #
        hpixc  = np.concatenate(hpixt)
        wlc    = np.concatenate(wlt)
        deltac = np.concatenate(deltalt)
        #
        maski  = np.zeros(npix, '?')
        deltai = np.zeros(npix)
        wlci   = np.zeros(npix)
        #
        maski[hpixc]   = True
        deltai[hpixc]  = deltac
        wlci[hpixc]    = wlc
        #
        map_i       = hp.ma(deltai * wlci)
        map_i.mask  = np.logical_not(maski)
        cljks[i]    = hp.anafast(map_i.filled(), lmax=lmax)/sfj
    #
    hpixt   = hpixl.copy()
    wlt     = wl.copy()
    deltalt = deltal.copy()
    #
    hpixc  = np.concatenate(hpixt)
    wlc    = np.concatenate(wlt)
    deltac = np.concatenate(deltalt)
    #
    maski  = np.zeros(npix, '?')
    deltai = np.zeros(npix)
    wlci   = np.zeros(npix)
    #
    maski[hpixc]   = True
    deltai[hpixc]  = deltac
    wlci[hpixc]    = wlc
    #
    map_i      = hp.ma(maski.astype('f8'))
    map_i.mask = np.logical_not(maski)
    clmask = hp.anafast(map_i.filled(), lmax=lmax)
    sf = ((2*np.arange(clmask.size)+1)*clmask).sum()/(4.*np.pi) 

    map_i       = hp.ma(deltai * wlci)
    map_i.mask  = np.logical_not(maski)
    cljks[-1]   = hp.anafast(map_i.filled(), lmax=lmax)/sf   # entire footprint
    #
    clvar = np.zeros(len(cljks[-1]))
    for i in range(njack):
        clvar += (cljks[-1] - cljks[i])*(cljks[-1] - cljks[i])
    clvar *= (njack-1)/njack
    return dict(clerr=np.sqrt(clvar), cljks=cljks, clmaskj=clmaskj, clmask=clmask, sf=sf, sfj=sfj)




def split_jackknife(hpix, weight, label, features, njack=20):
    '''
        split_jackknife(hpix, weight, label, features, njack=20)
        split healpix-format data into k equi-area regions
        hpix: healpix index shape = (N,)
        weight: weight associated to each hpix 
        label: label associated to each hpix
        features: features associate to each pixel shape=(N,M) 
    '''
    f = weight.sum() // njack
    hpix_L = []
    hpix_l = []
    frac_L = []
    frac    = 0
    label_L = []
    label_l = []
    features_L = []
    features_l = []
    w_L = []
    w_l = []
    #
    #
    for i in range(hpix.size):
        frac += weight[i]            
        hpix_l.append(hpix[i])
        label_l.append(label[i])
        w_l.append(weight[i])
        features_l.append(features[i])
        #
        #
        if frac >= f:
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            label_L.append(label_l)
            w_L.append(w_l)
            features_L.append(features_l)
            frac    = 0
            features_l  = []
            w_l     = []
            hpix_l = []
            label_l = []
        elif (i == hpix.size-1) and (frac > 0.9*f):
            hpix_L.append(hpix_l)
            frac_L.append(frac)
            label_L.append(label_l)
            w_L.append(w_l)
            features_L.append(features_l)            
    return hpix_L, w_L, label_L, features_L #, frac_L

def concatenate(A, ID):
    # combine A[i] regions for i in ID 
    AA = [A[i] for i in ID]
    return np.concatenate(AA)
    
def combine(hpix, fracgood, label, features, DTYPE, IDS):
    # uses concatenate(A,ID) to combine different attributes
    size = np.sum([len(hpix[i]) for i in IDS])
    zeros = np.zeros(size, dtype=DTYPE)
    zeros['hpind']     = concatenate(hpix, IDS)
    zeros['fracgood'] = concatenate(fracgood, IDS)
    zeros['features'] = concatenate(features, IDS)
    zeros['label']    = concatenate(label, IDS)
    return zeros

    
def split2KfoldsSpatially(data, k=5, shuffle=True, random_seed=123):
    '''
        split data into k contiguous regions
        for training, validation and testing
    '''
    P, W, L, F = split_jackknife(data['hpind'],data['fracgood'],
                                data['label'], data['features'], 
                                 njack=k)
    DTYPE = data.dtype
    np.random.seed(random_seed)
    kfold = KFold(k, shuffle=shuffle, random_state=random_seed)
    index = np.arange(k)
    kfold_data = {'test':{}, 'train':{}, 'validation':{}}
    arrs = P, W, L, F, DTYPE
    for i, (nontestID, testID) in enumerate(kfold.split(index)):
        foldname = 'fold'+str(i)
        validID  = np.random.choice(nontestID, size=testID.size, replace=False)
        trainID  = np.setdiff1d(nontestID, validID)
        kfold_data['test'][foldname]       = combine(*arrs, testID)
        kfold_data['train'][foldname]      = combine(*arrs, trainID)
        kfold_data['validation'][foldname] = combine(*arrs, validID)
    return kfold_data    




def split2Kfolds(data, k=5, shuffle=True, random_seed=123):
    '''
        split data into k randomly chosen regions
        for training, validation and testing
    '''
    np.random.seed(random_seed)
    kfold = KFold(k, shuffle=shuffle, random_state=random_seed)
    index = np.arange(data.size)
    kfold_data = {'test':{}, 'train':{}, 'validation':{}}
    for i, (nontestID, testID) in enumerate(kfold.split(index)):
        #
        #
        foldname = 'fold'+str(i)
        validID  = np.random.choice(nontestID, size=testID.size, replace=False)
        trainID  = np.setdiff1d(nontestID, validID)
        #
        #
        kfold_data['test'][foldname]       = data[testID]
        kfold_data['train'][foldname]      = data[trainID]
        kfold_data['validation'][foldname] = data[validID]
    return kfold_data

def read_split_write(path2file, path2output, k, random=True):
    ''' 
    read path2file, splits the data either randomly or ra-dec
    then writes the data onto path2output
    '''
    DATA  = ft.read(path2file)
    if random:
        datakfolds = split2Kfolds(DATA, k=k)
    else:
        datakfolds = split2KfoldsSpatially(DATA, k=k)
    np.save(path2output, datakfolds)




def write(address, fname, data, fmt='txt'):
    if not os.path.exists(address):
        os.makedirs(address)
    if address[-1] != '/':
        address += '/'
    if fmt == 'txt':
        ouname = address+fname+'.dat'
        np.savetxt(ouname, data)
    elif fmt == 'npy':
        ouname = address+fname
        np.save(ouname, data)


def D(z, omega0):
    """
        Growth Function 
    """
    a = 1/(1+z)
    v = scs.cbrt(omega0/(1.-omega0))/a
    return a*d1(v)

def d1(v):
    """
        d1(v) = D(a)/a where D is growth function see. Einsenstein 1997 
    """
    beta  = np.arccos((v+1.-np.sqrt(3.))/(v+1.+np.sqrt(3.)))
    sin75 = np.sin(75.*np.pi/180.)
    sin75 = sin75**2
    ans   = (5./3.)*(v)*(((3.**0.25)*(np.sqrt(1.+v**3.))*(scs.ellipeinc(beta,sin75)\
            -(1./(3.+np.sqrt(3.)))*scs.ellipkinc(beta,sin75)))\
            +((1.-(np.sqrt(3.)+1.)*v*v)/(v+1.+np.sqrt(3.))))
    return ans

def growthrate(z,omega0):
    """
        growth rate f = dln(D(a))/dln(a)

    """
    a = 1/(1+z)
    v = scs.cbrt(omega0/(1.-omega0))/a
    return (omega0/(((1.-omega0)*a**3)+omega0))*((2.5/d1(v))-1.5)

def invadot(a, om_m=0.3, om_L=0.0, h=.696):
    om_r = 4.165e-5*h**-2 # T0 = 2.72528K
    answ = 1/np.sqrt(om_r/(a * a) + om_m / a\
            + om_L*a*a + (1.0-om_r-om_m-om_L))
    return answ

def invaadot(a, om_m=0.3, om_L=0.0, h=.696):
    om_r = 4.165e-5*h**-2 # T0 = 2.72528K
    answ = 1/np.sqrt(om_r/(a * a) + om_m / a\
            + om_L*a*a + (1.0-om_r-om_m-om_L))
    return answ/a


class camb_pk(object):
    
    #
    def __init__(self, h=0.675, omc=.268, omb=0.048, omk=0.0, num_massive_neutrinos=1,
           mnu=0.06, nnu=3.046, YHe=None, meffsterile=0, standard_neutrino_neff=3.046,
           TCMB=2.7255, tau=None, ns=0.95, As=2e-9):
        self.kwargs = dict(H0=h*100, ombh2=omb*h**2, omch2=omc*h**2, omk=omk, 
                          num_massive_neutrinos=num_massive_neutrinos,
                           mnu=mnu, nnu=nnu, YHe=YHe, meffsterile=meffsterile, 
                          standard_neutrino_neff=standard_neutrino_neff,
                           TCMB=TCMB, tau=tau)
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(**self.kwargs)
        self.pars.InitPower.set_params(ns=ns, As=As)
        
    def get_pk(self, z, kmax=.4, npoints=200):
        h = self.kwargs['H0']/100
        self.pars.set_matter_power(redshifts=[z], kmax=kmax)
        self.pars.NonLinear = model.NonLinear_none
        results = camb.get_results(self.pars)
        s8 = np.array(results.get_sigma8())
        print("s8 :", s8)
        # for nonlinear uncomment this, see http://camb.readthedocs.io/en/latest/CAMBdemo.html
        #pars.NonLinear = model.NonLinear_both
        #results = camb.get_results(pars)
        #results.calc_power_spectra(pars)
        #
        kh_nonlin,_, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=npoints)
        return kh_nonlin, np.ravel(pk_nonlin)
    
    def get_plk(self, z, kmax=.4, npoints=200, poles=[0,2,4], bias=1.0):
        k, pk = self.get_pk(z, kmax=kmax, npoints=npoints)
        h = self.kwargs['H0']/100
        omega0 = self.kwargs['ombh2'] / h**2
        beta = (1.0 / bias) * (growthrate(z, omega0))
        pks = []
        for pole in poles:
            rsd_factor = rsd(pole, beta=beta)
            pks.append(rsd_factor * bias**2 * pk)
        return k, np.column_stack(pks)
            

def rsd(l, ngauss=50, beta=.6):
    x, wx = scs.roots_legendre(ngauss)
    px    = scs.legendre(l)(x)
    rsd_int = 0.0
    for i in range(ngauss):
        rsd_int += wx[i] * px[i] * ((1.0 + beta * x[i]*x[i])**2)
    rsd_int *= (l + 0.5)
    return rsd_int
    
class cosmology(object):
    '''
       cosmology
       # see
       # http://www.astro.ufl.edu/~guzman/ast7939/projects/project01.html
       # or
       # https://arxiv.org/pdf/astro-ph/9905116.pdf
       # for equations, there is a typo in comoving-volume eqn
    '''    
    def __init__(self, om_m=1.0, om_L=0.0, h=.696):
        self.om_m = om_m
        self.om_L = om_L
        self.h    = h
        self.om_r = 4.165e-5*h**-2 # T0 = 2.72528K
        self.tH  = 9.778/h         # Hubble time : 1/H0 Mpc --> Gyr
        self.DH  = clight*1.e-5/h       # Hubble distance : c/H0
    
    def age(self, z=0):
        ''' 
            age of universe at redshift z [default z=0] in Gyr
        '''
        az = 1 / (1+z)
        answ,_ = integrate.quad(invadot, 0, az,
                               args=(self.om_m, self.om_L, self.h))
        return answ * self.tH
        
    def DCMR(self, z):
        '''
            comoving distance (line of sight) in Mpc
        '''
        az = 1 / (1+z)
        answ,_ = integrate.quad(invaadot, az, 1,
                               args=(self.om_m, self.om_L, self.h))
        return answ * self.DH
    
    def DA(self, z):
        '''
            angular diameter distance in Mpc
        '''
        az = 1 / (1+z)
        r = self.DCMR(z)
        om_k = (1.0-self.om_r-self.om_m-self.om_L)
        if om_k != 0.0:DHabsk = self.DH/np.sqrt(np.abs(om_k))
        if om_k > 0.0:
            Sr = DHabsk * np.sinh(r/DHabsk)
        elif om_k < 0.0:
            Sr = DHabsk * np.sin(r/DHabsk)
        else:
            Sr = r
        return Sr*az
    
    def DL(self, z):
        '''
            luminosity distance in Mpc
        '''
        az = 1 / (1+z)
        da = self.DA(z)
        return da / (az * az)

    def CMVOL(self, z):
        '''
            comoving volume in Mpc^3
        '''
        Dm = self.DA(z) * (1+z)
        om_k = (1.0-self.om_r-self.om_m-self.om_L)
        if om_k != 0.0:DHabsk = self.DH/np.sqrt(np.abs(om_k))
        if om_k > 0.0:
            Vc = DHabsk**2 * np.sqrt(1 + (Dm/DHabsk)**2) * Dm \
                 - DHabsk**3 * np.sinh(Dm/DHabsk)
            Vc *= 4*np.pi/2.
        elif om_k < 0.0:
            Vc = DHabsk**2 * np.sqrt(1 + (Dm/DHabsk)**2) * Dm \
                 - DHabsk**3 * np.sin(Dm/DHabsk)
            Vc *= 4*np.pi/2.
        else:
            Vc = Dm**3
            Vc *= 4*np.pi/3
        return Vc

def comvol(bins_edge, fsky=1, omega_c=.3075, hubble_param=.696):
    """
        calculate the comoving volume for redshift bins
    """
    universe = cosmology(omega_c, 1.-omega_c, h=hubble_param)
    vols = []
    for z in bins_edge:
        vol_i = universe.CMVOL(z) # get the comoving vol. @ redshift z
        vols.append(vol_i)
    # find the volume in each shell and multiply by footprint area
    vols  = np.array(vols) * fsky
    vols  = np.diff(vols) * universe.h**3            # volume in unit (Mpc/h)^3
    return vols

def nzhist(z, fsky, cosmology, bins=None, binw=0.01, weight=None):
    if bins is None:
        bins = np.arange(0.99*z.min(), 1.01*z.max(), binw)
    Nz, zedge = np.histogram(z, bins=bins, weights=weight)
    #zcenter = 0.5*(zedge[:-1]+zedge[1:])
    vol_hmpc3 = comvol(zedge, fsky=fsky, omega_c=cosmology['Om0'], hubble_param=cosmology['H0']/100.)
    return zedge, Nz/vol_hmpc3
    



#
"""
    a modified version of  ImagingLSS
    https://github.com/desihub/imaginglss/blob/master/imaginglss/analysis/tycho_veto.py

    veto objects based on a star catalogue.
    The tycho vetos are based on the email discussion at:
    Date: June 18, 2015 at 3:44:09 PM PDT
    To: decam-data@desi.lbl.gov
    Subject: decam-data Digest, Vol 12, Issue 29
    These objects takes a decals object and calculates the
    center and rejection radius for the catalogue in degrees.
    Note : The convention for veto flags is True for 'reject',
    False for 'preserve'.

    apply_tycho takes the galaxy catalog and appends a Tychoveto column
    the code works fine for ELG and LRGs. For other galaxy type, you need to adjust it!
"""

def BOSS_DR9(tycho):
    bmag = tycho['bmag']
    # BOSS DR9-11
    b = bmag.clip(6, 11.5)
    R = (0.0802 * b ** 2 - 1.86 * b + 11.625) / 60. #
    return R

def DECAM_LRG(tycho):
    vtmag = tycho['vtmag']
    R = 10 ** (3.5 - 0.15 * vtmag) / 3600.
    return R

DECAM_ELG = DECAM_LRG

def DECAM_QSO(tycho):
    vtmag = tycho['vtmag']
    # David Schlegel recommends not applying a bright star mask
    return vtmag - vtmag

def DECAM_BGS(tycho):
    vtmag = tycho['vtmag']
    R = 10 ** (2.2 - 0.15 * vtmag) / 3600.
    return R

def radec2pos(ra, dec):
    """ converting ra dec to position on a unit sphere.
        ra, dec are in degrees.
    """
    pos = np.empty(len(ra), dtype=('f8', 3))
    ra = ra * (np.pi / 180)
    dec = dec * (np.pi / 180)
    pos[:, 2] = np.sin(dec)
    pos[:, 0] = np.cos(dec) * np.sin(ra)
    pos[:, 1] = np.cos(dec) * np.cos(ra)
    return pos

def tycho(filename):
    """
    read the Tycho-2 catalog and prepare it for the mag-radius relation
    """
    dataf = ft.FITS(filename, lower=True)
    data = dataf[1].read()
    tycho = np.empty(len(data),
        dtype=[
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('vtmag', 'f8'),
            ('vmag', 'f8'),
            ('bmag', 'f8'),
            ('btmag', 'f8'),
            ('varflag', 'i8'),
            ])
    tycho['ra'] = data['ra']
    tycho['dec'] = data['dec']
    tycho['vtmag'] = data['mag_vt']
    tycho['btmag'] = data['mag_bt']
    vt = tycho['vtmag']
    bt = tycho['btmag']
    b = vt - 0.09 * (bt - vt)
    v = b - 0.85 * (bt - vt)
    tycho['vmag']=v
    tycho['bmag']=b
    dataf.close()
    return tycho


def txts_read(filename):
    obj = np.loadtxt(filename)
    typeobj = np.dtype([
              ('RA','f4'), ('DEC','f4'), ('COMPETENESS','f4'),
              ('rflux','f4'), ('rnoise','f4'), ('gflux','f4'), ('gnoise','f4'),
              ('zflux','f4'), ('znoise','f4'), ('W1flux','f4'), ('W1noise','f4'),
              ('W2flux','f4'), ('W2noise','f4')
              ])
    nobj = obj[:,0].size
    data = np.zeros(nobj, dtype=typeobj)
    data['RA'][:] = obj[:,0]
    data['DEC'][:] = obj[:,1]
    data['COMPETENESS'][:] = obj[:,2]
    data['rflux'][:] = obj[:,3]
    data['rnoise'][:] = obj[:,4]
    data['gflux'][:] = obj[:,5]
    data['gnoise'][:] = obj[:,6]
    data['zflux'][:] = obj[:,7]
    data['znoise'][:] = obj[:,8]
    data['W1flux'][:] = obj[:,9]
    data['W1noise'][:] = obj[:,10]
    data['W2flux'][:] = obj[:,11]
    data['W2noise'][:] = obj[:,12]
    #datas = np.sort(data, order=['RA'])
    return data

def veto(coord, center, R):
    """
        Returns a veto mask for coord. any coordinate within R of center
        is vet.
        Parameters
        ----------
        coord : (RA, DEC)
        center : (RA, DEC)
        R     : degrees
        Returns
        -------
        Vetomask : True for veto, False for keep.
    """
    #from sklearn.neighbors import KDTree
    pos_stars = radec2pos(center[0], center[1])
    R = 2 * np.sin(np.radians(R) * 0.5)
    pos_obj = radec2pos(coord[0], coord[1])
    tree = KDTree(pos_obj)
    vetoflag = ~np.zeros(len(pos_obj), dtype='?')
    arg = tree.query_radius(pos_stars, r=R)
    arg = np.concatenate(arg)
    vetoflag[arg] = False
    return vetoflag



def apply_tycho(objgal, galtype='LRG',dirt='/global/cscratch1/sd/mehdi/tycho2.fits'):
    # reading tycho star catalogs
    tychostar = tycho(dirt)
    #
    # mag-radius relation
    #
    if galtype == 'LRG' or galtype == 'ELG':    # so far the mag-radius relation is the same for LRG and ELG
        radii = DECAM_LRG(tychostar)
    else:
        sys.exit("Check the apply_tycho function for your galaxy type")
    #
    #
    # coordinates of Tycho-2 stars
    center = (tychostar['ra'], tychostar['dec'])
    #
    #
    # coordinates of objects (galaxies)
    coord = (objgal['ra'], objgal['dec'])
    #
    #
    # a 0.0 / 1.0 array (1.0: means the object is contaminated by a Tycho-2 star, so 0.0s are good)
    tychomask = (~veto(coord, center, radii)).astype('f4')
    objgal = rfn.append_fields(objgal, ['tychoveto'], data=[tychomask], dtypes=tychomask.dtype, usemask=False)
    return objgal

# def getcf(d):
#     # cut input maps based on PCC
#     from scipy.stats import pearsonr
#     # lbl = ['ebv', 'nstar'] + [''.join((s, b)) for s in ['depth', 'seeing', 'airmass', 'skymag', 'exptime'] for b in 'rgz']
#     cflist = []
#     indices = []
#     for i in range(d['train']['fold0']['features'].shape[1]):
#         for j in range(5):
#             fold = ''.join(['fold', str(j)])
#             cf = pearsonr(d['train'][fold]['label'], d['train'][fold]['features'][:,i])[0]
#             if np.abs(cf) >= 0.02:
#                 #print('{:s} : sys_i: {} : cf : {:.4f}'.format(fold, lbl[i], cf))
#                 indices.append(i)
#                 cflist.append(cf)
#     if len(indices) > 0:
#         indices = np.unique(np.array(indices))
#         return indices
#     else:
#         print('no significant features')
#         return None
#     cf = []
#     indices = []
#     for i in range(features.shape[1]):
#         cf.append(pearsonr(label, features[:,i]))
#         if np.abs(cf) > 0.0
# def change_coord(m, coord):
#     """ Change coordinates of a HEALPIX map
#     (c) dPol
#     https://stackoverflow.com/questions/44443498/
#     how-to-convert-and-save-healpy-map-to-different-coordinate-system

#     Parameters
#     ----------
#     m : map or array of maps
#       map(s) to be rotated
#     coord : sequence of two character
#       First character is the coordinate system of m, second character
#       is the coordinate system of the output map. As in HEALPIX, allowed
#       coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

#     Example
#     -------
#     The following rotate m from galactic to equatorial coordinates.
#     Notice that m can contain both temperature and polarization.
#     >>>> change_coord(m, ['G', 'C']
#     """
#     # Basic HEALPix parameters
#     npix = m.shape[-1]
#     nside = hp.npix2nside(npix)
#     ang = hp.pix2ang(nside, np.arange(npix))

#     # Select the coordinate transformation
#     rot = hp.Rotator(coord=reversed(coord))

#     # Convert the coordinates
#     new_ang = rot(*ang)
#     new_pix = hp.ang2pix(nside, *new_ang)

#     return m[..., new_pix]
