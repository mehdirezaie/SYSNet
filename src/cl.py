'''
    module to compute Cl angular power spectrum
'''
import numpy as np
import healpy as hp
from   time import time
from   tools import makedelta


def gauleg(ndeg, a=-1.0, b=1.0):
    '''
       Gauss-Legendre (default interval is [-1, 1])
    '''
    x, w = np.polynomial.legendre.leggauss(ndeg)
    # Translate x values from the interval [-1, 1] to [a, b]
    t = 0.5*(x + 1)*(b - a) + a
    w *= 0.5*(b - a)
    return t, w

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
    y     = np.polynomial.legendre.legval(x, c=coef, tensor=False)
    return y

def compute_cl(map1, weight1, mask1, lmax=None):
    nside = hp.get_nside(map1)
    t1    = time()
    log = '! ---- computing cl ------ \n'
    if lmax is None:
        lmax = 3*nside-1
        log += 'lmax is calculated from the input map(nside = %d) %d! \n'%(nside,lmax)
    #
    # will use gauss-leg points to do the integral
    x, w = np.polynomial.legendre.leggauss(lmax+1)
    log += 'generated the leg gauss points ... \n'
    #
    # map
    mapmasked       = hp.ma(map1 * weight1)
    mapmasked.mask  = np.logical_not(mask1)
    clmap           = hp.anafast(mapmasked.filled(), lmax=lmax)
    ximap           = cl2xi(clmap, x)
    log            += 'computed the map cl .... \n'
    #
    # weight
    maskmasked      = hp.ma(weight1)
    maskmasked.mask = np.logical_not(mask1)
    clmask          = hp.anafast(maskmasked.filled(), lmax=lmax)
    ximask          = cl2xi(clmask, x)
    log            += 'computed the weight cl .... \n'    
    #
    # correct for the mask
    xifinal         = ximap / ximask
    clfinal         = xi2cl(x, w, xifinal, lmax)
    log            += 'correct for the "mask" (xi_map / xi_weight) .... \n'    
    # update
    output = {}
    output['attr'] = {'nside':nside, 'lmax':lmax}
    output['cl']   = (np.arange(clfinal.size), clfinal)
    output['cl_u'] = clmap
    output['xi_mask'] = ximask
    output['xi']   = (np.rad2deg(np.arccos(x)), xifinal)
    log           += 'Finished C_l in %.2f [sec]'%(time()-t1)
    print(log)
    return output


def run_CL(OUF1, ELGMAP, RANMAP, WEIGHT, MASK, njack=0, LMAX=512):
    if njack==0:
        delta  = makedelta(ELGMAP, RANMAP, MASK, WEIGHT)
        lsscl  = compute_cl(delta, RANMAP, MASK, lmax=LMAX)
        np.save(OUF1, lsscl)
        print('save C_l in {}'.format(OUF1))
    else:
        print('jackknife C_l is not ready!!!')



### class AngularClustering2D(object):
# #     """
# #     Clustering Statistics Cell and Omega
# #     """
    
# #     def __init__(self, data, random, selection_function=None, mask=None, nside=256, hpmap=True):
# #         if hpmap:
# #             datamap = np.copy(data)
# #             randmap = np.copy(random)
# #         else:
# #             datamap = hpixsum(nside, data.RA, data.DEC, value=data.Weight)
# #             randmap = hpixsum(nside, random.RA, random.DEC, value=random.Weight)
# #         if selection_function is None:
# #             selection_function = np.ones(randmap.shape)
# #         delta   = np.zeros(datamap.shape)
# #         if mask is None:
# #             mask    = (randmap != 0.0) & (selection_function !=0.0)
# #         randsel = randmap * selection_function
# #         sf      = datamap[mask].sum() / randsel[mask].sum()
# #         delta[mask] = datamap[mask]/ (sf * randsel[mask]) - 1.0
# #         self.delta  = delta
# #         self.weight = randmap
# #         self.nside = nside
# #         self.mask  = mask
    
# #     def run(self, LMAX=None):
# #         if LMAX is None:
# #             LMAX           = 3*self.nside-1
# #         else:
# #             assert LMAX <=  3*self.nside-1
# #         x, w = np.polynomial.legendre.leggauss(LMAX)
# #         #map
# #         mapmasked       = hp.ma(self.delta * self.weight)
# #         mapmasked.mask  = np.logical_not(self.mask)
# #         clmap           = hp.anafast(mapmasked.filled(), lmax=LMAX)
# #         ximap           = cl2xi(clmap, x)
# #         #mask
# #         maskmasked      = hp.ma(self.weight)
# #         maskmasked.mask = np.logical_not(self.mask)
# #         clmask          = hp.anafast(maskmasked.filled(), lmax=LMAX)
# #         ximask          = cl2xi(clmask, x)
# #         # correct for the mask
# #         xifinal         = ximap / ximask
# #         clfinal         = xi2cl(x, w, xifinal, LMAX)
# #         # update
# #         output = {}
# #         output['attr'] = {'nside':self.nside, 'lmax':LMAX}
# #         output['cl'] = (np.arange(clfinal.size), clfinal)
# #         output['xi'] = (np.rad2deg(np.arccos(x)), xifinal)
# #         return output

# # #    def run_paircount(self, maxang=10):
# # #        bw = 3.*hp.nside2resol(self.nside)*180./3.1416  # 3x resol.
# # #        bins = np.arange(bw, maxang, bw)
# # #        #delta_i, rani = hp.ud_grade(delta, res), hp.ud_grade(weight, res)
# # #        theta, phi, deltam, fpixm = maps2pcinput(self.delta, self.weight, self.mask)
# # #        w = paircount(theta, phi, deltam, deltam, fpixm, np.deg2rad(bins))
# # #        binc = 0.5*(bins[1:]+bins[:-1])
# # #        return [binc, w[0]/w[1]]




# # # class AngularClustering2D(object):
# # #     """
# # #     Clustering Statistics Cell and Omega
# # #     """
    
# # #     def __init__(self, data, random, selection_function=None, mask=None, nside=256, hpmap=True):
# # #         if hpmap:
# # #             datamap = np.copy(data)
# # #             randmap = np.copy(random)
# # #         else:
# # #             datamap = hpixsum(nside, data.RA, data.DEC, value=data.Weight)
# # #             randmap = hpixsum(nside, random.RA, random.DEC, value=random.Weight)
# # #         if selection_function is None:
# # #             selection_function = np.ones(randmap.shape)
# # #         delta   = np.zeros(datamap.shape)
# # #         if mask is None:
# # #             mask    = (randmap != 0.0) & (selection_function !=0.0)
# # #         randsel = randmap * selection_function
# # #         sf      = datamap[mask].sum() / randsel[mask].sum()
# # #         delta[mask] = datamap[mask]/ (sf * randsel[mask]) - 1.0
# # #         self.delta  = delta
# # #         self.weight = randmap
# # #         self.nside = nside
# # #         self.mask  = mask
    
# # #     def run(self, LMAX=None):
# # #         if LMAX is None:
# # #             LMAX           = 3*self.nside-1
# # #         else:
# # #             assert LMAX <=  3*self.nside-1
# # #         x, w = np.polynomial.legendre.leggauss(LMAX)
# # #         #map
# # #         mapmasked       = hp.ma(self.delta * self.weight)
# # #         mapmasked.mask  = np.logical_not(self.mask)
# # #         clmap           = hp.anafast(mapmasked.filled(), lmax=LMAX)
# # #         ximap           = cl2xi(clmap, x)
# # #         #mask
# # #         maskmasked      = hp.ma(self.weight)
# # #         maskmasked.mask = np.logical_not(self.mask)
# # #         clmask          = hp.anafast(maskmasked.filled(), lmax=LMAX)
# # #         ximask          = cl2xi(clmask, x)
# # #         # correct for the mask
# # #         xifinal         = ximap / ximask
# # #         clfinal         = xi2cl(x, w, xifinal, LMAX)
# # #         # update
# # #         output = {}
# # #         output['attr'] = {'nside':self.nside, 'lmax':LMAX}
# # #         output['cl'] = (np.arange(clfinal.size), clfinal)
# # #         output['xi'] = (np.rad2deg(np.arccos(x)), xifinal)
# # #         return output

# # # #    def run_paircount(self, maxang=10):
# # # #        bw = 3.*hp.nside2resol(self.nside)*180./3.1416  # 3x resol.
# # # #        bins = np.arange(bw, maxang, bw)
# # # #        #delta_i, rani = hp.ud_grade(delta, res), hp.ud_grade(weight, res)
# # # #        theta, phi, deltam, fpixm = maps2pcinput(self.delta, self.weight, self.mask)
# # # #        w = paircount(theta, phi, deltam, deltam, fpixm, np.deg2rad(bins))
# # # #        binc = 0.5*(bins[1:]+bins[:-1])
# # # #        return [binc, w[0]/w[1]]





# # # def apply_window(clth, mask, weight=None, theory=True):
# # #     lmax = clth.size-1 
# # #     if weight is None:
# # #         weight = np.ones(mask.size)
# # #     x,w  = np.polynomial.legendre.leggauss(lmax)
# # #     xith = cl2xi(clth, x)
# # #     weightm = hp.ma(weight)
# # #     weightm.mask = np.logical_not(mask)
# # #     clw = hp.anafast(weightm.filled(), lmax=lmax)
# # #     xiw = cl2xi(clw, x)
# # #     if theory:
# # #         xif = xith * xiw
# # #     else:
# # #         xif = xith / xiw
# # #     clf = xi2cl(x, w, xif, lmax)
# # #     return clf, (x, xif)  


# # # def CL(map1, mask, map2=None, weight1=None, weight2=None, lmax=None, correction=False):
# # #     if lmax is None:
# # #         lmax = 3*hp.get_nside(map1)-1
# # #     # weight1
# # #     if weight1 is not None:
# # #         map1 *= weight1
# # #     else:
# # #         weight1 = np.ones(map1.size)

# # #     # map2 & weight2
# # #     if (map2 is not None): # map2 for cross-power
# # #         if weight2 is not None:
# # #             map2 *= weight2
# # #         else:
# # #             weight2 = np.ones(map2.size)
# # #         map2m = hp.ma(map2)
# # #         map2m.mask = np.logical_not(mask)
# # #         MAP2 = map2m.filled()
# # #         weight = weigh1 * weight2
# # #     else:
# # #         MAP2 = None
# # #         weight = weight1 

# # #     map1m = hp.ma(map1)   # map1
# # #     map1m.mask = np.logical_not(mask)
# # #     MAP1  = map1m.filled()

# # #     # Gauss points
# # #     x, w = np.polynomial.legendre.leggauss(lmax)

# # #     # cl map1
# # #     clmap = hp.anafast(MAP1, map2=MAP2, lmax=lmax)
# # #     ximap = cl2xi(clmap, x)
# # #     if not correction:
# # #         return clmap, (x, ximap)
# # #     else:
# # #         # mask
# # #         weightm  = hp.ma(weight)
# # #         weightm.mask = np.logical_not(mask)
# # #         clweight = hp.anafast(weightm.filled(), lmax=lmax)
# # #         xiweight = cl2xi(clweight, x)
# # #         # correct for the mask
# # #         xifinal  = ximap / xiweight
# # #         clfinal  = xi2cl(x, w, xifinal, lmax)
# # #         # output
# # #         return clfinal, (x, xifinal)

       
# # # def split_jackknife(hpix, weight, delta, sysm, njack=20):
# # #     f = weight.sum() // njack
# # #     hpix_L = []
# # #     hpix_l = []
# # #     frac_L = []
# # #     frac    = 0
# # #     delta_L = []
# # #     delta_l = []
# # #     sysm_L = []
# # #     sysm_l = []
# # #     w_L = []
# # #     w_l = []
# # #     #
# # #     #
# # #     for i in range(hpix.size):
# # #         frac += weight[i]            
# # #         hpix_l.append(hpix[i])
# # #         delta_l.append(delta[i])
# # #         w_l.append(weight[i])
# # #         sysm_l.append(sysm[i])
# # #         #
# # #         #
# # #         if frac >= f:
# # #             hpix_L.append(hpix_l)
# # #             frac_L.append(frac)
# # #             delta_L.append(delta_l)
# # #             w_L.append(w_l)
# # #             sysm_L.append(sysm_l)
# # #             frac    = 0
# # #             sysm_l  = []
# # #             w_l     = []
# # #             hpix_l   = []
# # #             delta_l = []
# # # #         elif i == theta.size-1:
# # # #             theta_L.append(theta_l)
# # # #             phi_L.append(phi_l)
# # # #             frac_L.append(frac)
# # # #             delta_L.append(delta_l)
# # # #             w_L.append(w_l)
# # #     return hpix_L, w_L, delta_L, sysm_L #, frac_L

# # # def makedelta(elgmap, ranmap, select_fun, mask):
# # #     delta = np.zeros(elgmap.size)
# # #     randc = ranmap * select_fun
# # #     sf    = (elgmap[mask].sum() / randc[mask].sum())
# # #     delta[mask] = elgmap[mask] / randc[mask] / sf - 1
# # #     return delta

# # # class CLJack(object):
# # #     def __init__(self, elgmap, ranmap, select_fun, mask, njack=20):
# # #         self.nside = hp.get_nside(elgmap)
# # #         delta = makedelta(elgmap, ranmap, select_fun, mask)
# # #         w = ranmap[mask]
# # #         hpix = np.argwhere(mask).flatten()
# # #         hpixl, wl, deltal,_ = split_jackknife(hpix, w, delta[mask], delta[mask], njack=njack)
# # #         #
# # #         self.hpix   = hpixl
# # #         self.weight = wl
# # #         self.delta  = deltal
# # #         self.njack  = len(hpixl)

# # #     def run(self):
# # #         npix = 12*self.nside**2
# # #         self.cell = {}
# # #         self.xil  = {}
# # #         for m in range(self.njack):
# # #             h = self.hpix.copy()
# # #             w = self.weight.copy()
# # #             d  = self.delta.copy()
# # #             h.pop(m)
# # #             w.pop(m)
# # #             d.pop(m)
# # #             hc = np.concatenate(h)
# # #             wc = np.concatenate(w)
# # #             dc = np.concatenate(d)
# # #             MASK   = np.zeros(npix, dtype='?')
# # #             WEIGHT = np.zeros(npix)
# # #             DELTA  = np.zeros(npix)
# # #             MASK[hc]   = True
# # #             WEIGHT[hc] = wc
# # #             DELTA[hc]  = dc
# # #             t1 = time()
# # #             cl, (t, xi) = CL(DELTA, MASK, weight1=WEIGHT)
# # #             print('sample{} done in {} s'.format(m, time()-t1))
# # #             self.cell[m] = cl
# # #             self.xil[m]  = (t, xi)
# # #         # all 
# # #         h = self.hpix.copy()
# # #         w = self.weight.copy()
# # #         d  = self.delta.copy()
# # #         hc = np.concatenate(h)
# # #         wc = np.concatenate(w)
# # #         dc = np.concatenate(d)
# # #         MASK   = np.zeros(npix, dtype='?')
# # #         WEIGHT = np.zeros(npix)
# # #         DELTA  = np.zeros(npix)
# # #         MASK[hc]   = True
# # #         WEIGHT[hc] = wc
# # #         DELTA[hc]  = dc
# # #         cl, (t, xi) = CL(DELTA, MASK, weight1=WEIGHT)
# # #         self.cell[-1] = cl
# # #         self.xil[-1]  = (t, xi)
# # #         clvar = np.zeros(cl.shape)
# # #         xivar = np.zeros(xi.shape)
# # #         for j in range(self.njack):
# # #             clvar += (self.cell[-1] - self.cell[j])**2
# # #             xivar += (self.xil[-1][1] - self.xil[j][1])**2
# # #         clvar *= (self.njack-1)/self.njack
# # #         xivar *= (self.njack-1)/self.njack
# # #         self.output = dict(cells=self.cell, xil=self.xil,
# # #                            celm=cl, celerr=np.sqrt(clvar),
# # #                            theta=np.rad2deg(np.arccos(t)),
# # #                            xim=xi, xierr=np.sqrt(xivar),
# # #                            njack=self.njack, nside=self.nside)



# # # def run_CLJack(ouname, elgmap, ranmap, select_fun, 
# # #                mask, sysm=None, njack=20):
# # #     if sysm is not None:
# # #        raise ValueError
# # #     else:
# # #        cljack = CLJack(elgmap, ranmap, select_fun, mask, njack=njack)
# # #        cljack.run()
# # #     np.save(ouname, cljack.ouput)
# # #     print('output is saved in {}'.format(ouname))



















# """
# class clustering(object):

    
#     def __init__(self, mapfile, maskfile=None):
#         self.mapfile  = mapfile
#         self.maskfile = maskfile
#         self.cl       = None
#         self.xi       = None
    
#     def get_stats(self, lmax=None):
#         if lmax is None:
#             nside          = hp.pixelfunc.get_nside(self.mapfile)
#             LMAX           = 3*nside-1
#         x, w = np.polynomial.legendre.leggauss(LMAX)
#         if self.maskfile is not None:
#             #map
#             mapmasked       = hp.ma(self.mapfile)
#             mapmasked.mask  = np.logical_not(self.maskfile)
#             clmap           = hp.anafast(mapmasked.filled(), lmax=LMAX)
#             ximap           = cl2xi(clmap, x)
#             #mask
#             maskmasked      = hp.ma(self.maskfile)
#             maskmasked.mask = np.logical_not(self.maskfile)
#             clmask          = hp.anafast(maskmasked.filled(), lmax=LMAX)
#             ximask          = cl2xi(clmask, x)
#             # correct for the mask
#             xifinal         = ximap / ximask
#             clfinal         = xi2cl(x, w, xifinal, LMAX)
#         else:
#             clfinal         = hp.anafast(self.mapfile, lmax=LMAX)
#             xifinal         = cl2xi(clmap, x)
#         # update
#         self.cl = (np.arange(clfinal.size), clfinal)
#         self.xi = (np.rad2deg(np.arccos(x)), xifinal)
    
#     def plot_maps(self, ttl='mollview'):
#         mp = hp.ma(self.mapfile)
#         if self.maskfile is not None:
#             mp.mask = np.logical_not(self.maskfile)
#             hp.mollview(mp.filled(), title=ttl)
#         else:
#             hp.mollview(mp, title=ttl)

#     def plot_stats(self, xlab=['l',r'$\theta$[deg]'],
#                    ylab=[r'l(l+1)C$_{l}$',r'$\omega$']):
#         el, cel = self.cl
#         t, om   = self.xi
#         plt.figure(figsize=(16,10))
#         plt.suptitle('Cell & $\omega$')
#         plt.subplot(121)
#         plt.xlabel(xlab[0]);plt.ylabel(ylab[0])
#         plt.plot(el, el*(el+1)*cel, 'b.', alpha=0.1)
#         plt.xscale('log')
#         plt.subplot(122)
#         plt.xlabel(xlab[1]);plt.ylabel(ylab[1])
#         plt.plot(t, om, 'b.');plt.loglog()


# #def maps2pcinput(galmap, ranmap, fo, returnw=False):
# #    '''
# #        function to extract theta, phi, delta, fpix from ngal, nran maps
# #        maps2pcinput(galmap, ranmap, fo, returnw=False)
# #    '''
# #    nside = hp.get_nside(galmap)
# #    delta, fpix = n2deln(galmap, ranmap, returnw=returnw)
# #    fofile = open(fo, 'w+')
# #    for i in np.argwhere(fpix != 0.0).flatten():
# #        teta, phi = hp.pixelfunc.pix2ang(nside, i, nest=False)
# #        fofile.write("%f %f %f %f \n"%(teta, phi, delta[i], fpix[i]))
# def get_xi_pc(delta, weight, res=512, maxang=180.):
#     bw = hp.nside2resol(res)*180./3.1416
#     bins = np.arange(bw, maxang, bw)
#     delta_i, rani = hp.ud_grade(delta, res), hp.ud_grade(weight, res)
#     theta, phi, deltam, fpixm = maps2pcinput(delta_i, rani)
#     w = paircount(theta, phi, deltam, fpixm, np.deg2rad(bins))
#     binc = 0.5*(bins[1:]+bins[:-1])
#     return [binc, w]

# def get_xi(delta, weight):
#     obj = clustering(delta*weight, weight)
#     obj.get_stats()
#     xistats = np.copy(obj.xi)
#     elcell  = np.copy(obj.cl)
#     del obj
#     return xistats, elcell



# def decra2hpix(nside, dec, ra):
#     pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
#     return pix

# def binstat(nside, ra, dec, value, func='count'):
#     pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
#     npix = hp.nside2npix(nside)
#     bins = [i for i in range(npix+1)]
#     w,_,_ = stats.binned_statistic(pix, value, statistic=func, bins=bins)
#     return w


# def pcxi2cl(x, xi, ndeg, nlmax):
#     '''
#         pair-counting omega to Cell
#     '''
#     xgau, wgau = gauleg(ndeg)
#     spl = iusp(x, xi)
#     ygau = spl(np.arccos(xgau))
#     cl = xi2cl(xgau, wgau, ygau, nlmax)
#     return cl

# def paircountingx2cl(d4, nside):
#     '''
        
#     '''
#     x = np.deg2rad(d4[:,0])
#     xi = d4[:,1]/d4[:,2]
#     ndeg = 3*nside - 1
#     nlmax = ndeg
#     cl = pcxi2cl(x, xi, ndeg, nlmax)
#     l = np.arange(0, cl.size)
#     return l, cl


# def map2cl(mapname,lmax=None,mask=None):
#     if lmax is None:
#         nside = hp.pixelfunc.get_nside(mapname)
#         LMAX = 3*nside-1
#     if mask is not None:
#         mapmasked = hp.ma(mapname)
#         mapmasked.mask = np.logical_not(mask)
#         cl = hp.anafast(mapmasked.filled(), lmax=LMAX)
#         return cl
#     else:
#         cl = hp.anafast(mapname, lmax=LMAX)
#         return cl

# def map2clwcorrect(inmap, inmask=None, res=256):
#     ndeg = 3*res
#     costheta, weights = gauleg(ndeg)
#     mask = None
#     if inmask != None:
#         mask = inmask.astype(np.bool)
#         clmask = map2cl(mask.astype(np.float), mask=mask)
#         omegamask = cl2xi(clmask, costheta)
#     #
#     #
#     #
#     clmap = map2cl(inmap,mask=mask)
#     omegamap = cl2xi(clmap, costheta)
#     #
#     #
#     if mask != None:
#         omegafinal = omegamap/omegamask
#         clfinal = xi2cl(costheta, weights, omegafinal, ndeg-1)
#         tw = [np.arccos(costheta), omegafinal]
#         lcl = [np.arange(len(clfinal)).astype('<f8'), clfinal]
#         return lcl, tw
#     else:
#         tw = [np.arccos(costheta), omegamap]
#         lcl = [np.arange(len(clmap)).astype('<f8'), clmap]
#         return lcl, tw

# def cat2map(fnmap, res=256, normalized=False):
#     dmap = fitsio.read(fnmap)
#     # if normalized:
#     #     omap = hpixsum(res, dmap['RA'], dmap['DEC'], value=dmap['COMP'])
#     # else:
#     #     omap = hpixsum(res, dmap['RA'], dmap['DEC'])
#     omap = hpixsum(res, dmap[:,0], dmap[:,1]) # in general RA-DEC
#     return omap.astype('<f8')


# def makedelta(galcat, rancat):
#     galmap = hpixsum(nside, galcat['ra'], galcat['dec'])
#     delta  = np.zeros(hp.nside2npix(nside))
#     mask   = rancat != 0.0
#     av     = galmap.sum()/rancat.sum()
#     delta[mask] = galmap[mask]/(rancat[mask]*av) - 1.0 
#     return delta

# def n2deln(galmap, ranmap, returnw=False):
#     ave = galmap.sum()/ranmap.sum()
#     delta = np.zeros(galmap.size)
#     arg = ranmap != 0.0
#     delta[arg] = galmap[arg]/(ave*ranmap[arg]) - 1.0
#     if returnw:
#         return delta, ranmap
#     maskmap = np.zeros(len(galmap))
#     maskmap[arg] = 1.0
#     return delta, maskmap

# def map2clus(galn, rann, res=256, oudir='./'):
#     galmap = cat2map(galn, res=res, normalized=False)
#     ranmap = cat2map(rann, res=res, normalized=True)
#     deltamap, mask = n2deln(galmap, ranmap)
#     lcl, tw = map2clwcorrect(deltamap, inmask=mask, res=res)
#     return lcl, tw

# def binner(X, Y, Xbins, statmode = 'mean'):
#     bin_stats,bin_edges,binnumber = stats.binned_statistic(X,Y,statistic=statmode,bins=Xbins)
#     bin_std,bin_edges,binnumber = stats.binned_statistic(X,Y,statistic=np.std,bins=Xbins)
#     bin_count,bin_edges,binnumber = stats.binned_statistic(X,Y,statistic='count',bins=Xbins)
#     errorbar = bin_std/np.sqrt(bin_count)
#     errorbarm = np.array(errorbar)
#     errorbarm[errorbar>=bin_stats] = bin_stats[errorbar>=bin_stats]*.9999999
#     bin_center = 0.5*(bin_edges[1:]+bin_edges[:-1])
#     return bin_center,bin_stats,[errorbarm,errorbar]


# def plotclw(lcl, tw, ou='figure.png'):
#     font = {'family' : 'monospace',
#             'weight' : 'normal',   #bold
#             'size'   : 15}
#     matplotlib.rc('font', **font)  # pass in the font dict as kwargs
#     ncol = 2
#     figsize = (8*ncol, 10)
#     fig, axes = plt.subplots(nrows=1, ncols=ncol, figsize=figsize, dpi=None, sharey=False)
#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                     wspace=0.2, hspace=None)
#     axes[0].set_yscale("log", nonposy='clip')
#     axes[0].set_ylabel(r'$C_{l}$')
#     axes[0].set_xscale("log", nonposy='clip')
#     axes[0].set_xlabel(r'$l$')
#     axes[0].set_xlim([1, 1e3])
#     axes[0].set_ylim([1e-7, 1e-1])
#     bn, bs, be = binner(lcl[0], lcl[1], 15)
#     axes[0].plot(lcl[0], lcl[1], 'b+', alpha=0.5)
#     axes[0].errorbar(bn, bs, yerr=be, color='g', marker='+', ls='None')
#     axes[1].set_yscale("log", nonposy='clip')
#     axes[1].set_ylabel(r'$w$')
#     axes[1].set_xscale("log", nonposy='clip')
#     axes[1].set_xlabel(r'$\theta$')
#     axes[1].set_xlim([0.1, 10.])
#     axes[1].set_ylim([1e-3, 1.])
#     axes[1].plot(np.rad2deg(tw[0]), tw[1], 'b+', alpha=0.5)
#     bn, bs, be = binner(np.rad2deg(tw[0]), tw[1], 1.0)
#     axes[1].errorbar(bn, bs, yerr=be, color='g', marker='+', ls='None')
#     plt.savefig(ou, bbox_inches='tight')
#     plt.show()
# #
# # redundant functions will be removed -- DEC,17,2017
# #
