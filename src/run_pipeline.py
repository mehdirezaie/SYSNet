#
#  pipeline that takes 
#  a galaxy healpix map
#  a random healpix map
#  a mask   healpix map
#  splits it if not provided 
#  does the linear/quadratic fit 
#  does a NN fit 
#
#  1) run Lin fit, nnbar w weights, Cl w weights
#  $>  mpirun -np 2 python run_pipeline.py --verbose --hpfit linfit
#       --nnbar nnbar-fit --clfile cllin --wmap weights.hp.256.fits

'''
python run_pipeline.py --galmap /Volumes/TimeMachine/data/eboss/eBOSS_DR7/eBOSSDR7.256.fits --ranmap /Volumes/TimeMachine/data/eboss/eBOSS_DR7/random.256.fits --photattrs /Volumes/TimeMachine/data/eboss/eBOSS_DR7/ngal.features.256.fits --oudir /Volumes/TimeMachine/data/eboss/eBOSS_DR7/clustering/ --nnbar eboss22wnn --verbose --mask /Volumes/TimeMachine/data/eboss/eBOSS_DR7/hpmask.256.fits --wmap /Volumes/TimeMachine/data/eboss/eBOSS_DR7/regression/nn-weights.hp256.fits
'''
import os
import numpy as np
from   time import time
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


labels = ['ebv', 'loghi', 'nstar']
labels += [''.join((s,'-',b)) for s in ['depth', 'seeing', 'skymag', 'exptime', 'mjd']\
           for b in 'rgz']
# greetings
print('Hi from %d'%rank)
if rank == 0:
    print('There are %d workers'%size)
    print('Features are {}'.format(labels))
    

if rank == 0:
    # import rank 0 modules
    import matplotlib as ml
    ml.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('font', family='serif')
    plt.rc('axes.spines', right=False, top=False)
    import healpy as hp
    import fitsio as ft
    from argparse import ArgumentParser
    ap = ArgumentParser(description='PHOT attr. correction pipeline')
    ap.add_argument('--galmap',    default='galaxy.hp.256.fits')
    ap.add_argument('--ranmap',    default='random.hp.256.fits')
    ap.add_argument('--splitdata', default='gal-feat.hp.256.k5.fits')
    ap.add_argument('--photattrs', default='phot-attrs.hp.256.fits')
    ap.add_argument('--wmap',      default='weights.hp.256.fits')
    ap.add_argument('--mask',      default='mask.hp.256.fits')
    ap.add_argument('--log',       default='analysis')
    ap.add_argument('--clfile',    default='none')
    ap.add_argument('--clsys',     default='none')    
    ap.add_argument('--corfile',   default='none')
    ap.add_argument('--corsys',   default='none')
    ap.add_argument('--nnbar',     default='none')
    ap.add_argument('--hpfit',     default='none')
    ap.add_argument('--oudir',     default='./output/')
    ap.add_argument('--axfit',     nargs='*', type=int,\
                                   default=[i for i in range(18)])
    ap.add_argument('--nbin',      default=20, type=int)
    ap.add_argument('--njack',     default=20, type=int)
    ap.add_argument('--nside',     default=256, type=int)
    ap.add_argument('--lmax',      default=512, type=int)
    ap.add_argument('--smooth',    action='store_true')
    ap.add_argument('--verbose',   action='store_true')
    ns = ap.parse_args()
    #
    #
    #fmt  = ['15', '30', '15']  # format list
    log  = 2*'\n'
    log += '!{:15s}{:30s}{:15s}!\n'.format(15*'=','verbose activated', 15*'=')
    log += '!{:15s}{:30s}{:15s}!\n'.format('','PHOT attr. correction pipeline','')
    log += '!{:15s}{:30s}{:15s}!\n'.format('','developed by Mehdi Rezaie', '')
    log += '!{:15s}{:30s}{:15s}!\n'.format('' ,'email : mr095415@ohio.edu','')
    log += '!{:60s}!\n'.format(60*'=')
    # 
    #
    log += '{:15s}\n'.format('INPUT PARAMS')
    args = ns.__dict__
    for (a,b) in zip(args.keys(), args.values()):
        log += '{:6s}{:15s} : {}\n'.format('', a, b) 
    # 
    #
    log += 2*'\n'
    log += '{:60s}\n'.format('Check if the inputs are correct')
    err_flag = False
    wrn_flag = False
    #
    #
    if not os.path.exists(ns.oudir):
        wrn_flag = True
        log += '{:15s} : {:30s}{:15s}\n'.format('oudir',  ns.oudir, 'X\n')
        log += '{:15s}{:14s}{}\n'.format('creating', 14*'.', ns.oudir)
        os.makedirs(ns.oudir)
    else:
        log += '{:15s} : {:30s}{:14s}OK\n'.format('directory', ns.oudir, 14*'.')
    #
    #
    for input_i in ['galmap', 'ranmap',\
                    'photattrs','mask']:
        if not os.path.isfile(args[input_i]):
            err_flag = True
            log += '{:15s} : {:30s}{:14s}err\n'.format(input_i,  args[input_i], 14*'.')
        else:
            log += '{:15s} : {:30s}{:14s}OK\n'.format(input_i, args[input_i], 14*'.')
    #
    #
    for input_i in ['splitdata',\
                    'wmap']:
        if not os.path.isfile(args[input_i]):
            wrn_flag = True
            log += '{:15s} : {:30s}{:14s}X\n'.format(input_i,  args[input_i], 14*'.')
        else:
            log += '{:15s} : {:30s}{:14s}OK\n'.format(input_i, args[input_i], 14*'.')

    # 
    #
    if ns.verbose:
        print(log)
    #
    # proceed if there is no err flag
    if err_flag:
        sys.exit('Exiting the program because there is an error!!!')
    elif wrn_flag:
        print('There is warning flag, anyhow the program will carry on!!!\n')
else:
    ns = None
ns = comm.bcast(ns, root=0)


# fit multivariate function
if (ns.hpfit != 'none'):
    if rank == 0:
        from scipy.optimize import curve_fit
        # log
        fo   = open(ns.oudir + ns.hpfit + '.log', 'w')
        fo.write(log)
        # model
        def model(x, *w):
            return np.dot(x, w)        
        #
        # read phot. attributes file, must have hpix, features
        feat, h   = ft.read(ns.photattrs, header=True, lower=True)
        feats     = feat['features'].shape
        mask1     = hp.read_map(ns.mask,   verbose=False).astype('bool')
        hpmask1   = np.in1d(feat['hpind'], np.argwhere(mask1).flatten())
        galmap    = hp.read_map(ns.galmap, verbose=False)
        ranmap    = hp.read_map(ns.ranmap, verbose=False)
        #
        # log
        log  = 2*'\n'
        log += '{:15s}{:40s}{:15s}\n'.format(15*'=',\
              'Fit a multivariate on Ngal-sys',15*'=')
        log += '{:15s} : \n'.format('HEADER of PHOT attrs')
        log += '{}\n'.format(h)
        log += '{:35s} : {}\n'.format('Phot attrs names', labels)
        log += '\n'
        log += '{:35s} : {}\n'.format('Shape of the phot. attrs', feats)
        log += '{:35s} : {}\n'.format('Total number of mask pixels',  mask1.sum())
        log += '{:35s} : {}\n'.format('Total number of feat pixels',  feats[0])
        log += '{:35s} : {}\n'.format('Total number of overlap mask', hpmask1.sum())
        #
        # multivariate fit on hp pixels
        hpix = feat['hpind'][hpmask1]
        x    = feat['features'][hpmask1][:,ns.axfit]
        xstat= np.mean(x, axis=0), np.std(x, axis=0)
        xs   = (x-xstat[0])/xstat[1]
        xsb  = np.concatenate([np.ones(xs.shape[0])[:,np.newaxis], xs],\
                             axis=1) # add bias feature
        avg  = ranmap[hpix].sum()/galmap[hpix].sum()
        y    = (galmap[hpix]/ranmap[hpix]) * avg
        ystat= np.mean(y, axis=0), np.std(y, axis=0)
        ys   = (y-ystat[0])/ystat[1]
        ye   = 1./(ranmap[hpix]/np.mean(ranmap[hpix]))  # Nran/Nran_exp
        #
        # model
        rmse = lambda y1, y2, noise: np.sqrt(np.mean(((y1-y2)/noise)**2))
        popt, pcov = curve_fit(model, xsb, ys,\
                    p0=[1.]+[0 for i in range(len(ns.axfit))],\
                    sigma=ye, absolute_sigma=True)
        yp   = model(xsb, *popt)
        wmap = np.zeros(12*ns.nside*ns.nside)
        wmap[hpix] = (yp*ystat[1]) + ystat[0]   # scale back
        hp.write_map(ns.wmap, wmap, overwrite=True, fits_IDL=False)
        np.save(ns.oudir + ns.hpfit, dict(params=(popt, pcov),\
               axfit=ns.axfit, xstat=xstat, ystat=ystat, avg=avg))
        #
        # log
        log += '{:35s} : {}\n'.format('avg nran/ngal', avg)
        log += '{:35s} : {}\n'.format('Index of features for fit', ns.axfit)
        log += '{:35s} : {}\n'.format('Features for fitting', [labels[i] for i in ns.axfit])
        log += '{:35s} : {}\n'.format('Best fit params', popt)
        log += '{:35s} : {}\n'.format('RMSE ', rmse(yp, ys, ye))
        log += '{:35s} : {}\n'.format('RMSE (baseline)', rmse(0, ys, ye))
        log += '{:35s} : {}\n'.format('Write wmap ', ns.wmap)   
        fo.write(log)
        if ns.verbose:
            print(log)
#      
# NNbar 
if (ns.nnbar != 'none'):
    if rank == 0:
        fo   = open(ns.oudir + ns.nnbar + '.log', 'w')
        fo.write(log)
        from nnbar import NNBAR
        log  = 2*'\n'
        log += '{:15s}{:40s}{:15s}\n'.format(15*'=',\
               'Look at Ngal vs. systematics',15*'=')
        #
        # read phot. attributes file, must have hpix, features
        feat, h = ft.read(ns.photattrs, header=True, lower=True)
        feats = feat['features'].shape
        #
        # read mask, galaxy and random map
        mask1   = hp.read_map(ns.mask,   verbose=False).astype('bool')
        mask2   = np.zeros_like(mask1).astype('bool')
        mask2[feat['hpind']] = True
        mask    = mask1 & mask2         # overlap 
        galmap  = hp.read_map(ns.galmap, verbose=False)
        ranmap  = hp.read_map(ns.ranmap, verbose=False)
        if not os.path.isfile(ns.wmap):
            wmap = None
            log += '{:35s} : {}\n'.format('Computing NNbar w/o wmap', ns.wmap)
        else: 
            wmap = hp.read_map(ns.wmap, verbose=False)
            log += '{:35s} : {}\n'.format('Computing NNbar with wmap', ns.wmap)
            if ns.smooth:
                log += '{:35s}\n'.format('Smoothing the wmap')
                wmap[~mask] = np.mean(wmap[mask]) # fill in empty pixels ?? required for smoothing 
                sdeg = np.deg2rad(0.25)           # smooth with sigma of 1/4 of a deg
                wmap = hp.sphtfunc.smoothing(wmap.copy(), sigma=sdeg)
        #
        # log
        log += '{:15s} : \n'.format('HEADER of PHOT attrs')
        log += '{}\n'.format(h)
        log += '{:35s} : {}\n'.format('Phot attrs names', labels)
        log += '\n'
        log += '{:35s} : {}\n'.format('Shape of the phot. attrs', feats)
        log += '{:35s} : {}\n'.format('Total number of mask pixels',  mask1.sum())
        log += '{:35s} : {}\n'.format('Total number of feat pixels',  mask2.sum())
        log += '{:35s} : {}\n'.format('Total number of overlap mask', mask.sum())      
        #
        # plot for nnbar vs. sys.
        #
        if feats[1] %  3 == 0:
            nrows = feats[1] // 3
        else:
            nrows = feats[1] // 3 + 1
            
        fig, ax = plt.subplots(ncols=3, nrows=nrows, sharey=True, figsize=(12, 3*nrows))
        plt.subplots_adjust(wspace=0.02, hspace=0.5)
        ax = ax.flatten()
        nnbar_res = {'nnbar':[], 'xlabels':labels}
        #
        # compute nnbar vs. systematics
        for i in range(feats[1]):
            sys_i = np.zeros_like(galmap)
            sys_i[feat['hpind']] = feat['features'][:,i]
            #
            Nnbar = NNBAR(galmap, ranmap, mask, sys_i, nbins=ns.nbin, selection=wmap)
            Nnbar.run(njack=20)
            nnbar_res['nnbar'].append(Nnbar.output)
            ax[i].errorbar(Nnbar.output['bin_edges'][:-1], Nnbar.output['nnbar'],
                          yerr=Nnbar.output['nnbar_err'])
            ax[i].axhline(1, linestyle=':', color='k')
            ax2 = ax[i].twinx()
            ax2.step(Nnbar.output['bin_edges'][:-1], Nnbar.output['area'],\
                    where='post', linestyle='--')
            ax2.set_yticks([])
            if i%3 == 0:ax[i].set_ylabel(r'N/$\overline{N}$')
            ax[i].set_xlabel(labels[i])
            ax[i].set_ylim(0.6, 1.4)
        #
        #
        log += '{:15s} : {}\n'.format('Outputs are under', ns.oudir)
        fo.write(log)
        plt.savefig(ns.oudir + ns.nnbar + '.png', bbox_inches='tight', dpi=300)
        np.save(ns.oudir + ns.nnbar, nnbar_res)
        if ns.verbose:print(log)


#
# C_l
if ns.clfile != 'none':
    if rank == 0:
        fo   = open(ns.oudir + ns.clfile + '.log', 'w')
        fo.write(log)
        log  = 2*'\n'
        log += '{:15s}{:40s}{:15s}\n'.format(15*'=',\
              'Compute the auto and cross C_l',15*'=')
        from utils import makedelta
        #
        # read phot. attributes file, must have hpix, features
        feat, h   = ft.read(ns.photattrs, header=True, lower=True)
        feats     = feat['features'].shape
        mask1     = hp.read_map(ns.mask,   verbose=False).astype('bool')
        mask2     = np.zeros_like(mask1).astype('bool')
        mask2[feat['hpind']] = True
        mask      = mask1 & mask2         # overlap 
        galmap    = hp.read_map(ns.galmap, verbose=False)
        ranmap    = hp.read_map(ns.ranmap, verbose=False)
        #
        # check if weight is available
        if not os.path.isfile(ns.wmap):
            wmap = None
            log += '{:35s} : {}\n'.format('Computing Cl w/o wmap', ns.wmap)
        else: 
            wmap = hp.read_map(ns.wmap, verbose=False)
            log += '{:35s} : {}\n'.format('Computing Cl with wmap', ns.wmap)
            if ns.smooth:
                log += '{:35s}\n'.format('Smoothing the wmap')
                wmap[~mask] = np.mean(wmap[mask]) # fill in empty pixels ?? required for smoothing 
                sdeg = np.deg2rad(0.25)           # smooth with sigma of 1/4 of a deg
                wmap = hp.sphtfunc.smoothing(wmap.copy(), sigma=sdeg)
        # 
        # construct delta 
        delta_ngal = makedelta(galmap, ranmap, mask, select_fun=wmap)
        #
        # compute C_l
        map_ngal       = hp.ma(delta_ngal * ranmap)
        map_ngal.mask  = np.logical_not(mask)
        cl_auto        = hp.anafast(map_ngal.filled(), lmax=ns.lmax)
        #
        # maps to do the cross correlation
        x    = feat['features'][:,ns.axfit]
        hpix = feat['hpind']
        #
        # log
        log += '{:15s} : \n'.format('HEADER of PHOT attrs')
        log += '{}\n'.format(h)
        log += '{:35s} : {}\n'.format('Phot attrs names', labels)
        log += '\n'
        log += '{:35s} : {}\n'.format('Shape of the phot. attrs', feats)
        log += '{:35s} : {}\n'.format('Total number of mask pixels',  mask1.sum())
        log += '{:35s} : {}\n'.format('Total number of feat pixels',  mask2.sum())
        log += '{:35s} : {}\n'.format('Total number of overlap mask', mask.sum())      
        log += '{:35s} : {}\n'.format('Compute the cross power spectra against', [labels[n] for n in ns.axfit])
    else:
        x         = None
        ranmap    = None
        mask      = None
        map_ngal  = None
        hpix      = None
        makedelta = None
       
    #
    # bcast
    x          = comm.bcast(x, root=0)
    map_ngal   = comm.bcast(map_ngal, root=0)
    mask       = comm.bcast(mask, root=0)
    ranmap     = comm.bcast(ranmap, root=0)
    hpix       = comm.bcast(hpix, root=0)
    makedelta  = comm.bcast(makedelta, root=0)
    from healpy import ma, anafast
    
    #
    # split the sysmaps (x) among workers
    if x.shape[1]%size ==0:
        my_size = x.shape[1] // size
    else:
        my_size = x.shape[1] // size + 1
    #
    #
    my_i  = rank*my_size
    my_f  = np.minimum(x.shape[1], (rank+1)*my_size)
    my_x  = x[:, my_i:my_f]
    my_cl = []
    for i in range(my_i, my_f):
        sys_i        = np.zeros_like(ranmap)
        sys_i[hpix]  = x[:, i]
        delta_sys    = makedelta(sys_i, ranmap, mask, select_fun=None, is_sys=True)
        map_sys      = ma(delta_sys * ranmap)
        map_sys.mask = np.logical_not(mask)
        my_cl_i      = anafast(map_ngal.filled(), map2=map_sys.filled(),\
                               lmax=ns.lmax)
        my_cl.append(my_cl_i)
    #
    #
    # gather the cross power spectra
    comm.Barrier()
    my_cl = comm.gather(my_cl, root=0)
    if rank == 0:
        all_cl = np.concatenate(my_cl, axis=0)
        colors = plt.cm.jet
        plt.figure()
        for j in range(all_cl.shape[0]):
            plt.plot(np.arange(all_cl.shape[1]), all_cl[j,:],\
                     label=r'Ngal$\times$%s'%labels[ns.axfit[j]], \
                     color=colors(j/all_cl.shape[0]))
        plt.plot(np.arange(cl_auto.size), cl_auto, color='k', label=r'Ngal$\times$Ngal')
        plt.xscale('log');plt.legend(ncol=2, bbox_to_anchor=(1.01,1.01))
        plt.ylabel(r'C$_{l}$');plt.xlabel('l')
        plt.savefig(ns.oudir + ns.clfile + '.png', bbox_inches='tight', dpi=300)
        All_cl = dict(cross=all_cl, auto=cl_auto, clabels=[labels[n] for n in ns.axfit])
        np.save(ns.oudir  + ns.clfile, All_cl)
        log   += '{:35s} : {}\n'.format('Outpus saved under', ns.oudir)
        fo.write(log)
        if ns.verbose:print(log)

if ns.clsys != 'none':
    if rank == 0:
        fo   = open(ns.oudir + ns.clsys + '.log', 'w')
        fo.write(log)
        log  = 2*'\n'
        log += '{:15s}{:40s}{:15s}\n'.format(15*'=',\
              'Compute the auto C_l for systematics',15*'=')
        from utils import makedelta
        #
        # read phot. attributes file, must have hpix, features
        feat, h   = ft.read(ns.photattrs, header=True, lower=True)
        feats     = feat['features'].shape
        mask1     = hp.read_map(ns.mask,   verbose=False).astype('bool')
        mask2     = np.zeros_like(mask1).astype('bool')
        mask2[feat['hpind']] = True
        mask      = mask1 & mask2         # overlap 
        
        ranmap    = hp.read_map(ns.ranmap, verbose=False)
        #
        # maps to do the cross correlation
        x    = feat['features'][:,ns.axfit]
        hpix = feat['hpind']
        #
        # log
        log += '{:15s} : \n'.format('HEADER of PHOT attrs')
        log += '{}\n'.format(h)
        log += '{:35s} : {}\n'.format('Phot attrs names', labels)
        log += '\n'
        log += '{:35s} : {}\n'.format('Shape of the phot. attrs', feats)
        log += '{:35s} : {}\n'.format('Total number of mask pixels',  mask1.sum())
        log += '{:35s} : {}\n'.format('Total number of feat pixels',  mask2.sum())
        log += '{:35s} : {}\n'.format('Total number of overlap mask', mask.sum())      
        log += '{:35s} : {}\n'.format('Compute the auto power spectra for systematics', [labels[n] for n in ns.axfit])
    else:
        x         = None
        ranmap    = None
        mask      = None
        hpix      = None
        makedelta = None
       
    #
    # bcast
    x          = comm.bcast(x, root=0)
    mask       = comm.bcast(mask, root=0)
    ranmap     = comm.bcast(ranmap, root=0)
    hpix       = comm.bcast(hpix, root=0)
    makedelta  = comm.bcast(makedelta, root=0)
    from healpy import ma, anafast
    
    #
    # split the sysmaps (x) among workers
    if x.shape[1]%size ==0:
        my_size = x.shape[1] // size
    else:
        my_size = x.shape[1] // size + 1
    #
    #
    my_i  = rank*my_size
    my_f  = np.minimum(x.shape[1], (rank+1)*my_size)
    my_x  = x[:, my_i:my_f]
    my_cl = []
    for i in range(my_i, my_f):
        sys_i        = np.zeros_like(ranmap)
        sys_i[hpix]  = x[:, i]
        delta_sys    = makedelta(sys_i, ranmap, mask, select_fun=None, is_sys=True)
        map_sys      = ma(delta_sys * ranmap)
        map_sys.mask = np.logical_not(mask)
        my_cl_i      = anafast(map_sys.filled(), lmax=ns.lmax)
        my_cl.append(my_cl_i)
    #
    #
    # gather the cross power spectra
    comm.Barrier()
    my_cl = comm.gather(my_cl, root=0)
    if rank == 0:
        all_cl = np.concatenate(my_cl, axis=0)
        colors = plt.cm.jet
        plt.figure()
        for j in range(all_cl.shape[0]):
            plt.plot(np.arange(all_cl.shape[1]), all_cl[j,:],\
                     label=r'%s'%labels[ns.axfit[j]], \
                     color=colors(j/all_cl.shape[0]))
        
        plt.xscale('log');plt.legend(ncol=2, bbox_to_anchor=(1.01,1.01))
        plt.ylabel(r'C$_{l}$');plt.xlabel('l')
        plt.savefig(ns.oudir + ns.clsys + '.png', bbox_inches='tight', dpi=300)
        All_cl = dict(cross=all_cl, auto=None, clabels=[labels[n] for n in ns.axfit])
        np.save(ns.oudir  + ns.clsys, All_cl)
        log   += '{:35s} : {}\n'.format('Outpus saved under', ns.oudir)
        fo.write(log)
        if ns.verbose:print(log)    
            
#
# Corr
if ns.corfile != 'none':    
    from xi import run_XI
    if rank == 0:        
        fo   = open(ns.oudir + ns.corfile + '.log', 'w')
        fo.write(log)
        log  = 2*'\n'
        log += '{:15s}{:40s}{:15s}\n'.format(15*'=',\
              'Compute the auto and cross xi',15*'=')
        from utils import makedelta
        #
        # read phot. attributes file, must have hpix, features
        feat, h   = ft.read(ns.photattrs, header=True, lower=True)
        feats     = feat['features'].shape
        mask1     = hp.read_map(ns.mask,   verbose=False).astype('bool')
        mask2     = np.zeros_like(mask1).astype('bool')
        mask2[feat['hpind']] = True
        mask      = mask1 & mask2         # overlap 
        galmap    = hp.read_map(ns.galmap, verbose=False)
        ranmap    = hp.read_map(ns.ranmap, verbose=False)
        #
        # check if weight is available
        if not os.path.isfile(ns.wmap):
            wmap = None
            log += '{:35s} : {}\n'.format('Computing Xi w/o wmap', ns.wmap)
        else: 
            wmap = hp.read_map(ns.wmap, verbose=False)
            log += '{:35s} : {}\n'.format('Computing Xi with wmap', ns.wmap)
            if ns.smooth:
                log += '{:35s}\n'.format('Smoothing the wmap')
                wmap[~mask] = np.mean(wmap[mask]) # fill in empty pixels ?? required for smoothing 
                sdeg = np.deg2rad(0.25)           # smooth with sigma of 1/4 of a deg
                wmap = hp.sphtfunc.smoothing(wmap.copy(), sigma=sdeg)
        # 
        # compute auto Xi
        xi_auto = run_XI(None, galmap, ranmap, wmap, mask, sysm=None, njack=ns.njack, Return=True)

        #
        # maps to do the cross correlation
        x    = feat['features'][:,ns.axfit]
        hpix = feat['hpind']
        #
        # log
        log += '{:15s} : \n'.format('HEADER of PHOT attrs')
        log += '{}\n'.format(h)
        log += '{:35s} : {}\n'.format('Phot attrs names', labels)
        log += '\n'
        log += '{:35s} : {}\n'.format('Shape of the phot. attrs', feats)
        log += '{:35s} : {}\n'.format('Total number of mask pixels',  mask1.sum())
        log += '{:35s} : {}\n'.format('Total number of feat pixels',  mask2.sum())
        log += '{:35s} : {}\n'.format('Total number of overlap mask', mask.sum())      
        log += '{:35s} : {}\n'.format('Compute the cross corr. functions against', [labels[n] for n in ns.axfit])
    else:
        x         = None
        ranmap    = None
        mask      = None
        galmap    = None
        hpix      = None
        wmap      = None
        
       
    #
    # bcast
    x          = comm.bcast(x, root=0)
    galmap     = comm.bcast(galmap, root=0)
    mask       = comm.bcast(mask,   root=0)
    ranmap     = comm.bcast(ranmap, root=0)
    hpix       = comm.bcast(hpix, root=0)
    wmap       = comm.bcast(wmap, root=0)
    
    
    #
    # split the sysmaps (x) among workers
    if x.shape[1]%size ==0:
        my_size = x.shape[1] // size
    else:
        my_size = x.shape[1] // size + 1
    #
    #
    my_i  = rank*my_size
    my_f  = np.minimum(x.shape[1], (rank+1)*my_size)
    my_x  = x[:, my_i:my_f]
    my_xi = []
    for i in range(my_i, my_f):
        sys_i        = np.zeros_like(ranmap)
        sys_i[hpix]  = x[:, i]
        my_xi_i      = run_XI(None, galmap, ranmap, wmap, mask, sysm=sys_i, njack=ns.njack, Return=True)
        my_xi.append(my_xi_i)
    #
    #
    # gather the cross power spectra
    comm.Barrier()
    my_xi = comm.gather(my_xi, root=0)
    if rank == 0:
        #print(my_xi, type(my_xi), len(my_xi))
        all_xi = np.concatenate(my_xi, axis=0)
        All_xi = dict(cross=all_xi, auto=xi_auto, clabels=[labels[n] for n in ns.axfit])
        np.save(ns.oudir  + ns.corfile, All_xi)
        log   += '{:35s} : {}\n'.format('Outpus saved under', ns.oudir)
        fo.write(log)
        if ns.verbose:print(log)
            
if ns.corsys != 'none':    
    from xi import run_XIsys
    if rank == 0:        
        fo   = open(ns.oudir + ns.corsys + '.log', 'w')
        fo.write(log)
        log  = 2*'\n'
        log += '{:15s}{:40s}{:15s}\n'.format(15*'=',\
              'Compute the auto corr for systematics',15*'=')
        from utils import makedelta
        #
        # read phot. attributes file, must have hpix, features
        feat, h   = ft.read(ns.photattrs, header=True, lower=True)
        feats     = feat['features'].shape
        mask1     = hp.read_map(ns.mask,   verbose=False).astype('bool')
        mask2     = np.zeros_like(mask1).astype('bool')
        mask2[feat['hpind']] = True
        mask      = mask1 & mask2         # overlap 
        ranmap    = hp.read_map(ns.ranmap, verbose=False)
        #
        # maps to do the cross correlation
        x    = feat['features'][:,ns.axfit]
        hpix = feat['hpind']
        #
        # log
        log += '{:15s} : \n'.format('HEADER of PHOT attrs')
        log += '{}\n'.format(h)
        log += '{:35s} : {}\n'.format('Phot attrs names', labels)
        log += '\n'
        log += '{:35s} : {}\n'.format('Shape of the phot. attrs', feats)
        log += '{:35s} : {}\n'.format('Total number of mask pixels',  mask1.sum())
        log += '{:35s} : {}\n'.format('Total number of feat pixels',  mask2.sum())
        log += '{:35s} : {}\n'.format('Total number of overlap mask', mask.sum())      
        log += '{:35s} : {}\n'.format('Compute the auto corr. functions for systematics', [labels[n] for n in ns.axfit])
    else:
        x         = None
        ranmap    = None
        mask      = None
        hpix      = None
        
       
    #
    # bcast
    x          = comm.bcast(x, root=0)
    mask       = comm.bcast(mask,   root=0)
    ranmap     = comm.bcast(ranmap, root=0)
    hpix       = comm.bcast(hpix, root=0)
    
    
    #
    # split the sysmaps (x) among workers
    if x.shape[1]%size ==0:
        my_size = x.shape[1] // size
    else:
        my_size = x.shape[1] // size + 1
    #
    #
    my_i  = rank*my_size
    my_f  = np.minimum(x.shape[1], (rank+1)*my_size)
    my_x  = x[:, my_i:my_f]
    my_xi = []
    for i in range(my_i, my_f):
        sys_i        = np.zeros_like(ranmap)
        sys_i[hpix]  = x[:, i]
        my_xi_i      = run_XIsys(None, sys_i, ranmap, mask, Return=True)
        my_xi.append(my_xi_i)
    #
    #
    # gather the cross power spectra
    comm.Barrier()
    my_xi = comm.gather(my_xi, root=0)
    if rank == 0:
        #print(my_xi, type(my_xi), len(my_xi))
        all_xi = np.concatenate(my_xi, axis=0)
        All_xi = dict(cross=all_xi, auto=None, clabels=[labels[n] for n in ns.axfit])
        np.save(ns.oudir  + ns.corsys, All_xi)
        log   += '{:35s} : {}\n'.format('Outpus saved under', ns.oudir)
        fo.write(log)
        if ns.verbose:print(log)            
