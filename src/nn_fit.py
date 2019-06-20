'''
    Regression with a Feed Forward Neural Network
    (c) mehdi rezaie
    Feb 21, 2019 
'''
import os
import numpy as np
import sys
import NN

from time import time





def getcf(d):
    from scipy.stats import pearsonr
    # lbl = ['ebv', 'nstar'] + [''.join((s, b)) for s in ['depth', 'seeing', 'airmass', 'skymag', 'exptime'] for b in 'rgz']
    cflist = []
    indices = []
    for i in range(d['train']['fold0']['features'].shape[1]):
        for j in range(5):
            fold = ''.join(['fold', str(j)])
            cf = pearsonr(d['train'][fold]['label'], d['train'][fold]['features'][:,i])[0]
            if np.abs(cf) >= 0.02:
                #print('{:s} : sys_i: {} : cf : {:.4f}'.format(fold, lbl[i], cf))
                indices.append(i)
                cflist.append(cf)
    if len(indices) > 0:
        indices = np.unique(np.array(indices))
        return indices
    else:
        print('no significant features')
        return None
#     cf = []
#     indices = []
#     for i in range(features.shape[1]):
#         cf.append(pearsonr(label, features[:,i]))
#         if np.abs(cf) > 0.0
   
def get_all(ablationlog):
    d = np.load(ablationlog).item()
    indices = None
    for il, l in enumerate(d['validmin']):
       m = (np.array(l) - d['RMSEall']) > 0.0
       #print(np.any(m), np.all(m))
       if np.all(m):
         #print(il, d['indices'][il])
         #print(il, [lbs[m] for m in d['indices'][il]])
         #break
         indices = d['indices'][il]
         break
       if (il == len(d['validmin'])-1) & (np.any(m)):
          indices = [d['indices'][il][-1]]       
    # return either None or indices
    return indices
 
#
if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()   
    #
    if rank == 0:
        log = '! ===== Regression with Neural Net ======\n'        
    else:
        log = None
    log    = comm.bcast(log, root=0)
    #
    if rank == 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        #
        #
        import os 
        import fitsio as ft
        import healpy as hp
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Neural Net regression')
        ap.add_argument('--input',  default='test_train_eboss_dr5-masked.npy')
        ap.add_argument('--output', default='test_train_eboss_dr5-maskednn.npy')
        ap.add_argument('--nside',  default=256, type=int)
        ap.add_argument('--axfit',  nargs='*',   type=int,\
                                    default=[i for i in range(18)])
        ap.add_argument('--ablog',  default='None')
        ns = ap.parse_args()
        NSIDE = ns.nside
        config = dict(nchain=1, batchsize=1024, nepoch=500, Units=[0],
                  tol=1.e-4, scale=0.0, learning_rate=0.001)
        # scales : 1000, 100, 50, 20, 10, 1, 5
        #
        log += 'reading input : {} with nside : {} \n'.format(ns.input, NSIDE)
        log += 'the fiducial config is {}\n'.format(config)
        data   = np.load(ns.input).item()
        if ns.ablog == 'cf':
            axfit  = getcf(data)
        elif os.path.isfile(ns.ablog):
            axfit  = get_all(ns.ablog)
        elif ns.ablog == 'None':
            axfit  = ns.axfit
        else:
            RaiseError("axis is not set correctly!")
        oupath = ns.output
        if not os.path.exists(ns.output):
            os.makedirs(ns.output)
    else:
        oupath = None
        data   = None
        config = None
        axfit  = None

    data   = comm.bcast(data,   root=0)
    oupath = comm.bcast(oupath, root=0)
    config = comm.bcast(config, root=0)
    axfit  = comm.bcast(axfit,  root=0)


    if rank == 0:
        if axfit is None:
            print('Rank %d : no correction is required!!! '%rank)
            print('Rank %d : will make a constant map ... '%rank)
            oumap  = np.ones(12*NSIDE*NSIDE)
            hp.write_map(oupath+'nn-weights.hp'+str(NSIDE)+'.fits', oumap, dtype=np.float64, fits_IDL=False, overwrite=True)
            sys.exit()
        else:
            print('Rank {} : will carry on with {} features '.format(rank, axfit))
    else:
        if axfit is None:
            sys.exit()
        else:
            print('Rank %d : will carry on ...'%rank)


    assert size == 5, 'required number of mpi tasks should be equal to number of folds'
    train_i = data['train']['fold'+str(rank)]
    test_i  = data['test']['fold'+str(rank)]
    valid_i = data['validation']['fold'+str(rank)]

    if rank==0:log+='Rank %d : train %d validation %d and test %d \n'%(rank, train_i.size, test_i.size, valid_i.size)
    comm.Barrier()

    #
    # train Num of Layers
    #
    valid_min  = []
    nlayers = [[0], [40], [20,20], [20, 10, 10], [10, 10, 10, 10]] #  
    if rank==0:
        log+='start the hyper-parameter training with the initial config {}\n'.format(config)
        log+='training num of layers {} \n'.format(nlayers)
        t1  = time()

    for nl_i in nlayers:
        config.update(Units=nl_i)
        net = NN.Netregression(train_i, valid_i, test_i, axfit)
        net.train_evaluate(**config)
        mse = []
        for a in net.epoch_MSEs:
            mse.append(a[2][:,2])
        MSE  = np.mean(np.column_stack(mse), axis=1)
        valid_min.append(np.min(MSE)/net.optionsdic['baselineMSE'][1])
        if rank==0:
            log+='rank{} finished {} in {} s\n'.format(rank, nl_i, time()-t1)

    VMIN    = np.array(valid_min)
    argbest = np.argmin(VMIN)
    nlbest  = nlayers[argbest]
    config.update(Units=nlbest)
    log += 'rank {} best nlayers is :: {}\n'.format(rank, nlbest)
    log += 'rank {} the updated config is {}\n'.format(rank, config)
    if rank ==0:
        print('num of layers is tuned')



    # train regularization scale

    valid_min  = []
    scales = np.array([0.001, 0.01, 0.1, 1.0, 10., 100., 1000.])
    if rank==0:
       log+='training reg. scale {} \n'.format(scales)
       t1 = time()
    for scale_i in scales:
        config.update(scale=scale_i)
        net = NN.Netregression(train_i, valid_i, test_i, axfit)
        net.train_evaluate(**config)
        mse = []
        for a in net.epoch_MSEs:
            mse.append(a[2][:,2])
        MSE  = np.mean(np.column_stack(mse), axis=1)
        valid_min.append(np.min(MSE)/net.optionsdic['baselineMSE'][1])
        if rank==0:
            log+='rank{} finished {} in {} s\n'.format(rank, scale_i, time()-t1)
    VMIN    = np.array(valid_min)
    argbest = np.argmin(VMIN)
    sclbest = scales[argbest]
    config.update(scale=sclbest)
    log += 'rank {} best scale is :: {}\n'.format(rank, sclbest)
    log += 'rank {} the updated config is {}\n'.format(rank, config)
    if rank ==0:print('regularization scale is tuned')    
 
    #
    # train batchsize
    #
    valid_min  = []
    bsizes = np.array([128, 256, 512, 1024, 2048, 4096])
    if rank==0:
        log+='training batchsize {} \n'.format(bsizes)
        t1 = time()
    for bsize in bsizes:
        config.update(batchsize=bsize)
        net = NN.Netregression(train_i, valid_i, test_i, axfit)
        net.train_evaluate(**config)
        mse = []
        for a in net.epoch_MSEs:
            mse.append(a[2][:,2])
        MSE  = np.mean(np.column_stack(mse), axis=1)
        valid_min.append(np.min(MSE)/net.optionsdic['baselineMSE'][1])
        if rank==0:
            log+='rank{} finished {} in {} s\n'.format(rank, bsize, time()-t1)
    VMIN    = np.array(valid_min)
    argbest = np.argmin(VMIN)
    bsbest = bsizes[argbest]
    config.update(batchsize=bsbest)
    log += 'rank {} best Batchsize is :: {}\n'.format(rank, bsbest)
    log += 'rank {} the updated config is {}\n'.format(rank, config)
    if rank ==0:print('batch size is tuned')
       
    if rank==0:
        log+='=======================================\n'
        log+='final run for the best hyper-parameters\n'
    config.update(nchain=10)
    log+='rank : {} BHPS: {}\n'.format(rank, config)
    
    net   = NN.Netregression(train_i, valid_i, test_i, axfit)
    net.train_evaluate(**config)
    net.savez(indir=oupath+'raw/', name='rank_'+str(rank))

    meanY, stdY = net.optionsdic['stats']['ystat']
    predP       = net.test.P
    
    y_avg = []
    for yi in net.chain_y:
        y_avg.append(yi[1].squeeze().tolist())    

    #predY = stdY*np.mean(np.array(y_avg), axis=0) + meanY
    predY = stdY*np.array(y_avg) + meanY

    comm.Barrier()
    log      = comm.gather(log, root=0)
    predY    = comm.gather(predY, root=0)
    predP    = comm.gather(predP, root=0)
    if rank ==0:
        hpix   = np.concatenate(predP)
        ngal   = np.concatenate(predY, axis=1)
        oudata = np.zeros(ngal.shape[1], dtype=[('hpind', 'i8'), \
                         ('weight',('f8', ngal.shape[0]))])

        oudata['hpind']   = hpix
        oudata['weight']  = ngal.T
        oumap             = np.zeros(12*NSIDE*NSIDE)
        oumap[hpix]      = np.median(ngal, axis=0)

        ft.write(oupath+'nn-weights'+str(NSIDE)+'.fits', oudata, clobber=True)
        hp.write_map(oupath+'nn-weights.hp'+str(NSIDE)+'.fits', oumap,\
                     fits_IDL=False, overwrite=True, dtype=np.float64)
        #
        #
        logfile = open(oupath+'nn-log.txt', 'w')
        LOG = ''
        for logi in log:
            LOG+= logi
            
        logfile.write(LOG)
        print(log)
