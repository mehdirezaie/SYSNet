'''
    Regression with a Feed Forward Neural Network
    (c) mehdi rezaie
    June 23, 2019 
    - run h-parameter training for each fold
    - ablation determines the input maps for each fold
'''
import os
import numpy as np
import NN
from time import time


   
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
    # initiate the log file
    if rank == 0:
        log = '! ===== Regression with Neural Net ======\n'        
    else:
        log = None
    log    = comm.bcast(log, root=0)
    #
    #
    if rank == 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        #
        #
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
        log += 'reading input : {} with nside : {} \n'.format(ns.input, NSIDE)
        log += 'the fiducial config is {}\n'.format(config)
        data   = np.load(ns.input).item()
        Ablog  = ns.ablog    
        Axfit  = ns.axfit
        oupath = ns.output
        if not os.path.exists(ns.output):
            os.makedirs(ns.output)
    else:
        oupath = None
        data   = None
        config = None
        Axfit  = None
        Ablog  = None


    #
    NCHAIN = 10        
    data   = comm.bcast(data,   root=0)
    oupath = comm.bcast(oupath, root=0)
    config = comm.bcast(config, root=0)
    Axfit  = comm.bcast(Axfit,  root=0)
    Ablog  = comm.bcast(Ablog,  root=0)    
    #
    #
    if os.path.isfile(Ablog+'_fold'+str(rank)+'.npy'):
        axfit  = get_all(Ablog+'_fold'+str(rank)+'.npy')
    elif Ablog == 'None':
        axfit  = Axfit
    else:
        RaiseError("axis is not set correctly!")
    print('rank {} with axes {}'.format(rank, axfit))
    log += 'rank {} with axes {}\n'.format(rank, axfit)
    
    # load data
    train_i = data['train']['fold'+str(rank)]
    test_i  = data['test']['fold'+str(rank)]
    valid_i = data['validation']['fold'+str(rank)]            
    if axfit is None:
        print('Rank %d : no correction is required!!! '%rank)
        print('Rank %d : will make a constant map ... '%rank)
        predP = test_i['hpind'] # pixel ID
        predY = np.ones((NCHAIN, predP.size))        
    else:        
        #
        # train Num of Layers
        #
        valid_min  = []
        nlayers = [[0], [40], [20,20], [20, 10, 10], [10, 10, 10, 10]] #  
        if rank==0:
            log+='start the hyper-parameter training with the initial config {}\n'.format(config)
            log+='training num of layers {} \n'.format(nlayers)
        for nl_i in nlayers:
            config.update(Units=nl_i)
            net = NN.Netregression(train_i, valid_i, test_i, axfit)
            net.train_evaluate(**config)
            mse = []
            for a in net.epoch_MSEs:
                mse.append(a[2][:,2])
            MSE  = np.mean(np.column_stack(mse), axis=1)
            valid_min.append(np.min(MSE)/net.optionsdic['baselineMSE'][1])
        #    
        VMIN    = np.array(valid_min)
        argbest = np.argmin(VMIN)
        nlbest  = nlayers[argbest]
        config.update(Units=nlbest)
        log += 'rank {} best nlayers is :: {}\n'.format(rank, nlbest)
        log += 'rank {} the updated config is {}\n'.format(rank, config)
        if rank ==0:
            print('num of layers is tuned')
        #
        # train regularization scale
        #
        valid_min  = []
        scales = np.array([0.001, 0.01, 0.1, 1.0, 10., 100., 1000.])
        if rank==0:
            log+='training reg. scale {} \n'.format(scales)
            
        for scale_i in scales:
            config.update(scale=scale_i)
            net = NN.Netregression(train_i, valid_i, test_i, axfit)
            net.train_evaluate(**config)
            mse = []
            for a in net.epoch_MSEs:
                mse.append(a[2][:,2])
            MSE  = np.mean(np.column_stack(mse), axis=1)
            valid_min.append(np.min(MSE)/net.optionsdic['baselineMSE'][1])
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
        for bsize in bsizes:
            config.update(batchsize=bsize)
            net = NN.Netregression(train_i, valid_i, test_i, axfit)
            net.train_evaluate(**config)
            mse = []
            for a in net.epoch_MSEs:
                mse.append(a[2][:,2])
            MSE  = np.mean(np.column_stack(mse), axis=1)
            valid_min.append(np.min(MSE)/net.optionsdic['baselineMSE'][1])

        VMIN    = np.array(valid_min)
        argbest = np.argmin(VMIN)
        bsbest = bsizes[argbest]
        config.update(batchsize=bsbest)
        log += 'rank {} best Batchsize is :: {}\n'.format(rank, bsbest)
        log += 'rank {} the updated config is {}\n'.format(rank, config)
        if rank==0:print('batch size is tuned')
        if rank==0:
            log+='=======================================\n'
            log+='final run for the best hyper-parameters\n'
        config.update(nchain=NCHAIN)
        log+='rank : {} BHPS: {}\n'.format(rank, config)

        net   = NN.Netregression(train_i, valid_i, test_i, axfit)
        net.train_evaluate(**config)
        net.savez(indir=oupath+'raw/', name='rank_'+str(rank))
        #
        meanY, stdY = net.optionsdic['stats']['ystat']
        predP       = net.test.P
        y_avg = []
        for yi in net.chain_y:
            y_avg.append(yi[1].squeeze().tolist())    
        #
        #predY = stdY*np.mean(np.array(y_avg), axis=0) + meanY
        predY = stdY*np.array(y_avg) + meanY
    #    
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