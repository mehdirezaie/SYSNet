'''
  Run ablation based on a linear multivariate model

'''

import numpy as np
import sys
import NN
from   time import time


#
CONFIG = dict(nchain=1, batchsize=1024,
              nepoch=500, Units=[0], tol=1.e-4, 
              learning_rate=0.001, scale=0.0)
# 
def fix(data, ax=list(range(18))):
    dt = np.dtype([('label', '>f8'), ('hpind', '>i8'), 
                   ('features', ('>f8', len(ax))),
                   ('fracgood', '>f8')])
    d_o = np.zeros(data.size, dtype=dt)
    if len(ax)==1:
        d_o['features'] = data['features'][:, ax].squeeze()
    else:
        d_o['features'] = data['features'][:, ax]
    d_o['label']    = data['label']
    d_o['hpind']     = data['hpind']
    d_o['fracgood'] = data['fracgood']
    return d_o


def get_my_chunk(size, rank, INDICES):
    # distribute files on different task ids
    # chunksize
    nfiles    = len(INDICES)
    remainder = nfiles % size
    chunksize = nfiles // size
    if remainder > 1:
        chunksize += 1
    if nfiles < size:
        chunksize = 1
    my_i     = rank*chunksize
    if rank == size-1:
        my_end = nfiles
    else:
        my_end = np.minimum(rank*chunksize + chunksize, nfiles)
    my_chunk = INDICES[my_i:my_end]
    return my_chunk


# ###

from mpi4py import MPI

# get the size, and the rank of each mpi task
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print('Hi from rank %d, total number of processes %d'%(rank, size))
    from glob import glob
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Run paircounts on mocks')
    ap.add_argument('--axfit',  nargs='*',   type=int,\
                                    default=[i for i in range(18)])    
    ap.add_argument('--data', default='../data/mocks/mocks5folds/'\
                    +'3dbox_nmesh1024_L5274.0_bias1.5_seed100hp256-ngal-featurs5fold.fits.npy')
    ap.add_argument('--output', default='../data/mocks/mocks5folds/test_ablation/')
    ap.add_argument('--rmses', default='ablation_rmse')
    ap.add_argument('--log', default='seed100.log')
    ap.add_argument('--rank', default='0')
    ns = ap.parse_args()
    INDICES  = ns.axfit
    #
    foldname = 'fold'+ns.rank
    data = np.load(ns.data).item()
    train = data['train'][foldname]
    test  = data['test'][foldname]
    valid = data['validation'][foldname]
    import os 
    if not os.path.exists(ns.output):
        print('creating ... ', ns.output)
        os.makedirs(ns.output)    
    del data
    LOGS = {'validmin':[], 'importance':[], 'indices':[]}
else:
    INDICES  = None
    train    = None
    test     = None
    foldname = None
    valid    = None

    
# bcast FILES
INDICES = comm.bcast(INDICES, root=0)
train   = comm.bcast(train, root=0)
test    = comm.bcast(test, root=0)
valid   = comm.bcast(valid, root=0)
foldname =comm.bcast(foldname, root=0)

# for filei in my_chunk:
#     print(filei.split('/')[-1])

INDICES_temp = INDICES.copy()
while len(INDICES_temp) > 1:
    #if rank==0:
    #    print('INDICES : {}'.format(INDICES_temp))
    my_chunk = get_my_chunk(size, rank, INDICES_temp)
    print('indices on rank {} are {}'.format(rank, my_chunk))
    #if rank==0:print('INDICES_temp', INDICES_temp)
    validmin   = []
    valid_rmse = []
    for j in my_chunk:
        All = INDICES_temp.copy()
        All.remove(j)
        if len(All)==0:
            continue
        train_2 = fix(train, ax=All)
        test_2  = fix(test,  ax=All)
        valid_2 = fix(valid, ax=All)
        t1 = time()
        net = NN.Netregression(train_2, valid_2, test_2)
        net.train_evaluate(**CONFIG)
        rmse = []
        for a in net.epoch_MSEs:
            rmse.append(np.sqrt(a[2][:,2]))
        RMSE  = np.column_stack(rmse)
        RMSEm = np.mean(RMSE, axis=1)
        #RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
        baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
        #valid_rmse.append([net.epoch_MSEs[0][2][:,0], RMSEm/baseline, RMSEe/baseline])
        valid_rmse.append([net.epoch_MSEs[0][2][:,0], RMSEm/baseline])
        validmin.append([np.min(RMSEm)])
        print('{} is done in {} s'.format(j, time()-t1))
        del rmse
        del train_2
        del valid_2
        del test_2
        del All
        del RMSE
        del RMSEm
        del net
     
    comm.Barrier()
    #
    valid_rmse=comm.gather(valid_rmse, root=0)
    validmin=comm.gather(validmin, root=0)
    #
    #comm.Barrier()
    if rank ==0:
        validmin = [validj for validi in validmin for validj in validi]
        arg      = np.argmin(validmin)
        np.save(ns.output+ns.rmses+str(INDICES_temp[arg])+'_'+foldname, valid_rmse)
        LOGS['validmin'].append(validmin)
        #print('valid mins are : {}'.format(validmin))
        print('removing {}th systematic'.format(INDICES_temp[arg]))
        LOGS['indices'].append(INDICES_temp.copy())
        LOGS['importance'].append(INDICES_temp[arg])
        INDICES_temp.pop(arg)
        print('new indices', INDICES_temp)
    else:
        INDICES_temp = None
    INDICES_temp = comm.bcast(INDICES_temp, root=0)
#
#
if rank == 0:
    net = NN.Netregression(train, valid, test)
    net.train_evaluate(**CONFIG)
    rmse = []
    for a in net.epoch_MSEs:
        rmse.append(np.sqrt(a[2][:,2]))
    RMSE  = np.column_stack(rmse)
    RMSEm = np.mean(RMSE, axis=1)
    #RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
    baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
    #valid_rmse = [net.epoch_MSEs[0][2][:,0], RMSEm/baseline, RMSEe/baseline]
    valid_rmse = [net.epoch_MSEs[0][2][:,0], RMSEm/baseline]
    LOGS['RMSEall'] = np.min(RMSEm)
    LOGS['baselineRMSE'] = baseline
    np.save(ns.output + ns.log +'_'+foldname, LOGS)
    np.save(ns.output + ns.rmses+ '_all'+'_'+foldname, valid_rmse)
