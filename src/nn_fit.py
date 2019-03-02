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


from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


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
    config = dict(nchain=1, batchsize=1024, nepoch=100, Units=[0],
              tol=0.0, scale=0.0, learning_rate=0.001)
    # scales : 1000, 100, 50, 20, 10, 1, 5
    #
    log  = '! ===== Regression with Neural Net ======\n'
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
     rmse = []
     for a in net.epoch_MSEs:
         rmse.append(np.sqrt(a[2][:,2]))
     RMSE  = np.column_stack(rmse)
     RMSEm = np.mean(RMSE, axis=1)
     baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
     #RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
     #valid_rmse[str(nl_i)] = [net.epoch_MSEs[0][2][:,0],\
     # RMSEm/baseline, RMSEe/baseline]
     valid_min.append(np.min(RMSEm)/baseline)
     if rank==0:
        log+='rank{} finished {} in {} s\n'.format(rank, nl_i, time()-t1)
     #plt.plot(np.arange(RMSEm.size), RMSEm/baseline, 
     #label='{}'.format(nl_i))
comm.Barrier()

#if rank ==0:
# #  plt.legend()
# #  plt.ylim(0.95, 1.05)
# #  plt.show() 
# #sys.exit()
# # valid_rmse = comm.gather(valid_rmse, root=0)

valid_min  = comm.gather(valid_min, root=0)

if rank == 0:
     VMIN = np.array(valid_min)
     Mean = VMIN.mean(axis=0)
     argbest = np.argmin(Mean)
     x = np.arange(VMIN.shape[1])
     plt.figure(figsize=(4,3))
     for i in range(VMIN.shape[0]):
         plt.plot(x, VMIN[i,:], color='grey', ls=':', alpha=0.5)
     plt.plot(x, Mean, color='red', ls='-',\
              label='Average across the folds')
     plt.axvline(x[argbest], ls='--', color='k',\
                 label='Capacity : CV estimate')
     plt.xticks(x, [str(l) for l in nlayers], rotation=45)
     plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
     plt.xlabel('Capacity')
     plt.legend()
     nlbest = nlayers[argbest]
     config.update(Units=nlbest)
     log += 'best nlayers is :: {}\n'.format(nlbest)
     log += 'the updated config is {}\n'.format(config)
     plt.savefig(oupath+'nlayers_validation.pdf', bbox_inches='tight')
else:
     config = None    
config = comm.bcast(config, root=0)
if rank == 0:print(config)
    

#
# train learning rate scale
#
valid_min  = []
lrates = np.array([0.0001, 0.001, 0.01])
if rank==0:
   log+='training learning rate {} \n'.format(lrates)
   t1 = time()
for lrate in lrates:
    config.update(learning_rate=lrate)
    net = NN.Netregression(train_i, valid_i, test_i, axfit)
    net.train_evaluate(**config)
    rmse = []
    for a in net.epoch_MSEs:
        rmse.append(np.sqrt(a[2][:,2]))
    RMSE  = np.column_stack(rmse)
    RMSEm = np.mean(RMSE, axis=1)
    baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
    valid_min.append(np.min(RMSEm)/baseline)
    if rank==0:
       log+='rank{} finished {} in {} s\n'.format(rank, lrate, time()-t1)
comm.Barrier()
valid_min  = comm.gather(valid_min, root=0)
if rank == 0:
    VMIN = np.array(valid_min)
    Mean = VMIN.mean(axis=0)
    argbest = np.argmin(Mean)
    plt.figure(figsize=(4,3))
    for i in range(VMIN.shape[0]):
        plt.plot(-np.log10(lrates), VMIN[i,:], color='grey',\
                  ls=':', alpha=0.5)

    plt.plot(-np.log10(lrates), Mean, color='red', ls='-',\
             label='Average across the folds')
    plt.axvline(-np.log10(lrates[argbest]), ls='--', color='k',\
                label=r'Learning Rate : CV estimate')
    plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
    plt.xlabel(r'$-\log(Learning Rate)$')
    plt.legend()    
    lrbest = lrates[argbest]
    config.update(learning_rate=lrbest)
    log += 'best learning rate is :: {}\n'.format(lrbest)
    log += 'the updated config is {}\n'.format(config)
    plt.savefig(oupath+'learningrate_validation.pdf', bbox_inches='tight')
else:
    config = None    
config = comm.bcast(config, root=0)    
if rank == 0:print(config)
#
# train batchsize
#
valid_min  = []
bsizes = np.array([512, 1024, 2048, 4096])
if rank==0:
   log+='training batchsize {} \n'.format(bsizes)
   t1 = time()
for bsize in bsizes:
     config.update(batchsize=bsize)
     net = NN.Netregression(train_i, valid_i, test_i, axfit)
     net.train_evaluate(**config)
     rmse = []
     for a in net.epoch_MSEs:
         rmse.append(np.sqrt(a[2][:,2]))
     RMSE  = np.column_stack(rmse)
     RMSEm = np.mean(RMSE, axis=1)
     baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
     valid_min.append(np.min(RMSEm)/baseline)
     if rank==0:
        log+='rank{} finished {} in {} s\n'.format(rank, bsize, time()-t1)
comm.Barrier()
valid_min  = comm.gather(valid_min, root=0)
if rank == 0:
     VMIN = np.array(valid_min)
     Mean = VMIN.mean(axis=0)
     argbest = np.argmin(Mean)
     plt.figure(figsize=(4,3))
     for i in range(VMIN.shape[0]):
         plt.plot(np.log10(bsizes), VMIN[i,:], color='grey', ls=':',\
                  alpha=0.5)

     plt.plot(np.log10(bsizes), Mean, color='red',   ls='-', \
             label='Average across the folds')
     plt.axvline(np.log10(bsizes[argbest]), ls='--', color='k',\
                 label=r'Batch Size : CV estimate')
     plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
     plt.xlabel(r'$\log(Batch Size)$')
     plt.legend()    
     bsbest = bsizes[argbest]
     config.update(batchsize=bsbest)
     log +='best Batchsize is :: {}\n'.format(bsbest)
     log += 'the updated config is {}\n'.format(config)
     plt.savefig(oupath+'batchsize_validation.pdf', bbox_inches='tight')
else:
     config = None     
config = comm.bcast(config, root=0)        
if rank == 0:print(config)    


# train regularization scale

valid_min  = []
scales = np.exp([-50.0, -40.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0])
if rank==0:
   log+='training reg. scale {} \n'.format(scales)
   t1 = time()
for scale_i in scales:
     config.update(scale=scale_i)
     net = NN.Netregression(train_i, valid_i, test_i, axfit)
     net.train_evaluate(**config)
     rmse = []
     for a in net.epoch_MSEs:
         rmse.append(np.sqrt(a[2][:,2]))
     RMSE  = np.column_stack(rmse)
     RMSEm = np.mean(RMSE, axis=1)
     baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
     valid_min.append(np.min(RMSEm)/baseline)
     if rank==0:
        log+='rank{} finished {} in {} s\n'.format(rank, scale_i, time()-t1)

comm.Barrier()
valid_min  = comm.gather(valid_min, root=0)
if rank == 0:
     VMIN = np.array(valid_min)
     Mean = VMIN.mean(axis=0)
     argbest = np.argmin(Mean)
     plt.figure(figsize=(4,3))
     for i in range(VMIN.shape[0]):
         plt.plot(-np.log(scales), VMIN[i,:], color='grey', 
                  ls=':', alpha=0.5)
     
     plt.plot(-np.log(scales), Mean, color='red', ls='-',
             label='Average across the folds')
     plt.axvline(-np.log(scales[argbest]), ls='--', color='k',
                 label=r'$\lambda$ : CV estimate')
     plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
     plt.xlabel(r'$-\ln(\lambda)$')
     plt.legend()
     sclbest = scales[argbest]
     config.update(scale=sclbest)
     log +='best scale is :: {}\n'.format(sclbest)
     log += 'the updated config is {}\n'.format(config)
     plt.savefig(oupath + 'scales_validation.pdf', bbox_inches='tight')
else:
     config = None
config = comm.bcast(config, root=0)

if rank == 0:print(config)
    
# train number of epochs

valid_min  = []
nepoch_max = 400
nepochs = np.arange(nepoch_max + 1)
config.update(nepoch=nepoch_max)
if rank==0:
   log+='training num of epochs up to {} \n'.format(nepoch_max)
   t1 = time()
net = NN.Netregression(train_i, valid_i, test_i, axfit)
net.train_evaluate(**config)
rmse = []
for a in net.epoch_MSEs:
    rmse.append(np.sqrt(a[2][:,2]))
RMSE  = np.column_stack(rmse)
RMSEm = np.mean(RMSE, axis=1)
#RMSEe = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
baseline = np.sqrt(net.optionsdic['baselineMSE'][1])
valid_min = RMSEm/baseline
if rank==0:
   log+='rank{} finished {} in {} s\n'.format(rank, nepochs[-1], time()-t1)

comm.Barrier()
valid_min  = comm.gather(valid_min, root=0)
if rank == 0:
    VMIN = np.array(valid_min)
    Mean = VMIN.mean(axis=0)
    argbest = np.argmin(Mean)
    plt.figure(figsize=(4,3))
    ymin = 1.e6
    for i in range(VMIN.shape[0]):
         plt.plot(nepochs, VMIN[i,:], color='grey', ls=':', alpha=0.5)
         ymin = np.minimum(np.min(VMIN[i, :]), ymin)
    plt.plot(nepochs, Mean, color='red', ls='-', 
            label='Average across the folds')
    plt.axvline(nepochs[argbest], ls='--', color='k',\
                label=r'Number of Epochs : CV estimate')
    plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
    plt.xlabel(r'Number of Epochs')
    plt.ylim(0.8*ymin, 1.2*ymin)
    plt.legend()    
    nebest = nepochs[argbest]
    config.update(nepoch=nebest)
    config.update(nchain=10)
    log += 'best nepoch is :: {}\n'.format(nebest)
    log += 'the updated config is {}\n'.format(config)
    plt.savefig(oupath+'nepochs_validation.pdf', bbox_inches='tight')
else:
    config = None    
config = comm.bcast(config, root=0)
if rank == 0:print(config)

if rank==0:
   log+='final run for the best hyper-parameters\n'
   log+='BHPS: {}\n'.format(config)
   t1  = time()

net   = NN.Netregression(train_i, valid_i, test_i, axfit)
net.train_evaluate(**config)

rmse  = []
for a in net.epoch_MSEs:
    rmse.append(np.sqrt(a[2][:,2]))

RMSE        = np.column_stack(rmse)
RMSEm       = np.mean(RMSE, axis=1)
RMSEe       = np.std(RMSE, axis=1)/np.sqrt(RMSE.shape[1])
baseline    = np.sqrt(net.optionsdic['baselineMSE'][1])
rmselist    = [net.epoch_MSEs[0][2][:,0], RMSEm/baseline, RMSEe/baseline]
meanY, stdY = net.optionsdic['stats']['ystat']
predP       = net.test.P
y_avg = []
for yi in net.chain_y:
    y_avg.append(yi[1].squeeze().tolist())    

#predY = stdY*np.mean(np.array(y_avg), axis=0) + meanY
predY = stdY*np.array(y_avg) + meanY


if rank==0: 
   log+='finished final run for rank {} in {} s \n'.format(rank, time()-t1)

comm.Barrier()
rmselist = comm.gather(rmselist, root=0)
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
    #oumap[hpix]       = np.mean(ngal, axis=0)
    oumap[hpix]      = np.median(ngal, axis=0)

    np.save(oupath+'nn-rmse', rmselist)
    ft.write(oupath+'nn-weights'+str(NSIDE)+'.fits', oudata, clobber=True)
    hp.write_map(oupath+'nn-weights.hp'+str(NSIDE)+'.fits', oumap,\
                 fits_IDL=False, overwrite=True, dtype=np.float64)
    log += 'write rmses in {}\n'.format(oupath+'nn-rmse')
    log += 'write hpix.weight in {}\n'\
           .format(oupath+'nn-weights'+str(NSIDE)+'.fits')
    log += 'write weight map in {}\n'\
           .format(oupath+'nn-weights.hp'+str(NSIDE)+'.fits')
    #
    #
    ymin = 1.e6
    plt.figure(figsize=(4,3))
    for l in range(len(rmselist)):
        x  = rmselist[l][0]
        y  = rmselist[l][1]
        ye = rmselist[l][2]
        plt.fill_between(x, y-ye, y+ye, color='grey', alpha=0.1)
        plt.plot(x, y, color='grey')
        ymin = np.minimum(np.min(y), ymin)

    plt.xlabel('Number of Epochs')
    plt.ylabel(r'$RMSE_{NN}/RMSE_{baseline}$')
    plt.ylim(0.8*ymin, 1.2*ymin)
    plt.savefig(oupath+'nn-rmse.pdf', bbox_inches='tight')
    log += 'write the rmse plot in {}'.format(oupath+'nn-rmse.pdf')
    logfile = open(oupath+'nn-log.txt', 'w')
    logfile.write(log)
