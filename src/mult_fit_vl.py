'''
    run the validation procedure on a dataset
    run it with
    mpirun -np 5 python validate.py 
'''
import os
import numpy as np
from time import time
import sys
HOME = os.getenv('HOME')
sys.path.append(HOME + '/github/DESILSS')
from scipy.optimize import curve_fit

def model(x, *theta):
    return theta[0] + np.matmul(x, np.array(theta[1:]))

def rmse(y1, y2, ye):
    return np.sqrt(np.mean(((y1-y2)/ye)**2))

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#
#
if rank == 0:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rc('font', family='serif')
    #
    #
    import fitsio as ft
    import healpy as hp
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Neural Net regression')
    ap.add_argument('--input',  default='test_train_eboss_dr5-masked.npy')
    ap.add_argument('--output', default='test_train_eboss_dr5-maskednn.npy')
    ap.add_argument('--nside', default=256, type=int)
    ns = ap.parse_args()
    NSIDE = ns.nside
    #
    #
    log  = '! ===== Regression with Neural Net ======\n'
    log += 'reading input : {} with nside : {} \n'.format(ns.input, NSIDE)
    data   = np.load(ns.input).item()
    oupath = ns.output
    import os 
    if not os.path.exists(ns.output):
        os.makedirs(ns.output)
else:
    oupath = None
    data   = None

data   = comm.bcast(data,   root=0)
oupath = comm.bcast(oupath, root=0)


assert size == 5, 'required number of mpi tasks should be equal to number of folds'
train_i = data['train']['fold'+str(rank)]
test_i  = data['test']['fold'+str(rank)]
valid_i = data['validation']['fold'+str(rank)]




if rank==0:
    log+='rank %d has train %d validation %d and test %d \n'%(rank, train_i.size, test_i.size, valid_i.size)
    t1 = time()
comm.Barrier()


# feature scaling
meanX, stdX = np.mean(train_i['features'], axis=0), np.std(train_i['features'], axis=0)
meanY, stdY = np.mean(train_i['label'], axis=0), np.std(train_i['label'], axis=0)
trainX      = (train_i['features']-meanX)/stdX
testX       = (test_i['features']-meanX)/stdX
trainY      = (train_i['label']-meanY)/stdY
testY       = (test_i['label']-meanY)/stdY

testYe    = 1./(test_i['fracgood']**0.5)
trainYe   = 1./(train_i['fracgood']**0.5)
trainXX   = np.concatenate([trainX, trainX*trainX], axis=1)
testXX    = np.concatenate([testX, testX*testX], axis=1)
lmodel    = 'lin'
ax        = testX.shape[1]
#print(ax, testXX.shape, trainXX.shape)
#sys.exit()#break
popt, pcov = curve_fit(model, trainX, trainY, p0=[0 for i in range(ax+1)],
                                 sigma=trainYe, absolute_sigma=True, method='lm')
 

predP = test_i['hpind']
predy = model(testX, *popt) 
predY = stdY*predy+ meanY

print("rank : {} training rmse : {}, test RMSE : {}"\
     .format(rank, rmse(trainY, model(trainX, *popt), trainYe), rmse(testY, predy, testYe)))
if rank==0: log+='finished final run for rank {} in {} s \n'.format(rank, time()-t1)

comm.Barrier()
predY    = comm.gather(predY, root=0)
predP    = comm.gather(predP, root=0)
if rank ==0:
    hpix = np.concatenate(predP)
    ngal = np.concatenate(predY)
    oudata = np.zeros(ngal.size, dtype=[('hpind', 'i8'), ('weight','f8')])
    oudata['hpind']   = hpix
    oudata['weight'] = ngal
    oumap  = np.zeros(12*NSIDE*NSIDE)
    oumap[hpix] = ngal
    ft.write(oupath+lmodel+'-weight'+str(NSIDE)+'.fits', oudata)
    hp.write_map(oupath+lmodel+'-weight.hp'+str(NSIDE)+'.fits', oumap, fits_IDL=False, overwrite=True)
    log += 'write hpix.weight in {}\n'.format(oupath+lmodel+'-weights'+str(NSIDE)+'.fits')
    log += 'write weight map in {}\n'.format(oupath+lmodel+'-weights.hp'+str(NSIDE)+'.fits')
    print(log)
    #
    #
    #
    #


'''
(py3p6) bash-3.2$ time bash analyze_dr7.sh 
rank : 1 training rmse : 0.9469504827767476, test RMSE : 0.9133453200685644
rank : 0 training rmse : 0.9450034936158515, test RMSE : 0.9531799131643774
rank : 2 training rmse : 0.9456280809032754, test RMSE : 0.9459668459463196
rank : 3 training rmse : 0.9453576480613363, test RMSE : 0.9518018843035201
rank : 4 training rmse : 0.9457293292588947, test RMSE : 0.9559985696874319
! ===== Regression with Neural Net ======
reading input : /Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.table.5.r.npy with nside : 256 
rank 0 has train 112353 validation 37452 and test 37452 
finished final run for rank 0 in 1.3762140274047852 s 
write hpix.weight in /Volumes/TimeMachine/data/DR7/results/regression/mult_all_vl/lin-weights256.fits
write weight map in /Volumes/TimeMachine/data/DR7/results/regression/mult_all_vl/lin-weight.hp256.fits
'''
