'''
   code to read the healpix map of log-normal mock
   add DR5 features
   -- updates
   June 26: divide the ngal by fracgood to correct for boundaries
   Oct  04: use fracgood from the mocks bc it has less cross-corr. with the sys. maps 
   Jan 16:
mpirun --oversubscribe -np 4 python add_features-split.py --hpmap /Volumes/TimeMachine/data/mocks/3dbox/ --ext */*.hp.256.fits --features /Volumes/TimeMachine/data/mocks/dr7mock-features.fits --split r
'''
import numpy as np
import healpy as hp
import fitsio as ft
import sys
from   utils import split2Kfolds, split2KfoldsSpatially


def read_write(path2file, path2output, DATA, split='r'):
    '''
        read path2file and appends the ngal as 
        label to path2output
    '''
    cat  = hp.read_map(path2file, verbose=False)
    ngal = cat[DATA['hpind']]
    nran = DATA['fracgood']
    labl = ngal/nran *(nran.sum()/ngal.sum())
    DATA['label'] = labl 
    if split == 'r':
        datakfolds = split2Kfolds(DATA, k=5)
    elif split == 's':
        datakfolds = split2KfoldsSpatially(DATA, k=5)
    else:
        raise RuntimeError("--split should be r or s")
    np.save(path2output, datakfolds)


def loop_filenames(filenames, DATA, split='r'):
    for file in filenames:
        inputf  = file
        outputf = file[:-5]+'.5.'+split+'.npy'
        print('working on ', outputf)
        read_write(inputf, outputf, DATA, split=split) 
        


# mpi
from mpi4py import MPI

# get the size, and the rank of each mpi task
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    from glob import glob
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Read BigFile mocks and write .dat')
    ap.add_argument('--hpmap',     default='/global/cscratch1/sd/mehdi/mocks/3dbox/')
    ap.add_argument('--ext',       default='') 
    ap.add_argument('--features',  default='/global/cscratch1/sd/mehdi/mocks/dr5mock-features.fits')
    ap.add_argument('--split',     default='r')
    ns = ap.parse_args()
    FILES = glob(ns.hpmap+ns.ext)
    DATA  = ft.read(ns.features)
    split = ns.split
    print('add features on %d data files'%len(FILES))
else:
    FILES = None
    DATA = None
    split= None
    
# bcast FILES
FILES = comm.bcast(FILES, root=0)
DATA  = comm.bcast(DATA, root=0)
split = comm.bcast(split, root=0)

#
# distribute files on different task ids
# chunksize
nfiles = len(FILES)
if np.mod(nfiles, size) == 0:
    chunksize = nfiles // size
else:
    chunksize = nfiles // size + 1
my_i     = rank*chunksize
my_end   = np.minimum((rank+1)*chunksize, nfiles)
my_chunk = FILES[my_i:my_end]

print('files on rank {} are {}'.format(rank, my_chunk))
loop_filenames(my_chunk, DATA, split)
