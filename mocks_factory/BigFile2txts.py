'''
   code to read the nbodykit BigFile and write it to a .dat file
   make_survey takes .dat input to make a survey mock out of 
   3d box mock
'''


import nbodykit.io.bigfile as bf
import numpy as np


def read_write(path2file, path2output):
    FILE = bf.BigFile(path2file)
    n    = FILE.size
    pos  = FILE.read('Position', 0, n)
    Pos  = pos['Position']
    np.savetxt(path2output, Pos, header='# position x, y, z')


def loop_filenames(filenames):
    for file in filenames:
        fn = file.split('/')[-1]
        textname = fn + '.dat'
        #print(file, file+'/'+textname)
        read_write(file, file+'/'+textname) 


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
    ap.add_argument('--path',   default='/global/cscratch1/sd/mehdi/mocks/3dbox/')
    ap.add_argument('--ext', default='*') 
    ns = ap.parse_args()
    FILES = glob(ns.path+ns.ext) # if seed 0 to 100 already done > use next line
    #FILES = [ns.path+'3dbox_nmesh1024_L5274.0_bias1.5_seed'+str(i) for i in range(101, 201)]
else:
    FILES = None

# bcast FILES
FILES = comm.bcast(FILES, root=0)



#
# distribute files on different task ids
# chunksize
nfiles = len(FILES)

if np.mod(nfiles, size) == 0:
    chunksize = nfiles // size
else:
    chunksize = nfiles // size + 1

my_i      = rank*chunksize
if rank*chunksize + chunksize > nfiles:
    my_end = None
else:
    my_end    = rank*chunksize + chunksize
my_chunk = FILES[my_i:my_end]


print('files on rank {} are {}'.format(rank, my_chunk))
for filei in my_chunk:
    print(filei.split('/')[-1])



#
# read BigFile and write in a .dat file  
#
loop_filenames(my_chunk)
