'''
   code to read the cut sky mocks and make them like a healpix map
   mpirun -np 2 python Cats2healpixmaps.py --path /Volumes/TimeMachine/data/mocks/dr5mocks/ --ext seed*/3dbox_nmesh1024_L5274.0_bias1.5_seed*.cat --nside 256 --zlim 0.7 1.2
'''


import healpy as hp
import numpy as np



def hpixsum(nside, ra, dec, value=None, nest=False):
    '''
        cc: Imaginglss Ellie and Yu
        make a healpix map from ra-dec
        hpixsum(nside, ra, dec, value=None, nest=False) 
    '''
    pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra), nest=nest)
    npix = hp.nside2npix(nside)
    w = np.bincount(pix, weights=value, minlength=npix)
    return w


def read_write(path2file, path2output, nside, zlim=None):
    data   = np.loadtxt(path2file)
    m      = np.ones(data[:,2].size, dtype='?')
    if zlim is not None:
        m &= (data[:,2]>=zlim[0]) & (data[:,2]< zlim[1])
    galmap = hpixsum(nside, data[m,0], data[m,1])
    hp.write_map(path2output, galmap, fits_IDL=False, dtype=np.float64, overwrite=True)
    del data

def loop_filenames(filenames, nside, zlim):
    for file in filenames:
        #fn = file.split('/')[-1]
        #inputf  = file + '/' + fn + '.cat'
        #outputf = file + '/' + fn + 'hp'+ str(nside) + 'v2.fits'
        inputf = file
        outputf = file[:-4] + 'hp'+str(nside)+'v0.fits' # v2 is with zcut
        read_write(inputf, outputf, nside, zlim) 
        


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
    ap.add_argument('--nside', type=int, default=256)
    ap.add_argument('--zlim', type=float, nargs='*', default=None)
    ns = ap.parse_args()
    #FILES = glob(ns.path+ns.ext) # already done for 1-100
    FILES = [ns.path+'seed'+str(i)+'/3dbox_nmesh1024_L5274.0_bias1.5_seed'+str(i)+'.cat' for i in range(1, 201)]
    NSIDE = ns.nside
    zlim = ns.zlim
    log  = '# --- create healpix maps with nside of {} ----\n'.format(NSIDE)
    log += '# PATH : {} nfiles : {} files : *{} \n'.format(ns.path, len(FILES), ns.ext)
    log += '# zlim : {}'.format(zlim)
    print(log)
else:
    FILES = None
    NSIDE = None
    zlim  = None


# bcast FILES
FILES = comm.bcast(FILES, root=0)
NSIDE = comm.bcast(NSIDE, root=0)
zlim  = comm.bcast(zlim, root=0)


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
# for filei in my_chunk:
#     print(filei.split('/')[-1])


#
# read BigFile and write in a .dat file  
#
loop_filenames(my_chunk, NSIDE, zlim)
