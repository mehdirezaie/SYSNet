'''
   utility to make 3D box log-normal mocks with Nbodykit (c) Nick Hand, Yu Feng
   Poisson samples from log normal density field, given a cosmology
   see http://nbodykit.readthedocs.io/en/stable/catalogs/mock-data.html#Log-normal-Mocks
'''

# what we need for mock catalog generation
import nbodykit.lab as nb
from nbodykit.cosmology import Planck15


# mpi
from mpi4py import MPI

# get the size, and the rank of each mpi task
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#
# rank 0 prepares the input parameters for nbodykit.lab.LogNormalCatalog
if rank == 0:
    from time import time
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Make Mock catalog using Nbodykit')
    ap.add_argument('--path',   default='/global/cscratch1/sd/mehdi/mocks/3dbox/')
    ap.add_argument('--nmesh', type=int, default=512)
    ap.add_argument('--length', type=float, default=5274.)
    ap.add_argument('--bias', type=float, default=1.)
    ap.add_argument('--seed', type=int, default=42)
    ns = ap.parse_args()
    #
    BIAS  = ns.bias
    NMESH = ns.nmesh
    BOX   = ns.length
    # create completely random seed
    import numpy as np
    np.random.seed(12345)
    seeds = np.random.randint(0, 4294967295, size=1000)
    SEED  = seeds[ns.seed]    
    #
    print(SEED)
    PATH  = ns.path+'3dbox_nmesh'+str(NMESH)+'_L'+str(BOX)+'_bias'+str(BIAS)+'_seed'+str(ns.seed)
    t_i   = time()
else:   
    BIAS  = None
    NMESH = None
    BOX   = None
    SEED  = None
    PATH  = None

# default parameters are redshift and nbar
# from Raichoor et. al. 2017
redshift = 0.85     
NBAR     = 1.947e-4 # h3/Mpc3

# bcast
BIAS     = comm.bcast(BIAS, root=0)    # bias 
NMESH    = comm.bcast(NMESH, root=0)   # number of mesh
BOX      = comm.bcast(BOX, root=0)     # boxsize in Mpc/h
SEED     = comm.bcast(SEED, root=0)    # seed for random sampling
PATH     = comm.bcast(PATH, root=0)    # path for writing the output

# cosmology
cosmo    = nb.cosmology.Planck15
Plin     = nb.cosmology.LinearPower(cosmo, redshift, transfer='CLASS')

# generate the catalog
cat      = nb.LogNormalCatalog(Plin=Plin, nbar=NBAR, BoxSize=BOX,
                               Nmesh=NMESH, bias=BIAS, seed=SEED)

# save the catalog, only Position is enough
cat.save(PATH, ['Position', 'Velocity'])
if rank == 0:
   print('finished in {} s'.format(time()-t_i))
