'''
   utility to compute the power spectrum from a 3D box log-normal mocks 
   with Nbodykit (c) Nick Hand, Yu Feng
   see http://nbodykit.readthedocs.io/en/stable/catalogs/mock-data.html#Log-normal-Mocks
'''

# 
import nbodykit.lab as nb
import nbodykit.io.bigfile as bf
import numpy as np
from nbodykit import setup_logging, style
setup_logging() # turn on logging to screen



from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()  


def read_BigFile(filename):
    FILE = bf.BigFile(filename)
    n    = FILE.size
    pos  = FILE.read('Position', 0, n)
    return nb.ArrayCatalog(pos)


# MR: very slow 
# BigFile is faster

# def read_txtfile(filename):
#     cat_i = open(filename, 'r')
#     lines = cat_i.readlines()
#     x = []
#     for line in lines:
#         x += [line.split(' ')]
#     x.pop(0)
#     X = np.array(x)
#     X = X.astype('f4')
#     cat_i.close()
#     del x
#     cat_f = np.zeros(X.shape[0], dtype=np.dtype([('Position', ('f4',3))]))
#     cat_f['Position'] = X
#     return nb.ArrayCatalog(cat_f)


from time import time
from argparse import ArgumentParser
ap = ArgumentParser(description='Power Spectrum Calculator using Nbodykit')
ap.add_argument('--input', default='3dbox_nmesh1024_L5274.0_bias1.5_seed1')
ap.add_argument('--output', default='power_3dbox_nmesh1024_L5274.0_bias1.5_seed1.json')
ap.add_argument('--boxsize', type=float, default=5274)
ap.add_argument('--nmesh', type=int, default=256)
ns = ap.parse_args()
#
t1    = time()
CAT   = read_BigFile(ns.input)
print('time to read the file', time()-t1)
OUT   = ns.output
BOX   = ns.boxsize
NMESH = ns.nmesh

t2 = time()
mesh = CAT.to_mesh(compensated=True, window='cic', position='Position', BoxSize=BOX, Nmesh=NMESH)
rpol = nb.FFTPower(mesh, mode='1d', kmin=0.0, poles=[0,2,4])
rpol.save(OUT)
print('finished in {} s'.format(time()-t2))
