'''
    code to select ELGs from sweepfiles
    1. salloc -N 1 -t 00:10:00 -q debug -C haswell
    2. module load python/3.6
    3. source activate craympi
    4. srun -n 16 python select_targets.py

    July 30: add brick_primary, tycho2inblob, decam_anymask for dr3

'''
import fitsio as ft
import numpy  as np

colnames = ['ra', 'dec', 
             'mw_transmission_g', 'mw_transmission_r','mw_transmission_z',
            'flux_g', 'flux_r', 'flux_z',
            'anymask_g', 'anymask_r', 'anymask_z' ,
            'brightstarinblob'] 
             
def unextinct_fluxes(objects):
    dtype = [('gflux', 'f8'), ('rflux', 'f8'), ('zflux', 'f8')]
    n     = len(objects)
    assert n > 0
    result = np.zeros(n, dtype=dtype)
    result['gflux'] = objects['flux_g'] / objects['mw_transmission_g']
    result['rflux'] = objects['flux_r'] / objects['mw_transmission_r']
    result['zflux'] = objects['flux_z'] / objects['mw_transmission_z']
    #result['gflux'] = objects['decam_flux'][:,1] / objects['decam_mw_transmission'][:,1]
    #result['rflux'] = objects['decam_flux'][:,2] / objects['decam_mw_transmission'][:,2]
    #result['zflux'] = objects['decam_flux'][:,4] / objects['decam_mw_transmission'][:,4]
    return result


def SGC_elg(gflux, rflux, zflux):
    elg = np.ones_like(gflux, dtype='?')
    
    elg &= gflux > 10**((22.5-22.825)/2.5)  # g<22.825
    elg &= gflux < 10**((22.5-21.825)/2.5)  # g>21.825
    

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    gflux = gflux.clip(0)
    
    elg &= gflux * 10**(0.457/2.5) < rflux**0.932 * zflux**0.068    # −0.068 × (r − z) + 0.457 < g − r
    elg &= rflux**1.112 < gflux * zflux**.112 * 10**(0.773/2.5)     # g − r < 0.112 × (r − z) + 0.773
    elg &= rflux**1.218 * 10**(0.571/2.5) < gflux**0.218 * zflux    # 0.218 × (g − r) + 0.571 < r − z 
    elg &= zflux < rflux**0.445 * gflux**0.555 * 10**(1.901/2.5)    # r − z < −0.555 × (g − r) + 1.901
    return elg
    

def NGC_elg(gflux, rflux, zflux):
    elg = np.ones_like(gflux, dtype='?')
    
    elg &= gflux > 10**((22.5-22.9)/2.5)  # g<22.9
    elg &= gflux < 10**((22.5-21.825)/2.5)  # g>21.825

    # Clip to avoid warnings from negative numbers raised to fractional powers.
    rflux = rflux.clip(0)
    zflux = zflux.clip(0)
    gflux = gflux.clip(0)
    
    elg &= gflux * 10**(0.457/2.5) < rflux**0.932 * zflux**0.068    # −0.068 × (r − z) + 0.457 < g − r
    elg &= rflux**1.112 < gflux * zflux**.112 * 10**(0.773/2.5)     # g − r < 0.112 × (r − z) + 0.773
    elg &= rflux**1.637 * 10**(0.399/2.5) < gflux**0.637 * zflux    # 0.637 × (g − r) + 0.399 < r − z 
    elg &= zflux < rflux**0.445 * gflux**0.555 * 10**(1.901/2.5)    # r − z < −0.555 × (g − r) + 1.901
    return elg
    
def isELG(gflux=None, rflux=None, 
          zflux=None, cap='SGC'):
    #----- Emission Line Galaxies
    if cap == 'SGC':
        return SGC_elg(gflux, rflux, zflux)
    elif cap == 'NGC':
        return NGC_elg(gflux, rflux, zflux)
    else:
        raise ValueError('cap unknown')

def select_cap(objects, cap='SGC'):
    mask = (objects['ra'] > 100) & (objects['ra'] < 290)
    if cap == 'NGC':
        return objects[mask]
    else:
        return objects[~mask]
        
    
def apply_cuts(objects, cap='SGC'):
    # select the cap, SGC default
    #objects = select_cap(objects, cap=cap)
    #if len(objects) == 0:
    #    print("this sweep did not have any object in selected cap {}".format(cap))
    #    return 0
    #un-extinct fluxes
    flux = unextinct_fluxes(objects)
    gflux = flux['gflux']
    rflux = flux['rflux']
    zflux = flux['zflux']
    #
    elg = isELG(zflux=zflux, rflux=rflux, gflux=gflux, cap=cap)
    if elg.sum() == 0:
        print("this sweep did not have any ELG")
        return 0, False
    return objects[elg], True

if __name__ == '__main__':
   # mpi
   from mpi4py import MPI
   # get the size, and the rank of each mpi task
   comm = MPI.COMM_WORLD
   size = comm.Get_size()
   rank = comm.Get_rank()

   if rank == 0:
      from glob import glob
      from argparse import ArgumentParser
      from   time import time
      ap = ArgumentParser(description='Run the selection through sweep or tractor files')
      ap.add_argument('--input',   default='/Volumes/TimeMachine/data/DR7/7.1/')
      ap.add_argument('--ext',     default='sweep-*.fits')
      ap.add_argument('--output',  default='/Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.fits')    
      ap.add_argument('--cap',     default='NGC') 
      ns = ap.parse_args()
      FILES = glob(ns.input+ns.ext)
      CAP = ns.cap
   else:
      FILES = None
      CAP = None

   # bcast file names
   FILES = comm.bcast(FILES, root=0)
   CAP   = comm.bcast(CAP, root=0)


   nfiles = len(FILES)
   # split files
   if rank == 0:print('Rank %d : total number of sweep files is %d'%(rank, nfiles))
   if np.mod(nfiles, size) == 0:
      chunksize = nfiles // size
   else:
      chunksize = nfiles // size + 1
   my_i   = rank*chunksize
   my_end = np.minimum(rank*chunksize + chunksize, nfiles)
   my_chunk = FILES[my_i:my_end]
   print('rank {} has {} files '.format(rank, len(my_chunk)))

   if rank==0:t1 = time()
   targets = []
   for i,s in enumerate(my_chunk):
       ds = ft.read(s, lower=True, columns=colnames)
       ts, flag = apply_cuts(ds, cap=CAP)
       if flag:
          targets.append(ts)
       del ds
       #print('rank-{} : {}/{} is done '.format(rank, i, len(my_chunk)))
   if rank==0:print("rank {} finished in {} sec ".format(rank, time()-t1))

   comm.Barrier()
   targets = comm.gather(targets, root=0)

   if rank==0:
      Targets = [t for l in targets for t in l]
      all_targets = np.concatenate(Targets)
      ft.write(ns.output, all_targets)
      print('Rank %d: %d targets are written into %s'%(rank, all_targets.size, ns.output))
