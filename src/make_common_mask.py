
from glob import glob
import healpy as hp
import numpy as np

mask = hp.read_map('/Volumes/TimeMachine/data/mocks/mask.hp.256.fits').astype('bool')
maps = glob('/Volumes/TimeMachine/data/mocks/3dbox/*/cp2p/cp2p_mask_*.hp.256.fits')
mask_neg = np.zeros(12*256*256, '?')
for n in maps:
    #print(n.split('/')[-1])
    mapi = hp.read_map(n, verbose=False).astype('bool')
    mask_neg |= mapi
    
noneg = (mask & (~mask_neg))
print('original mask: ', mask.sum(), 'after removing negative pixels :', noneg.sum())
hp.write_map('/Volumes/TimeMachine/data/mocks/mask.cut.hp.256.fits', noneg, dtype=np.float64, overwrite=True)