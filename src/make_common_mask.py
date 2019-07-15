
from glob import glob
import healpy as hp
import numpy as np


# mask = hp.read_map('/Volumes/TimeMachine/data/mocks/mask.hp.256.fits').astype('bool')
# maps = glob('cp2p//Volumes/TimeMachine/data/mocks/3dbox/*/cp2p_mask_*.hp.256.fits')

# mask_neg = np.zeros(12*256*256, '?')
# for n in maps:
#     #print(n.split('/')[-1])
#     mapi = hp.read_map(n, verbose=False).astype('bool')
#     mask_neg |= mapi
    
# noneg = (mask & (~mask_neg))
# print('original mask: ', mask.sum(), 'after removing negative pixels :', noneg.sum())
# hp.write_map('/Volumes/TimeMachine/data/mocks/mask.cut.hp.256.fits', noneg, dtype=np.float64, overwrite=True)







filename = '/Volumes/TimeMachine/data/mocks/mask.cut.hp.256.fits'
filename2 = '/Volumes/TimeMachine/data/mocks/mask.cut.w.hp.256.fits'    


weights  = glob('/Volumes/TimeMachine/data/mocks/3dbox/*/results/regression/*/*weights.hp256.fits')
weights += glob('/Volumes/TimeMachine/data/mocks/3dbox/*/cp2p/results/regression/*/*weights.hp256.fits')
print('total number of weights : %d'%len(weights))

mask = hp.read_map(filename, verbose=False).astype('bool')
for n in weights:
    wi = hp.read_map(n, verbose=False)
    maski = (wi > 0.5) & (wi < 2.0)
    mask &= maski   

hp.write_map(filename2, mask, overwrite=True, fits_IDL=False)

# test
#mask2 = hp.read_map(filename2, verbose=False).astype('bool')
#print(mask.sum(), mask2.sum(), np.array_equal(mask, mask2))
