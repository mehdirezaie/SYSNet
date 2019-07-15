
from glob import glob
import healpy as hp
import numpy as np

filename = '/Volumes/TimeMachine/data/DR7/mask.cut.hp.256.fits'
weights  = glob('/Volumes/TimeMachine/data/DR7/results/regression/*/*weights.hp256.fits')



mask    = hp.read_map('/Volumes/TimeMachine/data/DR7/mask.hp.256.fits', verbose=False).astype('bool')



for n in weights:
    print('map : %s '%n, end=' ')
    wi = hp.read_map(n, verbose=False)
    maski = (wi > 0.5) & (wi < 2.0)
    print(mask.sum(), maski.sum())
    mask &= maski    

    
hp.write_map(filename, mask, overwrite=True, fits_IDL=False)


print('total number of weights %d'%len(weights))
print('plain footprint %d'%mask.sum())
print('saving %s'%filename)


#map : /Volumes/TimeMachine/data/DR7/results/regression/mult_all/lin-weights.hp256.fits  187257 187251
#map : /Volumes/TimeMachine/data/DR7/results/regression/mult_all/quad-weights.hp256.fits  187251 187114
#map : /Volumes/TimeMachine/data/DR7/results/regression/mult_all_vl/lin-weights.hp256.fits  187108 187251
#map : /Volumes/TimeMachine/data/DR7/results/regression/nn_ab/nn-weights.hp256.fits  187108 186403
#map : /Volumes/TimeMachine/data/DR7/results/regression/nn_p/nn-weights.hp256.fits  186385 185914
#total number of weights 5
#plain footprint 185781
#saving /Volumes/TimeMachine/data/DR7/mask.cut.hp.256.fits
#
#real    0m10.265s
#user    0m2.066s
#sys     0m0.296s

