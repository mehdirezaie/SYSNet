
from glob import glob
import healpy as hp
import numpy as np

weights = glob('/Volumes/TimeMachine/data/DR7/results/regression/*/*weights.hp256.fits')
mask    = hp.read_map('/Volumes/TimeMachine/data/DR7/mask.hp.256.fits').astype('bool')

mask.sum()
for n in weights:
    wi = hp.read_map(n, verbose=False)
    maski = (wi > 0.5) & (wi < 2.0)
    mask &= maski
    print(mask.sum(), maski.sum())

hp.write_map('/Volumes/TimeMachine/data/DR7/mask.cut.hp.256.fits', mask)