"""
FEB 11, 2019
(py3p6) bash-3.2$ python mocks/mock_on_data.py /Volumes/TimeMachine/data/DR7_Feb10/sysmaps/DECaLS_DR7/nside256_oversamp4/features.fits  /Volumes/TimeMachine/data/mocks/mock.fracNgalhpmap.fits /Volumes/TimeMachine/data/mocks/dr7mock-features.fits /Volumes/TimeMachine/data/mocks/mock.hpmask.dr7.fits
data size 187257
NSIDE = 256
ORDERING = RING in fits file
INDXSCHM = IMPLICIT
mock > 0.0  99547
mock with imaging attrs  90024
saving ...  /Volumes/TimeMachine/data/mocks/dr7mock-features.fits
writing ...  /Volumes/TimeMachine/data/mocks/mock.hpmask.dr7.fits
"""

import numpy as np
import fitsio as ft
import sys
import healpy as hp
sys.path.append('/Users/rezaie/github/DESILSS')
#from tools import hpix2radec



# data
data,h = ft.read(sys.argv[1], header=True)
print('data size', data.size)

# mock fracmap
frac = hp.read_map(sys.argv[2])

mhpix = np.argwhere(frac>0).flatten()
print('mock > 0.0 ', mhpix.size)

# find the overlap
mockondata = np.in1d(data['hpix'], mhpix)
datamock = data[mockondata]
datamock['fracgood'] = frac[datamock['hpix']]  # replace the fracgood with the mocks 

print('mock with imaging attrs ', datamock.size)
h['Note'] = 'This is for the mocks'
ft.write(sys.argv[3], datamock, header=h, clobber=True)
print('saving ... ', sys.argv[3])

# make mask
mask = np.zeros(12*256*256)
mask[datamock['hpix']] = 1.0
hp.write_map(sys.argv[4], mask, fits_IDL=False, overwrite=True)
print('writing ... ', sys.argv[4])



