'''
    Code to make systematic maps given a ccd annotated file
    This calls some wrapper codes inside `desi_image_validation` developed by Marc Manera,
    which make use functions inside quicksip.py developed by Ashley J.Ross, Borris, ...
    
    Updates:
    Oct 11: deactivate airmass because DR7 is missing it    
    Jan 9: run for mjd_nobs for DR7
    python make_sysmaps.py --survey DECaLS --dr DR7 --localdir /Volumes/TimeMachine/data/DR7/sysmaps/
    Jan 10: run mjd for eBOSS chunks
    for chunk in 21 22 23 25;do python make_sysmaps.py --survey eBOSS --dr eboss${chunk} --localdir /Volumes/TimeMachine/data/ebo
ss/sysmaps/;echo $chunk is done;done
    Jan 11: run mjd for new updated version of eboss ccd files
for id in dr3 dr3_utah_ngc dr5-eboss dr5-eboss2; do python make_sysmaps.py --survey eBOSS --dr ${id} --localdir /Volumes/TimeMachine/data/eboss/sysmaps/;echo ${id} is done;done    

    This code is written by Marc Manera
    However its functionality has been reduced to 
    only generating the photometric maps from ccd files
    mysample is a class that facilitates reading the input
    parameters. The main task is done inside project_and_write_maps
'''
import os
import sys
import numpy as np
#import astropy.io.fits as pyfits # TOO SLOW
import healpy as hp
import numpy as np
import fitsio as ft

from time import time
from quicksip import *


COLNAMES = ['filter', 'photometric','blacklist_ok','bitmask','galdepth', 'ebv',
            'ccdskycounts', 'pixscale_mean','exptime', 'ccdzpt', 'filter',
            'fwhm','mjd_obs','exptime', #'airmass', #'ccdskymag',
             'ra', 'dec', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1',
            'cd1_2', 'cd2_1', 'cd2_2','width','height']

### ------------ A couple of useful conversions -----------------------

def zeropointToScale(zp):
    return 10.**((zp - 22.5)/2.5)	

def nanomaggiesToMag(nm):
    return -2.5 * (np.log(nm,10.) - 9.)

def Magtonanomaggies(m):
    return 10.**(-m/2.5+9.)
    #-2.5 * (log(nm,10.) - 9.)

def thphi2radec(theta,phi):
        return 180./np.pi*phi,-(180./np.pi*theta-90)
### ------------ SHARED CLASS: HARDCODED INPUTS GO HERE ------------------------
###    Please, add here your own harcoded values if any, so other may use them 

class mysample(object):
    """
    (c) Marc Manera
    This class mantains the basic information of the sample
    to minimize hardcoded parameters in the test functions

    Everyone is meant to call mysample to obtain information like 
         - path to ccd-annotated files   : ccds
         - zero points                   : zp0
         - magnitude limits (recm)       : recm
         - photoz requirements           : phreq
         - extintion coefficient         : extc
         - extintion index               : be
         - mask var eqv. to blacklist_ok : maskname
         - predicted frac exposures      : FracExp   
    Current Inputs are: survey, DR, band, localdir) 
         survey: DECaLS, MZLS, BASS
         DR:     DR3, DR4
         band:   g,r,z
         localdir: output directory
    """                                  


    def __init__(self, survey=None, dr=None, band=None, localdir=None, nside=256, **kwargs):
        """ 
        Initialize image survey, data release, band, output path
        Calculate variables and paths
        """   
        self.survey   = survey
        self.DR       = dr
        self.band     = band
        self.localdir = localdir 
        self.nside    = nside
        # Check bands
        if self.band not in ['r', 'g', 'z']: 
            raise RuntimeError("Band seems wrong options are 'g' 'r' 'z'")        
              
        # Check surveys
        if self.survey not in ['DECaLS', 'BASS', 'MZLS', 'eBOSS']:
            raise RuntimeError("Survey seems wrong options are 'DECAaLS' 'BASS' MZLS' ")

        # Annotated CCD paths  
        if(self.DR == 'DR3'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr3/'
            #self.ccds =inputdir+'ccds-annotated-decals.fits.gz'
            self.ccds ='/global/project/projectdirs/desi/users/mehdi/trunk/'\
                      +'dr3-ccd-annotated-nondecals-extra-decals.fits' # to include all 
            self.catalog = 'DECaLS_DR3'
            if(self.survey != 'DECaLS'): raise RuntimeError("Survey name seems inconsistent")
        elif(self.DR == 'DR4'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr4/'
            if (band == 'g' or band == 'r'):
                #self.ccds = inputdir+'ccds-annotated-dr4-90prime.fits.gz'
                self.ccds = inputdir+'ccds-annotated-bass.fits.gz'
                self.catalog = 'BASS_DR4'
                if(self.survey != 'BASS'): raise RuntimeError("Survey name seems inconsistent")
            elif(band == 'z'):
                #self.ccds = inputdir+'ccds-annotated-dr4-mzls.fits.gz'
                self.ccds = inputdir+'ccds-annotated-mzls.fits.gz'
                self.catalog = 'MZLS_DR4'
                if(self.survey != 'MZLS'): raise RuntimeError("Survey name seems inconsistent")
            else: raise RuntimeError("Input sample band seems inconsisent")
        elif(self.DR == 'DR5'):
            inputdir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr5/'
            self.ccds =inputdir+'ccds-annotated-dr5.fits.gz'
            self.catalog = 'DECaLS_DR5'
            if(self.survey != 'DECaLS'): raise RuntimeError("Survey name seems inconsistent")
        elif (self.DR == 'DR7'): 
            inputdir = '/Volumes/TimeMachine/data/DR7/'
            self.ccds = inputdir+'ccds-annotated-dr7.fits.gz'
            self.catalog = 'DECaLS_DR7'
        elif (self.DR in ['eboss21', 'eboss22', 'eboss23', 'eboss25']): # Jan 10, 2019 for eBOSS chunks
            inputdir     =  '/Volumes/TimeMachine/data/eboss/sysmaps/ccdfiles/'
            self.ccds    =  inputdir + 'survey-ccds.'+self.DR+'.fits.gz'
            self.catalog = self.survey+'_'+self.DR
        elif (self.DR in ['dr3', 'dr3_utah_ngc', 'dr5-eboss', 'dr5-eboss2', 'dr3_utah_sgc', 'eboss_combined']):
            inputdir     =  '/Volumes/TimeMachine/data/eboss/sysmaps/ccdfiles/'
            self.ccds    =  inputdir + 'survey-ccds-'+self.DR+'.fits'
            self.catalog = self.survey+'_'+self.DR
        else:
            raise RuntimeError("Data Realease seems wrong") 

#		
# irrelavant for production of photometric maps
#
#         Predicted survey exposure fractions 
#         if(self.survey =='DECaLS'):
#              # DECALS final survey will be covered by 
#              # 1, 2, 3, 4, and 5 exposures in the following fractions: 
#              self.FracExp=[0.02,0.24,0.50,0.22,0.02]
#         elif(self.survey == 'BASS'):
#             # BASS coverage fractions for 1,2,3,4,5 exposures are:
#             self.FracExp=[0.0014,0.0586,0.8124,0.1203,0.0054,0.0019]
#         elif(self.survey == 'MZLS'):
#              # For MzLS fill factors of 100% with a coverage of at least 1, 
#              # 99.5% with a coverage of at least 2, and 85% with a coverage of 3.
#              self.FracExp=[0.005,0.145,0.85,0,0]
#         else:
#              raise RuntimeError("Survey seems to have wrong options for fraction of exposures ")

        #Bands inputs
        # extc = {'r':2.165, 'z':1.211, 'g':3.214} # galactic extinction correction
        if band == 'g':
            self.be = 1
            self.extc = 3.214  #/2.751
            self.zp0 = 25.08
            self.recm = 24.
            self.phreq = 0.01
        if band == 'r':
            self.be = 2
            self.extc = 2.165  #/2.751
            self.zp0 = 25.29
            self.recm = 23.4
            self.phreq = 0.01
        if band == 'z':
            self.be = 4
            self.extc = 1.211  #/2.751
            self.zp0 = 24.92
            self.recm = 22.5
            self.phreq = 0.02

# ------------------------------------------------------------------




# ------------------------------------------------------------------
# ------------ VALIDATION TESTS ------------------------------------
# ------------------------------------------------------------------
# Note: part of the name of the function should startw with number valXpX 

def generate_maps(comm, sample):
    '''
       generate ivar, airmass, seeing, count and sky brightness map
       OCt 11: deactivate airmass
    '''
    nside     = sample.nside       # Resolution of output maps
    nsideSTR  = str(nside)    # same as nside but in string format
    nsidesout = None   # if you want full sky degraded maps to be written
    ratiores  = 4       # Superresolution/oversampling ratio, simp mode doesn't allow anything other than 1
    mode      = 1           # 1: fully sequential, 2: parallel then sequential, 3: fully parallel
    pixoffset = 0      # How many pixels are being removed on the edge of each CCD? 15 for DES.
    oversamp  = str(ratiores)       # ratiores in string format
    
    band           = sample.band
    catalogue_name = sample.catalog
    fname          = sample.ccds    
    localdir       = sample.localdir
    extc           = sample.extc

    #Read ccd file 
    rank = comm.Get_rank()
    if rank==0:
        print('Rank %d working on %s'%(rank, fname))
        tbdata = ft.read(fname, lower=True)
        common_cols = np.intersect1d(COLNAMES, tbdata.dtype.names)
        tbdata = tbdata[common_cols]
    else:
        tbdata = None

    # bcast
    tbdata  = comm.bcast(tbdata, root=0)
    sfilter = tbdata['filter'].astype(str)

    if rank == 0:
        print('Rank %d : broadcasting is done'%rank)
    #exit()
    # ------------------------------------------------------
    # Obtain indices that satisfy filter / photometric cuts
    #
    auxstr='band_'+band
    sample_names = [auxstr]
    if(sample.DR in ['DR3', 'DR5', 'eboss21', 'eboss22', 'eboss23', 'eboss25']):
        inds = np.where((sfilter == band) & (tbdata['photometric'] == True) & (tbdata['blacklist_ok'] == True)) 
    elif(sample.DR == 'DR4'):
        inds = np.where((sfilter == band) & (tbdata['photometric'] == True) & (tbdata['bitmask'] == 0)) 
    elif (sample.DR in ['DR5', 'DR7', 'dr3', 'dr3_utah_ngc', 'dr3_utah_sgc', 'dr5-eboss', 'dr5-eboss2', 'eboss_combined']):
        inds = np.where(sfilter == band)
    #Read data 
    #obtain invnoisesq here, including extinction 
    nmag = Magtonanomaggies(tbdata['galdepth']-extc*tbdata['ebv'])/5.
    ivar= 1./(nmag*nmag)
    
    #
    # compute CCDSKYMAG. since it's missing in DR 7
    ccdskymag = -2.5*np.log10(tbdata['ccdskycounts']/tbdata['pixscale_mean']/tbdata['pixscale_mean']/tbdata['exptime'])\
              + tbdata['ccdzpt']

    # What properties do you want mapped?
    # Each each tuple has [(quantity to be projected, weighting scheme, operation),(etc..)] 
    propertiesandoperations = [ ('ivar', '', 'total'),
                                ('count', '', 'fracdet'),
                                ('fwhm',    '', 'mean'),
                                #('airmass', '', 'mean'),   # no airmass in DR 7, so COMMENT it out
                                ('exptime', '', 'total'),
                                ('ccdskymag', '', 'mean'),
                                ('mjd_obs', '', 'min'),
                                ('mjd_obs', '', 'mean'),
                                ('mjd_obs', '', 'max')
                              ]
 
    # What properties to keep when reading the images? 
    # Should at least contain propertiesandoperations and the image corners.
    # MARCM - actually no need for ra dec image corners.   
    # Only needs ra0 ra1 ra2 ra3 dec0 dec1 dec2 dec3 only if fast track appropriate
    # quicksip subroutines were implemented 
    #propertiesToKeep = [ 'filter', 'airmass', 'fwhm','mjd_obs','exptime','ccdskymag',\
    #                     'ra', 'dec', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1',\
    #                    'cd1_2', 'cd2_1', 'cd2_2','width','height']
    propertiesToKeep = [ 'filter', 'fwhm','mjd_obs','exptime', #'airmass', #'ccdskymag',
                         'ra', 'dec', 'crval1', 'crval2', 'crpix1', 'crpix2', 'cd1_1',
                        'cd1_2', 'cd2_1', 'cd2_2','width','height']
    
    # Create big table with all relevant properties. 
    #tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar], names = propertiesToKeep\
    #+ [ 'ivar'])
    tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep] + [ivar, ccdskymag],
                                        names = propertiesToKeep + [ 'ivar', 'ccdskymag'])
    #tbdata = np.core.records.fromarrays([tbdata[prop] for prop in propertiesToKeep],\
    #names = propertiesToKeep)
 
    # Read the table, create Healtree, project it into healpix maps, and write these maps.
    # Done with Quicksip library, note it has quite a few hardcoded values
    #(use new version by MARCM for BASS and MzLS) 
    # project_and_write_maps_simp(mode, propertiesandoperations, tbdata,
    #catalogue_name, outroot, sample_names, inds, nside)
    project_and_write_maps(mode, propertiesandoperations, tbdata,
                           catalogue_name, localdir, sample_names, inds,
                           nside, ratiores, pixoffset, nsidesout)

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank==0:
        print('Hi from rank %d, number of MPI processes %d'%(rank, size))
        from   time import time
        from argparse import ArgumentParser
        ap = ArgumentParser(description='systematic maps generating routine')
        ap.add_argument('--survey',     default='DECaLS')
        ap.add_argument('--dr',         default='DR7')
        ap.add_argument('--localdir',   default='/Volumes/TimeMachine/data/DR7/sysmaps/')
        ap.add_argument('--nside',      default=256, type=int)
        ap.add_argument('--bands',      nargs='*', type=str, default=['r', 'g', 'z'])
        ns = ap.parse_args()    
        dics = ns.__dict__
        print('INPUTS : ')
        for keyi in dics.keys():
            print('{:15s}: {}'.format(keyi, dics[keyi]))
        BANDS = ns.bands
    else:
        BANDS = None
        dics  = None
    #
    # Bcast and Scatter bands
    dics  = comm.bcast(dics,    root=0)
    BANDS = comm.scatter(BANDS, root=0)
    #
    # Run
    if rank==0:t1=time()
    Mysample = mysample(band=BANDS, **dics)
    generate_maps(comm, Mysample)
    if rank==0:print('Rank %d finished in %f secs'%(rank, time()-t1))
