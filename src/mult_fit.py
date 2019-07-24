'''
    This code 
    1. takes an observation data
    2. fits the ngal/nbar vs. a set of systematics linear or quadratic
    3. contaminates a set of mocks (plot an example of the contamination model)
    4. look at the ngal histrogram from the mocks and compare it with the observation    
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('font', family='serif')
import fitsio as ft
import numpy as np
from scipy.optimize import curve_fit

def quad(x, *theta):
    return theta[0] + np.matmul(x, np.array(theta[1:]))
def lin(x, *theta):
    return theta[0] + np.matmul(x, np.array(theta[1:]))

class DATA(object):
    def __init__(self, INPUT, axfit=list(range(18)), OUTPUT='./dr5linquadfit', split=False):
        self.axfit = axfit
        if split:
            datai = np.load(INPUT).item()
            data  = np.concatenate([datai['test']['fold'+str(i)] for i in range(5)])
        else:
            data = ft.read(INPUT)
        self.hpix = data['hpind']
        self.XSTATS = (np.mean(data['features'], axis=0), np.std(data['features'], axis=0))
        self.f = (data['features']-self.XSTATS[0])/self.XSTATS[1]
        if len(axfit)==1:
            self.X = self.f[:,self.axfit]#.squeeze()
            self.XX = np.concatenate([self.X, self.X*self.X], axis=1)
        else:
            self.X = self.f[:,self.axfit]
            self.XX = np.concatenate([self.X, self.X*self.X], axis=1)
        #self.Y  = data['label']/np.average(data['label'], weights=data['fracgood'])
        self.Y  = data['label']
        self.Ye = (1./data['fracgood'])**0.5        
        log  = '! ---- Fit a linear/quadratic multivariate to data ------\n'
        log += 'reading {} \n'.format(INPUT)
        log += 'number of data : {} Number of systematic maps : {}\n'.format(*data['features'].shape)
        log += 'feature indices to be used : {}\n'.format(axfit)
        popt, pcov = curve_fit(quad, self.XX, self.Y, p0=[0 for i in range(2*len(axfit)+1)],
                                 sigma=self.Ye, method='lm', absolute_sigma=True)
        self.quad  = (popt, pcov)
        popt, pcov = curve_fit(lin, self.X, self.Y, p0=[0 for i in range(len(axfit)+1)],
                                 sigma=self.Ye, method='lm', absolute_sigma=True)
        self.lin   = (popt, pcov)
        rmse = lambda y1, y2, noise: np.sqrt(np.mean(((y1-y2)/noise)**2))
        #
        #
        baseline_model = np.mean(self.Y)
        rmse_baseline  = rmse(baseline_model, self.Y, self.Ye)
        rmse_lin       = rmse(lin(self.X, *self.lin[0]), self.Y, self.Ye)
        rmse_quad      = rmse(quad(self.XX, *self.quad[0]), self.Y, self.Ye)
        log += 'baseline model has RMSE of {}\n'.format(rmse_baseline)
        log += 'linear fit has RMSE of {}\n'.format(rmse_lin)
        log += 'quadratic fit has RMSE of {}\n'.format(rmse_quad)
        log += 'write results on {}'.format(OUTPUT)
        oudata = {'rmses':[rmse_baseline, rmse_lin, rmse_quad], 'log':log,
                  'params':{'lin':self.lin, 'quad':self.quad}, 'baseline':baseline_model,
                  'xstats':self.XSTATS, 'axfit':self.axfit}
        np.save(OUTPUT, oudata)
        print(log) # show results
        
        
        
    def plotbfitparam(self, path4png='bfitdr5.pdf'):
        print('plotting bfit plot under {}'.format(path4png))
        c = ['cornflowerblue','crimson']
        self.LABELS = ['EBV', 'logHI', 'NStar']+[l+s for l in ['depth','seeing','skymag','exptime', 'mjd'] for s in 'rgz']
        labels = [self.LABELS[i] for i in self.axfit]
        f = plt.figure(figsize=(10,5))
        plt.subplots_adjust(hspace=0.1)
        gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax1.set_ylim(0.81, 1.1)
        ax1.errorbar(np.arange(len(self.lin[0])), self.lin[0], np.diag(self.lin[1])**0.5,
                     marker='.', color=c[1], ls='-')
        ax1.errorbar(np.arange(len(self.quad[0])), self.quad[0], np.diag(self.quad[1])**0.5,
                     marker='+', color=c[0], ls=':')
        ax1.axhline(0, ls='--', color='k',alpha=0.5)
        ax1.xaxis.tick_top()
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticks([])
        ax1.text(0.8, 0.5, 'quadratic', color=c[0], transform=ax1.transAxes, size=20)
        ax1.text(0.8, 0.0, 'linear', color=c[1], transform=ax1.transAxes, size=20)
        #ax1.grid()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.xticks(np.arange(2*len(self.axfit)+1), ['b']+[l+s for s in ['','$^{2}$'] for l in labels], rotation=90)
        ax2.errorbar(np.arange(len(self.lin[0])), self.lin[0], np.diag(self.lin[1])**0.5,
                     marker='.', color=c[1], ls='-')
        ax2.errorbar(np.arange(len(self.quad[0])), self.quad[0], np.diag(self.quad[1])**0.5,
                     marker='+', color=c[0], ls=':')
        ax2.set_ylim(-0.06, 0.07)
        ax2.axhline(0, color='k', ls='--', alpha=0.5)
        ax2.set_ylabel(r'$\Theta_{i}$', size=18)
        #ax2.grid()
        d = 0.01
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d/2, +d/2), **kwargs)        # top-left diagonal
        kwargs.update(transform=ax2.transAxes)            # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d/2, 1 + d/2), **kwargs)  # bottom-left diagonal
        plt.savefig(path4png, bbox_inches='tight', dpi=300)
        
    def plotnnbardata(self, path4nnbar='nnbardr5fit.pdf'):
        print('plotting nnbar plot under {}'.format(path4nnbar))        
        from scipy.stats import binned_statistic
        def binit(x, values=None):
            y1, x1, _ = binned_statistic(x, values=values)
            n1, _, _ = binned_statistic(x, values=values, statistic='count')
            s1, _, _ = binned_statistic(x, values=values, statistic=np.std)
            x  = x1[:-1] 
            ye = s1/np.sqrt(n1)
            mask = (y1 != 0.0 ) & ~np.isnan(y1) & (ye !=0.0) & (~np.isnan(ye))
            return y1[mask], x[mask], ye[mask]
        # 
        f,a = plt.subplots(ncols=3, nrows=6,
                          figsize=(15, 30), sharey=True)
        plt.subplots_adjust(wspace=0.02)
        a = a.flatten()
        f.delaxes(a[-1])
        c = ['grey','crimson', 'cornflowerblue']
        x = self.f*self.XSTATS[1] + self.XSTATS[0]
        inds = np.random.choice(np.arange(x.shape[0]), size=1000, replace=False)
        for i in range(x.shape[1]):
            a[i].set_xlabel(self.LABELS[i])
            a[i].set_ylim(0,2)
            # scatters
            a[i].scatter(x[inds,i], self.Y[inds], color=c[0], alpha=0.02, marker='.') # data
            a[i].scatter(x[inds,i], lin(self.X[inds], *self.lin[0]), 
                         color=c[1], alpha=0.02, marker='+')
            a[i].scatter(x[inds,i],quad(self.XX[inds], *self.quad[0]),
                         color=c[2], alpha=0.02, marker='*')
            # binned
            y1, x1, s1 = binit(x[:,i], values=lin(self.X, *self.lin[0]))
            a[i].errorbar(x1, y1, yerr=s1, color=c[1], marker='+', ls=':')
            y1, x1, s1 = binit(x[:,i], values=quad(self.XX, *self.quad[0]))
            a[i].errorbar(x1, y1, yerr=s1, color=c[2], marker='*', ls='--')
            y1, x1, s1 = binit(x[:,i], values=self.Y)
            a[i].errorbar(x1, y1, yerr=s1, color=c[0], marker='.', ls='-')
            if i%3==0:a[i].set_ylabel(r'Ngal/$\overline{Ngal}$')
        # add legends        
        for j,(c,l) in enumerate(zip(c, ['DR5 data', 'linear-fit', 'quadratic-fit'])):
            a[0].text(0.5, 0.94-j*0.06, l, color=c, transform=a[0].transAxes)
        plt.savefig(path4nnbar, bbox_inches='tight', dpi=300)
    def savemodels(self, path4models, nside=256):
        print('saving  '+path4models+'lin-weights.hp'+str(nside)+'.fits')
        print('saving  '+path4models+'quad-weights.hp'+str(nside)+'.fits')
        import healpy as hp
        lin_model  = lin(self.X, *self.lin[0])
        quad_model = quad(self.XX, *self.quad[0])
        mapi = np.zeros(12*nside**2)
        mapi[self.hpix] = lin_model
        hp.write_map(path4models+'lin-weights.hp'+str(nside)+'.fits', mapi, fits_IDL=False, dtype=np.float64, overwrite=True)
        mapj = np.zeros(12*nside**2)
        mapj[self.hpix] = quad_model
        hp.write_map(path4models+'quad-weights.hp'+str(nside)+'.fits', mapj, fits_IDL=False, dtype=np.float64, overwrite=True)
#
from argparse import ArgumentParser
ap = ArgumentParser(description='Multivariate linear/quadratic regression')
ap.add_argument('--input',  default='/Volumes/TimeMachine/data/DR7/eBOSS.ELG.NGC.DR7.table.5.r.npy')
ap.add_argument('--axfit',     nargs='*', type=int, default=[i for i in range(18)])
ap.add_argument('--output', default='/Volumes/TimeMachine/data/DR7/results/regression/')
ap.add_argument('--split',  action='store_true')
ap.add_argument('--plots',  action='store_true')
ap.add_argument('--nside',  type=int, default=256)
ns = ap.parse_args()

print('INPUTS :')
dics = ns.__dict__
for keyi in dics.keys():
    print('%20s : %s'%(keyi, dics[keyi]))

import os 
if not os.path.exists(ns.output):
    os.makedirs(ns.output)
    
fitdata = DATA(ns.input, ns.axfit, ns.output+'regression_log', split=ns.split)
if ns.plots:
   fitdata.plotbfitparam(ns.output+'bfit_params')
   fitdata.plotnnbardata(ns.output+'nnbar.pdf')

fitdata.savemodels(ns.output, nside=ns.nside) # only address
