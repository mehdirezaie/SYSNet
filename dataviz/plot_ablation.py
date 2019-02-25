import matplotlib
matplotlib.use('Agg') # png/pdf backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def ablation_plot(filename):    
    params = {
    'axes.spines.right':False,
    'axes.spines.top':False,
    'axes.labelsize': 12,
    #'text.fontsize': 8,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': True,
    'figure.figsize': [6, 4], # 11 9
    'font.family':'serif'}
    plt.rcParams.update(params)
    ab1 = np.load(filename).item()
    #
    INDICES = ab1['indices']
    VALUES  = ab1['validmin']
    FEAT    = ab1['importance'] + [i for i in range(18)\
                  if i not in ab1['importance']]

    matric_dict = {}
    for i in range(len(INDICES)):
        for j in range(len(VALUES[i])):
            matric_dict[str(i)+'-'+str(INDICES[i][j])] = VALUES[i][j]

    matric = np.zeros(shape=(18, 18))
    for i in range(17):
        for j, sys_i in enumerate(FEAT):
            if str(i)+'-'+str(sys_i) in matric_dict.keys():
                matric[i,j] = (matric_dict['%d-%d'%(i,sys_i)][0]/ab1['RMSEall'])-1.#-ab1['baselineRMSE']
    matric *= 1.e4
    bands = ['r','g','z']
    labels = ['ebv','logHI','nstar']
    labels += ['depth-'+b for b in bands]
    labels += ['seeing-'+b for b in bands]
    #labels += ['airmass-'+b for b in bands]
    labels += ['skymag-'+b for b in bands]
    labels += ['exptime-'+b for b in bands]
    labels += ['mjd-'+b for b in bands]
    xlabels = [labels[j] for j in FEAT]
    mask = ~np.zeros_like(matric, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = False
    mask[matric==0.0] = False
    vmin = np.minimum(np.abs(np.min(matric)), np.abs(np.max(matric))) #* 0.1
    # Set up the matplotlib figure
    f, ax = plt.subplots()
    #plt.title('Correlation Matrix of DR5')
    # Generate a custom diverging colormap
    kw = dict(mask=~mask, cmap=plt.cm.seismic_r, xticklabels=xlabels,
               yticklabels=xlabels[::-1], 
               center=0.0, vmin=-1.*vmin, vmax=vmin, 
               square=True, linewidths=.5, 
               cbar_kws={"shrink": .5, 
               "label":r'$10^{4} \times \delta$RMSE'})
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matric, **kw)
    ax.set_xticklabels(xlabels, rotation=80)
    ax.set_yticks([])
    ax.xaxis.tick_top()
    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="b", lw=2)
    t = ax.text(0.5, 0.2, "Importance",
                ha="center", va="center", rotation=0,
                transform=ax.transAxes,
                bbox=bbox_props)
    bb = t.get_bbox_patch()
    bb.set_boxstyle("rarrow", pad=0.6)
    t1 = ax.text(0.2, 0.5, "Iteration",
                ha="center", va="center", rotation=-90,
                transform=ax.transAxes,
                bbox=bbox_props)
    bb1 = t1.get_bbox_patch()
    bb1.set_boxstyle("rarrow", pad=0.6)
    ou = ''.join([filename[:-4], '.pdf']) # drop .npy
    print('save ... ', ou)
    plt.savefig(ou, bbox_inches='tight')


import sys


ablation_plot(sys.argv[1])

