# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:21:05 2023

@author: porri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameter_freq import Timescape, LCDM, Milne
import matplotlib.cm as mcm

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig_width_pt = 244.0  # Get this from LaTeX using \the\columnwidth
text_width_pt = 508.0 # Get this from LaTeX using \the\textwidth

inches_per_pt = 1.0/72.27                         # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0                # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt*1.5        # width in inches
fig_width_full = text_width_pt*inches_per_pt*1.5  # 17
fig_height =fig_width_full*golden_mean*0.4        # height in inches
fig_size = [fig_width_full,fig_height]            # (9,5.5)   # (9, 4.5)

plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
axlabelfontsize = 14

save = '_'
zmin = np.linspace(0,0.1,41)
half = int(len(zmin) / 5*4)
print(zmin[half])
models = ['TS', 'LCDM', 'Milne']
fctns = {'TS': Timescape, 'LCDM': LCDM, 'Milne': Milne}
colors = {'TS': 'k', 'Milne': 'C0', 'LCDM': 'C1'}
colors = {'TS': 'C0', 'LCDM': 'C3', 'Milne': 'k'}
modelnames = {'TS': 'Timescape', 'Milne': 'Milne', 'LCDM': r'$\Lambda$CDM'}
variables = {'omega':r'$\Omega_{M0}$', 'a':r'$\alpha$', 'b':r'$\beta$', 'x':r'$x_1^{}$', 'c':r'$c$', 
             'M':r'$M$', 'ax':r'$\alpha x_1^{}$', 'bc':r'$-\beta c$', 'Da':r'$\Delta \alpha$', 
             'Db':r'$\Delta \beta$', 'Dx':r'$\Delta x_1^{}$', 'Dc':r'$\Delta c$', 'Dax':r'$\Delta (\alpha x_1^{})$', 
             'Dbc':r'$\Delta (-\beta c)$'}
comparenames = {'PP_1690': 'P+1690', 'PP_1690jla': 'P+580', 
                'PP_JLA_Common/JLA_comm': 'JLA580', 'PP_JLA_Common/JLA': 'JLA740', 
                'PP_1690jlamargfull': 'P+580, Marginalised', 'PP_1690margfull': 'P+1690, Marginalised'}
alllines = ['-', '--', '-.', ':', (0, (3, 2, 1, 2, 1, 2)), (0, (4, 1, 4, 1, 1, 1))]
alllines = ['dashed', '-', '-', 'dashed', 'dashed', '-']

prefixes = ['Pantheon/Build/PP_1690random1000_', 'Pantheon/Build/PP_1690random580marg_', 
            'Pantheon/Build/PP_JLA_Common/JLA_740random580_']
meanlines = {p: alllines[prefixes.index(p)] for p in prefixes}
meannames = {'Pantheon/Build/PP_1690random1000_': r'RS, $N = 1000$', 
             'Pantheon/Build/PP_1690random580marg_' : r'RS of JLA, $N = 580$', 
             'Pantheon/Build/PP_JLA_Common/JLA_740random580_' : r'RS, Marg., $N=1000$', } # differences between _zacP_ and _1537random_ are negligible (< 2.5e-16)

comparefiles = ['PP_1690', 'PP_JLA_Common/JLA', 'PP_JLA_Common/JLA_comm', 'PP_1690jla', 'PP_1690jlamargfull', 'PP_1690margfull'] 

inputfiles = [str(i) for i in range(50)]

# -------------------------------------- READ DATA ------------------------------------------------------------------------------

def getvalues(path, fctn, ref=None, cols=zmin):
    data = pd.DataFrame(fctn(np.loadtxt(path)), index=['omega', 'a', 'b', 'x', 'c', 'M'], columns=cols).T
    data['ax'] = data.a * data.x
    data['bc'] = -1 * data.b * data.c
    try:
        data['Da'] = data.a - ref.a
        data['Db'] = data.b - ref.b
        data['Dx'] = data.x - ref.x
        data['Dc'] = data.c * ref.c
        data['Dax'] = data.ax - ref.ax
        data['Dbc'] = data.bc - ref.bc
    except:
        if len(cols) == len(zmin):
            print('reference:', path)
    return data

def getmodeldata(model, allfilenames, prefix=prefixes[0], cols=zmin):
    if len(comparefiles) == 0:
        return pd.concat({f: getvalues(prefix + f + '_' + model + '.txt', fctns[model], cols=cols) for f in allfilenames}, axis=1)
    ref = getvalues('Pantheon/Build/' + comparefiles[0] + '_' + model + '.txt', fctns[model], cols=zmin)
    if len(cols) != len(ref.index):
        return pd.concat({f: getvalues(prefix + f + '_' + model + '.txt', fctns[model], cols=cols) for f in allfilenames}, axis=1)
    return pd.concat({f: getvalues(prefix + f + '_' + model + '.txt', fctns[model], ref) for f in allfilenames}, axis=1)

fulldata = {m: getmodeldata(m, inputfiles) for m in models}
meandata = {m: fulldata[m].groupby(axis=1, level=1).median() for m in fulldata.keys()}
stddata = {m: fulldata[m].groupby(axis=1, level=1).std(ddof = 1) for m in fulldata.keys()}

comparedata = {m: getmodeldata(m, comparefiles, prefix='Pantheon/Build/') for m in models}

zminreduced = np.array([0.01, 0.033, 0.05, 0.075, 0.1])
#-------------------------------------- read data from other prefixes -----------------------------------------------------------

if len(prefixes) > 1:
    otherfulldata = {p: {m: getmodeldata(m, inputfiles, p, cols = (zminreduced if p == prefixes[1] else zmin)) for m in fulldata.keys()} for p in prefixes[1:]}
    othermeandata = {p: {m: otherfulldata[p][m].groupby(axis=1, level=1).median() for m in fulldata.keys()} for p in otherfulldata.keys()}
    otherstddata = {p: {m: otherfulldata[p][m].groupby(axis=1, level=1).std() for m in fulldata.keys()} for p in otherfulldata.keys()}

###################################### random subsamples including sample size analysis #####################
# need comparefiles = ['PP_1690', 'PP_fullJLA', 'PP_JLAPanthPlus', 'PP_1690jla'] and prefixes = ['Pantheon/Build/PP_1690random_', 'Pantheon/Build/PP_1690random580_', 'Pantheon/Build/CompareRandomSeeded/JLA_anto_'] or similar and np.abs(showallfiles) < 3

zmincompplot = zmin[zmin >= 0.01]

samplesize = [580, 660, 750, 875, 1000, 1125, 1250, 1375]
seeds = np.arange(50)
def colorsK(k):
    maximum = max(samplesize)
    minimum = min(samplesize)
    return mcm.get_cmap('nipy_spectral')(0.1 + (k - minimum) / (maximum - minimum) * 0.8)
fulldataK = {m: {k: getmodeldata(m, [str(k) + '_' + str(s) for s in seeds], prefix='Pantheon/samplesizefiles/PP_1690random', cols=zminreduced) for k in samplesize} for m in models}
meandataK = {m: {k: fulldataK[m][k].groupby(axis=1, level=1).median() for k in samplesize} for m in models}
stddataK = {m: {k: fulldataK[m][k].groupby(axis=1, level=1).std() for k in samplesize} for m in models}

vs = ['a', 'b', 'x', 'c', 'ax', 'bc']
vs = ['a', 'b', 'x', 'c']
# vs = ['ax', 'bc']
models = ['TS', 'LCDM']
fig = plt.figure('paper', figsize=(fig_width_full, fig_height*len(vs)))
for i, v in enumerate(vs):
    ylim = {'x':[-0.15, 0.2], 'c':[-0.045, -0.005], 'a':[0.12, 0.20], 'b':[2.95, 4.4], 'ax':[-0.02, 0.03], 'bc':[0.025, 0.25]}[v]
    # print(v, ylim)
#         fig.add_subplot(len(vs), 3, i*3 + 1)
    plt.subplot2grid((len(vs), 30), (i, 0), colspan=10)
    p = prefixes[2]
    for m in models:
        for f in ['PP_fullJLA', 'PP_JLAPanthPlus']:
            labelfm = comparenames[f] + ',\n' + {'TS': 'Timescape', 'LCDM': '$\Lambda$CDM'}[m]
            ls = alllines[1 + comparefiles.index(f)]
            plt.plot(zmin, comparedata[m].loc[:, (f, v)], linestyle=ls, color=colors[m], label = labelfm)
        labelm = m
        labelp = meannames[p] + ',\n' + {'TS': 'Timescape', 'LCDM': '$\Lambda$CDM'}[m]
        plt.plot(zmin, othermeandata[p][m].loc[:, v], lw=2, color=colors[m], label = labelp, linestyle = (0, (3, 2, 1, 2, 1, 2)))
        plt.fill_between(zmin, othermeandata[p][m].loc[:, v] - otherstddata[p][m].loc[:, v], othermeandata[p][m].loc[:, v] + otherstddata[p][m].loc[:, v], lw=0, color=colors[m], alpha=0.3)
    if i == len(vs)-1:
        leg = plt.legend(loc='lower right', bbox_to_anchor=(0.9, -1.53))
    if i == 0:
        plt.title('JLA')
    plt.ylim(*ylim)
    plt.ylabel(fontsize=axlabelfontsize, ylabel=variables[v])
    # yticks, _ = plt.yticks()
    yticks = np.linspace(*ylim, 5)
    print(yticks, ylim, v)
    if i > 0:
        plt.yticks(yticks[:-1])
    else:
        plt.yticks(yticks)
    plt.xlim(0, 0.1)
    xticks, _ = plt.xticks()
    if i < len(vs) - 1:
        plt.xticks(xticks, [])
    else:
        plt.xticks(xticks[:-1])
    plt.xlabel(fontsize=axlabelfontsize, xlabel=r'$z_{\mathrm{min}}^{}$')
    plt.grid()

    plt.subplot2grid((len(vs), 30), (i, 10), colspan=10)
    p = prefixes[1]
    for m in models:
        for f in ['PP_1690', 'PP_1690jla']:
            labelfm = comparenames[f] + ',\n' + {'TS': 'Timescape', 'LCDM': '$\Lambda$CDM'}[m]
            ls = alllines[1 + comparefiles.index(f)]
            plt.plot(zmin, comparedata[m].loc[:, (f, v)], linestyle=ls, color=colors[m], label = labelfm)
        labelm = m
        labelp = meannames[prefixes[0]] + ',\n' + {'TS': 'Timescape', 'LCDM': '$\Lambda$CDM'}[m]
        plt.plot(zmin, meandata[m].loc[:, v], lw=2, color=colors[m], label = labelp, linestyle = (0, (3, 2, 1, 2, 1, 2)))
        plt.fill_between(zmin, meandata[m].loc[:, v] - stddata[m].loc[:, v], meandata[m].loc[:, v] + stddata[m].loc[:, v], lw=0, color=colors[m], alpha=0.3)
        labelp = meannames[p] if models.index(m) == 0 else ''
    if i == len(vs)-1:
        leg = plt.legend(loc='lower right', bbox_to_anchor=(0.82, -1.53))
    if i == 0:
        plt.title('Pantheon+')
    plt.ylim(*ylim)
    plt.yticks(yticks, [])
    plt.xlim(0, 0.1)
    if i < len(vs) - 1:
        plt.xticks(xticks, [])
    else:
        plt.xticks(xticks[:-1])
    plt.xlabel(fontsize=axlabelfontsize, xlabel=r'$z_{\mathrm{min}}^{}$')
    plt.grid()



    plt.subplot2grid((len(vs), 30), (i, 20), colspan=10)
    p = prefixes[1]
    
    for m in models:
        for f in ['PP_1690margfull', 'PP_1690jlamargfull']:
            labelfm = comparenames[f] + ',\n' + {'TS': 'Timescape', 'LCDM': '$\Lambda$CDM'}[m]
            margf = f.split('margfull')[0]
            ls = alllines[1 + comparefiles.index(margf)]
            plt.plot(zmin, comparedata[m].loc[:, (f, v)], linestyle=ls, color=colors[m], label = labelfm)
        labelp = meannames[prefixes[1]] + ',\n' + {'TS': 'Timescape', 'LCDM': '$\Lambda$CDM'}[m]
        plt.plot(zminreduced, othermeandata[p][m].loc[:, v], lw=2, color=colors[m], label = labelp, alpha=1, ls=(0, (3, 2, 1, 2, 1, 2)))
        plt.fill_between(zminreduced, othermeandata[p][m].loc[:, v] - otherstddata[p][m].loc[:, v], othermeandata[p][m].loc[:, v] + otherstddata[p][m].loc[:, v], lw=0, color=colors[m], alpha=0.3)
    if i == len(vs)-1:
        leg = plt.legend(loc='lower right', bbox_to_anchor=(0.9, -1.52))
    if i == 0:
        plt.title('Pantheon+ Marginalised')
    plt.ylim(*ylim)
    plt.yticks(yticks, [])
    plt.xlim(0, 0.1)
    if i < len(vs) - 1:
        plt.xticks(xticks, [])
    else:
        plt.xticks(xticks[:-1])
    plt.xlabel(fontsize=axlabelfontsize, xlabel=r'$z_{\mathrm{min}}^{}$')
    plt.grid()

plt.subplots_adjust(wspace=0.5, hspace = 0.05 if len(vs) > 1 else None)
# plt.savefig(bbox_inches='tight', fname='fig4_sample_comparison_plot_' + str(len(vs)) + 'vars.pdf', format = 'pdf')

plt.show()
