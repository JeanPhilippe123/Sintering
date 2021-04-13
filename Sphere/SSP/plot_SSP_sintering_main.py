# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:18:05 2021

@author: jplan58
"""

import matplotlib.pyplot as plt
import numpy as np
import plot_SSP_sintering_class as ps

radius = [176E-6,176E-6]
deltas = np.linspace(radius[0]/100,2*radius[0],100).round(6)
pols = [[1,0,0,0],[1,1,0,90]]
label = ['Linear','Circular']
#====================== SSP vs Sintering ======================
#Figure 1: B vs Delta vs pol
fig,ax = plt.subplots(1,1,figsize=[10,6])
for delta in deltas:
    ssp = ps.Simulation_SSP('t4', radius, delta, 1_000_000, 1_000_000, 1.0, pols[0])
    ps.plot_B(ssp,fig,ax)
    ax.set_xlabel('Sintering (%)',fontsize=15)
    ax.set_ylabel('B',fontsize=15)
ps.plot_B(ssp,fig,ax)
fig.suptitle('B vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\B_vs_sintering.png',format='png')

#Figure 2: SSA vs Delta vs pol
fig,ax = plt.subplots(1,1,figsize=[10,6])
for delta in deltas:
    ssp = ps.Simulation_SSP('t4', radius, delta, 1_000_000, 1_000_000, 1.0, pols[0])
    ps.plot_SSA(ssp,fig,ax)
    ax.set_xlabel('Sintering (%)',fontsize=15)
    ax.set_ylabel('SSA',fontsize=15)
ps.plot_SSA(ssp,fig,ax)
fig.suptitle('SSA vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\SSA_vs_sintering.png',format='png')

#Figure 3: g vs Delta vs pol
fig,ax = plt.subplots(1,1,figsize=[10,6])
for delta in deltas:
    ssp = ps.Simulation_SSP('t4', radius, delta, 1_000_000, 1_000_000, 1.0, pols[0])
    ps.plot_g(ssp,fig,ax)
    ax.set_xlabel('Sintering (%)',fontsize=15)
    ax.set_ylabel('g',fontsize=15)
ps.plot_g(ssp,fig,ax)
fig.suptitle('g vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\g_vs_sintering.png',format='png')
#%%
#Figure 4: phase function DOPC vs Delta correction source
fig,ax = plt.subplots(1,1,figsize=[10,6])
for i in range(0,len(deltas),20):
    ssp = ps.Simulation_SSP('t4', radius, deltas[i], 1_000_000, 1_000_000, 1.0, [1,1,0,90])
    Sint_str = str(round(100*ssp.prop['sintering']))+'%'
    ps.plot_pf_DOPC_ref_source(ssp,fig,ax,marker='',linestyle='-',label='Sintering: ' + Sint_str)
    ax.set_xlabel('cos$\Theta$',fontsize=15)
    ax.set_ylabel('DOPC',fontsize=15)
ax.legend(fontsize=15)
fig.suptitle('DOPC phase function vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\DOPC_vs_sintering.png',format='png')
#%%
#Figure 5: phase function DOPL vs Delta correction source
fig,ax = plt.subplots(1,1,figsize=[10,6])
for i in range(0,len(deltas),20):
    ssp = ps.Simulation_SSP('t4', radius, deltas[i], 1_000_000, 1_000_000, 1.0, [1,0,0,0])
    Sint_str = str(round(100*ssp.prop['sintering']))+'%'
    ps.plot_pf_DOPL_ref_source(ssp,fig,ax,marker='',linestyle='-',label='Sintering: ' + Sint_str)
    ax.set_xlabel('cos$\Theta$',fontsize=15)
    ax.set_ylabel('DOPL',fontsize=15)
ax.legend(fontsize=15)
fig.suptitle('DOPL phase function vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\DOPL_vs_sintering.png',format='png')
#%%
#Figure 6: phase function vs Delta correction source
fig,ax = plt.subplots(1,1,figsize=[10,6])
for i in range(0,len(deltas),20):
    ssp = ps.Simulation_SSP('t4', radius, deltas[i], 1_000_000, 1_000_000, 1.0, [1,0,0,0])
    Sint_str = str(round(100*ssp.prop['sintering']))+'%'
    ps.plot_intensity_rt(ssp,fig,ax,marker='',linestyle='-',label='Sintering: ' + Sint_str)
    ax.set_xlabel('cos$\Theta$',fontsize=15)
    ax.set_ylabel('Intensity',fontsize=15)
ax.legend(fontsize=15)
fig.suptitle('Intensity phase function vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\Intensities_vs_sintering.png',format='png')
