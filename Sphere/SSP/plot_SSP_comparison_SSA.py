# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:27:11 2021

@author: jplan58
"""

import matplotlib.pyplot as plt
import numpy as np
import plot_SSP_sintering_class as ps

radius_sphere = np.linspace(60E-6,600E-6,10)
radius_spheres = np.expand_dims(radius_sphere,axis=0).repeat(2,axis=0).transpose()
delta = radius_sphere*7/8
pols = [[1,0,0,0],[1,1,0,90]]
label = ['Linear','Circular']
#%%
#Figure 1: phase function DOPC vs Delta correction source
fig,ax = plt.subplots(1,1,figsize=[10,6])
for i in range(0,len(delta)):
    ssp = ps.Simulation_SSP('test_SSA', radius_spheres[i], delta[i], 1_000_000, 1_000_000, 1.0, [1,1,0,90])
    Sint_str = str(round(100*ssp.prop['sintering']))+'%'
    ps.plot_pf_DOPC_ref_source(ssp,fig,ax,marker='',linestyle='-',label='SSA: '+str(round(ssp.prop['SSA_stereo'],1)))
    ax.set_xlabel('cos$\Theta$',fontsize=15)
    ax.set_ylabel('DOPC',fontsize=15)
ax.legend(fontsize=15)
fig.suptitle('DOPC phase function vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\DOPC_vs_sintering.png',format='png')
#%%
#Figure 2: phase function DOPL vs Delta correction source
fig,ax = plt.subplots(1,1,figsize=[10,6])
for i in range(0,len(delta)):
    ssp = ps.Simulation_SSP('test_SSA', radius_spheres[i], delta[i], 1_000_000, 1_000_000, 1.0, [1,0,0,0])
    Sint_str = str(round(100*ssp.prop['sintering']))+'%'
    ps.plot_pf_DOPL_ref_source(ssp,fig,ax,marker='',linestyle='-',label='SSA: '+str(round(ssp.prop['SSA_stereo'],1)))
    ax.set_xlabel('cos$\Theta$',fontsize=15)
    ax.set_ylabel('DOPL',fontsize=15)
ax.legend(fontsize=15)
fig.suptitle('DOPL phase function vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\DOPL_vs_sintering.png',format='png')
#%%
#Figure 3: phase function vs Delta correction source
fig,ax = plt.subplots(1,1,figsize=[10,6])
for i in range(0,len(delta)):
    ssp = ps.Simulation_SSP('test_SSA', radius_spheres[i], delta[i], 1_000_000, 1_000_000, 1.0, [1,0,0,0])
    Sint_str = str(round(100*ssp.prop['sintering']))+'%'
    ps.plot_intensity_rt(ssp,fig,ax,marker='',linestyle='-',label='SSA: '+str(round(ssp.prop['SSA_stereo'],1)))
    ax.set_xlabel('cos$\Theta$',fontsize=15)
    ax.set_ylabel('Intensity',fontsize=15)
ax.legend(fontsize=15)
fig.suptitle('Intensity phase function vs sintering',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\Intensities_vs_sintering.png',format='png')