# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:40:32 2021

@author: jplan58
"""

import matplotlib.pyplot as plt
import plot_SSP_sintering_class as ps

radius = [176E-6]
delta = 0E-6
pols = [[0,0,0,0],[1,0,0,0],[0,1,0,0],[1,1,0,0]]

#Figure 1: Comparaison entre les intensités
fig,ax = plt.subplots(2,2,figsize=[15,9],sharex=True,sharey=True)
ax = ax.flatten()
for i in range(0,len(pols)):
    ssp = ps.Simulation_SSP('Comparison_mie', radius, delta, 1_000_000, 1_000_000, 1.0, pols[i])
    ps.plot_intensity_rt(ssp,fig,ax[i],marker='',linestyle='-',color='b',label='Raytracing '+ssp.pol_str)
    ps.plot_intensity_hg(ssp,fig,ax[i],factor=1/4,marker='',linestyle='-',color='r',label='Henyey Greenstein')
    if i==1:
        ps.plot_intensity_mie_par(ssp,fig,ax[i],factor=1/4,marker='',linestyle='-',color='k',label='Mie parallele')
    if i==2:
        ps.plot_intensity_mie_per(ssp,fig,ax[i],factor=1/4,marker='',linestyle='-',color='k',label='Mie perpendiculaire')
    if i==0:
        ps.plot_intensity_mie(ssp,fig,ax[i],factor=1/4,marker='',linestyle='-',color='k',label='Mie')
    if i==3:
        ps.plot_intensity_mie(ssp,fig,ax[i],factor=1/4,marker='',linestyle='-',color='k',label='Mie 45$\degree$')
    ax[i].legend()
    ax[i].set_xlabel('cos$\Theta$',fontsize=15)
    ax[i].set_ylabel('Intensity',fontsize=15)
ax[0].set_title('Unpolarized',fontsize=15)
ax[1].set_title('Parallel',fontsize=15)
ax[2].set_title('Perpendicular',fontsize=15)
ax[3].set_title('45$\degree$',fontsize=15)
fig.suptitle('Difference in intensities for different plan of polarization \
 between \n Mie theory, Henyey-Greenstein and Zemax Raytracing for a sphere',fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\Intensities_for_sphere_sp_polarization.png',format='png')

#Figure 2: Comparaison entre les DOPs 
fig,ax = plt.subplots(2,2,figsize=[15,9],sharex=True)
ax = ax.flatten()
ssp = ps.Simulation_SSP('Comparison_mie', radius, delta, 1_000_000, 1_000_000, 1.0, [0,0,0,0])
ps.plot_scattering_pf_mie(ssp,fig,ax[0],S11=True,marker='',linestyle='-',color='b',label='Mie')
ps.plot_scattering_pf_mie(ssp,fig,ax[1],S12=True,marker='',linestyle='-',color='b',label='Mie')
ps.plot_scattering_pf_mie(ssp,fig,ax[2],S33=True,marker='',linestyle='-',color='b',label='Mie')
ps.plot_scattering_pf_mie(ssp,fig,ax[3],S34=True,marker='',linestyle='-',color='b',label='Mie')
ssp = ps.Simulation_SSP('Comparison_mie', radius, delta, 1_000_000, 1_000_000, 1.0, [0,0,0,0])
ps.plot_intensity_rt(ssp,fig,ax[0],factor=4,marker='',linestyle='-',color='k',label='Raytracing')
ssp = ps.Simulation_SSP('Comparison_mie', radius, delta, 1_000_000, 1_000_000, 1.0, [0,0,0,0])
ps.plot_pf_DOPL(ssp,fig,ax[1],factor=-1,color='k',label='Raytracing')
ssp = ps.Simulation_SSP('Comparison_mie', radius, delta, 1_000_000, 1_000_000, 1.0, [1,1,0,0])
ps.plot_pf_DOP45(ssp,fig,ax[2],color='k',label='Raytracing')
ssp = ps.Simulation_SSP('Comparison_mie', radius, delta, 1_000_000, 1_000_000, 1.0, [1,1,0,0])
ps.plot_pf_DOPC(ssp,fig,ax[3],color='k',label='Raytracing')
ax[0].set_yscale('log')
ax[1].set_ylim(-1.05,1.05)
ax[2].set_ylim(-1.05,1.05)
ax[3].set_ylim(-1.05,1.05)
ax[0].set_title('Intensity vs mu')
ax[1].set_title('S12: Unpolarized incident light viewed with a linear polarizer')
ax[2].set_title('S33: 45$\degree$ polarized light incident light viewed with a 45$\degree$ polarizer')
ax[3].set_title('S34: 45$\degree$ polarized light incident light viewed with a Circular polarizer')
for i in range(0,len(pols)):
    ax[i].legend()
    ax[i].set_ylabel('DOP',fontsize=15)
    ax[i].set_xlabel('cos$\Theta$',fontsize=15)
ax[0].set_ylabel('Intensity',fontsize=15)
fig.suptitle('Difference in degree of polarization for different plan of incidence polarization \n \
between Mie theory and Zemax Raytracing for a sphere', fontsize=18)
fig.savefig('Z:\Sintering\Plots\SSPvsSintering_13_02_2021\DOPs_for_sphere_sp_polarization.png',format='png')
