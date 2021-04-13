# -*- coding: utf-8 -*-
"""
Created on Tue Dec 8 13:59:26 2020

@author: jplan58
"""
import matplotlib.pyplot as plt
import plot_simulations_class as ps
import numpy as np
import miepython
import sys

path_SSP = 'Z:\Sintering\Sphere\SSP'
sys.path.insert(2,path_SSP)
import plot_SSP_sintering_class as pSSP

if __name__ == '__main__':
    plt.close('all')
    #Selecting parameters
    radius=[500E-6]
    delta=1415E-6
    pol_vector = [[1,0,0,0],[1,1,0,90]]
    properties_predict=[]
    c=['b','r','g','k','y']


    #Find g, SSA and B value
    #Figure 1: B vs Delta vs pol
    wlums = [1.1,1.3,1.5]
    imag = 1.5E-6
    g=[]
    B=[]
    fig = plt.subplots(1,1,figsize=[10,6])
    for i,wlum in enumerate(wlums):
        ssp = pSSP.Simulation_SSP('sim_article_B_g_vs_pol_v2', radius, 0., 1_000_000, 1_000_000, wlum, [1,1,0,90])
        g=round(ssp.properties()['g'],2)
        B=round(ssp.properties()['B'],2)
        mc = ps.Simulation_Monte_Carlo('sim_article_B_g_vs_pol_v2', 100_000, radius[0], delta, g, wlum, pol_vector[0], B=B)
        ps.plot_I_transmittance(mc,fig,plt,label=str(wlum)+' '+str(imag),c='-'+c[i],linewidth=3)
        plt.yscale('log')
        # ax.ylim([1E-3,2])
        # ax.xlabel('Radius (m)')
        # ax.ylabel('Intensity (W)')
    plt.title('')
    plt.legend(fontsize=20)
    plt.xlabel('Depth (m)',fontsize=20)
    plt.ylabel('Intensity (W)',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
#%%
    #Figure 2
    #Irradiance transmission vs index
    fig,ax = plt.subplots(2,1,figsize=[10,6],sharex=True)
    for i in range(0,len(wlums)):
        ssp = pSSP.Simulation_SSP('sim_article_B_g_vs_pol_v2', radius, 0., 1_000_000, 1_000_000, wlums[i], [1,1,0,90])
        g=round(ssp.properties()['g'],2)
        B=round(ssp.properties()['B'],2)
        for pol in pol_vector:
            mc = ps.Simulation_Monte_Carlo('sim_article_B_g_vs_pol_v2', 100_000, radius[0], delta, g, wlums[i], pol, B=B)
            if pol == [1,0,0,0]:
                ps.plot_DOPL_transmittance(mc,fig,ax[0],label=str(wlum)+' '+str(imag),c='-'+c[i],linewidth=3)
            if pol == [1,1,0,90]:
                ps.plot_DOPC_transmittance(mc,fig,ax[1],label=str(wlum)+' '+str(imag),c='-'+c[i],linewidth=3)
    ax[0].set_title('DOPL',fontsize=20)
    ax[1].set_title('DOPC',fontsize=20)
    ax[1].set_xlabel('Depth (m)',fontsize=20)
    ax[0].set_ylabel('DOP',fontsize=20)
    ax[1].set_ylabel('DOP',fontsize=20)
    ax[0].legend(fontsize=15)
    ax[1].legend(fontsize=15)
    xticks=[0.,0.02,0.04,0.06,0.08,0.1]
    yticks=[0.,0.2,0.4,0.6,0.8,1.]
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(yticks,fontsize=15)
    ax[1].set_xticks(xticks)
    ax[1].set_yticks(yticks)
    ax[1].set_xticklabels(xticks,fontsize=15)
    ax[1].set_yticklabels(yticks,fontsize=15)
    fig.savefig('Z:\Sintering\Plots\plot_article\DOP_vs_n.png',format='png')
    # ax.set_ylim([1E-3,2])
    # ax.set_xlabel('Radius (m)')
    # ax.set_ylabel('Intensity (W)')

