# -*- coding: utf-8 -*-
"""
Created on Tue Dec 8 13:59:26 2020

@author: jplan58
"""
import matplotlib.pyplot as plt
import plot_simulations_class as ps
import numpy as np

if __name__ == '__main__':
    plt.close('all')
    #Selecting parameters
    radius=[66E-6,88E-6,176E-6,88E-6,88E-6]
    deltas=[287E-6,382.66E-6,765.33E-6,347.7E-6,482.1E-6]
    shapes = zip(radius,deltas)
    wlums=[0.8,0.9,1.0]
    pol_vectors = [[1,0,0,0],[1,1,0,90]]
    properties_predict=[]
    wlum = wlums[2]
    c=['b','r','g','k','y']

    #Figure 1
    #Irradiance reflectance vs radius
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    for ind_shape in [0,1,2]:
        mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[1])
        sphere = ps.Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[1])
        ps.plot_Stokes_reflectance_I(mc,fig,ax,label=mc.label,c=c[ind_shape])
        ps.plot_Stokes_reflectance_I(sphere,fig,ax,label=sphere.label,c='--'+c[ind_shape])
        ax.set_xlim([0,0.04])
        ax.set_ylim([1E-3,2])
        ax.set_xlabel('Radius (m)')
        ax.set_ylabel('Intensity (W)')
    fig.suptitle('Reflectance vs radius')
    
    #Figure 2
    #Longueur d'onde 2
    compare_shapes=[[0,1,2],[2,3,4]]
    for compare_shape in compare_shapes:
        fig,ax = plt.subplots(2,2,figsize=[10,6],sharex=True)
        for ind_shape in compare_shape:
            mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[0])
            sphere = ps.Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[0])
            ps.plot_DOPL_radius_reflectance(sphere,fig,ax[0,0],label=sphere.label,c=c[ind_shape])
            ps.plot_DOPL_radius_reflectance(mc,fig,ax[0,0],label=mc.label,c='--'+c[ind_shape])
            mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[1])
            sphere = ps.Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[1])
            ps.plot_DOPC_radius_reflectance(sphere,fig,ax[0,1],label=sphere.label,c=c[ind_shape])
            ps.plot_DOPC_radius_reflectance(mc,fig,ax[0,1],label=mc.label,c='--'+c[ind_shape])
            ps.plot_MOPL_reflectance(sphere,fig,ax[1,0],label=sphere.label,c=c[ind_shape])
            ps.plot_MOPL_reflectance(mc,fig,ax[1,0],label=mc.label,c='--'+c[ind_shape])
            ax[1,0].legend(loc='upper left',prop={'size': 8})
            ps.plot_lair_reflectance(sphere,fig,ax[1,1],label=sphere.label,c=c[ind_shape])
            ps.plot_lair_reflectance(mc,fig,ax[1,1],mc.label,c='--'+c[ind_shape])
            ax[1,1].legend(loc='upper left',prop={'size': 8})
        ax[0,0].set_xlim([0,.04])
        ax[0,1].set_xlim([0,.04])
        ax[1,0].set_xlim([0,.04])
        ax[1,1].set_xlim([0,.04])
        ax[1,0].set_ylim([0,.3])
        ax[1,1].set_ylim([0,.2])
        fig.suptitle('DOPs vs $L_{air}$/MOPL')

    #Figure 3 Irradiance tranmition
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    for ind_shape in [0,1,2]:
        mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[1])
        sphere = ps.Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[1])
        ps.plot_irradiance(mc,mc.label,fig,ax,c=['-.'+c[ind_shape],c[ind_shape]],TI=True,Tartes=True)
        ps.plot_irradiance(sphere,sphere.label,fig,ax,c=['--'+c[ind_shape]],TI=True,Tartes=False)
    ax.legend(loc='lower left', prop={'size': 10})
    ax.set_ylim([1E-3,2E0])
    ax.set_xlim([0,0.1])
    ax.set_xlabel('Distance from source (m)')
    ax.set_ylabel('Irradiance (W)')
    fig.suptitle('Irradiance transmission')
    
    #Figure 4
    #DOPC, DOPL, transmittance
    fig,ax = plt.subplots(2,figsize=[10,6],sharex=True)
    for ind_shape in [0,1,2]:
        mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[1])
        sphere = ps.Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[1])
        ps.plot_DOPC_transmittance(sphere,fig,ax[0],sphere.label,c=c[ind_shape])
        ps.plot_DOPC_transmittance(mc,fig,ax[0],mc.label,c='--'+c[ind_shape])
        mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlums[1], pol_vectors[0])
        sphere = ps.Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlums[1], pol_vectors[0])
        ps.plot_DOPL_transmittance(sphere,fig,ax[1],c=c[ind_shape],label=sphere.label)
        ps.plot_DOPL_transmittance(mc,fig,ax[1],c='--'+c[ind_shape],label=mc.label)
    ax[0].set_xlim([0,0.04])
    ax[1].set_xlim([0,0.04])
    ax[1].set_xlabel('Distance from source (m)')
    fig.suptitle('DOPs transmittance')
    
    #Figure 5 I, DOP, DOPL, DOPC
    #Map DOP reflectance
    for pol in pol_vectors:
        fig,ax = plt.subplots(2,4,sharex=True,sharey=True,figsize=[10,6])
        fig.suptitle('Map DOPs Reflectance' + str(pol),fontsize=16)
        mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[0], deltas[0], 0.89, wlums[1], pol)
        sphere = ps.Simulation_Sphere('test3_sphere', radius[0], deltas[0], 100_000, 1000, wlums[1], pol)
        cb = ps.map_reflectance_I(mc,fig,ax[0,0])
        cb.ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        cb.ax.set_xlabel('W')
        ps.map_reflectance_DOPL(mc,fig,ax[0,1])
        ps.map_reflectance_DOP45(mc,fig,ax[1,0])
        ps.map_reflectance_DOPC(mc,fig,ax[1,1])
        cb = ps.map_reflectance_I(sphere,fig,ax[0,2])
        cb.ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ps.map_reflectance_DOPL(sphere,fig,ax[0,3])
        ps.map_reflectance_DOP45(sphere,fig,ax[1,2])
        ps.map_reflectance_DOPC(sphere,fig,ax[1,3])
        fig.text(0.22, 0.9, mc.label,fontsize=12)
        fig.text(0.62, 0.9, sphere.label,fontsize=12)
        ax[0,0].set_yticks(np.arange(-0.01,0.011,0.005))
        ax[0,0].set_xticks(np.arange(-0.01,0.011,0.006))
        ax[0,0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax[0,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    
    #Figure 6 I, Q, U, V
    #Map Stokes reflectance
    for pol in pol_vectors:
        fig,ax = plt.subplots(2,4,sharex=True,sharey=True,figsize=[10,6])
        fig.suptitle('Map Stokes Reflectance' + str(pol),fontsize=16)
        mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[0], deltas[0], 0.89, wlums[1], pol)
        sphere = ps.Simulation_Sphere('test3_sphere', radius[0], deltas[0], 100_000, 1000, wlums[1], pol)
        cb = ps.map_Stokes_reflectance_I(mc,fig,ax[0,0])
        cb.ax.set_xlabel('W')
        ps.map_Stokes_reflectance_Q(mc,fig,ax[0,1])
        ps.map_Stokes_reflectance_U(mc,fig,ax[1,0])
        ps.map_Stokes_reflectance_V(mc,fig,ax[1,1])
        ps.map_Stokes_reflectance_I(sphere,fig,ax[0,2])
        ps.map_Stokes_reflectance_Q(sphere,fig,ax[0,3])
        ps.map_Stokes_reflectance_U(sphere,fig,ax[1,2])
        ps.map_Stokes_reflectance_V(sphere,fig,ax[1,3])
        fig.text(0.22, 0.9, mc.label,fontsize=12)
        fig.text(0.62, 0.9, sphere.label,fontsize=12)
        ax[0,0].set_yticks(np.arange(-0.01,0.011,0.005))
        ax[0,0].set_xticks(np.arange(-0.01,0.011,0.006))
        ax[0,0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax[0,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    #Figure 7
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    fig.suptitle('Irradiance transmission vs wlum')
    for ind_wlum in range(0,len(wlums)):
        mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[0], deltas[0], 0.89, wlums[ind_wlum], pol_vectors[1])
        sphere = ps.Simulation_Sphere('test3_sphere', radius[0], deltas[0], 100_000, 100, wlums[ind_wlum], pol_vectors[1])
        ps.plot_irradiance(mc,mc.label,fig,ax,c=[c[ind_wlum],'--'+c[ind_wlum]],TI=True,Tartes=True)
        # plot_irradiance(sphere,sphere.label,fig,ax,TI=True,Tartes=True)
    ax.legend(loc='lower left')
    ax.set_ylim([1E-3,2E0])
    ax.set_xlim([0,0.1])
    ax.set_xlabel('Distance from source (m)')
    ax.set_ylabel('Irradiance (W)')
        
    #Figure 8
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    fig.suptitle('Irradiance vs $\mu_s\'$')
    for ind_shape in [0,1,2]:
        mc = ps.Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlums[0], pol_vectors[1])
        sphere = ps.Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlums[0], pol_vectors[1])
        # plot_irradiance(sphere,sphere.label,fig,ax,c=[c[ind_shape],'-.'+c[ind_shape]],TI=True,Tartes=True,x_axis='musp_theo')
        ps.plot_irradiance(mc,sphere.label,fig,ax,c=[c[ind_shape],'-.'+c[ind_shape]],TI=True,Tartes=True,x_axis='musp_theo')
    ax.set_xlabel('$\mu_s\' \ (m^-1)$')
    ax.set_ylabel('Irradiance (W)')
    ax.set_xlim([0,30])
    ax.set_ylim([1E-1,1.5])
    ax.legend(loc='lower left')
    
    #%%
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    mc_msp = ps.Simulation_Monte_Carlo('test3_mc_msp', 100_000, radius[0], deltas[0], 0.89, wlum, pol_vectors[1])
    sphere = ps.Simulation_Sphere('test3_sphere', radius[0], deltas[0], 100_000, 100, wlum, pol_vectors[1])
    ps.plot_DOPC_radius_reflectance(sphere,fig,ax,label=sphere.label,c=c[0])
    ps.plot_DOPC_radius_reflectance(mc,fig,ax,label=mc.label,c='--'+c[0])
    ps.plot_DOPC_radius_reflectance(mc_msp,fig,ax,label=mc_msp.label,c='-.'+c[0])
    
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    ps.plot_DOPC_transmittance(mc_msp,fig,ax,mc.label,c=c[0])
    ps.plot_DOPC_transmittance(mc,fig,ax,mc.label,c=c[0])
    ps.plot_DOPC_transmittance(sphere,fig,ax,sphere.label,c='--'+c[0])