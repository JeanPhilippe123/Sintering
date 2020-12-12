# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:59:26 2020

@author: jplan58
"""
import matplotlib.pyplot as plt
import plot_simulations as ps

if __name__ == '__main__':
    plt.close('all')
    #Selecting parameters
    radius=[66E-6,88E-6,176E-6,88E-6,88E-6]
    deltas=[287E-6,382.66E-6,765.33E-6,347.7E-6,482.1E-6]
    shapes = zip(radius,deltas)
    wlums=[0.8,0.9,1.0]
    pol = [[1,1,0,90],[1,0,0,0]]
    properties_predict=[]
    sim_mc = ps.Simulation_Monte_Carlo('test3_mc', 10_000, radius[0], deltas[0], 0.89, wlums[0], pol[0])
    
    #Get properties
    sim_mc.properties()
    
    #Plot irradiance
    fig,ax = plt.subplots(figsize=[10,6])
    datas=ps.plot_irradiance(sim_mc,'MC',fig,ax)
    fig.suptitle('Irradiance')

    #Map stokes reflectance
    fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=[10,6])
    fig.suptitle('Map Stokes Reflectance')
    cb1 = ps.map_Stokes_reflectance_I(sim_mc,fig,ax[0,0])
    cb2 = ps.map_Stokes_reflectance_Q(sim_mc,fig,ax[0,1])
    cb3 = ps.map_Stokes_reflectance_U(sim_mc,fig,ax[1,0])
    cb4 = ps.map_Stokes_reflectance_V(sim_mc,fig,ax[1,1])

    #Map DOP reflectance
    fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=[10,6])
    fig.suptitle('Map DOP Reflectance')
    cb1 = ps.map_DOP_reflectance_I(sim_mc,fig,ax[0,0])
    cb2 = ps.map_DOP_reflectance_DOPL(sim_mc,fig,ax[0,1])
    cb3 = ps.map_DOP_reflectance_DOP45(sim_mc,fig,ax[1,0])
    cb4 = ps.map_DOP_reflectance_DOPC(sim_mc,fig,ax[1,1])
    
    #DOPs radius reflectance
    fig,ax = plt.subplots(2,2,figsize=[10,6])
    fig.suptitle('Map DOPs vs radius reflectance')
    ps.map_nrays_radius(sim_mc,fig,ax[0,0],label='MC')
    ps.map_DOPL_radius(sim_mc,fig,ax[0,1],label='MC')
    ps.map_DOP45_radius(sim_mc,fig,ax[1,0],label='MC')
    ps.map_DOPC_radius(sim_mc,fig,ax[1,1],label='MC')

    #MOPL radius reflectance
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    fig.suptitle('MOPL vs radius reflectance')
    ps.plot_MOPL_reflectance(sim_mc,fig,ax,label='MC')
    
    #Lair radius reflectance
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    fig.suptitle('MOPL vs radius reflectance')
    ps.plot_lair_reflectance(sim_mc,fig,ax,label='MC')