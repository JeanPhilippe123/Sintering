# -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 11:40:24 2021

@author: jplan58
"""

import matplotlib.pyplot as plt
import plot_simulations_class as ps
import numpy as np

if __name__ == '__main__':
    plt.close('all')
    #Selecting parameters
    # radius=[350E-6]
    # deltas=[990E-6]
    radius=[1500E-6]
    deltas=[4250E-6]
    shapes = zip(radius,deltas)
    wlums=[0.633]
    pol_vectors = [[1,0,0,0]]
    properties_predict=[]
    wlum = wlums[0]
    c=['b','r','g','k','y']

    #Figure 1
    #Irradiance reflectance vs radius
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    mc = ps.Simulation_Monte_Carlo('simulation_labo', 100_000, radius[0], deltas[0], 0.823, wlum, pol_vectors[0])
    ps.plot_I_transmittance(mc,fig,ax,mc.label)
    # ps.plot_irradiance(mc,mc.label,fig,ax,c=['-.'+c[0],c[0]],TI=True,Tartes=True)
    ax.legend()
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Intensity (W)')
    fig.suptitle('Transmition vs radius')
    
    # fig,ax = plt.subplots(1,1,figsize=[10,6])
    
    #150um
    # depth = np.array([0,0.005,0.008])+0.005
    # I = [480,40,3.5]
    #1500um
    depth = np.array([1,2,3,4,5])/100-0.005
    I = np.array([6450,1830,1155,422,129])/10000000
    
    ax.semilogy(depth,I,'.',)
