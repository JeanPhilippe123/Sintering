# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:50:13 2021
@author: jplan58
"""

import matplotlib.pyplot as plt
import numpy as np
import os

class Simulation_SSP:
    path = os.path.join(os.sep, os.path.dirname(os.path.realpath(__file__)), '')
    def __init__(self,name,radius,Delta,numrays,numrays_stereo,wlum,pol):
        self.name = name
        self.radius = np.array(radius)
        self.num_spheres = len(radius)
        self.Delta = Delta
        self.numrays = numrays
        self.numrays_stereo = numrays_stereo
        self.wlum = wlum
        self.jx,self.jy,self.phasex,self.phasey = np.array(pol)
        self.pathDatas = os.path.join(self.path,'Simulations',self.name)
        
        self.properties_string = '_'.join([name,str(numrays),str(tuple(radius)),str(Delta),str(tuple(pol))])
        self.properties_string_plot = '_'.join([self.properties_string,str(self.wlum)])

        self.pathDatas = os.path.join(self.path,'Simulations',self.name)
        self.path_plot = os.path.join(self.pathDatas,'Results_plots')
        self.prop = self.properties()
        
        self.prop = self.properties()
        self.prop['sintering'] = calculate_pourcentage_sintering(self)
        # self.label = ', '.join(['MC',str(round(self.prop['SSA_theo'],2)),str(round(self.prop['density_theo'],2))])
        
    def properties(self):
        #Monte-Carlo
        self.path_properties = os.path.join(self.path_plot,self.properties_string_plot+'_properties.npy')
        
        #Open properties
        properties = np.load(self.path_properties,allow_pickle=True).item()
        return properties
    
    def __del__(self):
        del self

def calculate_pourcentage_sintering(self):
    if self.Delta <= self.radius[0]+self.radius[1]:
        
        #Volume à enlever
        r_1=self.radius[0]
        r_2=self.radius[0]
        d=self.Delta
        V_inter=np.pi*(r_1+r_2-d)**2*(d**2+2*d*r_2-3*r_2**2+2*d*r_1+6*r_2*r_1-3*r_1**2)/(12*d)
    
    Vol_sphere = 4*np.pi*self.radius[0]**3/3

    return V_inter/Vol_sphere

def plot_B(self,fig,ax):
    B = self.prop['B']
    sintering = self.prop['sintering']
    
    ax.plot(sintering,B,'b.')
    
def plot_g(self,fig,ax):
    g = self.prop['g']
    sintering = self.prop['sintering']
    
    ax.plot(sintering,g,'b.')

def plot_SSA(self,fig,ax):
    SSA = self.prop['SSA_stereo']
    sintering = self.prop['sintering']
    
    ax.plot(sintering,SSA,'b.')
    
if __name__ == '__main__':
    radius = [176E-6,176E-6]
    deltas = np.linspace(radius[0]/100,2*radius[0],100).round(6)
    
    #====================== SSP vs Sintering ======================
    #Figure 1: B vs Delta
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    for delta in deltas:
        ssp = Simulation_SSP('Sintering1', radius, delta, 1_000_000, 1_000_000, 0.76, [1,1,0,90])
        plot_B(ssp,fig,ax)
        ax.set_xlabel('Sintering (%)')
        ax.set_ylabel('B')
    fig.suptitle('B vs Sintering')
    
    #Figure 2: SSA vs Delta
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    for delta in deltas:
        ssp = Simulation_SSP('Sintering1', radius, delta, 1_000_000, 1_000_000, 0.76, [1,1,0,90])
        plot_SSA(ssp,fig,ax)
        ax.set_xlabel('Sintering (%)')
        ax.set_ylabel('SSA')
    fig.suptitle('SSA vs Sintering')
    
    #Figure 3: g vs Delta
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    for delta in deltas:
        ssp = Simulation_SSP('Sintering1', radius, delta, 1_000_000, 1_000_000, 0.76, [1,1,0,90])
        plot_g(ssp,fig,ax)
        ax.set_xlabel('Sintering (%)')
        ax.set_ylabel('g')
    fig.suptitle('g vs Sintering')
