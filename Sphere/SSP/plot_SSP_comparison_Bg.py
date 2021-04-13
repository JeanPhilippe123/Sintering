# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:27:11 2021

@author: jplan58
"""

import matplotlib.pyplot as plt
import numpy as np
import plot_SSP_sintering_class as ps

wlums=np.linspace(1.3,1.4,10)[:2]
radius = [500E-6]
delta = 0
pols = [[1,0,0,0],[1,1,0,90]]
label = ['Linear','Circular']

def plot_B_vs_n(self,fig,ax,**kwargs):
    B = self.prop['B']
    wlum = self.prop['wlum']
    ax.plot(wlum,B,self.color,**kwargs)
    
def plot_g_vs_n(self,fig,ax,**kwargs):
    g = self.prop['g']
    wlum = self.prop['wlum']
    ax.plot(wlum,g,self.color,**kwargs)

#Figure 1: B vs Delta vs pol
fig,ax = plt.subplots(1,1,figsize=[10,6])
for wlum in wlums:
    ssp = ps.Simulation_SSP('test_bg_3', radius, delta, 1_000_000, 1_000_000, wlum, pols[1])
    plot_B_vs_n(ssp,fig,ax)
xticks=[1.0,1.1,1.2,1.3,1.4,1.5]
# yticks=[1.0,1.05,1.1,1.15,1.3]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xticks,fontsize=15)
ax.set_yticklabels(yticks,fontsize=15)
ax.set_xlabel('Real index of refraction',fontsize=15)
ax.set_ylabel('Absorption enhancement parameter (B)',fontsize=15)
fig.savefig('Z:\Sintering\Plots\plot_article\B_vs_n.png',format='png')

#Figure 2: g vs Delta vs pol
fig,ax = plt.subplots(1,1,figsize=[10,6])
for wlum in wlums:
    ssp = ps.Simulation_SSP('test_bg_3', radius, delta, 1_000_000, 1_000_000, wlum, pols[1])
    plot_g_vs_n(ssp,fig,ax)
    ax.set_xlabel('Real index of refraction',fontsize=15)
    ax.set_ylabel('Asymmetry parameter (g)',fontsize=15)
xticks=[1.0,1.1,1.2,1.3,1.4,1.5]
yticks=[0.7,0.75,0.8,0.85,0.9,0.95,1.]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xticks,fontsize=15)
ax.set_yticklabels(yticks,fontsize=15)
fig.savefig('Z:\Sintering\Plots\plot_article\g_vs_n.png',format='png')
