# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:50:13 2021
@author: jplan58
"""

import matplotlib.pyplot as plt
import numpy as np
import os, sys

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
        self.pol = pol
        self.jx,self.jy,self.phasex,self.phasey = np.array(pol)
        self.pathDatas = os.path.join(self.path,'Simulations',self.name)
        
        self.properties_string = '_'.join([name,str(numrays),str(tuple(radius)),str(Delta),str(tuple(pol))])
        self.properties_string_plot = '_'.join([self.properties_string,str(self.wlum)])

        self.pathDatas = os.path.join(self.path,'Simulations',self.name)
        self.path_plot = os.path.join(self.pathDatas,'Results_plots')
        self.prop = self.properties()
        
        self.prop = self.properties()
        self.prop['sintering'] = calculate_pourcentage_sintering(self)
        self.color, self.pol_str = self.decode_polarization()
        
    def properties(self):
        #Monte-Carlo
        self.path_properties = os.path.join(self.path_plot,self.properties_string_plot+'_properties.npy')
        
        #Open properties
        properties = np.load(self.path_properties,allow_pickle=True).item()
        return properties

    def decode_polarization(self):
        if self.pol == [0,0,0,0]:
            color='k.'
            pol_str='not polorized'
        elif self.pol == [1,0,0,0]:
            color='b.'
            pol_str='Horizontale'
        elif self.pol == [0,1,0,0]:
            color='r.'
            pol_str='Verticale'
        elif self.pol == [1,1,0,0]:
            color='y.'
            pol_str='45$\degree$'
        elif self.pol == [1,1,0,90]:
            color='k.'
            pol_str='Circulaire'
        else:
            print('Select a valid polarization')
            sys.exit()
        return color, pol_str
            
    def __del__(self):
        del self

def calculate_pourcentage_sintering(self):
    if len(self.radius) == 2:
        if self.Delta <= self.radius[0]+self.radius[1]:
            
            #Volume à enlever
            r_1=self.radius[0]
            r_2=self.radius[0]
            d=self.Delta
            V_inter=np.pi*(r_1+r_2-d)**2*(d**2+2*d*r_2-3*r_2**2+2*d*r_1+6*r_2*r_1-3*r_1**2)/(12*d)
            Vol_sphere = 4*np.pi*self.radius[0]**3/3
            return V_inter/Vol_sphere
        else: return 0 #Sphere doesn't touch sintering=0
    else:
        return 0 #Unique sphere sintering=0

def plot_B(self,fig,ax,**kwargs):
    B = self.prop['B']
    sintering = self.prop['sintering']*100
    ax.plot(sintering,B,self.color,**kwargs)
    
def plot_g(self,fig,ax,**kwargs):
    g = self.prop['g']
    sintering = self.prop['sintering']*100
    ax.plot(sintering,g,self.color,**kwargs)
    
def plot_SSA(self,fig,ax,**kwargs):
    SSA = self.prop['SSA_stereo']
    sintering = self.prop['sintering']*100
    ax.plot(sintering,SSA,self.color,**kwargs)

def plot_pf_DOPL(self,fig,ax,factor=1,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_phase_function_DOP.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    ax.plot(datas['mu'],datas['DOPL']*factor,**kwargs)

def plot_pf_DOP45(self,fig,ax,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_phase_function_DOP.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    ax.plot(datas['mu'],datas['DOP45'],**kwargs)

def plot_pf_DOPC(self,fig,ax,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_phase_function_DOP.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    ax.plot(datas['mu'],datas['DOPC'],**kwargs)

def plot_pf_DOPL_ref_source(self,fig,ax,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_DOP_ref_source.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    mu = datas['mu']
    DOPL = datas['DOPL']
    ax.plot(mu,DOPL,**kwargs)
    
def plot_pf_DOPC_ref_source(self,fig,ax,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_DOP_ref_source.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    mu = datas['mu']
    DOPC = datas['DOPC']
    ax.plot(mu,DOPC,**kwargs)

def plot_intensity_mie_per(self,fig,ax,factor,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_intensities.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    mu = datas['mu']
    I_per_mie = datas['I_per_mie']*factor
    ax.semilogy(mu[:-1],I_per_mie,self.color,**kwargs)

def plot_intensity_mie_par(self,fig,ax,factor,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_intensities.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    mu = datas['mu']
    I_par_mie = datas['I_par_mie']*factor
    ax.semilogy(mu[:-1],I_par_mie,self.color,**kwargs)

def plot_intensity_mie(self,fig,ax,factor=1,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_intensities.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    mu = datas['mu']
    I_mie = datas['I_mie']*factor
    ax.semilogy(mu[:-1],I_mie,self.color,**kwargs)

def plot_intensity_hg(self,fig,ax,factor=1,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_intensities.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    mu = datas['mu']
    I_hg = datas['I_hg']*factor
    ax.semilogy(mu,I_hg,self.color,**kwargs)
    
def plot_intensity_rt(self,fig,ax,factor=1,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_intensities.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    mu = datas['mu']
    I_rt = datas['I_rt']*factor
    ax.semilogy(mu[:-1],I_rt,**kwargs)
    
def plot_scattering_pf_mie(self,fig,ax,S11=False,S12=False,S33=False,S34=False,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_scattering_matrix_mie.npy')
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    mu = datas['mu']
    if S11 == True:
        S11 = datas['S11']
        ax.plot(mu,S11,self.color,**kwargs)
    if S12 == True:
        S12 = datas['S12']
        S11 = datas['S11']
        ax.plot(mu,S12/S11,self.color,**kwargs)
    if S33 == True:
        S33 = datas['S33']
        S11 = datas['S11']
        ax.plot(mu,S33/S11,self.color,**kwargs)
    if S34 == True:
        S34 = datas['S34']
        S11 = datas['S11']
        ax.plot(mu,S34/S11,self.color,**kwargs)
        
if __name__ == '__main__':
    radius = [176E-6,176E-6]
    deltas = np.linspace(radius[0]/100,2*radius[0],30).round(6)
    pol = [1,0,0,0]
    
    #====================== SSP vs Sintering ======================
    #Figure 1: B vs Delta
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    for delta in deltas:
        ssp = Simulation_SSP('t3', radius, delta, 100_000, 100_000, 1.0, pol)
        plot_intensity_rt(ssp,fig,ax)
