# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:32:31 2020

@author: jplan58
"""
import matplotlib.pyplot as plt
import numpy as np
import os

class Simulation_Monte_Carlo:
    path = os.path.join(os.sep, os.path.dirname(os.path.realpath(__file__)), '')
    def __init__(self,name,numrays,radius,Delta,g,wlum,pol,Random_pol=False,diffuse_light=False):
        self.numrays = numrays
        self.radius = radius
        self.Delta = Delta
        self.g = g
        self.B_theo = 1.2521
        self.wlum = wlum
        self.pol = pol
        self.Random_pol = Random_pol
        self.diffuse_light = diffuse_light
        self.name = name
        self.pathDatas = os.path.join(self.path,'Monte_Carlo','Simulations',self.name)
        self.path_plot = os.path.join(self.pathDatas,'Results_plots')
        if self.diffuse_light == True:
            self.diffuse_str = 'diffuse'
        else:
            self.diffuse_str = 'not_diffuse'
        self.properties_string = '_'.join([name,str(numrays),str(radius),str(Delta),str(self.B_theo),str(g),str(tuple(pol)),self.diffuse_str])
        
    def properties(self):
        #Monte-Carlo
        self.path_properties = os.path.join(self.path_plot,self.properties_string+'_properties.npy')
        
        #Open properties
        self.dict_properties = np.load(self.path_properties,allow_pickle=True)
        return
    
    def __del__(self):
        del self

def plot_irradiance(self,name,fig,ax,TI=True,UI=True,DI=True,Tartes=True):
    path = os.path.join(self.path_plot,self.properties_string+'_plot_irradiances.npy')
    
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    #Plot data
    if TI==True and Tartes==True:
        ax.semilogy(datas['depth'],datas['irradiance_down_tartes']+datas['irradiance_up_tartes'],label='Total Irradiance tartes')
    if TI==True:
        ax.semilogy(datas['depth'],datas['Irradiance_rt'],label='Total Irradiance raytracing '+name)
    if UI==True and Tartes==True:
        ax.semilogy(datas['depth'],datas['irradiance_up_tartes'],label='Downwelling Irradiance tartes')
    if DI==True and Tartes==True:
        ax.semilogy(datas['depth'],datas['irradiance_down_tartes'],label='Upwelling Irradiance tartes')
    if UI==True:
        ax.semilogy(datas['depth'],datas['Irradiance_down_rt'],label='Downwelling Irradiance raytracing '+name)
    if DI==True:
        ax.semilogy(datas['depth'],datas['Irradiance_up_rt'],label='Upwelling Irradiance raytracing '+name)
    ax.legend(loc='upper right')
    return

def map_Stokes_reflectance_I(self,fig,ax):
    return map_Stokes_reflectance(self,fig,ax,0,'I')
def map_Stokes_reflectance_Q(self,fig,ax):
    return map_Stokes_reflectance(self,fig,ax,1,'Q')
def map_Stokes_reflectance_U(self,fig,ax):
    return map_Stokes_reflectance(self,fig,ax,2,'U')
def map_Stokes_reflectance_V(self,fig,ax):
    return map_Stokes_reflectance(self,fig,ax,3,'V')

def map_Stokes_reflectance(self,fig,ax,datas_ind,datas_str):
    path = os.path.join(self.path_plot,self.properties_string+'_map_Stokes_reflectance.npy')
    
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    #Plot data
    plt = ax.pcolormesh(datas['x'],datas['y'],datas['array_Stokes'][datas_ind])
    cb = fig.colorbar(plt,ax=ax)
    ax.set_title(datas_str,c='w',x=0.1,y=0.8)
    return cb

def map_DOP_reflectance_I(self,fig,ax):
    return map_DOP_reflectance(self,fig,ax,0,'I')
def map_DOP_reflectance_DOPL(self,fig,ax):
    return map_DOP_reflectance(self,fig,ax,1,'DOPL')
def map_DOP_reflectance_DOP45(self,fig,ax):
    return map_DOP_reflectance(self,fig,ax,2,'DOP45')
def map_DOP_reflectance_DOPC(self,fig,ax):
    return map_DOP_reflectance(self,fig,ax,3,'DOPC')

def map_DOP_reflectance(self,fig,ax,datas_ind,datas_str):
    path = os.path.join(self.path_plot,self.properties_string+'_map_DOP_reflectance.npy')
    
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    #Plot data
    plt = ax.pcolormesh(datas['x'],datas['y'],datas['array_DOPs'][datas_ind])
    cb = fig.colorbar(plt,ax=ax)
    ax.set_title(datas_str,c='w',x=0.15,y=0.8)
    return cb

def map_DOPL_radius(self,fig,ax,label):
    return map_DOP_radius(self,fig,ax,'DOPL',label)
def map_DOP45_radius(self,fig,ax,label):
    return map_DOP_radius(self,fig,ax,'DOP45',label)
def map_DOPC_radius(self,fig,ax,label):
    return map_DOP_radius(self,fig,ax,'DOPC',label)
def map_nrays_radius(self,fig,ax,label):
    return map_DOP_radius(self,fig,ax,'numberRays',label)
        
def map_DOP_radius(self,fig,ax,DOP_str,label):
    path = os.path.join(self.path_plot,self.properties_string+'_DOPs_radius_reflectance.npy')
    
    #Load data
    df_DOP = np.load(path,allow_pickle=True).item()['df_DOP']
    ax.plot(df_DOP['radius'],df_DOP[DOP_str],label=label)
    ax.legend()
    
    #Plot data
    ax.set_title(DOP_str)
    return 

def plot_MOPL_reflectance(self,fig,ax,label):
    path = os.path.join(self.path_plot,self.properties_string+'_MOPL_radius_reflectance.npy')
    
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    ax.plot(datas['radius'],datas['MOPL'],label=label)
    ax.legend()
    
    #Plot data
    ax.set_ylabel('MOPL (m)')
    ax.set_xlabel('radius (m)')
    return 

def plot_lair_reflectance(self,fig,ax,label):
    path = os.path.join(self.path_plot,self.properties_string+'_lair_radius_reflectance.npy')
    
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    ax.plot(datas['radius'],datas['lair'],label=label)
    ax.legend()
    
    #Plot data
    ax.set_ylabel('length air (m)')
    ax.set_xlabel('radius (m)')
    return 

if __name__ == '__main__':
    plt.close('all')
    #Selecting parameters
    radius=[66E-6,88E-6,176E-6,88E-6,88E-6]
    deltas=[287E-6,382.66E-6,765.33E-6,347.7E-6,482.1E-6]
    shapes = zip(radius,deltas)
    wlums=[0.8,0.9,1.0]
    pol = [[1,1,0,90],[1,0,0,0]]
    properties_predict=[]
    sim_mc = Simulation_Monte_Carlo('test3_mc', 10_000, radius[0], deltas[0], 0.89, wlums[0], pol[0])
    
    #Get properties
    sim_mc.properties()
    
    #Plot irradiance
    fig,ax = plt.subplots(figsize=[10,6])
    datas=plot_irradiance(sim_mc,'MC',fig,ax)
    fig.suptitle('Irradiance')

    #Map stokes reflectance
    fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=[10,6])
    fig.suptitle('Map Stokes Reflectance')
    cb1 = map_Stokes_reflectance_I(sim_mc,fig,ax[0,0])
    cb2 = map_Stokes_reflectance_Q(sim_mc,fig,ax[0,1])
    cb3 = map_Stokes_reflectance_U(sim_mc,fig,ax[1,0])
    cb4 = map_Stokes_reflectance_V(sim_mc,fig,ax[1,1])

    #Map DOP reflectance
    fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=[10,6])
    fig.suptitle('Map DOP Reflectance')
    cb1 = map_DOP_reflectance_I(sim_mc,fig,ax[0,0])
    cb2 = map_DOP_reflectance_DOPL(sim_mc,fig,ax[0,1])
    cb3 = map_DOP_reflectance_DOP45(sim_mc,fig,ax[1,0])
    cb4 = map_DOP_reflectance_DOPC(sim_mc,fig,ax[1,1])
    
    #DOPs radius reflectance
    fig,ax = plt.subplots(2,2,figsize=[10,6])
    fig.suptitle('Map DOPs vs radius reflectance')
    map_nrays_radius(sim_mc,fig,ax[0,0],label='MC')
    map_DOPL_radius(sim_mc,fig,ax[0,1],label='MC')
    map_DOP45_radius(sim_mc,fig,ax[1,0],label='MC')
    map_DOPC_radius(sim_mc,fig,ax[1,1],label='MC')

    #MOPL radius reflectance
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    fig.suptitle('MOPL vs radius reflectance')
    plot_MOPL_reflectance(sim_mc,fig,ax,label='MC')
    
    #Lair radius reflectance
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    fig.suptitle('MOPL vs radius reflectance')
    plot_lair_reflectance(sim_mc,fig,ax,label='MC')