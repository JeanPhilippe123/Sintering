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
    def __init__(self,name,numrays,radius,Delta,g,wlum,pol,B=1.2521,Random_pol=False,diffuse_light=False):
        self.numrays = numrays
        self.radius = radius
        self.Delta = Delta
        self.g = g
        self.B_theo = B
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
        self.properties_string_plot = '_'.join([self.properties_string,str(self.wlum)])
        self.prop = self.properties()
        self.label = ', '.join(['MC',str(round(self.prop['SSA_theo'],2)),str(round(self.prop['density_theo'],2))])
        
    def properties(self):
        #Monte-Carlo
        self.path_properties = os.path.join(self.path_plot,self.properties_string_plot+'_properties.npy')
        
        #Open properties
        properties = np.load(self.path_properties,allow_pickle=True).item()
        return properties
    
    def __del__(self):
        del self
        
class Simulation_Sphere:
    path = os.path.join(os.sep, os.path.dirname(os.path.realpath(__file__)), '')
    def __init__(self,name,radius,Delta,numrays,numrays_stereo,wlum,pol,Random_pol=False,diffuse_light=False):
        self.numrays = numrays
        self.numrays_stereo = numrays_stereo
        self.radius = radius
        self.Delta = Delta
        self.wlum = wlum
        self.pol = pol
        self.Random_pol = Random_pol
        self.diffuse_light = diffuse_light
        self.name = name
        self.pathDatas = os.path.join(self.path,'Sphere','Volume','Simulations',self.name)
        self.path_plot = os.path.join(self.pathDatas,'Results_plots')
        if self.diffuse_light == True:
            self.diffuse_str = 'diffuse'
        else:
            self.diffuse_str = 'not_diffuse'
        self.properties_string = '_'.join([name,str(numrays),str(radius),str(Delta),str(tuple(pol)),self.diffuse_str])
        self.properties_string_plot = '_'.join([self.properties_string,str(self.wlum)])
        self.prop = self.properties()
        self.label = ', '.join(['Sphere',str(round(self.prop['SSA_theo'],2)),str(round(self.prop['density_theo'],2))])
        
    def properties(self):
        #Monte-Carlo
        self.path_properties = os.path.join(self.path_plot,self.properties_string_plot+'_properties.npy')
        
        #Open properties
        properties = np.load(self.path_properties,allow_pickle=True).item()
        return properties
    
def plot_irradiance(self,label,fig,ax,c=['-','--'],TI=True,Tartes=False,x_axis=''):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_irradiances.npy')
    
    def normalized(array):
        return array/max(array)
    
    def find_ind(array):
        ind = list(array).index(1.0)
        return ind

    #Load data
    datas = np.load(path,allow_pickle=True).item()
    
    depth = datas['depth']

    #Ponderation for musp on x_axis
    if x_axis=='musp_theo':
        pond = self.prop[x_axis]
        depth = datas['depth']*pond
    
    #Plot data
    if TI==True and Tartes==True:
        irradiance = normalized(datas['irradiance_down_tartes']+datas['irradiance_up_tartes'])
        # ind = find_ind(irradiance)
        # depth = depth-depth[ind]
        ax.semilogy(depth,irradiance,c[1],label='Total Irradiance tartes')
    if TI==True:
        irradiance = normalized(datas['Irradiance_rt'])
        # ind = find_ind(irradiance)
        # depth = depth-depth[ind]
        ax.semilogy(depth,irradiance,c[0],label='Total Irradiance raytracing '+label)
    # if UI==True and Tartes==True:
    #     ax.semilogy(datas['depth'],normalized(datas['irradiance_up_tartes']),c,label='Downwelling Irradiance tartes')
    # if DI==True and Tartes==True:
    #     ax.semilogy(datas['depth'],normalized(datas['irradiance_down_tartes']),c,label='Upwelling Irradiance tartes')
    # if UI==True:
    #     ax.semilogy(datas['depth'],normalized(datas['Irradiance_down_rt']),c,label='Downwelling Irradiance raytracing '+label)
    # if DI==True:
    #     ax.semilogy(datas['depth'],normalized(datas['Irradiance_up_rt']),c,label='Upwelling Irradiance raytracing '+label)
    # ax.set_yscale('linear')
    return


def plot_Stokes_reflectance_I(self,fig,ax,label,c='-'):
    return plot_Stokes_reflectance(self,fig,ax,0,label,c)
def plot_Stokes_reflectance_Q(self,fig,ax,label,c='-'):
    return plot_Stokes_reflectance(self,fig,ax,1,label,c)
def plot_Stokes_reflectance_U(self,fig,ax,label,c='-'):
    return plot_Stokes_reflectance(self,fig,ax,2,label,c)
def plot_Stokes_reflectance_V(self,fig,ax,label,c='-'):
    return plot_Stokes_reflectance(self,fig,ax,3,label,c)

def plot_Stokes_reflectance(self,fig,ax,datas_ind,label,c):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_map_Stokes_reflectance.npy')
    df = np.load(path,allow_pickle=True).item()
    
    #Calculate radius
    df['radius'] = np.sqrt((df['x']-df['x'].mean())**2+(df['y']-df['y'].mean())**2)

    bins = np.linspace(0,0.1,100)
    n_rays, radius = np.histogram(df['radius'], bins=bins)
    Stokes_bins, radius = np.histogram(df['radius'], weights=df['array_Stokes'][datas_ind], bins=bins)

    #Normalize
    # Stokes_bins = Stokes_bins/max(Stokes_bins)
    ax.semilogy(radius[:-1],Stokes_bins,c,label=label)
    
    ax.legend()
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
    path = os.path.join(self.path_plot,self.properties_string_plot+'_map_Stokes_reflectance.npy')
    
    #Load data
    datas = np.load(path,allow_pickle=True).item()
    
    #Set bins
    bin_max=0.01
    mean_x = datas['x'].mean()
    mean_y = datas['y'].mean()
    bins = (np.linspace(-bin_max,bin_max,100)+mean_x,np.linspace(-bin_max,bin_max,100)+mean_y)
    
    #Histogram of x,y vs Stokes
    Stokes, x_bins, y_bins = np.histogram2d(datas['x'], datas['y'], weights=datas['array_Stokes'][datas_ind], bins=bins)
    x, y = np.meshgrid(x_bins, y_bins)
    
    #Plot
    plt = ax.pcolormesh(x-mean_x,y-mean_y,Stokes,cmap='PuBu')
    cb = fig.colorbar(plt,ax=ax,format='%.0e')
    ax.set_title(datas_str,c='k',x=0.1,y=0.8)
    return cb

def map_reflectance_I(self,fig,ax):
    return map_DOP_reflectance(self,fig,ax,0,'Intensity')
def map_reflectance_DOPL(self,fig,ax):
    return map_DOP_reflectance(self,fig,ax,1,'DOPL')
def map_reflectance_DOP45(self,fig,ax):
    return map_DOP_reflectance(self,fig,ax,2,'DOP45')
def map_reflectance_DOPC(self,fig,ax):
    return map_DOP_reflectance(self,fig,ax,3,'DOPC')

def map_DOP_reflectance(self,fig,ax,datas_ind,datas_str):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_map_DOP_reflectance.npy')
    
    #Load data
    datas = np.load(path,allow_pickle=True).item()

    #Set bins
    bin_max=0.01
    mean_x = datas['x'].mean()
    mean_y = datas['y'].mean()
    bins = (np.linspace(-bin_max,bin_max,100)+mean_x,np.linspace(-bin_max,bin_max,100)+mean_y)
    
    #Histogram of x,y vs Stokes
    I, x_bins, y_bins = np.histogram2d(datas['x'], datas['y'], weights=datas['array_Stokes'][0], bins=bins)
    Q, x_bins, y_bins = np.histogram2d(datas['x'], datas['y'], weights=datas['array_Stokes'][1], bins=bins)
    U, x_bins, y_bins = np.histogram2d(datas['x'], datas['y'], weights=datas['array_Stokes'][2], bins=bins)
    V, x_bins, y_bins = np.histogram2d(datas['x'], datas['y'], weights=datas['array_Stokes'][3], bins=bins)
    x, y = np.meshgrid(x_bins, y_bins)
    
    if datas_ind==0:
        #Intensity
        DOP=I
    if datas_ind==1:
        #DOPL
        DOP=np.true_divide(np.sqrt(Q**2), I, out=np.zeros_like(np.sqrt(Q**2)), where=I!=0)
    if datas_ind==2:
        #DOP45
        DOP=np.true_divide(np.sqrt(U**2), I, out=np.zeros_like(np.sqrt(U**2)), where=I!=0)
    if datas_ind==3:
        #DOPC
        DOP=np.true_divide(np.sqrt(V**2), I, out=np.zeros_like(np.sqrt(V**2)), where=I!=0)
        
    plt = ax.pcolormesh(x-mean_x,y-mean_y,DOP,cmap='PuBu')
    x, y = np.meshgrid(x_bins, y_bins)
    
    #Plot
    cb = fig.colorbar(plt,ax=ax)
    ax.set_title(datas_str,c='k',x=0.30,y=0.8)
    return cb

def plot_I_transmittance(self,fig,ax,label,c='-',**kwargs):
    return plot_Stokes_transmittance(self,fig,ax,1,'I',label,c,**kwargs)
def plot_Q_transmittance(self,fig,ax,label,c='-',**kwargs):
    return plot_Stokes_transmittance(self,fig,ax,2,'Q',label,c,**kwargs)
def plot_U_transmittance(self,fig,ax,label,c='-',**kwargs):
    return plot_Stokes_transmittance(self,fig,ax,3,'U',label,c,**kwargs)
def plot_V_transmittance(self,fig,ax,label,c='-',**kwargs):
    return plot_Stokes_transmittance(self,fig,ax,4,'V',label,c,**kwargs)
        
def plot_Stokes_transmittance(self,fig,ax,Stokes_int,Stokes_str,label,c,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_stokes_transmitance.npy')
    
    #Load data
    Stokes_dict = np.load(path,allow_pickle=True).item()
    ax.plot(Stokes_dict['depths'],Stokes_dict[Stokes_str],c,label=label,**kwargs)
    ax.legend()
    
    return 

def plot_DOPL_transmittance(self,fig,ax,label,c='-',**kwargs):
    return plot_DOP_transmittance(self,fig,ax,2,'DOPL',label,c,**kwargs)
def plot_DOP45_transmittance(self,fig,ax,label,c='-',**kwargs):
    return plot_DOP_transmittance(self,fig,ax,3,'DOP45',label,c,**kwargs)
def plot_DOPC_transmittance(self,fig,ax,label,c='-',**kwargs):
    return plot_DOP_transmittance(self,fig,ax,4,'DOPC',label,c,**kwargs)
        
def plot_DOP_transmittance(self,fig,ax,DOP_int,DOP_str,label,c,**kwargs):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_plot_DOP_transmitance.npy')
    
    #Load data
    df_DOP = np.load(path,allow_pickle=True).item()
    ax.plot(df_DOP['depths'],df_DOP['DOPs'][DOP_int],c,label=label,**kwargs)
    ax.legend()
    
    #Plot data
    ax.set_title(DOP_str)
    ax.set_ylabel('DOP')
    return 

def plot_DOPL_radius_reflectance(self,fig,ax,label,c='-'):
    return plot_DOP_radius_reflectance(self,fig,ax,'DOPL',label,c)
def plot_DOP45_radius_reflectance(self,fig,ax,label,c='-'):
    return plot_DOP_radius_reflectance(self,fig,ax,'DOP45',label,c)
def plot_DOPC_radius_reflectance(self,fig,ax,label,c='-'):
    return plot_DOP_radius_reflectance(self,fig,ax,'DOPC',label,c)
def plot_nrays_radius_reflectance(self,fig,ax,label,c='-'):
    return plot_DOP_radius_reflectance(self,fig,ax,'numberRays',label,c)
        
def plot_DOP_radius_reflectance(self,fig,ax,DOP_str,label,c):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_DOPs_radius_reflectance.npy')
    
    #Load data
    df_DOP = np.load(path,allow_pickle=True).item()['df_DOP']
    ax.plot(df_DOP['radius'],df_DOP[DOP_str],c,label=label)
    ax.legend()
    
    #Plot data
    ax.set_title(DOP_str)
    ax.set_ylabel('DOP')
    return 

def plot_MOPL_reflectance(self,fig,ax,label,c='-'):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_MOPL_radius_reflectance.npy')
    
    #Load data
    df_OPL = np.load(path,allow_pickle=True).item()['df_OPL']
    
    #Calculate MOPL
    bins = np.linspace(0,0.1,100)
    nrays, radius = np.histogram(df_OPL['radius'], weights=df_OPL['intensity'], bins=bins)
    OPL_bins, radius = np.histogram(df_OPL['radius'], weights=df_OPL['intensity']*df_OPL['OPL'], bins=bins)
    radius = radius[:-1]
    MOPL = np.true_divide(OPL_bins, nrays, out=np.zeros_like(nrays), where=nrays!=0)
    
    ax.plot(radius,MOPL,c,label=label)
    
    #Plot data
    ax.set_title('MOPL')
    ax.set_ylabel('MOPL (m)')
    ax.set_xlabel('radius (m)')
    return 

def plot_lair_reflectance(self,fig,ax,label,c='-'):
    path = os.path.join(self.path_plot,self.properties_string_plot+'_lair_radius_reflectance.npy')
    
    #Load data
    df_lair = np.load(path,allow_pickle=True).item()['df_lair']

    #Calculate lair
    plt.figure()
    bins = np.linspace(0,0.1,100)
    nrays, radius = np.histogram(df_lair['radius'], weights=df_lair['intensity'], bins=bins)
    lair_bins, radius = np.histogram(df_lair['radius'], weights=df_lair['intensity']*df_lair['lair'], bins=bins)
    radius = radius[:-1]
    lair = np.true_divide(lair_bins, nrays, out=np.zeros_like(nrays), where=nrays!=0)
    
    ax.plot(radius,lair,c,label=label)
    
    #Plot data
    ax.set_title('$L_{air}$')
    ax.set_ylabel('Length air (m)')
    ax.set_xlabel('radius (m)')
    return 

# def compare_prop(sim1,sim2):
#     print('=======================================')
#     # keys = ['radius','numrays','Delta','Depth','Reflectance','numrays_Reflectance','Tranmitance','numrays_transmittance'
#     #         'Error','numrays_Error','Lost','numrays_Lost','g_theo','gG_theo','density_theo','SSA_theo','mus_theo'
#     for key in sorted(sim1.prop):
#         print(key,sim1.prop[key])
#     print('=======================================')
#     for key in sorted(sim2.prop):
#         print(key,sim2.prop[key])
#     print('=======================================')
        
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
        mc = Simulation_Monte_Carlo('test3_mc_msp', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[1])
        sphere = Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[1])
        plot_Stokes_reflectance_I(mc,fig,ax,label=mc.label,c=c[ind_shape])
        plot_Stokes_reflectance_I(sphere,fig,ax,label=sphere.label,c='--'+c[ind_shape])
        ax.set_xlim([0,0.04])
        ax.set_ylim([1E-3,2])
        ax.set_xlabel('Radius (m)')
        ax.set_ylabel('Intensity (W)')
    fig.suptitle('Reflectance vs radius')
    
    #Figure 2
    #Longueur d'onde 2
    compare_shapes=[[0,1,2]]
    for compare_shape in compare_shapes:
        fig,ax = plt.subplots(2,2,figsize=[10,6],sharex=True)
        for ind_shape in compare_shape:
            mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[0])
            sphere = Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[0])
            plot_DOPL_radius_reflectance(sphere,fig,ax[0,0],label=sphere.label,c=c[ind_shape])
            plot_DOPL_radius_reflectance(mc,fig,ax[0,0],label=mc.label,c='--'+c[ind_shape])
            mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[1])
            sphere = Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[1])
            plot_DOPC_radius_reflectance(sphere,fig,ax[0,1],label=sphere.label,c=c[ind_shape])
            plot_DOPC_radius_reflectance(mc,fig,ax[0,1],label=mc.label,c='--'+c[ind_shape])
            plot_MOPL_reflectance(sphere,fig,ax[1,0],label=sphere.label,c=c[ind_shape])
            plot_MOPL_reflectance(mc,fig,ax[1,0],label=mc.label,c='--'+c[ind_shape])
            ax[1,0].legend(loc='upper left',prop={'size': 8})
            plot_lair_reflectance(sphere,fig,ax[1,1],label=sphere.label,c=c[ind_shape])
            plot_lair_reflectance(mc,fig,ax[1,1],mc.label,c='--'+c[ind_shape])
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
        mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[1])
        sphere = Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[1])
        plot_irradiance(mc,mc.label,fig,ax,c=['-.'+c[ind_shape],c[ind_shape]],TI=True,Tartes=True)
        plot_irradiance(sphere,sphere.label,fig,ax,c=['--'+c[ind_shape]],TI=True,Tartes=False)
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
        mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlum, pol_vectors[1])
        sphere = Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlum, pol_vectors[1])
        plot_DOPC_transmittance(sphere,fig,ax[0],sphere.label,c=c[ind_shape])
        plot_DOPC_transmittance(mc,fig,ax[0],mc.label,c='--'+c[ind_shape])
        mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlums[1], pol_vectors[0])
        sphere = Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlums[1], pol_vectors[0])
        plot_DOPL_transmittance(sphere,fig,ax[1],c=c[ind_shape],label=sphere.label)
        plot_DOPL_transmittance(mc,fig,ax[1],c='--'+c[ind_shape],label=mc.label)
    ax[0].set_xlim([0,0.04])
    ax[1].set_xlim([0,0.04])
    ax[1].set_xlabel('Distance from source (m)')
    fig.suptitle('DOPs transmittance')
    
    #Figure 5 I, DOP, DOPL, DOPC
    #Map DOP reflectance
    for pol in pol_vectors:
        fig,ax = plt.subplots(2,4,sharex=True,sharey=True,figsize=[10,6])
        fig.suptitle('Map DOPs Reflectance' + str(pol),fontsize=16)
        mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[0], deltas[0], 0.89, wlums[1], pol)
        sphere = Simulation_Sphere('test3_sphere', radius[0], deltas[0], 100_000, 1000, wlums[1], pol)
        cb = map_reflectance_I(mc,fig,ax[0,0])
        cb.ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        cb.ax.set_xlabel('W')
        map_reflectance_DOPL(mc,fig,ax[0,1])
        map_reflectance_DOP45(mc,fig,ax[1,0])
        map_reflectance_DOPC(mc,fig,ax[1,1])
        cb = map_reflectance_I(sphere,fig,ax[0,2])
        cb.ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        map_reflectance_DOPL(sphere,fig,ax[0,3])
        map_reflectance_DOP45(sphere,fig,ax[1,2])
        map_reflectance_DOPC(sphere,fig,ax[1,3])
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
        mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[0], deltas[0], 0.89, wlums[1], pol)
        sphere = Simulation_Sphere('test3_sphere', radius[0], deltas[0], 100_000, 1000, wlums[1], pol)
        cb = map_Stokes_reflectance_I(mc,fig,ax[0,0])
        cb.ax.set_xlabel('W')
        map_Stokes_reflectance_Q(mc,fig,ax[0,1])
        map_Stokes_reflectance_U(mc,fig,ax[1,0])
        map_Stokes_reflectance_V(mc,fig,ax[1,1])
        map_Stokes_reflectance_I(sphere,fig,ax[0,2])
        map_Stokes_reflectance_Q(sphere,fig,ax[0,3])
        map_Stokes_reflectance_U(sphere,fig,ax[1,2])
        map_Stokes_reflectance_V(sphere,fig,ax[1,3])
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
        mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[0], deltas[0], 0.89, wlums[ind_wlum], pol_vectors[1])
        sphere = Simulation_Sphere('test3_sphere', radius[0], deltas[0], 100_000, 100, wlums[ind_wlum], pol_vectors[1])
        plot_irradiance(mc,mc.label,fig,ax,c=[c[ind_wlum],'--'+c[ind_wlum]],TI=True,Tartes=True)
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
        mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[ind_shape], deltas[ind_shape], 0.89, wlums[0], pol_vectors[1])
        sphere = Simulation_Sphere('test3_sphere', radius[ind_shape], deltas[ind_shape], 100_000, 100, wlums[0], pol_vectors[1])
        # plot_irradiance(sphere,sphere.label,fig,ax,c=[c[ind_shape],'-.'+c[ind_shape]],TI=True,Tartes=True,x_axis='musp_theo')
        plot_irradiance(mc,sphere.label,fig,ax,c=[c[ind_shape],'-.'+c[ind_shape]],TI=True,Tartes=True,x_axis='musp_theo')
    ax.set_xlabel('$\mu_s\' \ (m^-1)$')
    ax.set_ylabel('Irradiance (W)')
    ax.set_xlim([0,30])
    ax.set_ylim([1E-1,1.5])
    ax.legend(loc='lower left')
    
    #%%
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    mc_msp = Simulation_Monte_Carlo('test3_mc_msp', 100_000, radius[0], deltas[0], 0.89, wlum, pol_vectors[1])
    mc = Simulation_Monte_Carlo('test3_mc', 100_000, radius[0], deltas[0], 0.89, wlum, pol_vectors[1])
    sphere = Simulation_Sphere('test3_sphere', radius[0], deltas[0], 100_000, 100, wlum, pol_vectors[1])
    plot_DOPC_radius_reflectance(sphere,fig,ax,label=sphere.label,c=c[0])
    plot_DOPC_radius_reflectance(mc,fig,ax,label=mc.label,c='--'+c[0])
    plot_DOPC_radius_reflectance(mc_msp,fig,ax,label=mc_msp.label,c='-.'+c[0])
    
    fig,ax = plt.subplots(1,1,figsize=[10,6])
    plot_DOPC_transmittance(mc_msp,fig,ax,mc.label,c=c[0])
    plot_DOPC_transmittance(mc,fig,ax,mc.label,c=c[0])
    plot_DOPC_transmittance(sphere,fig,ax,sphere.label,c='--'+c[0])
    
    #Table values

    # #Get properties
    # sim_mc.properties()
    # sim_sphere.properties()


    # #Map DOP reflectance
    # fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=[10,6])
    # fig.suptitle('Map DOP Reflectance')
    # cb1 = map_DOP_reflectance_I(sim_mc,fig,ax[0,0])
    # cb2 = map_DOP_reflectance_DOPL(sim_mc,fig,ax[0,1])
    # cb3 = map_DOP_reflectance_DOP45(sim_mc,fig,ax[1,0])
    # cb4 = map_DOP_reflectance_DOPC(sim_mc,fig,ax[1,1])
    
    # #DOPs radius reflectance
    # fig,ax = plt.subplots(2,2,figsize=[10,6])
    # fig.suptitle('Map DOPs vs radius reflectance')
    # map_nrays_radius(sim_mc,fig,ax[0,0],label='MC')
    # map_DOPL_radius(sim_mc,fig,ax[0,1],label='MC')
    # map_DOP45_radius(sim_mc,fig,ax[1,0],label='MC')
    # map_DOPC_radius(sim_mc,fig,ax[1,1],label='MC')

    # #MOPL radius reflectance
    # fig,ax = plt.subplots(1,1,figsize=[10,6])
    # fig.suptitle('MOPL vs radius reflectance')
    # plot_MOPL_reflectance(sim_mc,fig,ax,label='MC')
    
    # #Lair radius reflectance
    # fig,ax = plt.subplots(1,1,figsize=[10,6])
    # fig.suptitle('MOPL vs radius reflectance')
    # plot_lair_reflectance(sim_mc,fig,ax,label='MC')