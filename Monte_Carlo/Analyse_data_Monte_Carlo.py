"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import time as time
import tartes
import scipy.constants
import scipy.optimize
import concurrent.futures
import os
import psutil
import matplotlib.colors as colors

path_Vol_Raytrace = 'Z:\Sintering\Sphere\Volume'
sys.path.insert(1,path_Vol_Raytrace)
pathRaytraceDLL = 'C:\Zemax Files\ZOS-API\Libraries\Raytrace.DLL'
sys.path.insert(2,os.path.dirname(os.path.realpath(pathRaytraceDLL)))
import PythonNET_ZRDLoaderFull as init_Zemax
import Vol_Raytrace

class simulation_MC:
    path = os.path.join(os.sep, os.path.dirname(os.path.realpath(__file__)), '')
    max_segments = 4000
    pice = 917
    
    def __init__(self,name,numrays,radius,Delta,g,wlum,pol,Random_pol=False,diffuse_light=False):
        self.name = name
        self.numrays = numrays
        self.wlum = wlum
        self.radius = radius
        self.radius_source = self.radius*10        
        self.Delta = Delta
        self.calculate_density()
        self.calculate_SSA()
        self.calculate_mus()
        self.B_theo = 1.2521
        self.physical_porosity_theo = 0.80536
        self.ice_index,self.ice_complex = tartes.refice2016(self.wlum*1E-6)
        self.calculate_mua()
        self.g_theo = g
        self.gG_theo = self.g_theo*2-1
        self.jx,self.jy,self.phase_x,self.phase_y = np.array(pol)
        self.pathZMX = os.path.join(self.path,'Simulations', self.name)
        self.Random_pol=Random_pol
        self.diffuse_light = diffuse_light
        self.neff_stereo = 1.06
        if self.diffuse_light == True:
            self.tartes_dir_frac = 0
            self.diffuse_str = 'diffuse'
        else:
            self.tartes_dir_frac = 1
            self.diffuse_str = 'not_diffuse'
        
        self.name_ZRD = '_'.join([name,str(round(self.mus_theo,4)),str(self.B_theo),str(round(self.physical_porosity_theo,4)),str(g),str(pol),str(self.numrays),self.diffuse_str])+ ".ZRD"
        self.path_npy = os.path.join(self.pathZMX,'_'.join([name,str(round(self.mus_theo,4)),str(self.B_theo),str(round(self.physical_porosity_theo,4)),str(g),str(pol),str(self.numrays),self.diffuse_str])+ ".npy")
        self.path_ZRD = os.path.join(self.pathZMX,self.name_ZRD)
        
    def create_folder(self):
        if not os.path.exists(self.pathZMX):
            os.makedirs(self.pathZMX)
            print("Le répertoire " + str(self.name) +  " a été créé")
        else:
            print("Le répertoire " + str(self.name) +  " existe déjà")
    
    def find_source_object(self):
        Source_obj = np.where(self.array_objects() == np.array(['Source Ellipse']))[0] + 1
        return Source_obj
    
    def find_detector_object(self):
        Detector_obj = np.where(self.array_objects() == np.array(['Detector Rectangle']))[0] + 1
        return Detector_obj
    
    def array_objects(self):
        list_obj=[]
        for i in range(1,self.TheNCE.NumberOfObjects+1):
            Object = self.TheNCE.GetObjectAt(i)
            ObjectType = Object.TypeName
            ObjectMaterial = Object.Material 
            list_obj += [[ObjectType,ObjectMaterial]]
            array_obj = np.array(list_obj)
        return array_obj
    
    def Initialize_Zemax(self):
        self.zosapi = init_Zemax.PythonStandaloneApplication()
        self.BatchRayTrace = self.zosapi.BatchRayTrace 
        self.TheSystem = self.zosapi.TheSystem
        self.TheApplication = self.zosapi.TheApplication
        self.TheNCE = self.TheSystem.NCE
        
        self.ZOSAPI = self.zosapi.ZOSAPI
        self.ZOSAPI_NCE = self.ZOSAPI.Editors.NCE
        
        #Create File for non-sticky (ns)
        self.fileZMX = os.path.join(self.pathZMX, 'test_MC.zmx')
        
    def Load_File(self):
        start = time.time()
        self.TheSystem.LoadFile(self.fileZMX,False)
        self.TheNCE = self.TheSystem.NCE
        
        #Change wl in um
        self.TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = 1.0
        
        #Change Polarization
        Source_obj = self.find_source_object()[0]
        Source = self.TheNCE.GetObjectAt(Source_obj)
        
        Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par6).DoubleValue = self.radius_source #X Half width
        Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par7).DoubleValue = self.radius_source #Y Half width
        
        Source.SourcesData.RandomPolarization = self.Random_pol
        Source.SourcesData.Jx = self.jx
        Source.SourcesData.Jy = self.jy
        Source.SourcesData.XPhase = self.phase_x
        Source.SourcesData.YPhase = self.phase_y
        #Change radius of source to correspond with the zemax

        Rectangular_obj = self.TheNCE.GetObjectAt(1)
        Rectangular_obj_physdata = Rectangular_obj.VolumePhysicsData
        Rectangular_obj_physdata.Model = self.ZOSAPI_NCE.VolumePhysicsModelType.DLLDefinedScattering
        self.calculate_mua()
        print(Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.GetParameterName(2))
        Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.MeanPath = 1/(self.mus_theo+self.mua_theo)
        Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.g = self.g_theo
        Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.SetParameterValue(0,1-self.mua_theo/(self.mus_theo+self.mua_theo))
        
        self.TheSystem.SaveAs(self.fileZMX)
        end = time.time()
        print('Fichier loader en ',end-start)
    
    def shoot_rays(self):
        Source_obj = self.find_source_object()[0]
        Source_object = self.TheNCE.GetObjectAt(Source_obj)
        if self.diffuse_light==True:
            cosine = 1.0
        else:
            cosine = 100.
        cosine
        Source_object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par9).DoubleValue = cosine
            
        print('Raytrace')
        Vol_Raytrace.Shoot(self,'',self.numrays,self.path_npy,self.name_ZRD)
        pass    
    
    def Load_npyfile(self): 
        #Try loading the ray file
        try:
            #Filter les rayons avec plus de 4000 interactions
            self.df = Vol_Raytrace.Load_npy(self.path_npy)
            
            #Change pathLength
            self.df['pathLength'] = np.sqrt((((self.df[['x','y','z']].diff())**2).sum(1)))
            #Change intensity
            self.change_path_intensity()
            
            #Groupby for last segment of each ray
            df = self.df.groupby(self.df.index).agg({'hitObj':'last','segmentLevel':'last','intensity':'last'})
            #Reflectance
            filt_top_detector = df['hitObj'] == self.find_detector_object()[0]
            df_top = df[filt_top_detector]
            self.Reflectance = np.sum(df_top['intensity'])
            self.numrays_Reflectance = df_top.shape[0]
            #Transmitance
            filt_down_detector = df['hitObj'] == self.find_detector_object()[1]
            df_down = df[filt_down_detector]
            self.Transmitance = np.sum(df_down['intensity'])
            self.numrays_Transmitance = df_down.shape[0]
            #Error (max_segments)
            filt_error = df['segmentLevel'] == self.max_segments-1
            df_error = df[filt_error]
            self.Error = np.sum(df_error['intensity'])
            self.numrays_Error = df_error.shape[0]
            #Lost
            filt_Lost = (~filt_top_detector)&(~filt_down_detector)&(~filt_error)
            df_Lost = df[filt_Lost]
            self.Lost = np.sum(df_Lost['intensity'])
            self.numrays_Lost = df_Lost.shape[0]
            #Absorb (considering total intensity is 1)
            self.Absorb = np.abs(1-self.Reflectance-self.Transmitance-self.Error-self.Lost)
            self.numrays_Absorb = self.numrays-self.numrays_Reflectance-self.numrays_Transmitance-self.numrays_Error-self.numrays_Lost
        except FileNotFoundError:
            print('The raytrace npy file was not loaded, Please run raytrace')
            sys.exit()
            pass
        
    def change_path_intensity(self):
        if not hasattr(self, 'optical_porosity_theo'): self.calculate_porosity()
        #Calculate the intensity for each segments
        
        filt_ice = ((self.df['segmentLevel'] != 0) & (self.df['segmentLevel'].astype('float').diff(-1) == -1))
        
        #Calculate new intensity
        self.gamma = 4*np.pi*self.ice_complex/(self.wlum*1E-6)
        I_0 = 1./self.numrays
        #Ponderate pathlength for ice only (absorbing media)
        self.df.insert(15,'ponderation',0)
        self.df.loc[(filt_ice,'ponderation')] = 1-self.optical_porosity_theo
        pond_pL = self.df['ponderation']*self.df['pathLength']
        
        #Overwrite the intensity
        pathLength = pond_pL.groupby(pond_pL.index).cumsum()
        intensity = I_0*np.exp(-self.gamma*pathLength).values
        self.df['intensity'] = pd.DataFrame(intensity,dtype='float32',index=self.df.index)
        
        self.df = self.df.drop(columns = ['ponderation'])
        pass

    def calculate_porosity(self):
        #====================Porosité physique théorique====================
        if not hasattr(self, 'density_theo'): self.calculate_density()
            
        porosity = 1-self.density_theo/self.pice
        self.physical_porosity_theo = porosity
        
        #====================Porosité optique théorique====================
        self.optical_porosity_theo = porosity/(porosity+self.B_theo*(1-porosity))
    
    def calculate_mua(self):
        #Calculate mua theorique
        self.gamma = 4*np.pi*self.ice_complex/(self.wlum*1E-6)
        self.mua_theo = self.B_theo*self.gamma*(1-self.physical_porosity_theo)
        #Calculate mua with Tartes
        if hasattr(self,'ke_tartes') == False: return
        self.mua_tartes = -self.ke_tartes*np.log(self.alpha_tartes)/4
        
    def calculate_musp(self):
        self.musp_theo = self.mus_theo*(1-self.g_theo)
        pass
    
    def calculate_alpha(self):
        # filt_error = self.df['segmentLevel'] == self.max_segments-1
        # df_error = self.df[filt_error]
        # I_error = np.sum(df_error['intensity']*np.exp(-df_error['z']*self.ke_theo)/2)    
        
        self.alpha_rt = self.Reflectance #+ I_error
        self.alpha_tartes = float(tartes.albedo(self.wlum*1E-6,self.SSA_theo,self.density_theo,g0=self.g_theo,B0=self.B_theo,dir_frac=self.tartes_dir_frac))

    def up_irradiance_at_depth(self,depth):
        filt_detector = self.df['z'] <= depth
        df = self.df.loc[filt_detector]
        filt = ((df['segmentLevel'].diff() != 1.0)
                & (df['segmentLevel'] != 0))
        df_filtered = df[filt]
        return df_filtered
    
    def down_irradiance_at_depth(self,depth):
        filt_detector = self.df['z'] >= depth
        df = self.df.loc[filt_detector]
        filt = df['segmentLevel'].diff() != 1.0 
        df_filtered = df.loc[filt]
        return df_filtered
    
    def calculate_g_theo(self):
        self.g_theo = 0.89
        self.gG_theo = self.g_theo*2-1
    
    def calculate_g_rt(self):
        if not hasattr(self, 'ke_rt'): self.calculate_ke_rt()
        if not hasattr(self, 'alpha_rt'): self.calculate_alpha()
        
        #g calculé avec ke et alpha raytracing
        #Équation Quentin Libois 3.3 thèse
        self.g_rt = 1+8*self.ke_rt/(3*self.density_stereo*np.log(self.alpha_rt)*self.SSA_stereo)
        self.gG_rt = self.g_rt*2-1
        
    def ke_raytracing(self,depths_fit,intensity):
        [a,b], pcov=scipy.optimize.curve_fit(lambda x,a,b: a*x+b, depths_fit, np.log(intensity))
        return -a, b
    
    def calculate_ke_rt(self):
        #ke raytracing
        def linear_fit(depth,a,b):
            intensity = a*depth+b
            return intensity
        
        def I_ke_raytracing(depth):
            df = self.up_irradiance_at_depth(depth)
            irradiance_down = sum(df['intensity']*np.abs(df['n']))
            
            df = self.down_irradiance_at_depth(depth)
            irradiance_up = sum(df['intensity']*np.abs(df['n']))
            # print(irradiance_down+irradiance_up)
            return irradiance_down+irradiance_up
        
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        
        depths_fit = np.linspace(1/self.musp_theo,10/self.musp_theo,10)
        intensity_rt = list(map(I_ke_raytracing,depths_fit))
        self.ke_rt, self.b_rt = self.ke_raytracing(depths_fit,intensity_rt)
        
    def calculate_ke_theo(self):
        if not hasattr(self, 'density_theo'): self.calculate_density()
        if not hasattr(self, 'g_theo'): self.calculate_g_theo()
        if not hasattr(self, 'SSA_theo'): self.calculate_SSA()
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        
        depths_fit = np.linspace(1/self.musp_theo,10/self.musp_theo,10)
        
        #ke TARTES
        down_irr_profile, up_irr_profile = tartes.irradiance_profiles(
            self.wlum*1E-6,depths_fit,self.SSA_theo,self.density_theo,g0=self.g_theo,B0=self.B_theo,dir_frac=self.tartes_dir_frac,totflux=0.75)
        # plt.plot(depths_fit,np.exp(-self.ke_rt*depths_fit + b),label='fit raytracing')
        self.ke_tartes, self.b_tartes = self.ke_raytracing(depths_fit, down_irr_profile+up_irr_profile)

        #ke théorique
        #Imaginay part of the indice of refraction
        self.gamma = 4*np.pi*self.ice_complex/(self.wlum*1E-6)
        self.ke_theo = self.density_theo*np.sqrt(3*self.B_theo*self.gamma/(4*self.pice)*self.SSA_theo*(1-self.gG_theo))
        pass
    
    def calculate_density(self):
        #====================Densité physique théorique====================
        if np.sqrt(2)*self.radius > self.Delta/2:
            DensityRatio=0
        else:
            volsphere=4*np.pi*self.radius**3/3
            volcube=self.Delta**3
            DensityRatio = volsphere*4/volcube
        self.density_theo = DensityRatio*self.pice
    
    def calculate_SSA(self):
        #====================SSA théorique====================
        vol_sphere = 4*np.pi*self.radius**3/3
        air_sphere = 4*np.pi*self.radius**2
        self.SSA_theo = air_sphere/(vol_sphere*self.pice)
        
    def calculate_mus(self):
        self.mus_theo = self.density_theo*self.SSA_theo/2
        
    def calculate_MOPL(self):
        #Shoot rays for SSA
        if not hasattr(self, 'df'): self.Load_npyfile()
        if not hasattr(self, 'musp_theo'): self.calculate_musp()
        
        z_o = self.musp_theo**(-1)
        D = (3*(self.mua_theo+self.musp_theo))**(-1)
        self.MOPL_theo = self.neff_stereo*z_o/(2*np.sqrt(self.mua_theo*D))
        
        #Raytracing
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df_top = self.df[filt_top_detector]
        df_filt = self.df.loc[df_top.index]
        
        df_filt.insert(15,'OPL',np.nan)
        df_filt['pathLength'] = np.sqrt((((df_filt[['x','y','z']].diff())**2).sum(1)))
        df_filt = df_filt[~(df_filt['hitObj'] == 2)]
        df_filt['OPL'] = df_filt['pathLength']*self.neff_stereo
        df_OPL = df_filt.groupby(df_filt.index).agg({'OPL':sum,'intensity':'last'})
        self.MOPL_rt = np.average(df_OPL['OPL'],weights=df_OPL['intensity'])

    def time(self,text):
        t = time.localtime()
        self.date = str(t.tm_year)+"/"+str(t.tm_mon)+"/"+str(t.tm_mday)
        self.ctime = str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)
        print('Date : ',self.date,', Time : ',self.ctime,' ',text)
        
    def plot_time_reflectance(self):
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        if not hasattr(self, 'mua_rt'): self.calculate_mua()
        #if not hasattr(self, 'neff_rt'): self.calculate_neff()
        
        #Raytracing
        #Add a line of time for each segment
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df_top = self.df[filt_top_detector]
        df_filt = self.df.loc[df_top.index]
        df_filt.insert(15,'time',np.nan)
        df_filt['pathLength'] = np.sqrt((((df_filt[['x','y','z']].diff())**2).sum(1)))
        df_filt = df_filt[~((df_filt['segmentLevel']==1)|(df_filt['hitObj']==3))]
        v_medium = scipy.constants.c/self.ice_index
        df_filt['time'] = df_filt['pathLength']/v_medium
        
        #Time for each ray
        df = df_filt.groupby(df_filt.index).agg({'time':sum,'intensity':'last','hitObj':'last'})
        
        #Histogram of time vs intensity
        R_rt, bins = np.histogram(df['time'], weights=df_top['intensity'], bins=1000)
        t_rt = bins[:-1]
        
        #Théorique
        z_o = self.musp_theo**(-1)
        D = (3*(self.mua_theo+self.musp_theo))**(-1)
        t_theo = np.linspace(1E-16,1E-8,10000)
        c = scipy.constants.c/self.neff_stereo
        R_theo = (4*np.pi*D*c)**(-1/2)*z_o*t_theo**(-(3/2))*np.exp(-self.mua_theo*c*t_theo)*np.exp(-z_o**2/(4*D*c*t_theo))
        
        #Normalize
        R_rt = R_rt/max(R_rt)
        R_theo = R_theo/max(R_theo)
        
        #Change the offset on time theo so the 2 curves overlap (theo and rt)
        t_rt_offset = t_rt[(list(R_rt)).index(max(R_rt))]
        t_theo_offset = t_theo[(list(R_theo)).index(max(R_theo))]
        t_diff = t_rt_offset-t_theo_offset
        t_theo = t_theo+t_diff
        # np.average(t_theo*scipy.constants.c/self.neff_stereo,weights=R_theo)
        
        #Plot reflectance vs time
        plt.figure()
        plt.plot(bins[:-1],R_rt)
        plt.plot(t_theo,R_theo)
        #plt.xlim(1E-16,1E-12)
        plt.show()
        pass
    
    def calculate_Stokes_of_rays(self,df):
        if df.empty:
            #Return empty sequence
            return np.zeros(5)
        
        #Vecteur de polarization dans le plan othogonaux
        Ex = np.array(df['exr']+complex(0,1)*df['exi'])
        Ey = np.array(df['eyr']+complex(0,1)*df['eyi'])
        
        #Polariseur horizontale
        Efx=(1*Ex+0*Ey)
        Efy=(0*Ex+0*Ey)
        Iv=(Efx*Efx.conjugate()+Efy*Efy.conjugate()).real
        
        #Polariseur verticale
        Efx=(0*Ex+0*Ey)
        Efy=(0*Ex+1*Ey)
        Ih=(Efx*Efx.conjugate()+Efy*Efy.conjugate()).real

        #Polariseur à 45
        Efx=1/2*(Ex+Ey)
        Efy=1/2*(Ex+Ey)
        I45=(Efx*Efx.conjugate()+Efy*Efy.conjugate()).real
        
        #Polariseur à -45
        Efx=1/2*(Ex-Ey)
        Efy=1/2*(-Ex+Ey)
        Im45=(Efx*Efx.conjugate()+Efy*Efy.conjugate()).real

        #Polariseur Circulaire droit
        Efx=1/2*(Ex-complex(0,1)*Ey)
        Efy=1/2*(complex(0,1)*Ex+Ey)
        Icd=(Efx*Efx.conjugate()+Efy*Efy.conjugate()).real
        
        #Polariseur Circulaire gauche
        Efx=1/2*(Ex+complex(0,1)*Ey)
        Efy=1/2*(-complex(0,1)*Ex+Ey)
        Icg=(Efx*Efx.conjugate()+Efy*Efy.conjugate()).real
        
        I=Ih+Iv
        Q=Ih-Iv
        U=I45-Im45
        V=Icd-Icg
        return np.array([I,Q,U,V])
    
    def calculate_DOP_vs_xy(self,df):
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        
        #Calculate Stokes vs Radius
        XY_detector = 100/self.mus_theo
        bins = (np.linspace(-XY_detector,XY_detector,33),np.linspace(-XY_detector,XY_detector,33))
        
        #Histogram of time vs intensity
        n_rays, x_bins, y_bins = np.histogram2d(df['x'], df['y'], bins=bins)
        I, x_bins, y_bins = np.histogram2d(df['x'], df['y'], weights=I, bins=bins)
        Q, x_bins, y_bins = np.histogram2d(df['x'], df['y'], weights=Q, bins=bins)
        U, x_bins, y_bins = np.histogram2d(df['x'], df['y'], weights=U, bins=bins)
        V, x_bins, y_bins = np.histogram2d(df['x'], df['y'], weights=V, bins=bins)
        
        #Calculate DOPs
        #With handling division by zero
        # DOP=np.true_divide(np.sqrt(Q**2+U**2+V**2), I, out=np.zeros_like(np.sqrt(Q**2+U**2+V**2)), where=I!=0)
        # DOPLT=np.true_divide(np.sqrt(Q**2+U**2), I, out=np.zeros_like(np.sqrt(Q**2+U**2)), where=I!=0)
        DOPL=np.true_divide(np.sqrt(Q**2), I, out=np.zeros_like(np.sqrt(Q**2)), where=I!=0)
        DOP45=np.true_divide(np.sqrt(U**2), I, out=np.zeros_like(np.sqrt(U**2)), where=I!=0)
        DOPC=np.true_divide(np.sqrt(V**2), I, out=np.zeros_like(np.sqrt(V**2)), where=I!=0)
        
        return np.array([I,DOPL,DOP45,DOPC,n_rays]),x_bins,y_bins
    
    def plot_DOP_radius_top_detector(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df = self.df[filt_top_detector]
        
        #Plot datas
        fig, ax = plt.subplots(nrows=2,ncols=2)
        #return dataframe with dops colummns
        df_DOP = self.calculate_DOP_vs_radius(df)
        ax[0,0].plot(df_DOP['radius'],df_DOP['numberRays']/df_DOP['numberRays'].max()) #Intensity
        # ax[0,0].plot(df_DOP['radius'],df_DOP['intensity']/df_DOP['intensity'].max()) #Intensity
        ax[0,1].plot(df_DOP['radius'],df_DOP['DOPL']) #DOPL
        ax[1,0].plot(df_DOP['radius'],df_DOP['DOP45']) #DOP45
        ax[1,1].plot(df_DOP['radius'],df_DOP['DOPC']) #DOPC
        
        #Add rectangle
        max_y_Iplot = ax[0,0].get_ylim()[1]
        alpha=0.2
        ax[0,0].add_patch(plt.Rectangle((0,0),width=self.radius_source,height=max_y_Iplot,alpha=alpha))
        ax[0,1].add_patch(plt.Rectangle((0,0),width=self.radius_source,height=1.,alpha=alpha))
        ax[1,0].add_patch(plt.Rectangle((0,0),width=self.radius_source,height=1.,alpha=alpha))
        ax[1,1].add_patch(plt.Rectangle((0,0),width=self.radius_source,height=1.,alpha=alpha))
        
        #Set limits
        # ax[0,0].set_ylim(0,1.0)
        ax[0,1].set_ylim(-1.0,1.0)
        ax[1,0].set_ylim(-1.0,1.0)
        ax[1,1].set_ylim(-1.0,1.0)
        
        #Set titles
        ax[0,0].set_title('Number of Rays')
        ax[0,1].set_title('DOPL')
        ax[1,0].set_title('DOP45')
        ax[1,1].set_title('DOPC')
        fig.suptitle('DOPs vs Depth')
        pass
    
    def calculate_DOP_vs_radius(self,df):
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        
        #Calculate radius from source
        r = np.sqrt((df['x'])**2+(df['y'])**2)
        df.insert(15,'radius',r)
        
        #Calculate Stokes vs Radius
        bins = np.linspace(0,250/self.mus_theo,100)
        
        #Histogram of time vs intensity
        n_rays, radius = np.histogram(df['radius'], bins=bins)
        I, radius = np.histogram(df['radius'], weights=I, bins=bins)
        Q, radius = np.histogram(df['radius'], weights=Q, bins=bins)
        U, radius = np.histogram(df['radius'], weights=U, bins=bins)
        V, radius = np.histogram(df['radius'], weights=V, bins=bins)

        #Create Dataframe for the DOPs
        df_DOP = pd.DataFrame(data=np.transpose([I]),columns = ['intensity'])

        #Remove division by 0 by filter NaNs (Q,U and V are 0 anyway)
        I_df = df_DOP['intensity'] #Remove divide by zero exception with using df instead of array

        #Calculate DOPs
        DOP=np.sqrt(Q**2+U**2+V**2)/I_df
        DOPLT=np.sqrt(Q**2+U**2)/I_df
        DOPL=Q/I_df
        DOP45=U/I_df
        DOPC=V/I_df
        
        df_DOP.insert(len(df_DOP.columns),'numberRays',n_rays)
        df_DOP.insert(len(df_DOP.columns),'DOP',DOP)
        df_DOP.insert(len(df_DOP.columns),'DOPLT',DOPLT)
        df_DOP.insert(len(df_DOP.columns),'DOPL',DOPL)
        df_DOP.insert(len(df_DOP.columns),'DOP45',DOP45)
        df_DOP.insert(len(df_DOP.columns),'DOPC',DOPC)
        df_DOP[df_DOP.isna()] = 0

        df_DOP.insert(len(df_DOP.columns),'radius',radius[:-1])
        return df_DOP

    def map_DOP_top_detector(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df = self.df[filt_top_detector]
        
        #Plot datas
        fig, ax = plt.subplots(nrows=2,ncols=2)
        #return dataframe with dops colummns
        array_DOPs, x_bins, y_bins =  self.calculate_DOP_vs_xy(df)
        x, y = np.meshgrid(x_bins, y_bins)
        intensity = ax[0,0].pcolormesh(x, y, array_DOPs[0],cmap='PuBu')
        DOPL = ax[0,1].pcolormesh(x, y, array_DOPs[1],cmap='PuBu') #DOPL
        DOP45 = ax[1,0].pcolormesh(x, y, array_DOPs[2],cmap='PuBu') #DOP45
        DOPC = ax[1,1].pcolormesh(x, y, array_DOPs[3],cmap='PuBu') #DOPC
        
        #plot colormap
        fig.colorbar(intensity,ax=ax[0,0])
        fig.colorbar(DOPL,ax=ax[0,1])
        fig.colorbar(DOP45,ax=ax[1,0])
        fig.colorbar(DOPC,ax=ax[1,1])
        
        #Set titles
        ax[0,0].set_title('Intensity')
        ax[0,1].set_title('DOPL')
        ax[1,0].set_title('DOP45')
        ax[1,1].set_title('DOPC')
        fig.suptitle('DOPs vs Depth')
        pass
    
    def plot_irradiances(self):
        if not hasattr(self, 'musp_theo'): self.calculate_musp()
        if not hasattr(self, 'ke_rt'): self.calculate_ke_rt()
        if not hasattr(self, 'ke_theo'): self.calculate_ke_theo()
        
        #Irradiance raytracing
        def Irradiance_down(depth):
            df = self.down_irradiance_at_depth(depth)
            irradiance_down = sum(df['intensity']*np.abs(df['n']))
            return irradiance_down
        
        def Irradiance_up(depth):
            df = self.up_irradiance_at_depth(depth)
            irradiance_up = sum(df['intensity']*np.abs(df['n']))
            return irradiance_up
        
        depth = np.linspace(-0.001,25/self.musp_theo,50)
        irradiance_down_rt = np.array(list(map(Irradiance_down,depth)))
        irradiance_up_rt = np.array(list(map(Irradiance_up,depth)))
        
        #Irradiance TARTES
        if not hasattr(self, 'density_stereo'): self.calculate_density()
        if not hasattr(self, 'g_theo'): self.calculate_g()
        if not hasattr(self, 'SSA_theo'): self.calculate_SSA()
        irradiance_down_tartes, irradiance_up_tartes = tartes.irradiance_profiles(
            self.wlum*1E-6, depth, self.SSA_theo, density=self.density_theo,
            g0=self.g_theo,B0=self.B_theo,dir_frac=self.tartes_dir_frac,totflux=0.75)
        
        #Plot irradiance down
        plt.figure()
        plt.semilogy(depth,np.exp(-self.ke_rt*depth + self.b_rt),label='fit TARTES')
        plt.semilogy(depth,np.exp(-self.ke_tartes*depth + self.b_tartes),label='fit TARTES')
        plt.semilogy(depth,irradiance_down_tartes+irradiance_up_tartes, label='irradiance TARTES')
        plt.semilogy(depth,irradiance_down_rt, label='downwelling irradiance raytracing')
        plt.semilogy(depth,irradiance_up_rt, label='upwelling irradiance raytracing')
        plt.semilogy(depth,irradiance_down_rt+irradiance_up_rt, label='total irradiance raytracing')
        plt.xlabel('depth (m)')
        plt.ylabel('irradiance (W/m^2)')
        plt.legend()
        plt.title(self.name + '_numrays_' + str(self.numrays) + '_wlum_' + str(self.wlum) + '_alpha_' + str(round(self.alpha_rt,4)) + '_' + str(round(self.alpha_tartes,4)))
        plt.savefig(os.path.join(self.pathZMX,self.name + '_wlum_' + str(self.numrays) + '_wlum_' + str(self.wlum) + '_alpha_ ' + str(round(self.alpha_rt,4))+'_' + str(round(self.alpha_tartes,4))+'.png'),format='png')
        
    def properties(self):
        print('\n----------------------------------------------------\n')
        if hasattr(self, 'path_npy'): print('Simulation path npy file: ', self.path_npy)
        if hasattr(self, 'path_stereo_npy'): print('Simulation path npy file: ', self.path_stereo_npy)
        if hasattr(self, 'fileZMX'): print('Simulation path ZMX files: ', self.fileZMX)
        if hasattr(self, 'B_stereo'): print('Le B stéréologique: ' + str(round(self.B_stereo,4)))
        if hasattr(self, 'B_theo'): print('Le B théorique: ' + str(round(self.B_theo,4)))
        if hasattr(self, 'g_theo'): print('Le g théorique: ' + str(round(self.g_theo,4)))
        if hasattr(self, 'gG_theo'): print('Le gG théorique: ' + str(round(self.gG_theo,4)))
        if hasattr(self, 'g_rt'): print('Le g raytracing: ' + str(round(self.g_rt,4)))
        if hasattr(self, 'gG_rt'): print('Le gG raytracing: ' + str(round(self.gG_rt,4)))
        if hasattr(self, 'SSA_stereo'): print('La ù stéréologie: ' + str(round(self.SSA_stereo,4)))
        if hasattr(self, 'SSA_theo'): print('La SSA théorique: ' + str(round(self.SSA_theo,4)))
        if hasattr(self, 'density_theo'): print('La densité théorique: ' + str(round(self.density_theo,4)))
        if hasattr(self, 'density_stereo'): print('La densité stéréologique: ' + str(round(self.density_stereo,4)))
        if hasattr(self, 'physical_porosity_theo'): print('La porosité physique théorique: ' + str(round(self.physical_porosity_theo,5)))
        if hasattr(self, 'physical_porosity_stereo'): print('La porosité physique stéréologique: ' + str(round(self.physical_porosity_stereo,5)))
        if hasattr(self, 'optical_porosity_theo'): print('La porosité optique théorique: ' + str(round(self.optical_porosity_theo,5)))
        if hasattr(self, 'optical_porosity_stereo'): print('La porosité optique stéréologique: ' + str(round(self.optical_porosity_stereo,5)))
        if hasattr(self, 'mus_theo'): print('La mus théorique: ' + str(round(self.mus_theo,4)))
        if hasattr(self, 'mus_stereo'): print('La mus stéréologie: ' + str(round(self.mus_stereo,4)))
        if hasattr(self, 'musp_theo'): print('La musp théorique: ' + str(round(self.musp_theo,4)))
        if hasattr(self, 'musp_stereo'): print('La musp stéréologie: ' + str(round(self.musp_stereo,4)))
        if hasattr(self, 'mua_theo'): print('mua théorique: ' + str(round(self.mua_theo,6)))
        if hasattr(self, 'mua_tartes'): print('mua tartes: ' + str(round(self.mua_tartes,6)))
        if hasattr(self, 'alpha_theo'): print('alpha théorique: ' + str(round(self.alpha_theo,6)))
        if hasattr(self, 'alpha_stereo'): print('alpha stéréologique: ' + str(round(self.alpha_stereo,6)))
        if hasattr(self, 'alpha_rt'): print('alpha raytracing: ' + str(round(self.alpha_rt,6)))
        if hasattr(self, 'alpha_tartes'): print('alpha TARTES: ' + str(round(self.alpha_tartes,6)))
        if hasattr(self, 'ke_stereo'): print('ke stéréologique: ' + str(round(self.ke_stereo,6)))
        if hasattr(self, 'ke_theo'): print('ke théorique: ' + str(round(self.ke_theo,6)))
        if hasattr(self, 'ke_rt'): print('ke raytracing: ' + str(round(self.ke_rt,6)))
        if hasattr(self, 'ke_tartes'): print('ke TARTES: ' + str(round(self.ke_tartes,6)))
        if hasattr(self, 'MOPL_theo'): print('La MOPL stéréologique: ' + str(round(self.MOPL_theo,6)))
        if hasattr(self, 'MOPL_rt'): print('La MOPL raytracing: ' + str(round(self.MOPL_rt,6)))
        if hasattr(self, 'neff_stereo'): print('Le neff stéréologique: ' + str(round(self.neff_stereo,6)))
        if hasattr(self, 'neff_rt'): print('Le neff raytracing: ' + str(round(self.neff_rt,6)))
        if hasattr(self, 'Reflectance'): print('Reflectance raytracing: ' + str(round(self.Reflectance,6)) + ', NumRays: ' +str(self.numrays_Reflectance))
        if hasattr(self, 'Transmitance'): print('Transmitance raytracing: ' + str(round(self.Transmitance,6)) + ', NumRays: ' +str(self.numrays_Transmitance))
        if hasattr(self, 'Error'): print('Error raytracing: ' + str(round(self.Error,6)) + ', NumRays: ' +str(self.numrays_Error))
        if hasattr(self, 'Lost'): print('Lost raytracing: ' + str(round(self.Lost,6)) + ', NumRays: ' +str(self.numrays_Lost))
        if hasattr(self, 'Absorb'): print('Absorb raytracing: ' + str(round(self.Absorb,6)) + ', NumRays: ' +str(self.numrays_Absorb))
        
    def __del__(self):
        try:
            self.TheSystem.SaveAs(self.fileZMX)
            del self.zosapi
            self.zosapi = None
        except Exception:
            pass
    
plt.close('all')
properties=[]
if __name__ == '__main__':
    for wlum in [1.0]:
        print('Nombre de MB avalaible: ',psutil.virtual_memory()[1]/1E6)
        sim = simulation_MC('test1', 10000, 66E-6, 287E-6, 0.89, wlum, [1,1,0,-90], diffuse_light=False)
        sim.create_folder()
        sim.Initialize_Zemax()
        sim.Load_File()
        sim.shoot_rays()
        sim.Load_npyfile()
        sim.calculate_musp()
        sim.calculate_ke_theo()
        sim.calculate_ke_rt()
        sim.calculate_MOPL()
        sim.map_DOP_top_detector()
        sim.plot_DOP_radius_top_detector()
        sim.calculate_alpha()
        sim.calculate_mua()
        sim.properties()
        # sim.plot_irradiances()
        # sim.plot_time_reflectance()
        print('Nombre de MB avalaible: ',psutil.virtual_memory()[1]/1E6)
        # properties += [[sim.mua_theo,sim.alpha_rt,sim.MOPL_theo,sim.MOPL_rt,sim.Error]]
        # del sim