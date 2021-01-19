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
from dask.diagnostics import visualize
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

path_Sphere_Raytrace = 'Z:\Sintering\Sphere\Volume'
sys.path.insert(3,path_Sphere_Raytrace)
import Sphere_Raytrace

path_RaytraceDLL = 'C:\Zemax Files\ZOS-API\Libraries'
sys.path.insert(2,path_RaytraceDLL)
import PythonNET_ZRDLoader as init_Zemax

pathRaytraceDLL = 'Z:\Sintering\Glass Catalog'
sys.path.insert(1,os.path.dirname(os.path.realpath(pathRaytraceDLL)))
import Glass_Catalog

def profiler(func):
    def wrapper(*args,**kwargs):
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof,CacheProfiler() as cprof:
            func(*args,**kwargs)
        visualize([prof, rprof, cprof])
    return wrapper

class simulation_MC:
    path = os.path.join(os.sep, os.path.dirname(os.path.realpath(__file__)), '')
    max_segments = 4000
    
    def __init__(self,name,numrays,radius,Delta,g,wlum,pol,Random_pol=False,diffuse_light=False,source_radius=None,sphere_material='MY_ICE.ZTG',p_material=917):
        self.inputs = [name,numrays,radius,Delta,g,wlum,pol,Random_pol,diffuse_light]
        self.name = name
        self.numrays = numrays
        self.wlum = wlum
        self.radius = radius
        self.p_material = p_material
        
        #Same radius as in laboratory
        if source_radius==None:        
            self.source_radius = 10*self.radius
        else:
            self.source_radius = source_radius 
            
        self.Delta = Delta
        self.calculate_density()
        self.calculate_SSA()
        self.calculate_mus()
        self.B_theo = 1.2521
        self.calculate_porosity()

        mat = Glass_Catalog.Material(sphere_material)
        self.index_real,self.index_imag = mat.get_refractive_index(self.wlum)
        self.gamma = 4*np.pi*self.index_imag/(self.wlum*1E-6)

        # self.index_real,self.index_imag = tartes.refice2016(self.wlum*1E-6)
        self.calculate_mua()
        self.calculate_neff()
        self.g_theo = g
        self.gG_theo = self.g_theo*2-1
        self.jx,self.jy,self.phase_x,self.phase_y = np.array(pol)
        self.pathDatas = os.path.join(self.path,'Simulations',self.name)
        self.path_plot = os.path.join(self.pathDatas,'Results_plots')
        self.Random_pol=Random_pol
        self.diffuse_light = diffuse_light
        if self.diffuse_light == True:
            self.tartes_dir_frac = 0
            self.diffuse_str = 'diffuse'
        else:
            self.tartes_dir_frac = 1
            self.diffuse_str = 'not_diffuse'
            
        self.properties_string = '_'.join([name,str(numrays),str(radius),str(Delta),str(self.B_theo),str(g),str(tuple(pol)),self.diffuse_str])
        self.properties_string_plot = '_'.join([self.properties_string,str(self.wlum)])
        self.name_ZRD = self.properties_string + ".ZRD"
        
        self.path_parquet = os.path.join(self.pathDatas,'_'.join([self.properties_string,str(self.wlum)]) + "_.parquet")
        self.path_metadata = os.path.join(self.pathDatas,'_'.join([self.properties_string,'metadata'])+ ".npy")
        self.path_ZRD = os.path.join(self.pathDatas,self.name_ZRD)
        path_ZMX = os.path.dirname(os.path.dirname(self.pathDatas))
        self.fileZMX = os.path.join(path_ZMX, 'test_MC_msp.zmx')
        
        self.add_properties_to_dict('radius',self.radius)
        self.add_properties_to_dict('Delta',self.Delta)
        self.add_properties_to_dict('B_theo',self.B_theo)
        self.add_properties_to_dict('Pol',pol)
        self.add_properties_to_dict('Diffuse_light',self.diffuse_light)
        self.add_properties_to_dict('neff_theo',self.neff_theo)
        self.add_properties_to_dict('numrays',self.numrays)
        self.add_properties_to_dict('wlum',self.wlum)
        self.add_properties_to_dict('Random_pol',self.Random_pol)
        
        #Printing starting time
        t = time.localtime()
        self.date = str(t.tm_year)+"/"+str(t.tm_mon)+"/"+str(t.tm_mday)
        self.ctime = str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)
        print('Date : ',self.date,', Time : ',self.ctime)
        
        self.create_directory()
        
    def add_properties_to_dict(self,key,value):
        if not hasattr(self,'dict_properties'):
            self.dict_properties = {key:value}
        else:
            self.dict_properties[key] = value
        
    def create_directory(self):
        if not os.path.exists(self.pathDatas):
            os.makedirs(self.pathDatas)
            os.makedirs(self.path_plot)
            print("Le répertoire " + str(self.name) +  " a été créé")
        else:
            print("Le répertoire " + str(self.name) +  " existe déjà")
    
    def find_source_object(self):
        Source_obj = np.where(self.array_objects() == np.array(['Source Ellipse']))[0] + 1
        return Source_obj
    
    def find_detector_object(self):
        Detector_obj = np.where(self.array_objects() == np.array(['Detector Rectangle']))[0] + 1
        return Detector_obj

    def update_metadata(self):
        #Write metadata
        np.save(self.path_metadata,self.array_objects())
        
    def array_objects(self):
        #Check for metadata
        if os.path.exists(self.path_metadata):
            array_obj = np.load(self.path_metadata)
        #Check for Zemax open
        elif hasattr(self, 'zosapi'):
            list_obj=[]
            for i in range(1,self.TheNCE.NumberOfObjects+1):
                Object = self.TheNCE.GetObjectAt(i)
                ObjectType = Object.TypeName
                ObjectMaterial = Object.Material 
                list_obj += [[ObjectType,ObjectMaterial]]
                array_obj = np.array(list_obj)
        #Else return error
        else:
            print('There\'s no metadata file to get array object from and ZosAPI is not open.\
                  Please initialize Zemax')
            sys.exit()
        return array_obj
    
    def Initialize_Zemax(self):
        self.zosapi = init_Zemax.PythonStandaloneApplication()
        self.BatchRayTrace = self.zosapi.BatchRayTrace 
        self.TheSystem = self.zosapi.TheSystem
        self.TheApplication = self.zosapi.TheApplication
        self.TheNCE = self.TheSystem.NCE
        
        self.ZOSAPI = self.zosapi.ZOSAPI
        self.ZOSAPI_NCE = self.ZOSAPI.Editors.NCE
        
    def Load_File(self):
        self.Initialize_Zemax()
        start = time.time()
        self.TheSystem.LoadFile(self.fileZMX,False)
        self.TheNCE = self.TheSystem.NCE
        
        #Change wl in um
        self.TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = 1.0
        
        #Change Polarization
        Source_obj = self.find_source_object()[0]
        Source = self.TheNCE.GetObjectAt(Source_obj)
        
        Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par6).DoubleValue = self.source_radius #X Half width
        Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par7).DoubleValue = self.source_radius #Y Half width
        
        Source.SourcesData.RandomPolarization = self.Random_pol
        Source.SourcesData.Jx = self.jx
        Source.SourcesData.Jy = self.jy
        Source.SourcesData.XPhase = self.phase_x
        Source.SourcesData.YPhase = self.phase_y

        Rectangular_obj = self.TheNCE.GetObjectAt(1)
        Rectangular_obj_physdata = Rectangular_obj.VolumePhysicsData
        Rectangular_obj_physdata.Model = self.ZOSAPI_NCE.VolumePhysicsModelType.DLLDefinedScattering
        self.calculate_mua()
        
        # Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.DLL = 'Henyey-Greenstein-bulk.dll'
        #Change MSP
        Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.DLL = 'MSP_v4p2.dll'
        # Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.DLL = 'MSP_v4.dll'
        
        #MeanPath
        Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.MeanPath = 1/(self.mus_theo+self.mua_theo)
        #Transmission
        Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.SetParameterValue(0,1-self.mua_theo/(self.mus_theo+self.mua_theo))
        # #Radius (um)
        Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.SetParameterValue(1,self.radius*1E6)
        # #ice index
        Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.SetParameterValue(2,self.index_real)
        
        # g
        # Rectangular_obj_physdata.ModelSettings._S_DLLDefinedScattering.SetParameterValue(1,self.g_theo)
        
        self.TheSystem.SaveAs(self.fileZMX)
        end = time.time()
        print('Fichier loader en ',end-start)

        self.update_metadata()
        pass
    
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
        
        #Change path parquet to the real wavelength used during the raytrace
        path_parquet_list = self.path_parquet.split('_')
        path_parquet_list[-2] = '1.0'
        path_parquet = '_'.join(path_parquet_list)
        Sphere_Raytrace.Shoot(self,'',self.numrays,path_parquet,self.name_ZRD)
        pass    
    
    def Load_parquetfile(self): 
        #Try loading the ray file
        #Check if file at the specified wavlength exist..
        path_parquet_list = self.path_parquet.split('_')
        path_parquet_list[-2] = '1.0'
        path_parquet = '_'.join(path_parquet_list)
        #If not creates it
        if os.path.exists(self.path_parquet):
            self.df = Sphere_Raytrace.Load_parquet(self.path_parquet)
        #Check if file at 1.0 um has already been done
        elif os.path.exists(path_parquet):
            #Load File at another wavelength
            self.df = Sphere_Raytrace.Load_parquet(path_parquet)
            #Change pathLength
            filt = self.df['segmentLevel']==0
            self.df['pathLength'] = (~filt)*np.sqrt((((self.df[['x','y','z']].diff())**2).sum(1)))
            #Change intensity
            self.change_path_intensity()
            #Save changes in new path
            self.df.to_parquet(self.path_parquet)
            self.df = Sphere_Raytrace.Load_parquet(self.path_parquet)
        else:
            print('The raytrace parquet file was not loaded, Please run raytrace')
        
    def AOP(self):
        df = self.df.groupby(self.df.index).agg({'hitObj':'last','segmentLevel':'last','intensity':'last'}).compute()
        
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

        self.add_properties_to_dict('Reflectance',self.Reflectance)
        self.add_properties_to_dict('numrays_Reflectance',self.numrays_Reflectance)
        self.add_properties_to_dict('Transmitance',self.Transmitance)
        self.add_properties_to_dict('numrays_Transmitance',self.numrays_Transmitance)
        self.add_properties_to_dict('Error',self.Error)
        self.add_properties_to_dict('numrays_Error',self.numrays_Error)
        self.add_properties_to_dict('Lost',self.Lost)
        self.add_properties_to_dict('numrays_Lost',self.numrays_Lost)
        pass
    
    def change_path_intensity(self):
        if not hasattr(self, 'optical_porosity_theo'): self.calculate_porosity()
        
        #Calculate the intensity for each segments
        detector_obj_1,detector_obj_2 = self.find_detector_object()
        filt_ice = ((self.df['segmentLevel'] != 0) & (self.df['segmentLevel'].diff(-1) == -1))
        
        #Calculate new intensity
        self.gamma = 4*np.pi*self.index_imag/(self.wlum*1E-6)
        I_0 = 1./self.numrays
     
        #Create a ponderation for segment in material with optical density (absorbing media)
        self.df = self.df.assign(pathLengthIce = (filt_ice*(1-self.optical_porosity_theo)*(self.df.pathLength)))
        
        #Change pathLength for segment to cumulative pathLength
        self.df['pathLengthIce'] = self.df.groupby('numray')['pathLengthIce'].cumsum()
        
        #Overwrite the intensity
        self.df['intensity'] = I_0*np.exp(-self.gamma*self.df['pathLengthIce'])
        # self.df = self.df.persist()
        pass

    def calculate_porosity(self):
        #====================Porosité physique théorique====================
        if not hasattr(self, 'density_theo'): self.calculate_density()
            
        porosity = 1-self.density_theo/self.p_material
        self.physical_porosity_theo = porosity
        
        #====================Porosité optique théorique====================
        self.optical_porosity_theo = porosity/(porosity+self.B_theo*(1-porosity))
        self.add_properties_to_dict('physical_porosity_theo',self.physical_porosity_theo)
        self.add_properties_to_dict('optical_porosity_theo',self.optical_porosity_theo)

    def calculate_neff(self):
        porosity = self.physical_porosity_theo
        self.neff_theo = (porosity + self.index_real*self.B_theo*(1-porosity))/(porosity + self.B_theo*(1-porosity))
        pass        
        
    def calculate_mua(self):
        #Calculate mua theorique
        self.gamma = 4*np.pi*self.index_imag/(self.wlum*1E-6)
        self.mua_theo = self.B_theo*self.gamma*(1-self.physical_porosity_theo)
        #Calculate mua with Tartes
        if hasattr(self,'ke_tartes') == False: return
        self.mua_tartes = -self.ke_tartes*np.log(self.alpha_tartes)/4
        self.add_properties_to_dict('mua_theo',self.mua_theo)
        self.add_properties_to_dict('mua_tartes',self.mua_tartes)
        
    def calculate_musp(self):
        self.musp_theo = self.mus_theo*(1-self.g_theo)
        self.add_properties_to_dict('musp_theo',self.musp_theo)
        pass
    
    def calculate_alpha(self):
        if not hasattr(self, 'Reflectance'): self.AOP()
        
        self.alpha_rt = self.Reflectance
        self.alpha_tartes = float(tartes.albedo(self.wlum*1E-6,self.SSA_theo,self.density_theo,g0=self.g_theo,B0=self.B_theo,dir_frac=self.tartes_dir_frac))
        self.add_properties_to_dict('alpha_rt',self.alpha_rt)
        self.add_properties_to_dict('alpha_tartes',self.alpha_tartes)
    
    def calculate_g_theo(self):
        self.g_theo = 0.89
        self.gG_theo = self.g_theo*2-1
        self.add_properties_to_dict('g_theo',self.g_theo)
        self.add_properties_to_dict('gG_theo',self.gG_theo)
    
    def calculate_g_rt(self):
        if not hasattr(self, 'ke_rt'): self.calculate_ke_rt()
        if not hasattr(self, 'alpha_rt'): self.calculate_alpha()
        
        #g calculé avec ke et alpha raytracing
        #Équation Quentin Libois 3.3 thèse
        self.g_rt = 1+8*self.ke_rt/(3*self.density_theo*np.log(self.alpha_rt)*self.SSA_theo)
        self.gG_rt = self.g_rt*2-1
        self.add_properties_to_dict('g_rt',self.g_rt)
        self.add_properties_to_dict('gG_rt',self.gG_rt)
        
    def ke_raytracing(self,depths_fit,intensity):
        [a,b], pcov=scipy.optimize.curve_fit(lambda x,a,b: a*x+b, depths_fit, np.log(intensity))
        return -a, b
    
    def calculate_ke_rt(self):
        #ke raytracing
        def linear_fit(depth,a,b):
            intensity = a*depth+b
            return intensity
        
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        
        depths = np.linspace(2/self.musp_theo,25/self.musp_theo,10)
        Irradiance_rt = self.df.map_partitions(self.Irradiance,depths,meta=list)
        Irradiance_rt = Irradiance_rt.compute().sum(axis=0)
        self.ke_rt, self.b_rt = self.ke_raytracing(depths,Irradiance_rt)
        
        self.add_properties_to_dict('ke_rt',self.ke_rt)
        self.add_properties_to_dict('b_rt',self.b_rt)
    
    def calculate_ke_theo(self):
        if not hasattr(self, 'density_theo'): self.calculate_density()
        if not hasattr(self, 'g_theo'): self.calculate_g_theo()
        if not hasattr(self, 'SSA_theo'): self.calculate_SSA()
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        
        depths_fit = np.linspace(2/self.musp_theo,25/self.musp_theo,10)
        
        #ke TARTES
        down_irr_profile, up_irr_profile = tartes.irradiance_profiles(
            self.wlum*1E-6,depths_fit,self.SSA_theo,self.density_theo,g0=self.g_theo,B0=self.B_theo,dir_frac=self.tartes_dir_frac,totflux=0.75)
        
        # plt.plot(depths_fit,np.exp(-self.ke_rt*depths_fit + b),label='fit raytracing')
        self.ke_tartes, self.b_tartes = self.ke_raytracing(depths_fit, down_irr_profile+up_irr_profile)

        #ke théorique
        #Imaginay part of the indice of refraction
        self.gamma = 4*np.pi*self.index_imag/(self.wlum*1E-6)
        self.ke_theo = self.density_theo*np.sqrt(3*self.B_theo*self.gamma/(4*self.p_material)*self.SSA_theo*(1-self.gG_theo))
        self.add_properties_to_dict('ke_tartes',self.ke_tartes)
        self.add_properties_to_dict('b_tartes',self.b_tartes)
        self.add_properties_to_dict('ke_theo',self.ke_theo)
        pass
    
    def calculate_density(self):
        #====================Densité physique théorique====================
        if np.sqrt(2)*self.radius > self.Delta/2:
            DensityRatio=0
        else:
            volsphere=4*np.pi*self.radius**3/3
            volcube=self.Delta**3
            DensityRatio = volsphere*4/volcube
        self.density_theo = DensityRatio*self.p_material
        self.add_properties_to_dict('density_theo',self.density_theo)
    
    def calculate_SSA(self):
        #====================SSA théorique====================
        vol_sphere = 4*np.pi*self.radius**3/3
        air_sphere = 4*np.pi*self.radius**2
        self.SSA_theo = air_sphere/(vol_sphere*self.p_material)
        self.add_properties_to_dict('SSA_theo',self.SSA_theo)
        
    def calculate_mus(self):
        self.mus_theo = self.density_theo*self.SSA_theo/2
        self.add_properties_to_dict('mus_theo',self.mus_theo)
    
    def calculate_MOPL(self):
        #Shoot rays for SSA
        if not hasattr(self, 'df'): self.Load_parquetfile()
        if not hasattr(self, 'musp_theo'): self.calculate_musp()
        
        #Approximation de Paterson
        # z_o = self.musp_theo**(-1)
        # D = (3*(self.mua_theo+self.musp_theo))**(-1)
        # self.MOPL_theo = self.neff_theo*z_o/(2*np.sqrt(self.mua_theo*D))
        
        #Keep rays that touch 
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df_top = self.df[filt_top_detector]
        df_filt = self.df.loc[df_top.index]
        
        df_filt = df_filt[~(df_filt['hitObj'] == 2)]
        df_filt['OPL'] = df_filt['pathLength']*self.neff_theo
        df_OPL = df_filt.groupby(df_filt.index).agg({'OPL':sum,'intensity':'last'})
        self.MOPL_rt = np.average(df_OPL['OPL'],weights=df_OPL['intensity'])
        self.add_properties_to_dict('MOPL_rt',self.MOPL_rt)

    def plot_time_reflectance(self):
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        if not hasattr(self, 'mua_rt'): self.calculate_mua()
        
        #Raytracing
        #Add a line of time for each segment
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df_top = self.df[filt_top_detector]
        df_filt = self.df.loc[df_top.index]
        
        df_filt = df_filt[~((df_filt['segmentLevel']==1)|(df_filt['hitObj']==3))]
        v_medium = scipy.constants.c/self.index_real
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
        c = scipy.constants.c/self.neff_theo
        R_theo = (4*np.pi*D*c)**(-1/2)*z_o*t_theo**(-(3/2))*np.exp(-self.mua_theo*c*t_theo)*np.exp(-z_o**2/(4*D*c*t_theo))
        
        #Normalize
        R_rt = R_rt/max(R_rt)
        R_theo = R_theo/max(R_theo)
        
        #Change the offset on time theo so the 2 curves overlap (theo and rt)
        t_rt_offset = t_rt[(list(R_rt)).index(max(R_rt))]
        t_theo_offset = t_theo[(list(R_theo)).index(max(R_theo))]
        t_diff = t_rt_offset-t_theo_offset
        t_theo = t_theo+t_diff
        # np.average(t_theo*scipy.constants.c/self.neff_theo,weights=R_theo)
        
        #Plot reflectance vs time
        plt.figure()
        plt.plot(t_rt,R_rt)
        plt.plot(t_theo,R_theo)
        #plt.xlim(1E-16,1E-12)
        plt.show()
        
        #Save Datas to npy file
        path_npy = os.path.join(self.path_plot,'DOPs_time_reflectance.npy')
        self.create_npy(path_npy,t_rt=t_rt,R_rt=R_rt)
        pass
    
    def calculate_Stokes_of_rays(self,df):
        if len(df.index) == 0:
            #Return empty sequence
            return np.zeros(4)
        
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
    
    def calculate_Stokes_xy(self,df):
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        
        #Calculate Stokes vs Radius
        XY_detector = 100/self.mus_theo
        bins = (np.linspace(-XY_detector,XY_detector,100),np.linspace(-XY_detector,XY_detector,100))
        
        #Histogram of x,y vs intensity
        I, x_bins, y_bins = np.histogram2d(df['x'], df['y'], weights=I, bins=bins)
        Q, x_bins, y_bins = np.histogram2d(df['x'], df['y'], weights=Q, bins=bins)
        U, x_bins, y_bins = np.histogram2d(df['x'], df['y'], weights=U, bins=bins)
        V, x_bins, y_bins = np.histogram2d(df['x'], df['y'], weights=V, bins=bins)
        
        return np.array([I,Q,U,V]), x_bins, y_bins
    
    def calculate_DOP_vs_xy(self,df):
        [I,Q,U,V], x_bins, y_bins = self.calculate_Stokes_xy(df)
        
        #Calculate DOPs
        #With handling division by zero
        # DOP=np.true_divide(np.sqrt(Q**2+U**2+V**2), I, out=np.zeros_like(np.sqrt(Q**2+U**2+V**2)), where=I!=0)
        # DOPLT=np.true_divide(np.sqrt(Q**2+U**2), I, out=np.zeros_like(np.sqrt(Q**2+U**2)), where=I!=0)
        DOPL=np.true_divide(np.sqrt(Q**2), I, out=np.zeros_like(np.sqrt(Q**2)), where=I!=0)
        DOP45=np.true_divide(np.sqrt(U**2), I, out=np.zeros_like(np.sqrt(U**2)), where=I!=0)
        DOPC=np.true_divide(np.sqrt(V**2), I, out=np.zeros_like(np.sqrt(V**2)), where=I!=0)
        
        return np.array([I,DOPL,DOP45,DOPC]),x_bins,y_bins
    
    def plot_MOPL_radius_reflectance(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df_top = self.df[filt_top_detector]
        df_filt = self.df.loc[df_top.index]
        
        #Change pathLength
        filt = df_filt['segmentLevel']==0
        df_filt['pathLength'] = (~filt)*np.sqrt((((df_filt[['x','y','z']].diff())**2).sum(1)))

        #Calculate OPL for each ray
        df_filt['OPL'] = df_filt['pathLength']*self.neff_theo
        df_OPL = df_filt.groupby(df_filt.index).agg({'OPL':sum,'intensity':'last','x':'last','y':'last'}).compute()
        df_OPL['radius'] = np.sqrt((df_OPL['x'])**2+(df_OPL['y'])**2)
        
        #Calculate MOPL
        bins = np.linspace(0,250/self.mus_theo,100)
        nrays, radius = np.histogram(df_OPL['radius'], weights=df_OPL['intensity'], bins=bins)
        OPL_bins, radius = np.histogram(df_OPL['radius'], weights=df_OPL['intensity']*df_OPL['OPL'], bins=bins)

        #Calculate MOPL
        plt.figure()
        
        MOPL = OPL_bins/nrays
        radius = radius[:-1]
        plt.plot(radius,MOPL)
        plt.title('MOPL vs radius')
        plt.ylabel('MOPL (m)')
        plt.xlabel('Radius (m)')
        
        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_MOPL_radius_reflectance.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_MOPL_radius_reflectance.npy')
        self.create_npy(path_npy,df_OPL=df_OPL)
        
    def plot_lair_radius_reflectance(self):
        if not hasattr(self, 'optical_porosity_theo'): self.calculate_porosity()
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df_top = self.df[filt_top_detector]
        df_filt = self.df.loc[df_top.index]
        
        #Change pathLength
        filt = df_filt['segmentLevel']==0
        df_filt['pathLength'] = (~filt)*np.sqrt((((df_filt[['x','y','z']].diff())**2).sum(1)))

        #Calculate OPL for each ray
        df_filt = df_filt[~(df_filt['hitObj'] == 2)]
        df_filt['OPL'] = df_filt['pathLength']*self.neff_theo
        df_lair = df_filt.groupby(df_filt.index).agg({'OPL':sum,'intensity':'last','x':'last','y':'last'}).compute()
        
        #Calculate lair
        df_lair['lair'] = df_lair['OPL']/(1+self.index_real*(1-self.optical_porosity_theo)/self.optical_porosity_theo)
        
        #Calculate Radius for each ray
        df_lair['radius'] = np.sqrt((df_lair['x'])**2+(df_lair['y'])**2)
        
        #Calculate MOPL
        bins = np.linspace(0,250/self.mus_theo,100)
        nrays, radius = np.histogram(df_lair['radius'], weights=df_lair['intensity'], bins=bins)
        lair_bins, radius = np.histogram(df_lair['radius'], weights=df_lair['intensity']*df_lair['lair'], bins=bins)

        #Calculate MOPL
        plt.figure()
        
        lair = lair_bins/nrays
        radius = radius[:-1]
        plt.plot(radius,lair)
        plt.title('lair vs radius')
        plt.ylabel('lair (m)')
        plt.xlabel('Radius (m)')
        
        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_lair_radius_reflectance.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_lair_radius_reflectance.npy')
        self.create_npy(path_npy,df_lair=df_lair)
        return
    
    def plot_DOP_transmitance(self):
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        #Plot datas
        fig, ax = plt.subplots(nrows=2,ncols=2)
        depths = np.linspace(0,25/self.musp_theo,100)
        # Stokes_data = self.DOPs_at_depths(self.df.compute(),depths)
        Stokes = self.df.map_partitions(self.DOPs_at_depths,depths,meta=list).compute()
        
        # Mean all the partitions results together
        [I,Q,U,V] = Stokes.swapaxes(0,1).mean(axis=1).transpose()
        
        DOPs = self.calculate_DOP(I, Q, U, V)
        ax[0,0].plot(depths,DOPs[0]) #DOP
        ax[0,1].plot(depths,DOPs[2]) #DOPL
        ax[1,0].plot(depths,DOPs[3]) #DOP45
        ax[1,1].plot(depths,DOPs[4]) #DOPC
        
        #Set limits
        ax[0,0].set_ylim(0,1.0)
        ax[0,1].set_ylim(0,1.0)
        ax[1,0].set_ylim(0,1.0)
        ax[1,1].set_ylim(0,1.0)
        
        #Set titles
        ax[0,0].set_title('DOP')
        ax[0,1].set_title('DOPL')
        ax[1,0].set_title('DOP45')
        ax[1,1].set_title('DOPC')
        fig.suptitle('DOPs vs Depth')
        
        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_plot_DOP_transmitance.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_plot_DOP_transmitance.npy')
        self.create_npy(path_npy,depths=depths,DOPs=DOPs)
    
    def Transmitance_at_depth(self,df,depth):
        df_filt = df[df['z'] >= depth]
        df = df_filt[~df_filt.index.duplicated(keep='first')]
        return df
    
    def DOPs_at_depths(self,df,depths):
        """depths: list
        df: dataframe"""
        
        def Stokes(self,df,depth):
            df = self.Transmitance_at_depth(df,depth)
            [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
            I=np.mean(I)
            Q=np.mean(Q)
            U=np.mean(U)
            V=np.mean(V)
            return np.array([I,Q,U,V])
        
        Stokes_lam = lambda depth: Stokes(self,df,depth)
        Stokes_datas=np.array(list(map(Stokes_lam,depths)))
        return np.array([Stokes_datas])
    
    def calculate_DOP(self,I,Q,U,V):
        DOP=np.sqrt(Q**2+U**2+V**2)/I
        DOPLT=np.sqrt(Q**2+U**2)/I
        DOPL=np.sqrt(Q**2)/I
        DOP45=np.sqrt(U**2)/I
        DOPC=np.sqrt(V**2)/I
        return np.array([DOP, DOPLT, DOPL, DOP45, DOPC])
    
    def plot_DOP_radius_reflectance(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df = self.df[filt_top_detector].compute()
        
        #Plot datas
        fig, ax = plt.subplots(nrows=2,ncols=2)
        #return dataframe with dops colummns
        df_DOP = self.calculate_DOP_vs_radius(df)
        ax[0,0].plot(df_DOP['radius'],df_DOP['numberRays']/df_DOP['numberRays'].max()) #Intensity
        ax[0,1].plot(df_DOP['radius'],df_DOP['DOPL']) #DOPL
        ax[1,0].plot(df_DOP['radius'],df_DOP['DOP45']) #DOP45
        ax[1,1].plot(df_DOP['radius'],df_DOP['DOPC']) #DOPC
        
        #Add rectangle
        max_y_Iplot = ax[0,0].get_ylim()[1]
        alpha=0.2
        ax[0,0].add_patch(plt.Rectangle((0,0),width=self.source_radius,height=max_y_Iplot,alpha=alpha))
        ax[0,1].add_patch(plt.Rectangle((0,0),width=self.source_radius,height=1.,alpha=alpha))
        ax[1,0].add_patch(plt.Rectangle((0,0),width=self.source_radius,height=1.,alpha=alpha))
        ax[1,1].add_patch(plt.Rectangle((0,0),width=self.source_radius,height=1.,alpha=alpha))
        
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
        fig.suptitle('DOPs vs Radius Reflectance')
        
        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_DOPs_radius_reflectance.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_DOPs_radius_reflectance.npy')
        self.create_npy(path_npy,df_DOP=df_DOP)
        pass
    
    def calculate_DOP_vs_radius(self,df):
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        
        #Calculate radius from source
        r = np.sqrt((df['x'])**2+(df['y'])**2)
        df['radius'] = r
        
        #Calculate Stokes vs Radius
        bins = np.linspace(0,0.1,100)
        
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
        DOPL=np.sqrt(Q**2)/I_df
        DOP45=np.sqrt(U**2)/I_df
        DOPC=np.sqrt(V**2)/I_df
        
        df_DOP.insert(len(df_DOP.columns),'numberRays',n_rays)
        df_DOP.insert(len(df_DOP.columns),'DOP',DOP)
        df_DOP.insert(len(df_DOP.columns),'DOPLT',DOPLT)
        df_DOP.insert(len(df_DOP.columns),'DOPL',DOPL)
        df_DOP.insert(len(df_DOP.columns),'DOP45',DOP45)
        df_DOP.insert(len(df_DOP.columns),'DOPC',DOPC)
        df_DOP[df_DOP.isna()] = 0

        df_DOP.insert(len(df_DOP.columns),'radius',radius[:-1])
        return df_DOP
    
    def map_stokes_reflectance(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df = self.df[filt_top_detector].compute()
        
        #Plot datas
        fig, ax = plt.subplots(nrows=2,ncols=2)
        #return dataframe with dops colummns
        array_Stokes, x_bins, y_bins =  self.calculate_Stokes_xy(df)
        x, y = np.meshgrid(x_bins, y_bins)
        intensity = ax[0,0].pcolormesh(x, y, array_Stokes[0])
        DOPL = ax[0,1].pcolormesh(x, y, array_Stokes[1]) #DOPL
        DOP45 = ax[1,0].pcolormesh(x, y, array_Stokes[2]) #DOP45
        DOPC = ax[1,1].pcolormesh(x, y, array_Stokes[3]) #DOPC
        
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
        fig.suptitle('Map Stokes Reflectance')
        
        #Save data to npy file
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_map_Stokes_reflectance.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_map_Stokes_reflectance.npy')
        self.create_npy(path_npy,x_bins=x,y_bins=y,array_Stokes_bins=array_Stokes,x=df['x'],y=df['y'],array_Stokes=[I,Q,U,V])
        pass
    
    def map_DOP_reflectance(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df = self.df[filt_top_detector].compute()
        
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
        fig.suptitle('Map DOP Reflectance')
        
        #Save Datas to npy file
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_map_DOP_reflectance.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_map_DOP_reflectance.npy')
        self.create_npy(path_npy,x_bins=x,y_bins=y,array_DOPs_bins=array_DOPs,x=df['x'],y=df['y'],array_Stokes=[I,Q,U,V])
        pass
    
    def Irradiance(self,df,depths):
        def Irradiance_at_depth(df,depth):    
            df = df.query('((z<= {} & z.shift() >= {})|(z>= {} & z.shift() <= {}))&(segmentLevel!=0)'.format(depth,depth,depth,depth))
            irradiance = df['intensity'].mul(df['L'].abs()).sum()
            return irradiance
        Irradiance = lambda depth: Irradiance_at_depth(df,depth)
        Irradiance_rt=np.array(list(map(Irradiance,depths)))
        return np.array([Irradiance_rt])

    def Irradiance_up(self,df,depths):
        def Irradiance_up_at_depth(df,depth):    
            df = df.query('((z<= {} & z.shift() >= {})&(segmentLevel!=0))'.format(depth,depth))
            irradiance_up = df['intensity'].mul(df['L'].abs()).sum()
            return irradiance_up
        Irradiance_up = lambda depth: Irradiance_up_at_depth(df,depth)
        Irradiance_up_rt=np.array(list(map(Irradiance_up,depths)))
        return np.array([Irradiance_up_rt])
    

    def Irradiance_down(self,df,depths):
        def Irradiance_down_at_depth(df,depth):
            df = df.query('(z>= {} & z.shift() <= {})&(segmentLevel!=0)'.format(depth,depth))
            irradiance_down = df['intensity'].mul(df['L'].abs()).sum()
            return irradiance_down
        Irradiance_down = lambda depth: Irradiance_down_at_depth(df,depth)
        Irradiance_down_rt=np.array(list(map(Irradiance_down,depths)))
        return np.array([Irradiance_down_rt])
    
    def plot_irradiances(self):
        if not hasattr(self, 'musp_theo'): self.calculate_musp()
        if not hasattr(self, 'ke_rt'): self.calculate_ke_rt()
        if not hasattr(self, 'ke_tartes'): self.calculate_ke_theo()
        
        #Change pathLength
        filt = self.df['segmentLevel']==0
        self.df['pathLength'] = (~filt)*np.sqrt((((self.df[['x','y','z']].diff())**2).sum(1)))
        
        #Irradiance raytracing
        depth = np.linspace(-0.001,0.1,50)
        Irradiance_rt = self.df.map_partitions(self.Irradiance,depth,meta=list).compute().sum(axis=0)
        Irradiance_up_rt = self.df.map_partitions(self.Irradiance_up,depth,meta=list).compute().sum(axis=0)
        Irradiance_down_rt = self.df.map_partitions(self.Irradiance_down,depth,meta=list).compute().sum(axis=0)
        
        #Irradiance TARTES
        if not hasattr(self, 'density_theo'): self.calculate_density()
        if not hasattr(self, 'g_theo'): self.calculate_g()
        if not hasattr(self, 'SSA_theo'): self.calculate_SSA()
        irradiance_up_tartes, irradiance_down_tartes = tartes.irradiance_profiles(
            self.wlum*1E-6, depth, self.SSA_theo, density=self.density_theo,
            g0=self.g_theo,B0=self.B_theo,dir_frac=self.tartes_dir_frac,totflux=0.75)
        
        #Plot irradiance down
        plt.figure()
        plt.semilogy(depth,np.exp(-self.ke_rt*depth + self.b_rt),label='fit raytracing')
        plt.semilogy(depth,np.exp(-self.ke_tartes*depth + self.b_tartes),label='fit TARTES')
        plt.semilogy(depth,irradiance_down_tartes+irradiance_up_tartes, label='irradiance TARTES')
        plt.semilogy(depth,irradiance_down_tartes, label='irradiance down TARTES')
        plt.semilogy(depth,irradiance_up_tartes, label='irradiance up TARTES')
        plt.semilogy(depth,Irradiance_down_rt, label='downwelling irradiance raytracing')
        plt.semilogy(depth,Irradiance_up_rt, label='upwelling irradiance raytracing')
        plt.semilogy(depth,Irradiance_rt, label='total irradiance raytracing')
        plt.xlabel('depth (m)')
        plt.ylabel('irradiance (W/m^2)')
        plt.legend()
        
        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_plot_irradiances.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_plot_irradiances.npy')
        self.create_npy(path_npy,depth=depth,
                        irradiance_down_tartes=irradiance_down_tartes,
                        irradiance_up_tartes=irradiance_up_tartes,
                        Irradiance_rt=Irradiance_rt,
                        Irradiance_down_rt=Irradiance_down_rt,
                        Irradiance_up_rt=Irradiance_up_rt)
 
    def time(self,text):
        t = time.localtime()
        self.date = str(t.tm_year)+"/"+str(t.tm_mon)+"/"+str(t.tm_mday)
        self.ctime = str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)
        print('Date : ',self.date,', Time : ',self.ctime,' ',text)
        
    def export_properties(self):
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_properties.npy')
        np.save(path_npy,self.dict_properties)
        
    def create_npy(self,path,**kwargs):
        #Create File for simulation
        np.save(path,kwargs)
        pass
    
    def properties(self):
        print('\n----------------------------------------------------\n')
        for key in self.dict_properties.keys():
            print(key,' : ',self.dict_properties[key])
        
    def Close_Zemax(self):
        self.TheSystem.SaveAs(self.fileZMX)
        if hasattr(self, 'zosapi'): 
            del self.zosapi
            self.zosapi = None
        
    def __del__(self):
        try:
            self.Close_Zemax()
        except :
            pass

if __name__ == '__main__':
    plt.close('all')
    properties=[]
    for wlum in [1.0]:
        sim = simulation_MC('test4_mc', 10_000, 150E-6, 700E-6, 0.78, wlum, [1,1,0,90], diffuse_light=False)
        # sim = simulation_MC('test4_mc', 10_000, 88E-6, 382.66E-6, 0.89, wlum, [1,1,0,90], diffuse_light=False)
        sim.Load_File()
        sim.shoot_rays()
        sim.Close_Zemax()
        sim.Load_parquetfile()
        sim.AOP()
        sim.calculate_musp()
        sim.calculate_ke_theo()
        sim.calculate_MOPL()
        sim.calculate_alpha()
        sim.calculate_mua()
        sim.calculate_ke_rt()
        sim.plot_time_reflectance()
        sim.plot_DOP_transmitance()
        sim.map_stokes_reflectance()
        sim.map_DOP_reflectance()
        sim.plot_DOP_radius_reflectance()
        sim.plot_irradiances()
        sim.plot_MOPL_radius_reflectance()
        sim.plot_lair_radius_reflectance()
        sim.properties()
        sim.export_properties()
        # del sim