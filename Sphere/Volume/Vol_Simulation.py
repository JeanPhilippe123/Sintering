"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
# Import module
import numpy as np
import pandas as pd
import clr, os, sys
import time
from win32com.client import CastTo, constants
import matplotlib.pyplot as plt
import csv
import scipy.constants, scipy.optimize

#Import other files
pathRaytraceDLL = 'C:\Zemax Files\ZOS-API\Libraries\Raytrace.DLL'
sys.path.insert(1,os.path.dirname(os.path.realpath(pathRaytraceDLL)))
import PythonNET_ZRDLoaderFull as init_Zemax
import Vol_Raytrace
import matplotlib
import math
import random
import concurrent.futures
import tartes

class simulation:
    #Constants
    Usepol = True
    pice = 917
    NumModels = 1
    XY_half_width = 0.5
    Depth = 1.0
    max_segments = 4000
    posz_source = -0.0005
    NumArrayX = 3
    NumArrayY = 3
    NumArrayZ = 10
    
    path = os.path.join(os.sep, os.path.dirname(os.path.realpath(__file__)), '')
    
    def __init__(self,name,radius,delta,numrays,numrays_stereo,wlum,pol,Random_pol=False,diffuse_light=False):
        '''
        \r name = Nom donné à la simulation \r
        radius = Rayon des sphères \r
        Delta = Distances entre les centres des sphères \r
        Numrays = Nombre de rayons pour la simulation pour g et B (list array)\r
        numrays_stereo = Nombre de rayons pour la simulation SSA et B (list array)\r
        wlum = Nombre de rayons pour chaqune des simulations (list array)\r
        pol = Polarization entrante pour chaqune des simulations (jx,jy,px,py)
        '''
        self.name = name
        self.radius = np.array(radius)
        self.source_radius = self.radius*10
        self.num_spheres = len(radius)
        self.numrays = numrays
        self.DeltaX = delta
        self.DeltaY = self.DeltaX
        self.DeltaZ = self.DeltaY
        self.numrays_stereo = numrays_stereo
        self.wlum = wlum
        self.jx,self.jy,self.phase_x,self.phase_y = np.array(pol)
        self.dist_min_spheres = delta/np.sqrt(2)
        self.pathZMX = os.path.join(self.path,'Simulations', self.name)
        self.Random_pol = Random_pol
        self.Depth = self.calculate_depth_guess()
        self.XY_half_width = self.Depth/2
        self.diffuse_light = diffuse_light
        self.ice_index,self.ice_complex = tartes.refice2016(self.wlum*1E-6)
        
        if self.diffuse_light == True:
            self.tartes_dir_frac = 0
            self.diffuse_str = 'diffuse'
            self.cosine = 1.0
        else:
            self.tartes_dir_frac = 1
            self.diffuse_str = 'not_diffuse'
            self.cosine = 100.0
        
        self.name_ZRD = '_'.join([name,str(radius),str(delta),str(pol),str(self.numrays),self.diffuse_str])+ ".ZRD"
        self.name_stereo_ZRD = '_'.join([name,'stereo',str(radius),str(delta),str(self.numrays_stereo),self.diffuse_str])+ ".ZRD"
        
        self.path_stereo_npy = os.path.join(self.pathZMX,'_'.join([name,'stereo',str(radius),str(delta),str(self.numrays_stereo),self.diffuse_str])+ ".npy")
        self.path_npy = os.path.join(self.pathZMX,'_'.join([name,str(radius),str(delta),str(pol),str(self.numrays),self.diffuse_str])+ ".npy")
        self.path_stereo_ZRD = os.path.join(self.pathZMX, self.name_stereo_ZRD)
        self.path_ZRD = os.path.join(self.pathZMX,self.name_ZRD)
        
        #Writing in the text file properties
        self.write_in_txt()
        self.write_in_txt("======================================================")
        self.write_in_txt('Simulation :', name)
        self.write_in_txt('Radius :', radius)
        self.write_in_txt('Delta :', delta)
        self.write_in_txt('Numrays :', numrays)
        self.write_in_txt('Wavelength :', wlum)
        self.write_in_txt('Polarization :', pol)

        t = time.localtime()
        self.date = str(t.tm_year)+"/"+str(t.tm_mon)+"/"+str(t.tm_mday)
        self.ctime = str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)
        print('Date : ',self.date,', Time : ',self.ctime)
        self.write_in_txt('Time started', self.date + " " + self.ctime)
        self.write_in_txt()

    def create_folder(self):
        if not os.path.exists(self.pathZMX):
            os.makedirs(self.pathZMX)
            print("Le répertoire " + str(self.name) +  " a été créé")
        else:
            print("Le répertoire " + str(self.name) +  " existe déjà")
        
    def delete_null(self):
        self.list_null = []
        for i in range(1,self.TheNCE.NumberOfObjects+1):
            Object = self.TheNCE.GetObjectAt(i)
            ObjectType = Object.TypeName
            if ObjectType == 'Null Object':
                self.TheNCE.RemoveObjectAt(i)

    def find_ice_object(self):
        ice_obj = np.where(self.array_objects() == np.array(['MY_ICE.ZTG']))[0] + 1
        return ice_obj

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
    
    def filter_ice_air(self,df):
        Sphere_ice_obj = self.find_ice_object()
        filt_inter_ice = (df['hitObj'] == Sphere_ice_obj[0]) & (df['insideOf'] == 0)
        filt_ice = (df['insideOf'] == Sphere_ice_obj[0])
        filt_air = (df['insideOf'] == 0) & (df['segmentLevel'] != 0)
        return filt_ice, filt_air, filt_inter_ice
    
    def Create_filter_raytracing(self):
        # #Create filter that only consider rays touching spheres
        # list_ice_filter = []
        # Sphere_ice_obj = self.find_ice_object()
        # detector_obj = np.where(self.array_objects() == np.array(['Detector Rectangle']))[0]+ 1
        # for i in Sphere_ice_obj:
        #     list_ice_filter += ["H"+str(i)]
        self.filter_raytracing = ""#"Z|(("+"|".join(list_ice_filter)+")&(" + "L"+str(detector_obj[0]) + "|" + "L"+str(detector_obj[1]) + "))"
    
    def calculate_depth_guess(self):
        #Guess Ke so it can calculate depth required
        #Preselected depth
        depth = np.arange(0.00, self.Depth, 0.0001)  # from 0 to 1m depth every 0.1mm
        wl = np.array([self.wlum*1E-06])  # in m
        
        #Calculate pre-theorical SSA, density, g and B
        #Density
        if np.sqrt(2)*self.radius[0] > self.DeltaX/2:
            DensityRatio=0
        else:
            volsphere=4*np.pi*self.radius[0]**3/3
            volcube=self.DeltaX*self.DeltaY*self.DeltaZ
            DensityRatio = volsphere*4/volcube
        density = DensityRatio*self.pice
        #B
        B=1.25
        #g
        g=0.89
        #SSA
        vol_sphere = 4*np.pi*self.radius[0]**3/3
        air_sphere = 4*np.pi*self.radius[0]**2
        SSA = air_sphere/(vol_sphere*self.pice)
        
        #Calculate ke with a fit on intensity curve of TARTES datas
        def linear_fit(depth,a,b):
            intensity = a*depth+b
            return intensity
        
        def ke_raytracing(depths_fit,intensity):
            [a,b],pcov=scipy.optimize.curve_fit(lambda x,a,b: a*x+b, depths_fit, np.log(intensity))
            I_rt_fit=a*depths_fit+b
            return depths_fit, I_rt_fit, -a
                
        down_irr_profile, up_irr_profile = tartes.irradiance_profiles(
            wl,depth,SSA,density,g0=g,B0=B,dir_frac=1,totflux=1)
        
        depths_fit, I_rt_fit, ke_guess = ke_raytracing(depth, down_irr_profile+up_irr_profile)
        ke_guess = ke_guess
        
        #To imitate semi-infinite we must lost at max 0.1% on the z axis in transmitance
        intensity_max = 0.01
        Init_intensity = 1
        depth_min = -np.log(intensity_max/Init_intensity)/ke_guess
        return depth_min
        
    def Initialize_Zemax(self):
        self.zosapi = init_Zemax.PythonStandaloneApplication()
        self.BatchRayTrace = self.zosapi.BatchRayTrace 
        self.TheSystem = self.zosapi.TheSystem
        self.TheApplication = self.zosapi.TheApplication
        self.TheNCE = self.TheSystem.NCE
        
        self.ZOSAPI = self.zosapi.ZOSAPI
        self.ZOSAPI_NCE = self.ZOSAPI.Editors.NCE
        
        #Create File for non-sticky (ns)
        self.fileZMX = os.path.join(self.pathZMX, 'Vol_ns_' + str(self.radius) + '.zmx')
        
    def create_ZMX(self):
        self.TheSystem.New(False)
        self.TheSystem.SaveAs(self.fileZMX)
        self.TheSystem.MakeNonSequential()
        self.TheSystem.SystemData.Units.LensUnits = self.ZOSAPI.SystemData.ZemaxSystemUnits.Meters
        #Wavelength is changed in the function Shoot in Vol_Raytrace.py
        self.TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = self.wlum
        self.TheSystem.SystemData.NonSequentialData.MaximumIntersectionsPerRay = self.max_segments
        self.TheSystem.SystemData.NonSequentialData.MinimumRayIntensity = 1E-6
        self.TheSystem.SystemData.NonSequentialData.MaximumSegmentsPerRay = self.max_segments
        self.TheSystem.SystemData.NonSequentialData.MaximumNestedTouchingObjects = 8
        self.TheSystem.SystemData.NonSequentialData.SimpleRaySplitting = True
        self.TheSystem.SystemData.NonSequentialData.MaximumSourceFileRaysInMemory = 10000000
        self.TheSystem.SystemData.NonSequentialData.GlueDistanceInLensUnits = 1.0000E-10
        self.TheSystem.SaveAs(self.fileZMX)
        print('Fichier Créer')
    
    def Load_File(self):
        start = time.time()
        self.TheSystem.LoadFile(self.fileZMX,False)
        self.TheNCE = self.TheSystem.NCE
        
        #Change wl in um
        self.TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = self.wlum
        
        #Change Polarization
        Source_obj = self.find_source_object()[0]
        Source = self.TheNCE.GetObjectAt(Source_obj)
        Source.SourcesData.RandomPolarization = self.Random_pol
        Source.SourcesData.Jx = self.jx
        Source.SourcesData.Jy = self.jy
        Source.SourcesData.XPhase = self.phase_x
        Source.SourcesData.YPhase = self.phase_y
        Source = self.TheNCE.GetObjectAt(1)

        self.TheSystem.SaveAs(self.fileZMX)
        end = time.time()
        print('Fichier loader en ',end-start)
        self.write_in_txt('File Loaded')
        self.write_in_txt('')
        
    def create_source(self):
        #Créer la source avec un rectangle et 2 sphères
        self.TheNCE.InsertNewObjectAt(1)
        
        Source = self.TheNCE.GetObjectAt(1)
        
        # Type_Source = Source.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.SourceEllipse)
        # Source.ChangeType(Type_Source)
        Type_Source = Source.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.SourceEllipse)
        Source.ChangeType(Type_Source)
        
        Source.XPosition = self.XY_half_width
        Source.YPosition = self.XY_half_width
        Source.ZPosition = self.posz_source
        Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).IntegerValue = 100 #Layout Rays
        Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par6).DoubleValue = self.source_radius #X Half width
        Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par7).DoubleValue = self.source_radius #Y Half width
        Source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par9).DoubleValue = self.cosine #cosine

        Source.SourcesData.RandomPolarization = self.Random_pol
        Source.SourcesData.Jx = self.jx
        Source.SourcesData.Jy = self.jy
        Source.SourcesData.XPhase = self.phase_x
        Source.SourcesData.YPhase = self.phase_y
        
        self.delete_null()
        self.TheSystem.SaveAs(self.fileZMX)
        pass
    
    def create_detectors(self):
        #Créer le détecteur pour la tansmission
        self.TheNCE.InsertNewObjectAt(1)
        Object = self.TheNCE.GetObjectAt(1)
        Type = Object.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.DetectorRectangle)
        Object.ChangeType(Type)
        Object.Material = 'ABSORB'
            
        Object.XPosition = self.XY_half_width
        Object.YPosition = self.XY_half_width
        #Derrière le dernier grains de neige
        depth_z = self.DeltaZ*self.NumArrayZ*math.ceil(self.Depth/(self.DeltaZ*self.NumArrayZ))-self.radius
        Object.ZPosition = depth_z
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).DoubleValue = self.XY_half_width
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par2).DoubleValue = self.XY_half_width
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par3).IntegerValue = 1
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par4).IntegerValue = 1
        
        #Créer le détecteur pour la réflexion
        self.TheNCE.InsertNewObjectAt(1)
        Object = self.TheNCE.GetObjectAt(1)
        Type = Object.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.DetectorRectangle)
        Object.ChangeType(Type)
        Object.Material = 'ABSORB'
            
        Object.XPosition = self.XY_half_width
        Object.YPosition = self.XY_half_width
        Object.ZPosition = -0.0006
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).DoubleValue = self.XY_half_width
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par2).DoubleValue = self.XY_half_width
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par3).IntegerValue = 1
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par4).IntegerValue = 1
        
        self.delete_null()
        self.TheSystem.SaveAs(self.fileZMX)
        pass
        
    def create_model_sphere(self):
        #Créer la forme
        sphere_ZPosition = -0.001
        sphere_XPosition = 0
        
        self.TheNCE.InsertNewObjectAt(1)
        Object = self.TheNCE.GetObjectAt(1)
        Type = Object.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.Sphere)
        Object.ChangeType(Type)
        Object.Material = 'MY_ICE.ZTG'
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).DoubleValue = self.radius
        Object.XPosition = sphere_XPosition
        Object.YPosition = 0
        Object.ZPosition = sphere_ZPosition
        Object.DrawData.DoNotDrawObject = True
        Object.TypeData.RaysIgnoreObject = self.ZOSAPI_NCE.RaysIgnoreObjectType.Always
        
        self.delete_null()
        self.TheSystem.SaveAs(self.fileZMX)
        pass
    
    def get_arrays_info(self):
        #Space between each element of the same Array
        DeltaArrayX=self.DeltaX*self.NumArrayX
        DeltaArrayY=self.DeltaY*self.NumArrayY
        DeltaArrayZ=self.DeltaZ*self.NumArrayZ
        DeltaArray = np.array([DeltaArrayX,DeltaArrayY,DeltaArrayZ])
        
        #Number of Object of the same Array
        NumObjectPerArrayX = math.ceil(2*self.XY_half_width/(DeltaArrayX))
        NumObjectPerArrayY = math.ceil(2*self.XY_half_width/(DeltaArrayY))
        NumObjectPerArrayZ = math.ceil(self.Depth/(DeltaArrayZ))
        # NumObjectPerArrayX = 1
        # NumObjectPerArrayY = 1
        # NumObjectPerArrayZ = 1
        NumObjectPerArray = np.array([NumObjectPerArrayX,NumObjectPerArrayY,NumObjectPerArrayZ])
        
        #For cross arrays
        #First coordinate for the first element of the Array
        PosArrayX = np.linspace(0,DeltaArrayX,self.NumArrayX+1)[:-1] #+ Random_x
        PosArrayY = np.linspace(0,DeltaArrayY,self.NumArrayY+1)[:-1] #+ Random_y
        PosArrayZ = np.linspace(0,DeltaArrayZ,self.NumArrayZ+1)[:-1]
        #Cross in plane zx
        PosArrayX_xz = np.linspace(0,DeltaArrayX,self.NumArrayX+1)[:-1] + self.DeltaX/2 #+ Random_x
        PosArrayY_xz = np.linspace(0,DeltaArrayY,self.NumArrayY+1)[:-1] #+ Random_y
        PosArrayZ_xz = np.linspace(0,DeltaArrayZ,self.NumArrayZ+1)[:-1] + self.DeltaZ/2
        #Cross in plane zy
        PosArrayX_xy = np.linspace(0,DeltaArrayX,self.NumArrayX+1)[:-1] + self.DeltaX/2 #+ Random_x
        PosArrayY_xy = np.linspace(0,DeltaArrayY,self.NumArrayY+1)[:-1] + self.DeltaZ/2 # + Random_y
        PosArrayZ_xy = np.linspace(0,DeltaArrayZ,self.NumArrayZ+1)[:-1]
        #Cross in plane zx and zy
        PosArrayX_yz = np.linspace(0,DeltaArrayX,self.NumArrayX+1)[:-1] #+ Random_x
        PosArrayY_yz = np.linspace(0,DeltaArrayY,self.NumArrayY+1)[:-1] + self.DeltaZ/2 #+ Random_y
        PosArrayZ_yz = np.linspace(0,DeltaArrayZ,self.NumArrayZ+1)[:-1] + self.DeltaZ/2
        
        Positions = np.array([[PosArrayX,PosArrayY,PosArrayZ],
                              [PosArrayX_xz,PosArrayY_xz,PosArrayZ_xz],
                              [PosArrayX_yz,PosArrayY_yz,PosArrayZ_yz],
                              [PosArrayX_xy,PosArrayY_xy,PosArrayZ_xy]],dtype=object)
        
        #Randomize Positions
        Array_positions = self.randomize_positions(Positions,DeltaArray)
        
        #No Randomize
        # pos = np.array(np.meshgrid(PosArrayX,PosArrayY,PosArrayZ))
        # pos_xy = np.array(np.meshgrid(PosArrayX_xy,PosArrayY_xy,PosArrayZ_xy))
        # pos_xz = np.array(np.meshgrid(PosArrayX_xz,PosArrayY_xz,PosArrayZ_xz))
        # pos_yz = np.array(np.meshgrid(PosArrayX_yz,PosArrayY_yz,PosArrayZ_yz))
        # Array_positions = np.transpose(np.concatenate((pos,pos_xy,pos_xz,pos_yz),axis=1)).reshape(4*self.NumArrayY*self.NumArrayX*self.NumArrayZ,3)
        
        return Array_positions, NumObjectPerArray, DeltaArray
    
    def randomize_positions(self,Positions,DeltaArray):
        
        def Randomize(pos,pos_xy,pos_xz,pos_yz,filt,filt_xy,filt_xz,filt_yz):
            factor = (self.DeltaX/(2*np.sqrt(2))-self.radius)*2
            #Used DeltaX because DeltaX=DeltaY=DeltaZ
            pos_t = pos+filt*((np.random.rand(3,self.NumArrayY,self.NumArrayX,self.NumArrayZ)*factor)-factor/2)
            pos_xy_t = pos_xy+filt_xy*((np.random.rand(3,self.NumArrayY,self.NumArrayX,self.NumArrayZ)*factor)-factor/2)
            pos_xz_t = pos_xz+filt_xz*((np.random.rand(3,self.NumArrayY,self.NumArrayX,self.NumArrayZ)*factor)-factor/2)
            pos_yz_t = pos_yz+filt_yz*((np.random.rand(3,self.NumArrayY,self.NumArrayX,self.NumArrayZ)*factor)-factor/2)
            
            return pos_t,pos_xy_t,pos_xz_t,pos_yz_t
        
        def norm(pos1,pos2):
            posx1,posy1,posz1 = pos1
            posx2,posy2,posz2 = pos2
            norm = np.linalg.norm([posx1-posx2,posy1-posy2,posz1-posz2],axis=0)
            return norm
        
        def create_compare_array(pos,DeltaArray):
            posx,posy,posz = pos
            DeltaArrayX,DeltaArrayY,DeltaArrayZ = DeltaArray
            
            #Add line to posx in x axis
            posx_add_l = np.full((self.NumArrayY,self.NumArrayZ),posx[0,-1,0]-DeltaArrayX).reshape(self.NumArrayY,1,self.NumArrayZ)
            posx_add_r = np.full((self.NumArrayY,self.NumArrayZ),posx[0,0,0]+DeltaArrayX).reshape(self.NumArrayY,1,self.NumArrayZ)
            posx_c = np.append(posx_add_l,posx,axis=1)
            posx_c = np.append(posx_c,posx_add_r,axis=1)
            
            #Add line to posy in y axis
            posy_add_l = np.full((self.NumArrayX,self.NumArrayZ),posy[-1,0,0]-DeltaArrayY).reshape(1,self.NumArrayX,self.NumArrayZ)
            posy_add_r = np.full((self.NumArrayX,self.NumArrayZ),posy[0,0,0]+DeltaArrayY).reshape(1,self.NumArrayX,self.NumArrayZ)
            posy_c = np.append(posy_add_l,posy,axis=0)
            posy_c = np.append(posy_c,posy_add_r,axis=0)
            
            #Add line to posz in z axis
            posz_add_l = np.full((self.NumArrayX,self.NumArrayY),posz[0,0,-1]-DeltaArrayZ).reshape(self.NumArrayY,self.NumArrayX,1)
            posz_add_r = np.full((self.NumArrayX,self.NumArrayY),posz[0,0,0]+DeltaArrayZ).reshape(self.NumArrayY,self.NumArrayX,1)
            posz_c = np.append(posz_add_l,posz,axis=2)
            posz_c = np.append(posz_c,posz_add_r,axis=2)

            return posx_c,posy_c,posz_c
        
        def calculate_norm(pos1,pos_xy,pos_xz,pos_yz):
            #Function that calculate the norm with the pos1 vs pos xy,xz,yz
            #_ij indices indicate in which direction the shift will be
            #Shift Always occur in other plan than the plane of the two comparing spheres
            posx,posy,posz = pos1
            posx_xy,posy_xy,posz_xy = pos_xy
            posx_xz,posy_xz,posz_xz = pos_xz
            posx_yz,posy_yz,posz_yz = pos_yz

            #======Compare======
            #18 comparisons for normal array
            n = []
            #Creation of the compare_array
            posx_c,posy_c, posz_c = create_compare_array([posx,posy,posz],DeltaArray)
            #+y
            n += [norm([posx,posy,posz],[posx_c[:,2:,:],posy,posz])]
            #-y
            n += [norm([posx,posy,posz],[posx_c[:,:-2,:],posy,posz])]
            #+y
            n += [norm([posx,posy,posz],[posx,posy_c[2:,:,:],posz])]
            #-y
            n += [norm([posx,posy,posz],[posx,posy_c[:-2,:,:],posz])]
            #+z
            n += [norm([posx,posy,posz],[posx,posy,posz_c[:,:,:-2]])]
            #-z
            n += [norm([posx,posy,posz],[posx,posy,posz_c[:,:,2:]])]
            
            #====plan_xy=====
            posx_xy_c,posy_xy_c, posz_xy_c = create_compare_array([posx_xy,posy_xy,posz_xy],DeltaArray)
            if np.mean(posx) < np.mean(posx_xy): posx_xy_c = posx_xy_c[:,:-2,:]
            else: posx_xy_c = posx_xy_c[:,2:,:]
            if np.mean(posy) < np.mean(posy_xy): posy_xy_c = posy_xy_c[:-2,:,:]
            else: posy_xy_c = posy_xy_c[2:,:,:]
                
            #+x+y
            n += [norm([posx,posy,posz],[posx_xy,posy_xy,posz_xy])]
            #+x-y
            n += [norm([posx,posy,posz],[posx_xy,posy_xy_c,posz_xy])]
            #-x+y
            n += [norm([posx,posy,posz],[posx_xy_c,posy_xy,posz_xy])]
            #-x-y
            n += [norm([posx,posy,posz],[posx_xy_c,posy_xy_c,posz_xy])]
            
            #====plan_xz=====
            posx_xz_c,posy_xz_c, posz_xz_c = create_compare_array([posx_xz,posy_xz,posz_xz],DeltaArray)
            if np.mean(posx) < np.mean(posx_xz): posx_xz_c = posx_xz_c[:,:-2,:]
            else: posx_xz_c = posx_xz_c[:,2:,:]
            if np.mean(posz) < np.mean(posz_xz): posz_xz_c = posz_xz_c[:,:,:-2]
            else: posz_xz_c = posz_xz_c[:,:,2:]
            #+x+z
            n += [norm([posx,posy,posz],[posx_xz,posy_xz,posz_xz])]
            #+x-z
            n += [norm([posx,posy,posz],[posx_xz,posy_xz,posz_xz_c])]
            #-x+z
            n += [norm([posx,posy,posz],[posx_xz_c,posy_xz,posz_xz])]
            #-x-z
            n += [norm([posx,posy,posz],[posx_xz_c,posy_xz,posz_xz_c])]
            
            #====plan_yz=====
            posx_yz_c,posy_yz_c, posz_yz_c = create_compare_array([posx_yz,posy_yz,posz_yz],DeltaArray)
            if np.mean(posy) < np.mean(posy_yz): posy_yz_c = posy_yz_c[:-2,:,:]
            else: posy_yz_c = posy_yz_c[2:,:,:]
            if np.mean(posz) < np.mean(posz_yz): posz_yz_c = posz_yz_c[:,:,:-2]
            else: posz_yz_c = posz_yz_c[:,:,2:]
            #+y+z
            n += [norm([posx,posy,posz],[posx_yz,posy_yz,posz_yz])]
            #+y-z
            n += [norm([posx,posy,posz],[posx_yz,posy_yz,posz_yz_c])]
            #-y+z
            n += [norm([posx,posy,posz],[posx_yz,posy_yz_c,posz_yz])]
            #-y-z
            n += [norm([posx,posy,posz],[posx_yz,posy_yz_c,posz_yz_c])]

            return  np.array(n)
        
        #Positions
        [[PosArrayX,PosArrayY,PosArrayZ],
        [PosArrayX_xz,PosArrayY_xz,PosArrayZ_xz],
        [PosArrayX_yz,PosArrayY_yz,PosArrayZ_yz],
        [PosArrayX_xy,PosArrayY_xy,PosArrayZ_xy]] = Positions
        
        #Initial positions
        pos_i = np.array(np.meshgrid(PosArrayX,PosArrayY,PosArrayZ))
        pos_xy_i = np.array(np.meshgrid(PosArrayX_xy,PosArrayY_xy,PosArrayZ_xy))
        pos_xz_i = np.array(np.meshgrid(PosArrayX_xz,PosArrayY_xz,PosArrayZ_xz))
        pos_yz_i = np.array(np.meshgrid(PosArrayX_yz,PosArrayY_yz,PosArrayZ_yz))
        filt_i=np.ones((3,self.NumArrayY,self.NumArrayX,self.NumArrayZ),dtype=bool)
        filt_xy_i=np.ones((3,self.NumArrayY,self.NumArrayX,self.NumArrayZ),dtype=bool)
        filt_xz_i=np.ones((3,self.NumArrayY,self.NumArrayX,self.NumArrayZ),dtype=bool)
        filt_yz_i=np.ones((3,self.NumArrayY,self.NumArrayX,self.NumArrayZ),dtype=bool)
        
        move = 10
        
        for move in range(1,move+1):
            if move == 1:
                #Position for the move
                pos_m,pos_xy_m,pos_xz_m,pos_yz_m = pos_i,pos_xy_i,pos_xz_i,pos_yz_i
            
            #Position trying to randomize
            pos,pos_xy,pos_xz,pos_yz = pos_m,pos_xy_m,pos_xz_m,pos_yz_m
            filt,filt_xy,filt_xz,filt_yz = filt_i,filt_xy_i,filt_xz_i,filt_yz_i            
            
            #Checking if touches
            condition=False
            i=0
            j=0
            while condition != True:
                #Randomize position and associate it to a temporary variable
                pos_t,pos_xy_t,pos_xz_t,pos_yz_t = Randomize(pos, pos_xy, pos_xz, pos_yz, filt, filt_xy, filt_xz, filt_yz)
                # pos_t,pos_xy_t,pos_xz_t,pos_yz_t = pos,pos_xy,pos_xz,pos_yz
                
                #Calculate norm between each array
                norm_array = calculate_norm(pos_t,pos_xy_t,pos_xz_t,pos_yz_t)
                norm_array_xy = calculate_norm(pos_xy_t,pos_t,pos_yz_t,pos_xz_t)
                norm_array_xz = calculate_norm(pos_xz_t,pos_yz_t,pos_t,pos_xy_t)
                norm_array_yz = calculate_norm(pos_yz_t,pos_xz_t,pos_xy_t,pos_t)
                
                #Checking it touches and resize array
                filt_t = np.tile(np.any(norm_array < self.radius*2, axis=0),[3,1,1]).reshape(3,self.NumArrayY,self.NumArrayX,self.NumArrayZ)
                filt_xy_t = np.tile(np.any(norm_array_xy < self.radius*2, axis=0),[3,1,1]).reshape(3,self.NumArrayY,self.NumArrayX,self.NumArrayZ)
                filt_xz_t = np.tile(np.any(norm_array_xz < self.radius*2, axis=0),[3,1,1]).reshape(3,self.NumArrayY,self.NumArrayX,self.NumArrayZ)
                filt_yz_t = np.tile(np.any(norm_array_yz < self.radius*2, axis=0),[3,1,1]).reshape(3,self.NumArrayY,self.NumArrayX,self.NumArrayZ)
                
                #If it touches
                if filt_t.any()|filt_xy_t.any()|filt_xz_t.any()|filt_yz_t.any():
                    #If array has improved
                    count_t = sum(list(map(np.count_nonzero,(filt_t==True,filt_xy_t==True,filt_xz_t==True,filt_yz_t==True))))
                    count = sum(list(map(np.count_nonzero,(filt==True,filt_xy==True,filt_xz==True,filt_yz==True))))
                    if count_t<count:
                        i=0
                        #Associate new positions
                        pos = pos_t*~filt_t + pos*filt_t
                        pos_xy = pos_xy_t*~filt_xy_t + pos_xy*filt_xy_t
                        pos_xz = pos_xz_t*~filt_xz_t + pos_xz*filt_xz_t
                        pos_yz = pos_yz_t*~filt_yz_t + pos_yz*filt_yz_t
                        
                        #Associate new filter to temporary one because of improvement
                        filt, filt_xy, filt_xz, filt_yz = filt_t, filt_xy_t, filt_xz_t, filt_yz_t
                    else:
                        i+=1

                    if i>=10: 
                        i=0
                        j+=1
                        pos,pos_xy,pos_xz,pos_yz = pos_m,pos_xy_m,pos_xz_m,pos_yz_m
                        filt,filt_xy,filt_xz,filt_yz = filt_i,filt_xy_i,filt_xz_i,filt_yz_i
                        if j>=25:
                            print('Can\'t randomize more, %d moves have been done' %move)
                            # ax = plt.axes(projection='3d')
                            # ax.scatter3D(pos_i[0],pos_i[1],pos_i[2], cmap='Greens')
                            # ax.scatter3D(pos_xy_i[0],pos_xy_i[1],pos_xy_i[2], cmap='Greens')
                            # ax.scatter3D(pos_xz_i[0],pos_xz_i[1],pos_xz_i[2], cmap='Greens')
                            # ax.scatter3D(pos_yz_i[0],pos_yz_i[1],pos_yz_i[2], cmap='Greens')
                            positions = np.transpose(np.concatenate((pos,pos_xy,pos_xz,pos_yz),axis=1)).reshape(4*self.NumArrayY*self.NumArrayX*self.NumArrayZ,3)
                            return positions
                    # print(len(pos[filt]),len(pos_xy[filt_xy]),len(pos_xz[filt_xz]),len(pos_yz[filt_yz]))
                #If Ok
                else:
                    # Associate temporary position to real position
                    pos_m,pos_xy_m,pos_xz_m,pos_yz_m = pos_t,pos_xy_t,pos_xz_t,pos_yz_t
                    condition = True
            pos_i,pos_xy_i,pos_xz_i,pos_yz_i = pos_m,pos_xy_m,pos_xz_m,pos_yz_m
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(pos_i[0],pos_i[1],pos_i[2], cmap='Greens')
        # ax.scatter3D(pos_xy_i[0],pos_xy_i[1],pos_xy_i[2], cmap='Greens')
        # ax.scatter3D(pos_xz_i[0],pos_xz_i[1],pos_xz_i[2], cmap='Greens')
        # ax.scatter3D(pos_yz_i[0],pos_yz_i[1],pos_yz_i[2], cmap='Greens')
        positions = np.transpose(np.concatenate((pos,pos_xy,pos_xz,pos_yz),axis=1)).reshape(4*self.NumArrayY*self.NumArrayX*self.NumArrayZ,3)
        return positions
    
    def create_snow(self):
        self.create_model_sphere()
        self.Array_positions, self.NumObjectPerArray,self.DeltaArray = self.get_arrays_info()
        self.create_array(self.Array_positions[0],self.NumObjectPerArray,self.DeltaArray)
        self.create_array_copies()
        self.TheSystem.SaveAs(self.fileZMX)
    
    def create_array(self,Pos,Num,Delta):
        #Add object + make it an array
        self.TheNCE.InsertNewObjectAt(2)
        Object = self.TheNCE.GetObjectAt(2)
        Type = Object.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.Array)
        Object.ChangeType(Type)
        
        #Positions
        Object.XPosition = Pos[0]
        Object.YPosition = Pos[1]
        Object.ZPosition = Pos[2]
        
        # parentObject = self.find_ice_object()[0]
        #Parent Object
        # Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).IntegerValue = parentObject
        #Number Object in X direction
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par2).IntegerValue = Num[0]
        #Number Object in Y direction
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par3).IntegerValue = Num[1]
        #Number Object in Z direction
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par4).IntegerValue = Num[2]
        #Delta between Object in X direction
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par5).DoubleValue = Delta[0]
        #Delta between Object in Y direction
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par6).DoubleValue = Delta[1]
        #Delta between Object in Z direction
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par7).DoubleValue = Delta[2]
        self.TheSystem.SaveAs(self.fileZMX)
        pass
    
    def create_array_copies(self):
        def change_parent_obj(i,obj):
            Object = self.TheNCE.GetObjectAt(i+2)
            Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).IntegerValue = obj
        
        #Create a copie of the Array and change is position
        start = time.time()
        num_copies = len(self.Array_positions)
        for i in range(0,num_copies):
            self.TheNCE.CopyObjects(2,1,2)
            Object = self.TheNCE.GetObjectAt(3)
            Object.XPosition = self.Array_positions[i][0]
            Object.YPosition = self.Array_positions[i][1]
            Object.ZPosition = self.Array_positions[i][2]
        end = time.time()
        print(end-start)
        
        #Change the parent of the arrays using threading
        ice_obj = self.find_ice_object()[0]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0,num_copies+1):
                executor.submit(change_parent_obj,i,ice_obj)

        end = time.time()
        print(end-start)
        
        #Remove initial array so it is not doubled
        self.TheNCE.RemoveObjectAt(2)
        
    def shoot_rays(self):
        self.Create_filter_raytracing()
        
        Source_obj = self.find_source_object()[0]
        Source_object = self.TheNCE.GetObjectAt(Source_obj)
        Source_object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par9).DoubleValue = self.cosine
            
        print('Raytrace')
        Vol_Raytrace.Shoot(self,self.filter_raytracing,self.numrays,self.path_npy,self.name_ZRD)
        pass
    
    def shoot_rays_stereo(self):
        #Créer un filtre pour la glace
        self.Create_filter_raytracing()
        
        #Change les grains de neige en air (pour que les rayons n'intéragissent pas)
        Sphere_ice_obj = self.find_ice_object()
        for i in Sphere_ice_obj:
            Object = self.TheNCE.GetObjectAt(i)
            Object.Material = ''
            
        self.TheSystem.SaveAs(self.fileZMX)
        
        #Change cosine object for stereo
        Source_obj = self.find_source_object()[0]
        Source_object = self.TheNCE.GetObjectAt(Source_obj)
        cosine = 0.0
        Source_object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par9).DoubleValue = cosine
        
        #Calcul la stéréologie et retourne un fichier npy
        print('SSA Raytrace')
        Vol_Raytrace.Shoot(self,self.filter_raytracing,self.numrays_stereo,self.path_stereo_npy,self.name_stereo_ZRD)
        
        #Rechange les grains d'air en neige
        for i in Sphere_ice_obj:
            Object = self.TheNCE.GetObjectAt(i)
            Object.Material = 'MY_ICE.ZTG'
        
        self.TheSystem.SaveAs(self.fileZMX)
    
    def Load_npyfile(self):
        # Try loading the stereo file
        try: 
            self.df_stereo = Vol_Raytrace.Load_npy(self.path_stereo_npy)
        except FileNotFoundError:
            print('The stereo npy file was not loaded')
            pass
        #Try loading the ray file
        try:
            #Filter les rayons avec plus de 4000 interactions
            self.df = Vol_Raytrace.Load_npy(self.path_npy)
            
            #Change intensity
            self.change_path_intensity()
            #Groupby for last segment of each ray
            df = self.df.groupby(self.df.index).agg({'hitObj':'last','segmentLevel':'last','intensity':'last'})
            #Transmitance
            filt_top_detector = df['hitObj'] == self.find_detector_object()[0]
            df_top = df[filt_top_detector]
            self.Reflectance = np.sum(df_top['intensity'])
            self.numrays_Reflectance = df_top.shape[0]
            #Reflectance
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
            self.Absorb = 1-self.Reflectance-self.Transmitance-self.Error-self.Lost
            self.numrays_Absorb = self.numrays-self.numrays_Reflectance-self.numrays_Transmitance-self.numrays_Error-self.numrays_Lost
        except FileNotFoundError:
            print('The raytrace npy file was not loaded')
            pass
    
    def change_path_intensity(self):
        #Calculate the intensity for each segments
        filt_ice, filt_air, filt_inter_ice = self.filter_ice_air(self.df)

        #Calculate new intensity
        self.gamma = 4*np.pi*self.ice_complex/(self.wlum*1E-6)
        I_0 = 1./self.numrays
        #Ponderate pathlength for ice only (absorbing media)
        self.df.insert(15,'ponderation',0)
        self.df.loc[(filt_ice,'ponderation')] = 1
        pond_pL = self.df['ponderation']*self.df['pathLength']
        
        #Overwrite the intensity
        pathLength = pond_pL.groupby(pond_pL.index).cumsum()
        intensity = I_0*np.exp(-self.gamma*pathLength).values
        self.df['intensity'] = intensity
        
        self.df.drop(columns = ['ponderation'])
        pass

    def calculate_SSA(self):
        #====================SSA stéréologie====================
        #Shoot rays for SSA
        if not hasattr(self, 'df_stereo'):
            self.Load_npyfile()

        #Length
        filt_ice, filt_air, filt_inter_ice = self.filter_ice_air(self.df_stereo)
        numrays = self.df_stereo.index[-1]
        l_Ice_stereo = np.sum(self.df_stereo.loc[filt_ice, 'pathLength'])/numrays
        #Inter Air/Icef
        Num_ice_inter = self.df_stereo[filt_inter_ice].shape[0]/numrays
        lengthIce_per_seg = l_Ice_stereo/Num_ice_inter
        self.SSA_stereo = round(4/(self.pice*lengthIce_per_seg),6)
        self.write_in_txt('SSA stéréologie :', self.SSA_stereo)
        
        #====================SSA théorique====================
        vol_sphere = 4*np.pi*self.radius[0]**3/3
        air_sphere = 4*np.pi*self.radius[0]**2
        self.SSA_theo = air_sphere/(vol_sphere*self.pice)
        self.write_in_txt('SSA théorique:', self.SSA_theo)
        
    def calculate_density(self):
        #====================Densité physique théorique====================
        if np.sqrt(2)*self.radius[0] > self.DeltaX/2:
            DensityRatio=0
        else:
            volsphere=4*np.pi*self.radius[0]**3/3
            volcube=self.DeltaX*self.DeltaY*self.DeltaZ
            DensityRatio = volsphere*4/volcube
        self.density_theo = DensityRatio*self.pice
        self.write_in_txt('Densité théorique :', self.density_theo)
        
        
        #====================Densité physique stéréologie====================
        #Shoot rays for SSA if not already done
        if not hasattr(self, 'df_stereo'):
            self.Load_npyfile()
            
        filt_ice, filt_air, filt_inter_ice = self.filter_ice_air(self.df_stereo)
        #Remove first and last segment
        lengthIce = float(np.sum(self.df_stereo.loc[filt_ice, 'pathLength']))
        #Remove extra air segment du to source not beeing at origin
        lengthAir = float(np.sum(self.df_stereo.loc[filt_air, 'pathLength']))-np.abs(self.posz_source*self.numrays_stereo)
        self.density_stereo = round(lengthIce/(lengthAir+lengthIce),6)*self.pice
        self.write_in_txt('Densité stéreologique :', self.density_stereo)
    
    def calculate_porosity(self):
        #====================Porosité physique théorique====================
        if not hasattr(self, 'density_theo'): self.calculate_density()
        if not hasattr(self, 'B_stereo'): self.calculate_B()
            
        porosity = 1-self.density_theo/self.pice
        self.physical_porosity_theo = porosity
        
        #====================Porosité optique théorique====================
        self.optical_porosity_theo = porosity/(porosity+self.B_stereo*(1-porosity))
        
        #====================Porosité physique stéréologique====================
        #Shoot rays for SSA if not already done
        if not hasattr(self, 'df_stereo'):
            self.Load_npyfile()
        
        filt_ice_stereo, filt_air_stereo, filt_inter_ice_stereo = self.filter_ice_air(self.df_stereo)
        lengthIce = float(np.sum(self.df_stereo.loc[filt_ice_stereo, 'pathLength']))
        lengthAir = float(np.sum(self.df_stereo.loc[filt_air_stereo, 'pathLength']))
        self.physical_porosity_stereo = round(lengthAir/(lengthAir+lengthIce),6)
        
        #====================Porosité optique stéréologique====================
        self.optical_porosity_stereo = round(lengthAir/(lengthAir+self.B_stereo*lengthIce),6)
        
        self.write_in_txt('Porosité Physique théorique :', self.physical_porosity_theo)
        self.write_in_txt('Porosité Physique stéréologique :', self.physical_porosity_stereo)
        self.write_in_txt('Porosité Optique théorique :', self.optical_porosity_theo)
        self.write_in_txt('Porosité Optique stéréologique :', self.optical_porosity_stereo)
    
    def calculate_neff(self):
        df_filt = self.df[~((self.df['segmentLevel'] == 0)|(self.df['segmentLevel'] == 1))]
        #Calculate v mean
        self.neff_rt = sum(df_filt['pathLength']*df_filt['index'])/sum(df_filt['pathLength'])
        
        #Calculate neff stereo
        if not hasattr(self, 'B_stereo'): self.calculate_B()
        porosity = self.physical_porosity_stereo
        filt_ice, filt_air, filt_inter_ice = self.filter_ice_air(df_filt)
        n_ice = np.mean(df_filt['index'].loc[filt_ice])
        self.neff_stereo = (porosity + n_ice*self.B_stereo*(1-porosity))/(porosity + self.B_stereo*(1-porosity))
        pass
    
    def calculate_MOPL(self):
        #Shoot rays for SSA
        if not hasattr(self, 'df_stereo'):
            self.Load_npyfile()
        
        #Theorique
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        if not hasattr(self, 'mua_rt'): self.calculate_mua()
        if not hasattr(self, 'neff_rt'): self.calculate_neff()
        
        #Approximation de Paterson
        # z_o = self.musp_stereo**(-1)
        # D = (3*(self.mua_stereo+self.musp_stereo))**(-1)
        # self.MOPL_stereo = self.neff_stereo*z_o/(2*np.sqrt(self.mua_stereo*D))
        
        #Raytracing
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df_top = self.df[filt_top_detector]
        df_filt = self.df.loc[df_top.index]
        
        df_filt.insert(15,'OPL',np.nan)
        df_filt['OPL'] = df_filt['pathLength']*df_filt['index']
        df_OPL = df_filt.groupby(df_filt.index).agg({'pathLength':sum, 'OPL':sum, 'intensity':'last'})
        self.MOPL_rt = np.average(df_OPL['OPL'],weights=df_OPL['intensity'])
        
        #Stéréologique
        filt_ice_rt, filt_air_rt, filt_inter_ice_rt = self.filter_ice_air(self.df)
        n_ice = np.mean(self.df[filt_ice_rt]['index'])
        df_filt = self.df[filt_air_rt].groupby(self.df[filt_air_rt].index).agg({'pathLength':sum,'intensity':'last'})
        l_air_mean = np.average(df_filt['pathLength'],weights=df_filt['intensity'])
        self.MOPL_stereo = (n_ice*l_air_mean)/self.optical_porosity_stereo-(n_ice-1)*l_air_mean
        
        self.write_in_txt('MOPL raytracing:', self.MOPL_rt)
        self.write_in_txt('MOPL stéréologique:', self.MOPL_stereo)
        
    def calculate_g_theo(self):
        self.g_theo = 0.89
        self.gG_theo = self.g_theo*2-1
        self.write_in_txt('g théorique', self.g_theo)
        self.write_in_txt('gG théorique', self.gG_theo)
    
    def calculate_g_rt(self):
        if not hasattr(self, 'ke_rt'): self.calculate_ke_rt()
        if not hasattr(self, 'alpha_rt'): self.calculate_alpha()
        
        #g calculé avec ke et alpha raytracing
        #Équation Quentin Libois 3.3 thèse
        self.g_rt = 1+8*self.ke_rt/(3*self.density_stereo*np.log(self.alpha_rt)*self.SSA_stereo)
        self.gG_rt = self.g_rt*2-1
        
        self.write_in_txt('g raytracing', self.g_rt)
        self.write_in_txt('gG raytracing', self.gG_rt)
    
    def calculate_mus(self):
        if not hasattr(self, 'density_theo'): self.calculate_density()
        if not hasattr(self, 'SSA_theo'): self.calculate_SSA()
        self.mus_theo = self.density_theo*self.SSA_theo/2
        self.mus_stereo = self.density_stereo*self.SSA_stereo/2
        self.write_in_txt('mu_s stereo :', self.mus_stereo)
        self.write_in_txt('mu_s theo :', self.mus_theo)
        
    def calculate_musp(self):
        if not hasattr(self, 'mus_theo'): self.calculate_mus()
        if not hasattr(self, 'g_theo'): self.calculate_g_theo()
        self.musp_theo = self.mus_theo*(1-self.g_theo)
        self.musp_stereo = self.mus_stereo*(1-self.g_theo)
        self.write_in_txt('mu_s\' stereo :', self.musp_stereo)
        self.write_in_txt('mu_s\' theo :', self.musp_theo)
        
    def calculate_B(self):
        #Length of ice per rays
        filt_ice, filt_air, filt_inter_ice = self.filter_ice_air(self.df)
        num_inter = self.df[filt_inter_ice].shape[0]
        l_Ice_p = np.sum(self.df.loc[filt_ice]['pathLength'])/num_inter
        
        #Length of ice per rays for stereo
        filt_ice_stereo, filt_air_stereo, filt_inter_ice_stereo = self.filter_ice_air(self.df_stereo)
        num_inter_stereo = self.df_stereo[filt_inter_ice_stereo].shape[0]
        l_Ice_stereo = np.sum(self.df_stereo.loc[filt_ice_stereo, 'pathLength'])/num_inter_stereo
        
        #Calculate B stereological
        self.B_stereo = l_Ice_p/l_Ice_stereo

        #Calculate B raytracing
        if not hasattr(self, 'ke_theo'): self.calculate_ke_theo()
        if not hasattr(self, 'alpha_rt'): self.calculate_alpha()
        if not hasattr(self, 'optical_porosity_stereo'): self.calculate_porosity()

        self.B_theo = -self.ke_theo*np.log(self.alpha_theo)/(4*(1-self.physical_porosity_theo)*self.gamma)

        self.write_in_txt('B stéréologique :',round(self.B_stereo,4))
        self.write_in_txt('B stéréologique :',round(self.B_theo,4))
        pass

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
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for intensity_rt in executor.map(I_ke_raytracing, depths_fit):
        #         print(intensity_rt)
        intensity_rt = list(map(I_ke_raytracing,depths_fit))
        self.ke_rt, self.b_rt = self.ke_raytracing(depths_fit,intensity_rt)
        self.write_in_txt('ke raytrace:',round(self.ke_rt,4))
        
    def calculate_ke_theo(self):
        if not hasattr(self, 'density_theo'): self.calculate_density()
        if not hasattr(self, 'g_theo'): self.calculate_g_theo()
        if not hasattr(self, 'B_stereo'): self.calculate_B()
        if not hasattr(self, 'SSA_theo'): self.calculate_SSA()
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        
        depths_fit = np.linspace(1/self.musp_theo,10/self.musp_theo,10)
        #ke TARTES
        down_irr_profile, up_irr_profile = tartes.irradiance_profiles(
            self.wlum*1E-6,depths_fit,self.SSA_theo,self.density_theo,g0=self.g_theo,B0=self.B_stereo,dir_frac=self.tartes_dir_frac)
        # plt.plot(depths_fit,np.exp(-self.ke_rt*depths_fit + b),label='fit raytracing')
        self.ke_tartes, self.b_tartes = self.ke_raytracing(depths_fit, down_irr_profile+up_irr_profile)

        #ke stéréologique
        #Imaginay part of the indice of refraction
        self.gamma = 4*np.pi*self.ice_complex/(self.wlum*1E-6)
        self.ke_stereo = self.density_stereo*np.sqrt(3*self.B_stereo*self.gamma/(4*self.pice)*self.SSA_theo*(1-self.gG_theo))
        
        #ke théorique
        self.ke_theo = self.density_theo*np.sqrt(3*self.B_stereo*self.gamma/(4*self.pice)*self.SSA_theo*(1-self.gG_theo))
        
        # plt.plot(depths_fit,np.exp(-self.ke_tartes*depths_fit + b),label='fit TARTES')
        # plt.legend()
        self.write_in_txt('ke TARTES:',round(self.ke_tartes,4))
        self.write_in_txt('ke théorique:',round(self.ke_theo,4))
        self.write_in_txt('ke stéréologique:',round(self.ke_stereo,4))
        pass
    
    def calculate_alpha(self):
        if not hasattr(self, 'density_theo'): self.calculate_density()
        if not hasattr(self, 'g_theo'): self.calculate_g_theo()
        if not hasattr(self, 'SSA_theo'): self.calculate_SSA()
        if not hasattr(self, 'B_stereo'): self.calculate_B()
        
        #alpha raytracing
        #Intensity top detector
        filt_top = (self.df['hitObj'] == self.find_detector_object()[0])
        I_top_detector = np.sum(self.df[filt_top]['intensity'])
        
        #Rajoute l'intensité contenu dans les erreurs à alpha
        # filt_error = self.df['segmentLevel'] == self.max_segments-1
        # df_error = self.df[filt_error]
        # I_error = np.sum(df_error['intensity']*np.exp(-df_error['z']*self.ke_rt)/2)
        
        #Intensité total
        self.alpha_rt = I_top_detector #+ I_error
        
        #alpha TARTES
        self.alpha_tartes = float(tartes.albedo(self.wlum*1E-6,self.SSA_stereo,self.density_theo,g0=self.g_theo,B0=self.B_stereo,dir_frac=self.tartes_dir_frac))
        
        #alpha stéréologique
        self.gamma = 4*np.pi*self.ice_complex/(self.wlum*1E-6)
        self.alpha_stereo = np.exp(-4*np.sqrt(2*self.B_stereo*self.gamma/(3*self.pice*self.SSA_stereo*(1-self.g_theo))))

        #alpha théorique
        self.alpha_theo = np.exp(-4*np.sqrt(2*self.B_stereo*self.gamma/(3*self.pice*self.SSA_theo*(1-self.g_theo))))

        self.write_in_txt('alpha raytracing:',round(self.alpha_rt,4))
        self.write_in_txt('alpha TARTES:',round(self.alpha_tartes,4))
        self.write_in_txt('alpha stéréologique:',round(self.alpha_stereo,4))
        self.write_in_txt('alpha théorique:',round(self.alpha_theo,4))
        pass
    
    def calculate_mua(self):
        if not hasattr(self, 'B_stereo'): self.calculate_B()
        if not hasattr(self, 'optical_porosity_stereo'): self.calculate_porosity()
        if not hasattr(self, 'ke_rt'): self.calculate_ke_rt()
        if not hasattr(self, 'alpha_rt'): self.calculate_alpha()
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        
        #Calculate mua theorique
        self.gamma = 4*np.pi*self.ice_complex/(self.wlum*1E-6)
        self.mua_stereo = self.B_stereo*self.gamma*(1-self.physical_porosity_stereo)
        
        #Calculate mua with raytracing
        #Using alpha_theo because depend if direct illumination or diffuse
        self.mua_rt = -self.ke_rt*np.log(self.alpha_rt)/4
        
        #Calculate mua with Tartes
        self.mua_tartes = -self.ke_tartes*np.log(self.alpha_tartes)/4
        
        self.write_in_txt('mua théorique:',round(self.mua_stereo,4))
        self.write_in_txt('mua raytracing:',round(self.mua_rt,4))
        self.write_in_txt('mua raytracing:',round(self.mua_tartes,4))

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
    
    def transmitance_at_depth(self,depth):
        filt_detector = self.df['z'] >= depth
        df_filtered = self.df[filt_detector][~self.df[filt_detector].index.duplicated(keep='first')]
        return df_filtered
    
    def calculate_DOP_vs_depth(self,depth):
        df = self.transmitance_at_depth(depth)
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        
        I=np.mean(I)
        Q=np.mean(Q)
        U=np.mean(U)
        V=np.mean(V)
        
        DOP=np.sqrt(Q**2+U**2+V**2)/I
        DOPLT=np.sqrt(Q**2+U**2)/I
        DOPL=np.sqrt(Q**2)/I
        DOP45=np.sqrt(U**2)/I
        DOPC=np.sqrt(V**2)/I
        
        return np.array([DOP, DOPLT, DOPL, DOP45, DOPC])
    
    def calculate_DOP_vs_radius(self,df):
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        
        #Calculate radius from source
        r = np.sqrt((df['x']-self.XY_half_width)**2+(df['y']-self.XY_half_width)**2)
        df.insert(15,'radius',r)
        
        #Calculate Stokes vs Radius
        bins = np.linspace(0,self.XY_half_width,100)
        
        #Histogram of time vs intensity
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
        
        df_DOP.insert(len(df_DOP.columns),'DOP',DOP)
        df_DOP.insert(len(df_DOP.columns),'DOPLT',DOPLT)
        df_DOP.insert(len(df_DOP.columns),'DOPL',DOPL)
        df_DOP.insert(len(df_DOP.columns),'DOP45',DOP45)
        df_DOP.insert(len(df_DOP.columns),'DOPC',DOPC)
        df_DOP[df_DOP.isna()] = 0

        df_DOP.insert(len(df_DOP.columns),'radius',radius[:-1])
        return df_DOP
    
    def calculate_Stokes_xy(self,df):
        [I,Q,U,V] = self.calculate_Stokes_of_rays(df)
        
        #Calculate Stokes vs Radius
        bins = (np.linspace(0,self.XY_half_width*2,200),np.linspace(0,self.XY_half_width*2,200))
        
        #Histogram of time vs intensity
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
    
    def plot_DOP_transmitance(self):
        #Plot datas
        fig, ax = plt.subplots(nrows=2,ncols=2)
        depth = np.linspace(0,self.Depth,100)
        DOPs = np.array(list(map(self.calculate_DOP_vs_depth,depth))) 
        ax[0,0].plot(depth,DOPs[:,0]) #DOP
        ax[0,1].plot(depth,DOPs[:,2]) #DOPL
        ax[1,0].plot(depth,DOPs[:,3]) #DOP45
        ax[1,1].plot(depth,DOPs[:,4]) #DOPC
        
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
    
    def plot_DOP_radius_top_detector(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df = self.df[filt_top_detector]
        
        #Plot datas
        fig, ax = plt.subplots(nrows=2,ncols=2)
        #return dataframe with dops colummns
        df_DOP =  self.calculate_DOP_vs_radius(df)
        ax[0,0].plot(df_DOP['radius'],df_DOP['intensity']) #Intensity
        ax[0,1].plot(df_DOP['radius'],df_DOP['DOPL']) #DOPL
        ax[1,0].plot(df_DOP['radius'],df_DOP['DOP45']) #DOP45
        ax[1,1].plot(df_DOP['radius'],df_DOP['DOPC']) #DOPC
        
        #Set limits
        ax[0,1].set_ylim(0,1.0)
        ax[1,0].set_ylim(0,1.0)
        ax[1,1].set_ylim(0,1.0)
        
        #Set titles
        ax[0,0].set_title('Intensity')
        ax[0,1].set_title('DOPL')
        ax[1,0].set_title('DOP45')
        ax[1,1].set_title('DOPC')
        fig.suptitle('DOP map')
        pass
        
    def map_stokes_reflectance(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df = self.df[filt_top_detector]
        
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
        fig.suptitle('DOPs vs Depth')
        pass
    
    def map_DOP_reflectance(self):
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
    
    def plot_time_reflectance(self):
        if not hasattr(self, 'musp_stereo'): self.calculate_musp()
        if not hasattr(self, 'mua_rt'): self.calculate_mua()
        if not hasattr(self, 'neff_rt'): self.calculate_neff()
        
        #Raytracing
        #Add a line of time for each segment
        self.df.insert(15,'time',np.nan)
        v_medium = scipy.constants.c/self.df['index']
        self.df['time'] = self.df['pathLength']/v_medium
        
        #Time for each ray
        df = self.df.groupby(self.df.index).agg({'time':sum,'intensity':'last','hitObj':'last'})
        
        #Filt only reflectance rays
        filt_top_detector = df['hitObj'] == self.find_detector_object()[0]
        df_top = df[filt_top_detector]
        
        #Histogram of time vs intensity
        R_rt, bins = np.histogram(df_top['time'], weights=df_top['intensity'], bins=1000)
        t_rt = bins[:-1]
        
        #Théorique
        z_o = self.musp_stereo**(-1)
        D = (3*(self.mua_stereo+self.musp_stereo))**(-1)
        t_theo = np.linspace(1E-12,t_rt[-1],100000)
        c = scipy.constants.c/self.neff_stereo
        R_theo = (4*np.pi*D*c)**(-1/2)*z_o*t_theo**(-3/2)*np.exp(-self.mua_stereo*c*t_theo)*np.exp(-z_o**2/(4*D*c*t_theo))
        
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
        plt.xlim(1E-12,max(t_rt)/2)
        pass
    
        
    
    def map_top_detector(self):
        filt_top_detector = self.df['hitObj'] == self.find_detector_object()[0]
        df_top_detector = self.df[filt_top_detector]
        df_top_detector.plot(x='x',y='y',kind='scatter')
        print('Numrays top detector',df_top_detector.shape[0])
        pass
    
    def map_down_detector(self):
        filt_down_detector = self.df['hitObj'] == self.find_detector_object()[1]
        df_down_detector = self.df[filt_down_detector]
        df_down_detector.plot(x='x',y='y',kind='scatter')
        print('Numrays down detector',df_down_detector.shape[0])
        pass
    
    def map_detector(self,depth):
        df = self.transmitance_at_depth(self.df,depth)
        df.plot(x='x', y='y', kind='scatter')
        print('Numrays on detector',df.shape[0])
        pass
    
    def plot_irradiances(self):
        #Irradiance raytracing
        def Irradiance_down(depth):
            df = self.down_irradiance_at_depth(depth)
            irradiance_down = sum(df['intensity']*np.abs(df['n']))
            return irradiance_down
        
        def Irradiance_up(depth):
            df = self.up_irradiance_at_depth(depth)
            irradiance_up = sum(df['intensity']*np.abs(df['n']))
            return irradiance_up
        
        depth = np.linspace(-0.001,25/self.musp_theo,200)
        irradiance_down_rt = np.array(list(map(Irradiance_down,depth)))
        irradiance_up_rt = np.array(list(map(Irradiance_up,depth)))
        
        #Irradiance TARTES
        if not hasattr(self, 'density_stereo'): self.calculate_density()
        if not hasattr(self, 'g_theo'): self.calculate_g_theo()
        if not hasattr(self, 'SSA_stereo'): self.calculate_SSA()
        irradiance_down_tartes, irradiance_up_tartes = tartes.irradiance_profiles(
            self.wlum*1E-6, depth, self.SSA_stereo, density=self.density_stereo,
            g0=self.g_theo,B0=self.B_stereo,dir_frac=self.tartes_dir_frac,totflux=1)
        
        #Plot irradiance down
        plt.figure()
        plt.semilogy(depth,np.exp(-self.ke_rt*depth + self.b_rt),label='fit total irradiance raytracing')
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
    
    def write_in_txt(self,properties='',value=''):
        with open('Properties_Simulations_SSP.txt','a') as f:
            f.write(properties +' '+ str(value)+'\n')
            
    def time(self,text=''):
        t = time.localtime()
        self.date = str(t.tm_year)+"/"+str(t.tm_mon)+"/"+str(t.tm_mday)
        self.ctime = str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)
        print('Date : ',self.date,', Time : ',self.ctime,' ',text)
        
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
        if hasattr(self, 'SSA_stereo'): print('La SSA stéréologie: ' + str(round(self.SSA_stereo,4)))
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
        if hasattr(self, 'mua_stereo'): print('mua stéréologique: ' + str(round(self.mua_stereo,6)))
        if hasattr(self, 'mua_rt'): print('mua raytracing: ' + str(round(self.mua_rt,6)))
        if hasattr(self, 'mua_tartes'): print('mua tartes: ' + str(round(self.mua_tartes,6)))
        if hasattr(self, 'alpha_theo'): print('alpha théorique: ' + str(round(self.alpha_theo,6)))
        if hasattr(self, 'alpha_stereo'): print('alpha stéréologique: ' + str(round(self.alpha_stereo,6)))
        if hasattr(self, 'alpha_rt'): print('alpha raytracing: ' + str(round(self.alpha_rt,6)))
        if hasattr(self, 'alpha_tartes'): print('alpha TARTES: ' + str(round(self.alpha_tartes,6)))
        if hasattr(self, 'ke_stereo'): print('ke stéréologique: ' + str(round(self.ke_stereo,6)))
        if hasattr(self, 'ke_theo'): print('ke théorique: ' + str(round(self.ke_theo,6)))
        if hasattr(self, 'ke_rt'): print('ke raytracing: ' + str(round(self.ke_rt,6)))
        if hasattr(self, 'ke_tartes'): print('ke TARTES: ' + str(round(self.ke_tartes,6)))
        if hasattr(self, 'MOPL_stereo'): print('La MOPL stéréologique: ' + str(round(self.MOPL_stereo,6)))
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
# simulation(name, radius, delta, numrays, numrays_stereo, wlum, pol)
properties=[]
for wlum in [1.0]:
    # sim = simulation('test1', [65E-6], 287E-6, 1000, 100, wlum, [1,1,0,0], diffuse_light=False)
    sim = simulation('test2', [66E-6], 287E-6, 10000, 1000, wlum, [1,1,0,90], diffuse_light=False)
    # sim = simulation('test1', [36.32E-6], 220E-6, 1000, 1000, 1.0, [1,1,0,90], diffuse_light=True)
    sim.create_folder()
    sim.Initialize_Zemax()
    sim.Load_File()
    # sim.create_ZMX()
    # sim.create_source()
    # sim.create_detectors()
    # sim.create_snow()
    # sim.shoot_rays_stereo()
    # sim.shoot_rays()
    sim.Load_npyfile()
    # sim.calculate_g_theo()
    # sim.calculate_g_rt()
    # sim.calculate_B()
    # sim.calculate_SSA()
    # sim.calculate_density()
    # sim.calculate_mus()
    # sim.calculate_mua()
    # sim.calculate_musp()
    # sim.calculate_MOPL()
    # sim.calculate_neff()
    # sim.calculate_porosity()
    # sim.calculate_alpha()
    # sim.calculate_ke_theo()
    # sim.calculate_ke_rt()
    # sim.reflectance_transmitance()
    # sim.map_top_detector()
    # sim.map_down_detector()
    # sim.plot_DOP_radius_top_detector()
    # sim.plot_irradiances()
    # sim.map_DOP_reflectance()
    sim.map_stokes_reflectance()
    # sim.plot_time_reflectance()
    sim.properties()
    # properties += [[sim.mua_stereo,sim.alpha_rt,sim.alpha_stereo,sim.MOPL_stereo,sim.MOPL_rt,sim.Error]]
    # del sim