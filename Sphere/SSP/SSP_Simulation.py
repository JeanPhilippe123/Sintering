"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""
#Import module
import numpy as np
import pandas as pd
import clr, os, sys
import time
from win32com.client import CastTo, constants
import matplotlib.pyplot as plt
#Import other files
pathRaytraceDLL = 'C:\Zemax Files\ZOS-API\Libraries\Raytrace.DLL'
sys.path.insert(1,os.path.dirname(os.path.realpath(pathRaytraceDLL)))
import PythonNET_ZRDLoaderFull as init_Zemax
import SSP_Raytrace
import matplotlib

class simulation:
    #Constants
    Interactionslimit = 4000
    Usepol = True
    pice = 917
    NumModels = 1
    path = os.path.join(os.sep, os.path.dirname(os.path.realpath(__file__)), '')
    
    def __init__(self,name,radius,dist,numrays,numrays_SSA,wlum,pol):
        ''' 
        \r name = Nom donné à la simulation \r
        radius = Rayon des sphères \r
        dist = Distances entre les centres des sphères \r
        Numrays = Nombre de rayons pour la simulation pour g et B (list array)\r
        Numrays_SSA = Nombre de rayons pour la simulation SSA et B (list array)\r
        wlum = Nombre de rayons pour chaqune des simulations (list array)\r
        pol = Polarization entrante pour chaqune des simulations
        '''
        self.name = name
        self.radius = np.array(radius)
        self.num_spheres = len(radius)
        self.dist = dist
        self.numrays = numrays
        self.numrays_SSA = numrays_SSA
        self.wlum = wlum
        self.jx,self.jy,self.phasex,self.phasey = np.array(pol)
        
        if self.num_spheres == 1:
            self.Source_radius = self.radius[0]
        elif self.num_spheres == 2:
            self.Source_radius = (self.radius[0] + self.radius[1] + self.dist)/2
            
        #Writing in the text file properties
        self.write_in_txt()
        self.write_in_txt("======================================================")
        self.write_in_txt('Simulation :', name)
        self.write_in_txt('Radius :', radius)
        self.write_in_txt('Distance :', dist)
        self.write_in_txt('Numrays :', numrays)
        self.write_in_txt('Wavelength :', wlum)
        self.write_in_txt('Polarization :', pol)

        t = time.localtime()
        date = str(t.tm_year)+"/"+str(t.tm_mon)+"/"+str(t.tm_mday)
        ctime = str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)
        self.write_in_txt('Time started', date + " " + ctime)
        self.write_in_txt()

    def create_folder(self):
        self.pathZMX = os.path.join(self.path,'Simulations',self.name)
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
                # self.list_null += [i]
        # map(self.TheNCE.RemoveObjectAt,self.list_null)
    
    def array_objects(self):
        list_obj=[]
        for i in range(1,self.TheNCE.NumberOfObjects+1):
            Object = self.TheNCE.GetObjectAt(i)
            ObjectType = Object.TypeName
            ObjectMaterial = Object.Material
            # print(i, ObjectType)
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
        
        self.fileZMX = os.path.join(self.pathZMX,'g_calculator_' + str(self.radius) + '.zmx')
        
    def Create_ZMX(self): 
        self.TheSystem.New(False)
        self.TheSystem.SaveAs(self.fileZMX)
        self.TheSystem.MakeNonSequential()
        self.TheSystem.SystemData.Units.LensUnits = self.ZOSAPI.SystemData.ZemaxSystemUnits.Meters
        self.TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = self.wlum
        self.TheSystem.SystemData.NonSequentialData.MaximumIntersectionsPerRay = 4000
        self.TheSystem.SystemData.NonSequentialData.MaximumSegmentsPerRay = 4000
        self.TheSystem.SystemData.NonSequentialData.MaximumNestedTouchingObjects = 8
        self.TheSystem.SystemData.NonSequentialData.SimpleRaySplitting = True
        self.TheSystem.SystemData.NonSequentialData.GlueDistanceInLensUnits = 1.0000E-10
        self.TheSystem.SaveAs(self.fileZMX)
        print('Fichier Créer')
    
    def Load_File(self):
        self.TheSystem.LoadFile(self.fileZMX,False)
        self.TheNCE = self.TheSystem.NCE
        print('Fichier loader')
        self.write_in_txt('File Loaded')
        self.write_in_txt('')
        
    def create_2_spheres(self):
        #Créer la forme
        if self.num_spheres == 1:
            sphere_ZPosition = [0]
        elif self.num_spheres == 2:
            sphere_ZPosition = [self.dist/2,-self.dist/2]
        else:
            print('Update code to create more than 2 spheres')
            return
        
        for i in range(0,self.num_spheres):
            self.TheNCE.InsertNewObjectAt(1)
            Object = self.TheNCE.GetObjectAt(1)
            Type = Object.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.Sphere)
            Object.ChangeType(Type)
            Object.Material = 'MY_ICE.ZTG'
            Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).DoubleValue = self.radius[i]
            Object.XPosition = 0 #sphere_ZPosition[i]
            Object.YPosition = 0 #sphere_ZPosition[i]
            Object.ZPosition = sphere_ZPosition[i]

        self.delete_null()
        self.TheSystem.SaveAs(self.fileZMX)
        
    def create_source(self):
        #Créer la source avec un rectangle et 2 sphères
        self.TheNCE.InsertNewObjectAt(1)
        
        Source_source = self.TheNCE.GetObjectAt(1)
        
        # Type_Source_source = Source_source.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.SourceEllipse)
        # Source_source.ChangeType(Type_Source_source)
        Type_Source_source = Source_source.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.SourceDLL)
        Source_source.ChangeType(Type_Source_source)
        Source_source.Comment = 'intern_sphere.dll'
        Source_source.TiltAboutX = 90
        # Source_source.TiltAboutY = 0
        # Source_source.TiltAboutZ = 0
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).IntegerValue = 100 #Layout Rays
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par6).DoubleValue = 3.0 * self.Source_radius #Radius
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par7).DoubleValue = self.Source_radius #Apparent maximum radius
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par8).DoubleValue = 180
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par9).DoubleValue = 180
        Source_source.DrawData.DoNotDrawObject = True

        self.delete_null()
        self.TheSystem.SaveAs(self.fileZMX)

    def create_detector(self):
        #Créer le détecteur
        self.TheNCE.InsertNewObjectAt(1)
        Object = self.TheNCE.GetObjectAt(1)
        Type = Object.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.DetectorPolar)
        Object.ChangeType(Type)
        Object.Material = 'ABSORB'
            
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par2).DoubleValue = 6.0 * self.Source_radius
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par3).IntegerValue = 10
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par4).IntegerValue = 12
        
        self.delete_null()
        self.TheSystem.SaveAs(self.fileZMX)
        pass
    
    def Create_filter_Ice(self):
        #Create filter that only consider rays touching spheres
        list_ice_filter = []
        Sphere_ice_obj = self.find_ice_object()
        detector_obj = np.where(self.array_objects() == np.array(['Detector Polar']))[0][0] + 1
        for i in Sphere_ice_obj:
            list_ice_filter += ["H"+str(i)]
        self.filter_Ice = "("+"|".join(list_ice_filter)+")" + "&L"+str(detector_obj)
        return None
    
    def find_ice_object(self):
        Sphere_ice_obj = np.where(self.array_objects() == np.array(['MY_ICE.ZTG']))[0] + 1
        return Sphere_ice_obj

    def shoot_rays(self):
        self.Create_filter_Ice()
        
        print('Raytrace')
        self.npyPath = SSP_Raytrace.Shoot(self,self.name,self.filter_Ice,self.numrays)
        
        #Load the npy file into a dataframe
        self.df = SSP_Raytrace.Load_npy(self.npyPath)
        
    def shoot_rays_SSA(self):
        #Créer un filtre pour la glace
        self.Create_filter_Ice()
        
        #Change les grains de neige en air (pour que les rayons n'intéragissent pas)
        Sphere_ice_obj = self.find_ice_object()
        for i in Sphere_ice_obj:
            Object = self.TheNCE.GetObjectAt(i)
            Object.Material = ''
            
        self.TheSystem.SaveAs(self.fileZMX)
        
        #Calcul la stéréologie et retourne un fichier npy
        print('Raytrace SSA')
        self.npyPath_SSA = SSP_Raytrace.Shoot(self,self.name+'_stereo_SSA',self.filter_Ice,self.numrays_SSA)
        
        #Rechange les grains d'air en neige
        for i in Sphere_ice_obj:
            Object = self.TheNCE.GetObjectAt(i)
            Object.Material = 'MY_ICE.ZTG'
         
        self.TheSystem.SaveAs(self.fileZMX)
    
    def filter_ice_air(self,df):
        Sphere_ice_obj = self.find_ice_object()
        if self.num_spheres == 1:
            filt_inter_ice = (df['hitObj'] == Sphere_ice_obj[0]) & (df['insideOf'] == 0)
            filt_ice = (df['insideOf'] == Sphere_ice_obj[0])
            # filt_air = (df['insideOf'] == 0)
            return filt_ice, filt_inter_ice
        elif self.num_spheres == 2:    
            filt_inter_ice = ((df['hitObj'] == Sphere_ice_obj[0]) | (df['hitObj'] == Sphere_ice_obj[1])) & (self.df_stereo['insideOf'] == 0)
            filt_ice = (df['insideOf'] == Sphere_ice_obj[0]) | (df['insideOf'] == Sphere_ice_obj[1])
            # filt_air = (df['insideOf'] == 0)
            return filt_ice, filt_inter_ice
        else:
            print('Update code for calculating stereo_SSA with more than 2 spheres')
            return 
        
    def stereo_SSA(self):
        #Calcul de la stéréologie et retourner fichier npy
        self.shoot_rays_SSA()
        
        #Load the npy file into a dataframe
        self.df_stereo = SSP_Raytrace.Load_npy(self.npyPath_SSA)
        
        self.filt_ice, self.filt_inter_ice = self.filter_ice_air(self.df_stereo)
        
        #Length of ice
        numrays = self.df_stereo.index[-1]
        self.l_Ice_stereo = np.sum(self.df_stereo.loc[self.filt_ice, 'pathLength'])/numrays
        #Inter Air/Ice
        Num_ice_inter = self.df_stereo[self.filt_inter_ice].shape[0]/numrays
        lengthIce_per_seg = self.l_Ice_stereo/Num_ice_inter
        self.SSA_stereo = round(4/(self.pice*lengthIce_per_seg),6)
        self.write_in_txt('SSA stéréologie :', self.SSA_stereo)
        
        #SSA théorique
        pice=917
        if self.num_spheres == 1:
            Vol_sphere = 4*np.pi*self.radius[0]**3/3
            Air_sphere = 4*np.pi*self.radius[0]**2
        elif self.num_spheres == 2:
            #Calcul du Volume et de l'air à enlever
            if self.dist <= self.radius[0]+self.radius[1]:
                #Volume à enlever
                r_1=self.radius[0]
                r_2=self.radius[0]
                d=self.dist
                V_inter=np.pi*(r_1+r_2-d)**2*(d**2+2*d*r_2-3*r_2**2+2*d*r_1+6*r_2*r_1-3*r_1**2)/(12*d)
                
                # Air à enlever
                # x_1 et x_2: distance entre le centre de la sphère et le point de contact en x
                x_1 = (d**2-r_1**2+r_2**2)/(2*d)
                x_2 = (d**2+r_1**2-r_2**2)/(2*d)
                h_1 = r_1 - x_1
                h_2 = r_2 - x_2
                Air_inter_1 = 2*np.pi*r_1*h_1
                Air_inter_2 = 2*np.pi*r_2*h_2
                #Formule trouvé:
                #https://undergroundmathematics.org/circles/cutting-spheres/
                #solution#:~:text=We%20have%20a%20section%20of,n%3D2%CF%80rh.
            else:
                V_inter = 0
                Air_inter_1 = 0
                Air_inter_2 = 0
            
            Vol_sphere_1 = 4*np.pi*self.radius[0]**3/3
            Vol_sphere_2 = 4*np.pi*self.radius[1]**3/3
            
            Air_sphere_1 = 4*np.pi*self.radius[0]**2
            Air_sphere_2 = 4*np.pi*self.radius[1]**2
            
            Vol_sphere = Vol_sphere_1 + Vol_sphere_2 - V_inter
            Air_sphere = Air_sphere_1 + Air_sphere_2 - Air_inter_1 - Air_inter_2
        else:
            print('Update code for num_spheres > 2')
        
        self.SSA_theo = round(Air_sphere/(Vol_sphere*pice),6)
        self.write_in_txt('SSA théorique :', self.SSA_stereo)
        
    def calculate_B(self):
        self.filt_ice, self.filt_inter_ice = self.filter_ice_air(self.df)
        #Length of ice per rays
        numrays = self.df.index[-1]
        self.l_Ice_p = np.sum(self.df.loc[self.filt_ice]['pathLength'])/numrays
        
        #Calculate B
        self.B = self.l_Ice_p/self.l_Ice_stereo
        self.write_in_txt('B :',round(self.B,4))
        
    def calculate_g(self):
        #Return angles vs Intensity
        df_angles = self.phase_function() #angles
        filt_notna = df_angles['Angle'].notna()
        self.gG = np.average(np.cos(df_angles[filt_notna]['Angle']),weights=df_angles[filt_notna]['intensity'])
        self.g = (self.gG+1)/2
        self.write_in_txt('g :',round(self.g,4))
        self.write_in_txt('gG :',round(self.gG,4))
    
    def phase_function(self):
        #Return angles and intensity for each rays
        filt_source = self.df['segmentLevel'] == 0
        filt_detector = self.df['hitObj'] == (np.where(self.array_objects() == ['Detector Polar'])[0][0]+1)
        
        v1 = self.df[filt_source][['l','m','n']]
        v2 = self.df[filt_detector][['l','m','n','intensity']]
        v_12 = v1[['l','m','n']]*v2[['l','m','n']]
        dot_product = v_12['l']+v_12['m']+v_12['n']
        
        df_angles = pd.DataFrame(np.arccos(dot_product))
        df_angles.columns = ['Angle']
        df_angles['intensity'] = v2['intensity']
        
        return df_angles
    
    def plot_phase_function(self):
        df_angles = self.phase_function()
        fig, ax = plt.subplots()
        df_angles['Angle'].hist(ax=ax, bins=100,bottom=0.1,weights=df_angles['intensity'])
        ax.set_yscale('log')
    
    def write_in_txt(self,properties='',value=''):
        with open('Properties_Simulations_SSP.txt','a') as f:
            f.write(properties +' '+ str(value)+'\n')
            
    def properties(self):
        print('Path: ',self.path)
        try:
            print('Simulation path files: ', self.path)
        except NameError: pass

        try:
            print('Simulation path ZMX files: ', self.fileZMX)
        except AttributeError:
            print('No Simulation files path')

        try:
            print('Les B: ' + str(round(self.B,4)))
        except AttributeError: pass
        try:
            print('Les g: ' + str(round(self.g,6)))
        except AttributeError: pass
        try:
            print('Les gG: ' + str(round(self.gG,10)))
        except AttributeError: pass
        try:
            print('Les SSA stereo : ' + str(self.SSA_stereo))
        except AttributeError: pass
        try:
            print('Les SSA théorique : ' + str(self.SSA_theo))
        except AttributeError: pass

    def __del__(self):
        try:
            self.TheSystem.SaveAs(self.fileZMX)
            del self.zosapi
            self.zosapi = None
        except Exception:
            pass

plt.close('all')
# (name,radius,dist,numrays,numrays_SSA,wlum,pol)
sim = simulation('test3', [65E-6], 0.0, 50000, 50000, 1.0, [0.1,0.3,30,90])
sim.create_folder()
sim.Initialize_Zemax()
# sim.Load_File()
sim.Create_ZMX()
sim.create_detector()
sim.create_source()
sim.create_2_spheres()
sim.array_objects()
# print(sim.dict_obj)
sim.stereo_SSA()
sim.shoot_rays()
sim.calculate_B()
sim.calculate_g()
# sim.plot_phase_function()
# sim.calculate_B()
sim.properties()
# plt.close('all')
# del sim