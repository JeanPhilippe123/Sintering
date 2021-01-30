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
import matplotlib
import miepython

pathRaytraceDLL = 'Z:\Sintering\Glass_Catalog'
sys.path.insert(2,os.path.realpath(pathRaytraceDLL))
import Glass_Catalog

path_Sphere_Raytrace = 'Z:\Sintering\Sphere\Volume'
sys.path.insert(3,path_Sphere_Raytrace)
import Sphere_Raytrace

class simulation:
    #Constants
    Interactionslimit = 4000
    Usepol = True
    pice = 917
    NumModels = 1
    path = os.path.join(os.sep, os.path.dirname(os.path.realpath(__file__)), '')
    
    def __init__(self,name,radius,Delta,numrays,numrays_stereo,wlum,pol,Random_Pol=False,sphere_material='MY_ICE.ZTG'):
        ''' 
        \r name = Nom donné à la simulation \r
        radius = Rayon des sphères \r
        Delta = Distances entre les centres des sphères \r
        Numrays = Nombre de rayons pour la simulation pour g et B (list array)\r
        numrays_stereo = Nombre de rayons pour la simulation SSA et B (list array)\r
        wlum = Nombre de rayons pour chaqune des simulations (list array)\r
        pol = Polarization entrante pour chaqune des simulations
        '''
        self.name = name
        self.sphere_material = sphere_material
        self.radius = np.array(radius)
        self.num_spheres = len(radius)
        self.Delta = Delta
        self.numrays = numrays
        self.numrays_stereo = numrays_stereo
        self.wlum = wlum
        self.Random_Pol = Random_Pol
        self.jx,self.jy,self.phase_x,self.phase_y = np.array(pol)
        self.pathDatas = os.path.join(self.path,'Simulations',self.name)
        
        mat = Glass_Catalog.Material(self.sphere_material)
        self.index_real,self.index_imag = mat.get_refractive_index(self.wlum)
        self.index = complex(self.index_real,self.index_imag)
        self.size_parameter = 2*np.pi*self.radius/(self.wlum*1E-6)
        # self.index_real, self.index_imag = tartes.refice2016(self.wlum*1E-6)
        # self.gamma = 4*np.pi*self.index_imag/(self.wlum*1E-6)
        
        if self.num_spheres == 1:
            self.Source_radius = self.radius[0]
        elif self.num_spheres == 2:
            self.Source_radius = (self.radius[0] + self.radius[1] + self.Delta)/2
            
        self.properties_string = '_'.join([name,str(numrays),str(tuple(radius)),str(Delta),str(tuple(pol))])
        self.properties_string_stereo = '_'.join([name,'stereo',str(numrays_stereo),str(tuple(radius)),str(Delta),str(tuple(pol))])
        self.properties_string_plot = '_'.join([self.properties_string,str(self.wlum)])
        self.name_ZRD = self.properties_string + ".ZRD"
        self.name_stereo_ZRD = self.properties_string_stereo + ".ZRD"
        
        self.path_parquet_stereo = os.path.join(self.pathDatas,'_'.join([self.properties_string_stereo,str(self.wlum)]) + "_.parquet")
        self.path_parquet = os.path.join(self.pathDatas,'_'.join([self.properties_string,str(self.wlum)]) + "_.parquet")
        self.path_metadata = os.path.join(self.pathDatas,'_'.join([self.properties_string,'metadata'])+ ".npy")
        self.path_ZRD = os.path.join(self.pathDatas,self.name_ZRD)
        self.fileZMX = os.path.join(self.pathDatas, '_'.join(['SSP',self.properties_string])+'.zmx')
        self.path_plot = os.path.join(self.pathDatas,'Results_plots')
            
        self.add_properties_to_dict('radius',self.radius)
        self.add_properties_to_dict('Delta',self.Delta)
        self.add_properties_to_dict('Pol',pol)
        self.add_properties_to_dict('numrays',self.numrays)
        self.add_properties_to_dict('numrays',self.numrays_stereo)
        self.add_properties_to_dict('wlum',self.wlum)
        
        #Printing starting time
        t = time.localtime()
        self.date = str(t.tm_year)+"/"+str(t.tm_mon)+"/"+str(t.tm_mday)
        self.ctime = str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)
        print('Date : ',self.date,', Time : ',self.ctime)
   
    def find_source_object(self):
        Source_obj = np.where(self.array_objects() == np.array(['Source DLL']))[0] + 1
        return Source_obj
    
    def find_detector_object(self):
        Detector_obj = np.where(self.array_objects() == np.array(['Detector Polar']))[0] + 1
        return Detector_obj
    
    def create_directory(self):
        if not os.path.exists(self.pathDatas):
            os.makedirs(self.pathDatas)
            os.makedirs(self.path_plot)
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
    
    def update_metadata(self):
        #Write metadata
        np.save(self.path_metadata,self.get_array_objects_zemax())
    
    def get_array_objects_zemax(self):
        list_obj=[]
        for i in range(1,self.TheNCE.NumberOfObjects+1):
            Object = self.TheNCE.GetObjectAt(i)
            ObjectType = Object.TypeName
            ObjectMaterial = Object.Material 
            list_obj += [[ObjectType,ObjectMaterial]]
            array_obj = np.array(list_obj)
        return array_obj
            
    def array_objects(self):
        #Check for metadata
        if os.path.exists(self.path_metadata):
            array_obj = np.load(self.path_metadata)
        #Check for Zemax open
        elif hasattr(self, 'zosapi'):
            array_obj = self.get_array_objects_zemax()
        #Else return error
        else:
            print('There\'s no metadata file to get array object from and ZOSAPI is not open.\
                  Please initialize Zemax') 
            sys.exit()
        return array_obj
    
    def Initialize_Zemax(self):
        self.create_directory()
        self.zosapi = init_Zemax.PythonStandaloneApplication()
        self.BatchRayTrace = self.zosapi.BatchRayTrace 
        self.TheSystem = self.zosapi.TheSystem
        self.TheApplication = self.zosapi.TheApplication
        self.TheNCE = self.TheSystem.NCE
        
        self.ZOSAPI = self.zosapi.ZOSAPI
        self.ZOSAPI_NCE = self.ZOSAPI.Editors.NCE
        
    def Create_ZMX(self):
        self.Initialize_Zemax()
        self.TheSystem.New(False)
        self.TheSystem.SaveAs(self.fileZMX)
        self.TheSystem.MakeNonSequential()
        self.TheSystem.SystemData.Units.LensUnits = self.ZOSAPI.SystemData.ZemaxSystemUnits.Meters
        self.TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = self.wlum
        self.TheSystem.SystemData.Polarization.Method = self.ZOSAPI.SystemData.PolarizationMethod.ZAxisMethod
        self.TheSystem.SystemData.NonSequentialData.MaximumIntersectionsPerRay = 4000
        self.TheSystem.SystemData.NonSequentialData.MaximumSegmentsPerRay = 4000
        self.TheSystem.SystemData.NonSequentialData.MaximumNestedTouchingObjects = 8
        self.TheSystem.SystemData.NonSequentialData.SimpleRaySplitting = True
        self.TheSystem.SystemData.NonSequentialData.GlueDistanceInLensUnits = 1.0000E-10
        self.TheSystem.SaveAs(self.fileZMX)
        print('Fichier Créer')
    
    def Load_File(self):
        self.Initialize_Zemax()
        self.TheSystem.LoadFile(self.fileZMX,False)
        self.TheNCE = self.TheSystem.NCE
        print('Fichier loader')
        
    def create_spheres(self):
        #Créer la forme
        if self.num_spheres == 1:
            sphere_ZPosition = [0]
        elif self.num_spheres == 2:
            sphere_ZPosition = [self.Delta/2,-self.Delta/2]
        else:
            print('Update code to create more than 2 spheres')
            return
        
        for i in range(0,self.num_spheres):
            self.TheNCE.InsertNewObjectAt(1)
            Object = self.TheNCE.GetObjectAt(1)
            Type = Object.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.Sphere)
            Object.ChangeType(Type)
            Object.Material = self.sphere_material
            Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).DoubleValue = self.radius[i]
            Object.XPosition = 0 #sphere_ZPosition[i]
            Object.YPosition = 0 #sphere_ZPosition[i]
            Object.ZPosition = sphere_ZPosition[i]
        
        self.delete_null()
        self.update_metadata()
        self.TheSystem.SaveAs(self.fileZMX)
        
    def create_source(self):
        #Créer la source avec un rectangle et 2 sphères
        self.TheNCE.InsertNewObjectAt(1)
        
        Source_source = self.TheNCE.GetObjectAt(1)
        
        # Type_Source_source = Source_source.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.SourceEllipse)
        # Source_source.ChangeType(Type_Source_source)
        Type_Source_source = Source_source.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.SourceDLL)
        Source_source.ChangeType(Type_Source_source)
        # Source_source.Comment = 'line_ponderate_source.dll'
        Source_source.Comment = 'intern_sphere_line_v22.dll'
        # Source_source.ZPosition = -self.Source_radius*1.1
        # Source_source.TiltAboutX = 0
        # Source_source.TiltAboutY = 0
        # Source_source.TiltAboutZ = 0
        
        Source_source.SourcesData.RandomPolarization = self.Random_Pol
        Source_source.SourcesData.Jx = self.jx
        Source_source.SourcesData.Jy = self.jy
        Source_source.SourcesData.XPhase = self.phase_x
        Source_source.SourcesData.YPhase = self.phase_y

        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par1).IntegerValue = 10000 #Layout Rays
        # Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par6).DoubleValue = self.Source_radius
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par6).DoubleValue = self.Source_radius*3 #Radius
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par7).DoubleValue = self.Source_radius  #Apparent maximum radius
        # Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par7).DoubleValue = self.Source_radius+self.Source_radius-self.Delta #Apparent maximum radius
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par8).DoubleValue = 360
        Source_source.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par9).DoubleValue = 180
        Source_source.DrawData.DoNotDrawObject = True

        self.delete_null()
        self.update_metadata()
        self.TheSystem.SaveAs(self.fileZMX)

    def create_detector(self):
        #Créer le détecteur
        self.TheNCE.InsertNewObjectAt(1)
        Object = self.TheNCE.GetObjectAt(1)
        Type = Object.GetObjectTypeSettings(self.ZOSAPI_NCE.ObjectType.DetectorPolar)
        Object.ChangeType(Type)
        Object.Material = 'ABSORB'
            
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par2).DoubleValue = self.Source_radius*100 #Radius
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par3).IntegerValue = 10 #Pixel radial
        Object.GetObjectCell(self.ZOSAPI_NCE.ObjectColumn.Par4).IntegerValue = 12 #Pixel angulaire
        
        self.delete_null() 
        self.update_metadata()
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
        Sphere_ice_obj = np.where(self.array_objects() == np.array([self.sphere_material]))[0] + 1
        return Sphere_ice_obj

    def shoot_rays(self):
        self.Create_filter_Ice()
        
        print('Raytrace')
        Sphere_Raytrace.Shoot(self,'',self.numrays,self.path_parquet,self.name_ZRD)
        
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
        Sphere_Raytrace.Shoot(self,'',self.numrays,self.path_parquet_stereo,self.name_stereo_ZRD)
        
        #Rechange les grains d'air en neige
        for i in Sphere_ice_obj:
            Object = self.TheNCE.GetObjectAt(i)
            Object.Material = self.sphere_material
         
        self.TheSystem.SaveAs(self.fileZMX)
        
    def Load_parquetfile(self):
        #Try loading the ray file
        #Check if file at the specified wavelength exist..

        # Load stereo files
        if os.path.exists(self.path_parquet_stereo):
            self.df_stereo = Sphere_Raytrace.Load_parquet(self.path_parquet_stereo).compute()
        else:
            print('The raytrace stereo parquet file was not loaded, Please run raytrace')
            sys.exit()

        if os.path.exists(self.path_parquet):
            self.df = Sphere_Raytrace.Load_parquet(self.path_parquet).compute()
        else:
            print('The raytrace parquet file was not loaded, Please run raytrace')
            sys.exit()
        pass
    
    def filter_ice_air(self,df):
        Sphere_ice_obj = self.find_ice_object()
        if self.num_spheres == 1:
            filt_inter_ice = (df['hitObj'] == Sphere_ice_obj[0]) & (df['insideOf'] == 0)
            filt_ice = (df['insideOf'] == Sphere_ice_obj[0])
            return filt_ice, filt_inter_ice
        
        elif self.num_spheres == 2:
            filt_inter_ice = ((df['hitObj'] == Sphere_ice_obj[0]) | (df['hitObj'] == Sphere_ice_obj[1])) & (self.df_stereo['insideOf'] == 0)
            filt_ice = (df['insideOf'] == Sphere_ice_obj[0]) | (df['insideOf'] == Sphere_ice_obj[1])
            return filt_ice, filt_inter_ice
        
        else:
            print('Update code for calculating stereo_SSA with more than 2 spheres')
            return 
        
    def calculate_SSA(self):
        self.filt_ice, self.filt_inter_ice = self.filter_ice_air(self.df_stereo)
        
        #Length of ice
        numrays = self.df_stereo.index[-1]
        self.l_Ice_stereo = np.sum(self.df_stereo.loc[self.filt_ice, 'pathLength'])/numrays
        #Inter Air/Ice
        Num_ice_inter = self.df_stereo[self.filt_inter_ice].shape[0]/numrays
        lengthIce_per_seg = self.l_Ice_stereo/Num_ice_inter
        self.SSA_stereo = round(4/(self.pice*lengthIce_per_seg),6)
        self.add_properties_to_dict('SSA_stereo',self.SSA_stereo)
        
        #SSA théorique
        pice=917
        if self.num_spheres == 1:
            Vol_sphere = 4*np.pi*self.radius[0]**3/3
            Air_sphere = 4*np.pi*self.radius[0]**2
        elif self.num_spheres == 2:
            #Calcul du Volume et de l'air à enlever
            if self.Delta <= self.radius[0]+self.radius[1]:
                #Volume à enlever
                r_1=self.radius[0]
                r_2=self.radius[0]
                d=self.Delta
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
        self.add_properties_to_dict('SSA_theo',self.SSA_theo)
        
    def calculate_B(self):
        if not hasattr(self, 'l_Ice_stereo'): self.calculate_SSA()
        
        self.filt_ice, self.filt_inter_ice = self.filter_ice_air(self.df)
        #Length of ice per rays
        numrays = self.df.index[-1]
        self.l_Ice_p = np.sum(self.df.loc[self.filt_ice]['pathLength'])/numrays
        
        #Calculate B
        self.B = self.l_Ice_p/self.l_Ice_stereo
        self.add_properties_to_dict('B',self.B)
        
    def calculate_g(self):
        #Return angles vs Intensity
        filt_source = self.df['hitObj'] == self.find_source_object()[0]
        filt_detector = self.df['hitObj'] == self.find_detector_object()[0]
        Source = self.df.loc[filt_source]
        Detector = self.df.loc[filt_detector]
        
        df_angles = self.pol_phase_function(Source,Detector) #angles
        filt_notna = df_angles['Angle'].notna()
        self.gG = np.average(np.cos(df_angles[filt_notna]['Angle']),weights=df_angles[filt_notna]['intensity'])
        self.g = (self.gG+1)/2
        
        Qext, Qsca, Qback, g = miepython.mie(self.index,self.size_parameter)
        self.g_mie = g[0]
        self.add_properties_to_dict('g',self.g)
        self.add_properties_to_dict('g mie',self.g_mie)
        self.add_properties_to_dict('gG',self.gG)
        
    def pol_phase_function(self,Source,Detector):
        def basis_change(df):
            #Correction à la polarisation pour que les axes de polarisations (changement de base
            #Vectors
            v1=np.array([df.L,df.M,df.N])
            v2=np.cross([df.L_source,df.M_source,df.N_source],v1,axis=0)
            v3=np.cross(v1,v2,axis=0)
            
            #Normalise
            v1=v1/np.linalg.norm(v1,axis=0)
            v2=v2/np.linalg.norm(v2,axis=0)
            v3=v3/np.linalg.norm(v3,axis=0)
            
            #Matrice de changement de base pour des vecteurs (i,j,k) (sphérique) vers (x,y,z) (cartésien)
            spherical_to_cartesian=np.array([v2,v3,v1]).transpose()
            
            #Matrice de changement de base pour des vecteur (x,y,z) (cartésien) vers (i,j,k) (sphérique)
            cartesian_to_spherical=np.linalg.inv(spherical_to_cartesian)
            
            #Changement de base de b2i vers polV
            Ex=df.exr+1j*df.exi
            Ey=df.eyr+1j*df.eyi
            Ez=df.ezr+1j*df.ezi
            jones_vector = np.array([Ex,Ey,Ez]).transpose()
            pol=np.zeros([len(jones_vector),3],dtype=complex)
            for i in range(0,len(jones_vector)):
                pol[i]=np.dot(cartesian_to_spherical[i],jones_vector[i])
            # print(cartesian_to_spherical)
            # print(pol)
            return pol
        
        df_angles = pd.DataFrame()
        df_angles['mu'] = np.sum(Source[['L','M','N']]*Detector[['L','M','N']],axis=1)
        df_angles['Angle'] = np.arccos(round(df_angles.mu,7)) #round -> Remove float inacurracy
        df_angles['intensity'] = Detector['intensity']
        
        Source = Source.rename(columns={'L':'L_source','N':'N_source','M':'M_source'})
        df = Detector.join(Source[['L_source','M_source','N_source']])
        
        start = time.time()
        pol = basis_change(df).transpose()
        # pol = np.array(df.apply(basis_change,axis=1).tolist()).transpose()
        end = time.time()
        print(end-start)
        
        df_angles['Ex'] = pol[0]
        df_angles['Ey'] = pol[1]
        df_angles['Ez'] = pol[2]
        return df_angles
    
    def calculate_Stokes_of_rays(self,df):
        if len(df.index) == 0:
            #Return empty sequence
            return np.zeros(4)
        
        #Vecteur de polarization dans le plan othogonaux
        Ex = np.array(df.Ex)
        Ey = np.array(df.Ey)
        
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

        df['I'] = list(I)
        df['Q'] = list(Q)
        df['U'] = list(U)
        df['V'] = list(V)
        return df
    
    def calculate_DOP(self,Stokes):
        [I,Q,U,V] = Stokes
        #Create Dataframe for the DOPs
        df_DOP = pd.DataFrame(data=np.transpose([I]),columns = ['intensity'])

        #Remove division by 0 by filter NaNs (Q,U and V are 0 anyway)
        I_df = df_DOP['intensity'] #Remove divide by zero exception with using df instead of array
        
        DOP=np.sqrt(Q**2+U**2+V**2)/I_df
        DOPLT=np.sqrt(Q**2+U**2)/I_df
        DOPL=Q/I_df
        DOP45=U/I_df
        DOPC=V/I_df
        return [DOP, DOPLT, DOPL, DOP45, DOPC]
    
    def calculate_hist_stokes(self,df,bins):
        I, mu = np.histogram(df['mu'], bins=np.linspace(-1,1,bins), weights=df['I'])
        Q, mu = np.histogram(df['mu'], bins=np.linspace(-1,1,bins), weights=df['Q'])
        U, mu = np.histogram(df['mu'], bins=np.linspace(-1,1,bins), weights=df['U'])
        V, angle = np.histogram(df['mu'], bins=np.linspace(-1,1,bins), weights=df['V'])
        return mu[:-1], [I,Q,U,V]
    
    def get_stokes(self,bins):
        if not hasattr(self, 'df_stokes'):
            filt_source = self.df['hitObj'] == self.find_source_object()[0]
            filt_detector = self.df['hitObj'] == self.find_detector_object()[0]
            Source = self.df.loc[filt_source]
            Detector = self.df.loc[filt_detector]
            
            df = self.pol_phase_function(Source,Detector)
            self.df_stokes = self.calculate_Stokes_of_rays(df)
            self.df_stokes.dropna(inplace=True)
        
        #Calulate Stokes
        mu, Stokes = self.calculate_hist_stokes(self.df_stokes,bins)
        
        return mu, Stokes
    
    def plot_phase_function_stokes(self,bins=100):
        if not hasattr(self, 'g'): self.calculate_g()
        
        mu, [I,Q,U,V] = self.get_stokes(bins)
        fig, ax = plt.subplots(nrows=2,ncols=2)
        [[ax_I,ax_Q],[ax_U,ax_V]] = ax
        fig.suptitle('Phase function (Stokes)')
        
        ax_I.plot(mu,I)
        ax_Q.plot(mu,Q)
        ax_U.plot(mu,U)
        ax_V.plot(mu,V)
        
    def plot_intensities(self,bins=100):
        if not hasattr(self, 'g'): self.calculate_g()
            
        fig,ax = plt.subplots()
        mu,I_rt = self._plot_rt_intensity(ax,bins)
        mu,[I_mie,I_par_mie,I_per_mie] = self._plot_mie_intensity(ax,bins)
        mu,I_hg = self._plot_hg_intensity(self.g,ax,bins)
        ax.legend()

        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_plot_intensities.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_plot_intensities.npy')
        self.create_npy(path_npy,mu=mu,I_rt=I_rt,I_par_mie=I_par_mie,I_per_mie=I_per_mie,I_mie=I_mie,I_hg=I_hg)
        return
        
    def _plot_rt_intensity(self,ax,bins):
        mu, [I,_,_,_] = self.get_stokes(bins)

        ax.plot(mu,I,label = 'Intensity Raytracing')
        return mu,I
        
    def _plot_mie_intensity(self,ax,bins):
        mu,[I_par,I_per],[S11,_,_,_] = self.miepython_intensities(bins=bins)
        I = (I_par+I_per)/2
        
        #Normalize
        ax.semilogy(mu,I_par, label = 'Mie Ipar')
        ax.semilogy(mu,I_per, label = 'Mie Iper')
        ax.semilogy(mu,I, label = 'Mie S11')
        return mu,[I,I_par,I_per]
    
    def _plot_hg_intensity(self,g,ax,bins):
        def hg(g,cosangle):
            return (1/(4*np.pi))*(1-g**2)/(1+g**2-2*g*cosangle)**(3/2)
        
        mu = np.linspace(-1,1,bins)
        intensity_hg = hg(g,mu)
        
        ax.semilogy(mu,intensity_hg, label = 'Henyey-Greenstein')
        return mu,intensity_hg

    def plot_phase_function_DOP(self,bins=100):
        mu, Stokes = self.get_stokes(bins)
        DOPs = self.calculate_DOP(Stokes)
        DOP, DOPLT, DOPL, DOP45, DOPC = DOPs
        
        fig, ax = plt.subplots(nrows=2,ncols=2)
        [[ax_DOP,ax_DOPL],[ax_DOP45,ax_DOPC]] = ax
        fig.suptitle('Phase function (DOPs)')

        ax_DOP.plot(mu,DOP)
        ax_DOPL.plot(mu,DOPL)
        ax_DOP45.plot(mu,DOP45)
        ax_DOPC.plot(mu,DOPC)

        ax_DOP.set_ylim(-1,1)
        ax_DOPL.set_ylim(-1,1)
        ax_DOP45.set_ylim(-1,1)
        ax_DOPC.set_ylim(-1,1)
        
        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_phase_function_DOP.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_phase_function_DOP.npy')
        self.create_npy(path_npy,mu=mu,DOP=DOP,DOPL=DOPL,DOP45=DOP45,DOPC=DOPC)
        
    def plot_scattering_matrix_mie(self,bins=100):
        mu, [_, _], [S11,S12,S33,S34] = self.miepython_intensities(bins)

        fig, ax = plt.subplots(nrows=2,ncols=2)
        [[ax_S11,ax_S12],[ax_S33,ax_S34]] = ax
        fig.suptitle('Scattering matrix mie (DOPs)')
        
        ax_S11.semilogy(mu,S11)
        ax_S12.plot(mu,S12/S11)
        ax_S33.plot(mu,S33/S11)
        ax_S34.plot(mu,S34/S11)
        ax_S11.set_title('S11',x=0.2,y=0.8)
        ax_S12.set_title('S12',x=0.2,y=0.8)
        ax_S33.set_title('S33',x=0.2,y=0.8)
        ax_S34.set_title('S34',x=0.2,y=0.8)
    
        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_plot_scattering_matrix_mie.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_plot_scattering_matrix_mie.npy')
        self.create_npy(path_npy,mu=mu,S11=S11,S12=S12,S33=S33,S34=S34)
        
    def miepython_intensities(self,bins):
        bins -= 1 
        
        mu = np.linspace(-1,1,bins)
        x = 2*np.pi*self.radius[0]/(self.wlum*1E-6)
        S1,S2 = miepython.mie_S1_S2(self.index, x, mu)
        qext, qsca , qback, g = miepython.mie(self.index,x)
        # norm = np.sqrt(qext * x**2 * np.pi)
        S1_ = S1
        S2_ = S2
        
        S11 = (abs(S2_)**2+abs(S1_)**2)/2
        S12 = (abs(S2_)**2-abs(S1_)**2)/2
        S33 = (S2_*S1_.conjugate()).real
        S34 = (S2_*S1_.conjugate()).imag
        
        return mu,[abs(S1)**2,abs(S2)**2],[S11,S12,S33,S34]
    
    def plot_DOPs_source(self):
        filt_source = self.df.hitObj == self.find_source_object()[0]
        df_source = self.df.loc[filt_source]
        
        #Calculate Source output polarization
        df = df_source.assign(mu = np.sum(df_source[['L','M','N']]*[0,0,1],axis=1))
        df['Ex'] = df.exr+1j*df.exi
        df['Ey'] = df.eyr+1j*df.eyi
        df['Ez'] = df.ezr+1j*df.ezi
        
        print('Pol',df['exr'].mean(),df['eyr'].mean(),df['ezr'].mean())
        print('Angle',df['L'].mean(),df['M'].mean(),df['N'].mean())
        
        # Detector
        # Detector = df_source
        # Detector.L = 0
        # Detector.M = 0
        # Detector.N = 1
        
        # df = self.pol_phase_function(df_source,Detector)
        df = self.calculate_Stokes_of_rays(df)
        mu,Stokes = self.calculate_hist_stokes(df,100)
        DOP = self.calculate_DOP(Stokes)
        
        plt.figure()
        plt.plot(mu,DOP[0],label='DOP')
        plt.plot(mu,DOP[1],label='DOPLT')
        plt.plot(mu,DOP[2],label='DOPL')
        plt.plot(mu,DOP[3],label='DOP45')
        plt.plot(mu,DOP[4],label='DOPC')
        plt.legend()
        return
    
    def plot_source_output(self):
        filt_source = self.df['hitObj'] == self.find_source_object()[0]
        df_source = self.df[filt_source]
        
        bins = np.linspace(-self.Source_radius,self.Source_radius,100)
        num_x,x = np.histogram(df_source['x'],bins=bins)
        num_y,y = np.histogram(df_source['y'],bins=bins)
        
        fig, ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(x[:-1]/self.Source_radius,num_x,label='x')
        ax[1].plot(y[:-1]/self.Source_radius,num_y,label='y')
        ax[0].set_xlabel('x coordinate normalized')
        ax[0].set_ylabel('% of rays')
        ax[1].set_xlabel('y coordinate normalized')
        ax[1].set_ylabel('% of rays')
        plt.legend()
        
        #Save Datas to npy file
        plt.savefig(os.path.join(self.path_plot,self.properties_string_plot+'_plot_source_output.png'),format='png')
        path_npy = os.path.join(self.path_plot,self.properties_string_plot+'_plot_source_output.npy')
        self.create_npy(path_npy,x=x[:-1],y=y[:-1],num_x=num_x,num_y=num_y,source_radius=self.Source_radius)

    def add_properties_to_dict(self,key,value):
        if not hasattr(self,'dict_properties'):
            self.dict_properties = {key:value}
        else:
            self.dict_properties[key] = value
    
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

if __name__=='__main__':
    plt.close('all')
    # (name,radius,Delta,numrays,numrays_stereo,wlum,pol)
    sim = simulation('test5', [150E-6], 100E-6, 1_000_000, 1_000_000, 0.76, [0,1,0,0], Random_Pol=False)
    # sim = simulation('test4', [176E-6], 0, 100_000, 100_000, 0.76, [1,1,0,0])
    sim.Create_ZMX()
    sim.create_detector()
    sim.create_source()
    sim.create_spheres()
    sim.shoot_rays_SSA()
    sim.shoot_rays()
    sim.Close_Zemax()
    sim.Load_parquetfile()
    sim.calculate_SSA()
    sim.calculate_B()
    sim.calculate_g()
    sim.plot_phase_function_stokes()
    sim.plot_phase_function_DOP()
    sim.plot_intensities(bins=100)
    # sim.plot_scattering_matrix_mie(bins=100)
    # sim.plot_source_output()
    # sim.plot_DOPs_source()
    sim.properties()
    # sim.export_properties()
    # plt.close('all')
    # del sim