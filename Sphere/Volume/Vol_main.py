from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache
import time
import MakeSnowSphereND
import numpy as np
#import MakeSnowCylinder
import MakeSimulationSphereND
import DensitySphereCalculatorInfiniteND
import os

if __name__ == '__main__':
###########Parameters##############
###### À définir comme liste pour la première boucle (doivent tous avoir la même longueur) #######

    New = [True, True, True, True, True, True, True, True, True, True, True, True]
    
    #Aspect Ratio est défini comme étant hauteur/2*base
    radius=np.array([55E-6,150E-6,350E-6,150E-6,150E-6])
    
    #Nombre d'hexagone différents
    #Profondeur. (largueur est 5 x profondeur)
    Depth = [0.05,0.05,0.05,0.05,0.05]
    
    #Espace entre chaque hexagone
    #Minimum Delta/2=hauteur
    DeltaX = [155.5635E-6,424.2641E-6,989.9495E-6,534.539E-6,673.47637E-6]
    DeltaY = DeltaX
    DeltaZ = DeltaY
    
    #Nombre d'intéraction maximal
    Interactionslimit=[4000,4000,4000,4000,4000]
    
    #Nombre de rayons lancés pour chaque polarisation
    Numraysl=[1,10000]
    
####À définir comme liste ou pas pour le lancement de rayon ou c'est un paramètre fixe###########    
    dossier='Sphere7ND'
    
    jxl = [1,1]
    jyl = [0,1]
    phasexl = [0,0]
    phaseyl = [0,-90]
    filtl = '!Z'
    RandomPl = [False]
    Usepol = True
    wl=0.633
    pice=917 #Kg/m**3
    g=0.787
    
    NumArrayX = 10
    NumArrayY = 3
    NumArrayZ = 3
    
    NumModels = 1
    
    #Création du dossier dans portal
    if not os.path.exists("\\\\usagers.copl.ulaval.ca\\usagers\\jplan58\\Desktop\\Portal\\" + str(dossier) + "\\"):
        os.makedirs("\\\\usagers.copl.ulaval.ca\\usagers\\jplan58\\Desktop\\Portal\\" + str(dossier) + "\\")
        print("Le répertoire " + str(dossier) +  " a été créé")
    else:
        print("Le répertoire " + str(dossier) +  " existe déjà")
    
    #Calcul densité
    Densitél=np.zeros(len(radius))
    for i in range(0,len(radius)):
        Densitél[i] = DensitySphereCalculatorInfiniteND.DensityFinder(radius[i], DeltaX[i], DeltaY[i], DeltaZ[i], pice)
    
    #SSA CALCULATION
    #Récupération de la hauteur et de la base dans le nom de fichier
    Volume=4*np.pi*radius**3/3
    #base*2+6*côtés
    Aire=4*np.pi*radius**2
    
    Masse=Volume*pice
    SSA=Aire/Masse
    mus=SSA*Densitél/2/1000 #mm**-1
    musp=SSA*Densitél/2*(1-g)/1000 #mm**-1
    
    
    Q=2
    pair=1.225
    pice=917
    fw=pice/(pair/Densitél+pice) #Densitél=Vice/Vair
    muss=3*Q*pair*fw/(4*radius*pice)
    
    for i in range(0,len(radius)):
        print('  SSA :' + str(SSA[i]),'  Densité volumique :'+str(Densitél[i]/pice))
        print(' Mus :'+str(mus[i]),' Musp :'+str(musp[i]),' Muss :'+str(muss[i]),'\n')
        
    for i in range(0,len(radius)):
        path= "\\\\usagers2.copl.ulaval.ca\\usagers\\jplan58\\"
        filezmx='Sphere '+str(round(Densitél[i],4))+' '+str(Depth[i])+' '+str(round(SSA[i],3))+' '+str(radius[i])
        print(filezmx)
        #Noms des fichiers de sorties de polarisation
        filezrdl = ['test','Sphere_C_'+str(Depth[i])+'_'+str(round(SSA[i],3))+'_'+str(round(Densitél[i],4))+'_'+str(radius[i])]
        
        #Temps du début de la simulation
        start = time.time()
        st = time.localtime()
        print('Simulation commencé à:', st.tm_hour,st.tm_min,st.tm_sec)

        print('ok1')
        if i == 0:
            #Initiation de OpticStudio
            zosapi = PythonStandaloneApplication()
            TheSystem = zosapi.TheSystem
            TheApplication = zosapi.TheApplication
    
        NewFile = path + filezmx + ".zmx"
        
        if New[i] == True:
            #Création du fichier
            TheSystem.New(False)
            TheSystem.SaveAs(NewFile)
            TheSystem.MakeNonSequential()
            sysUnits = TheSystem.SystemData.Units
            sysUnits.LensUnits = constants.ZemaxSystemUnits_Meters
            
            density = MakeSnowSphereND.MakeSnowGreatAgain(TheSystem,radius[i],
                                                             NumModels,Depth[i],DeltaX[i],DeltaY[i],DeltaZ[i],
                                                             NumArrayX,NumArrayY,NumArrayZ,Numraysl[0],pice)
            print('La densité est: ',density)
            TheSystem.SaveAs(NewFile)
            end = time.time()
            print('La création de fichier a pris: ' + str(end - start))
        else:
            TheSystem.LoadFile(NewFile,False)
            end = time.time()
            print('Loader le fichier a pris: ' + str(end - start))
        TheSystem.SaveAs(NewFile)
        MakeSimulationSphereND.MakeSimulation(TheSystem,path,filezmx,filezrdl,Interactionslimit[i],Numraysl,jxl,jyl,phasexl,phaseyl,
                       filtl,RandomPl,Usepol,wl,dossier)
        TheSystem.SaveAs(NewFile)
        print('ok2')
    del zosapi
    zosapi = None