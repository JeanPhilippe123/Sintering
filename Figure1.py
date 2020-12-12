"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""

"""
Intensity vs Radius from source to confirm result from the Monte-Carlo
Intensity is in log
"""

#Import important libraries
import numpy as np
import os
import matplotlib.pyplot as plt

class plots:
    def __init__(self,name,numrays,radius,Delta,g,wlum,pol,Random_pol=False,diffuse_light=False):
        self.Random_pol = Random_pol
        self.diffuse_light = diffuse_light
        self.name = name
        self.numrays = numrays
        self.radius = radius
        self.Delta = Delta
        self.g = g
        self.wlum = wlum
        self.pol = pol
        self.path_main = 
        self.path_mc = 
    def find_path(self):
    def Load_data(self):
    def irradiance_plot(self):
    def map_stokes_reflectance(self):
    def map_DOPs_reflectance(self):
        
#Close all previous opened plots
plt.close('all')

#Path for datas plots
path_main = os.path.dirname(os.path.realpath(__file__))
path_mc = os.path.join()
# path_spheres = 

#Import datas from Monte-Carlo
[x_mc,y_mc] = np.load(path_mc)
#Import datas from Spheres datas
[x_sp,y_sp] = np.load(path_spheres)

#Create Figures and plot datas
fig,ax = plt.subplots(1,1)
ax.plot(x_mc,y_mc,'r.-',label = 'Monte-Carlo')
ax.plot(x_sp,y_sp,'b.-',label = 'Spheres')

#Plot legend
ax.legend()

#set titles and labels
ax.set_title('DOPL')
ax.set_ylabel('log(Intensity)')
ax.set_xlabel('Radius (mm)')
fig.suptitle('Confirmation du Monte-Carlo')