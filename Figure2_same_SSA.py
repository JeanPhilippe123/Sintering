"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""

"""
Comparing DOPL,DOPC, l_air and MOPL for different porosity and SSA while keeping the same \mu_s
"""

#Import important libraries
import numpy as np
import matplotlib.pyplot as plt

#Close all previous opened plots
plt.close('all')

"""For the same SSA"""
#Random paths
path_mc = "/Users/JP/Documents/Python/Sintering_test/test_MC_datas.npy"
path_spheres = "/Users/JP/Documents/Python/Sintering_test/test_Spheres_datas.npy"

#Save random datas
x_mc = np.random.rand(10)
y_mc = np.random.rand(10)
x_sp = np.random.rand(10)
y_sp = np.random.rand(10)
np.save(path_mc,[x_mc,y_mc])
np.save(path_spheres,[x_sp,y_sp])
    
#Import datas
[x_mc,y_mc] = np.load(path_mc)
[x_sp,y_sp] = np.load(path_spheres)

#Create Figures and plot datas
fig,ax = plt.subplots(2,2,num=1)
ax[0,0].plot(x_mc,y_mc,'r.-',label = 'Monte-Carlo for $SSA_1$')
ax[0,0].plot(x_mc,y_mc,'r.-',label = 'Monte-Carlo for $SSA_2$')
#...
ax[0,0].plot(x_sp,y_sp,'b.-',label = 'Spheres for $SSA_2$')
#...
ax[0,1].plot(x_mc,y_mc,'r.-',label = 'Monte-Carlo for $SSA_2$')
#...
ax[0,1].plot(x_sp,y_sp,'b.-',label = 'Spheres for $SSA_2$')
#...
ax[1,0].plot(x_mc,y_mc,'r.-',label = 'Monte-Carlo for $SSA_3$')
#...
ax[1,0].plot(x_sp,y_sp,'b.-',label = 'Spheres for $SSA_3$')
#...
ax[1,1].plot(x_mc,y_mc,'r.-',label = 'Monte-Carlo for $SSA_4$')
#...
ax[1,1].plot(x_sp,y_sp,'b.-',label = 'Spheres for $SSA_4$')
#...

#Plot legend
ax[0,0].legend()

#Set y limit
ax[0,0].set_ylim(0,1)
ax[0,1].set_ylim(0,1)

#Set titles and labels
ax[0,0].set_title('DOPL vs Radius')
ax[0,1].set_title('DOPC vs Radius')
ax[1,0].set_title('MOPL vs Radius')
ax[1,1].set_title('$l_{air}$ vs Radius')
ax[0,0].set_ylabel('DOPL')
ax[0,0].set_xlabel('Radius (mm)')
ax[0,1].set_ylabel('DOPC')
ax[0,1].set_xlabel('Radius (mm)')
ax[1,0].set_ylabel('MOPL')
ax[1,0].set_xlabel('Radius (mm)')
ax[1,1].set_ylabel('$l_{air}$')
ax[1,1].set_xlabel('Radius (mm)')
fig.suptitle('Optical vs physical parameter')
