""" 
Sert uniquement à gérer SSP_Simulation
Gérer les simulations avec la for loop
"""

import numpy as np
from SSP_Simulation import simulation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    #Inputs :
    radius = [176E-6,176E-6]
    delta = np.linspace(radius[0]/100,2*radius[0],100).round(6)
    pols = [[1,0,0,0],[1,1,0,90]]
    # pols = [[1,0,0,0],[0,1,0,0],[1,1,0,0]]
    # delta = [0.]
    for i in range(0,len(delta)):
        for pol in pols:
            # (name,radius,Delta,numrays,numrays_stereo,wlum,pol)
            sim = simulation('t4', radius, delta[i], 1_000_000, 1_000_000, 1.0, pol, Random_Pol=False)
            # print('Delta :', delta[i])
            # sim.Create_ZMX()
            # sim.create_detector()
            # sim.create_source()
            # sim.create_object()
            # sim.shoot_rays_SSA()
            # sim.shoot_rays()
            # sim.Close_Zemax()
            sim.Load_parquetfile()
            sim.plot_stokes_ref_source()
            sim.plot_DOP_ref_source()
            sim.calculate_SSA()
            sim.calculate_B()
            sim.calculate_g()
            sim.plot_phase_function_stokes()
            sim.plot_phase_function_DOP()
            sim.plot_intensities(bins=100)
            sim.export_properties()
            sim.properties()
            plt.close('all')
            del sim