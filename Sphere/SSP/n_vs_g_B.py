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
    pol = [1,1,0,90]
    wlums=np.linspace(1.0,1.5,100)
    # wlums=[wlums[i] for i in [4,9,15,21,26,30,31,32]]
    # 4
    for i,wlum in enumerate(wlums):
        # (name,radius,Delta,numrays,numrays_stereo,wlum,pol)
        sim = simulation('test_bg_3', [500E-6], 0, 1_000_000, 1_000_000, wlum, pol, Random_Pol=False, sphere_material='CUSTOM_MATERIAL')
        sim.Create_ZMX()
        sim.create_detector()
        sim.create_source()
        sim.create_object()
        sim.shoot_rays_stereo()
        sim.shoot_rays()
        sim.Close_Zemax()
        print('number : ',i)
        # sim.Load_parquetfile()
        # sim.plot_stokes_ref_source()
        # sim.plot_DOP_ref_source()
        # sim.calculate_SSA()
        # sim.calculate_B()
        # sim.calculate_g()
        # sim.plot_phase_function_stokes()
        # sim.plot_phase_function_DOP()
        # sim.plot_intensities(bins=100)
        # sim.export_properties()
        # sim.properties()
        # plt.close('all')
        # del sim