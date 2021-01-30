import SSP_Simulation as SIM
import numpy as np

""" 
Sert uniquement à gérer SSP_Simulation
Gérer les simulations avec la for loop
"""

from SSP_Simulation import simulation

if __name__ == '__main__':
    #Inputs :
    radius = [176E-6,176E-6]
    delta = np.linspace(radius[0]/100,2*radius[0],100).round(6)
    pols = [[1,0,0,0],[0,1,0,0],[1,1,0,0]]
    
    for i in range(0,len(delta)):
        for pol in pols:
            # (name,radius,Delta,numrays,numrays_stereo,wlum,pol)
            sim = simulation('Sintering1', radius, delta[i], 1_000_000, 1_000_000, 0.78, pol, Random_Pol=False)
            print('Delta :', delta[i])
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
            # sim.properties()
            sim.export_properties()
            plt.close('all')
            del sim