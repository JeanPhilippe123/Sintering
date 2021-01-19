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
    
    for i in range(79,len(delta)):
        # (name,radius,Delta,numrays,numrays_stereo,wlum,pol)
        sim = simulation('Sintering1', [176E-6,176E-6], delta[i], 1_000_000, 1_000_000, 0.76, [1,1,0,90])
        print('Delta :', delta[i])
        # sim = simulation('test4', [176E-6], 0, 100_000, 100_000, 0.76, [1,1,0,0])
        # sim.create_directory()
        # sim.Load_File()
        sim.Create_ZMX()
        sim.create_detector()
        sim.create_source()
        sim.create_spheres()
        sim.shoot_rays_SSA()
        sim.shoot_rays()
        sim.Close_Zemax()
        sim.calculate_SSA()
        sim.calculate_B()
        sim.calculate_g()
        sim.plot_phase_function()
        # sim.properties()
        sim.export_properties()
        # plt.close('all')
        del sim