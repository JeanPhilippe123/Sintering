import SSP_Simulation as SIM
import numpy as np

""" 
Sert uniquement à gérer SSP_Simulation
Gérer les simulations avec la for loop
"""
if __name__ == '__main__':
    #Inputs :
    Inputs={'Name': np.array(['test3']),
            'Radius': np.array([[0.0003]]),
            'Distance': np.array([0.1]),
            'Numrays': np.array([1000]),
            'Numrays SSA': np.array([1000]),
            'Wavelength': np.array([1.0]),
            'Polarization': np.array([[1,1,0,0]])}
    
    for i in range(0,len(radius))
    plt.close('all')
    # (name,radius,Delta,numrays,numrays_stereo,wlum,pol)
    sim = simulation('test4', [176E-6,176E-6], 100E-6, 100_000, 100_000, 0.76, [1,1,0,90])
    # sim = simulation('test4', [176E-6], 0, 100_000, 100_000, 0.76, [1,1,0,0])
    # sim.create_directory()
    # sim.Load_File()
    sim.Create_ZMX()
    sim.create_detector()
    sim.create_source()
    sim.create_spheres()
    sim.shoot_rays_SSA()
    sim.shoot_rays()
    sim.calculate_SSA()
    sim.calculate_B()
    sim.calculate_g()
    sim.plot_phase_function()
    sim.properties()
    # plt.close('all')
    # del sim