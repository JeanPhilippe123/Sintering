import SSP_Simulation as SIM
import numpy as np

""" 
Sert uniquement à gérer SSP_Simulation
Gérer les simulations avec la for loop
"""
if __name__ == '__main__':
    #Inputs :
    Inputs={'Name': np.array(['test3','test3']),
            'Radius': np.array([[0.0003],[0.0003]]),
            'Distance': np.array([0.1,0.1]),
            'Numrays': np.array([1000,1000]),
            'Numrays SSA': np.array([1000,1000]),
            'Wavelength': np.array([0.199,0.199]),
            'Polarization': np.array([[0.1,0.3,30,90],[0.1,0.3,30,90]])}
    
    for i in range(0,len(Inputs['Name'])):
        sim = SIM.simulation(Inputs['Name'][i],Inputs['Radius'][i],
              Inputs['Distance'][i],Inputs['Numrays'][i],
              Inputs['Numrays SSA'][i],Inputs['Wavelength'][i],
              Inputs['Polarization'][i])
        sim.create_folder()
        sim.Initialize_Zemax()
        # sim.Load_File()
        sim.Create_ZMX()
        sim.create_detector()
        sim.create_source()
        sim.create_2_spheres()
        sim.array_objects()
        sim.stereo_SSA()
        sim.shoot_rays()
        sim.calculate_g()
        sim.plot_phase_function()
        # sim.calculate_B()
        sim.properties()
        # del sim