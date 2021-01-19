# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:15:51 2020

@author: jplan58
"""
import Monte_Carlo_Class as MC
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.close('all')
    properties=[]

    #Selecting parameters
    # radius=[66E-6,88E-6,176E-6,88E-6,88E-6]
    # deltas=[287E-6,382.66E-6,765.33E-6,347.7E-6,482.1E-6]
    radius=[88E-6,176E-6,88E-6,88E-6]
    deltas=[382.66E-6,765.33E-6,347.7E-6,482.1E-6]
    # radius=[88E-6,176E-6,88E-6,88E-6]
    # deltas=[382.66E-6,765.33E-6,347.7E-6,482.1E-6]
    shapes = zip(radius,deltas)
    wlums=[1.0]
    # wlums=[0.8,0.9]
    # pol_vector = [[1,0,0,0]]
    # pol_vector = [[1,1,0,90]]
    pol_vector = [[1,1,0,90],[1,0,0,0]]
    
    #Simulating for choseen parameters
    for shape in shapes:
        for wlum in wlums:
            for pol in pol_vector:
                sim = MC.simulation_MC('test4_mc_msp', 100_000, shape[0], shape[1], 0.89, wlum, pol, diffuse_light=False)
                print(sim.inputs)
                # sim.create_directory()
                sim.Load_File()
                sim.shoot_rays()
                sim.Close_Zemax()
                sim.Load_parquetfile()
                sim.AOP()
                sim.calculate_mus()
                sim.calculate_ke_theo()
                sim.calculate_MOPL()
                sim.calculate_alpha()
                sim.calculate_mua()
                sim.calculate_ke_rt()
                sim.map_stokes_reflectance()
                sim.map_DOP_reflectance()
                sim.plot_DOP_radius_reflectance()
                sim.plot_DOP_transmitance()
                sim.plot_irradiances()
                sim.plot_MOPL_radius_reflectance()
                sim.plot_lair_radius_reflectance()
                sim.export_properties()
                # sim.properties()
                del sim
