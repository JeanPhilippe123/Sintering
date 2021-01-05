# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:15:51 2020

@author: jplan58
"""
import Monte_Carlo_Class as MC
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pol_vector = [[1,0,0,0],[1,1,0,90]]
    for pol in pol_vector:
        sim = MC.simulation_MC('test3_mc_msp_B270', 100_000, 150E-6, 425E-6, 0.89, 0.76, pol, diffuse_light=False, sphere_material='B270')
        print(sim.inputs)
        sim.create_directory()
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
        sim.properties()
        del sim
