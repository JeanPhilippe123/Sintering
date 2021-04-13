# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:15:51 2020

@author: jplan58
"""
import Monte_Carlo_Class as MC
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # pol_vector = [[1,0,0,0],[1,1,0,90]]
    pol_vector = [[1,0,0,0]]
    for pol in pol_vector:
        sim = MC.simulation_MC('simulation_labo', 10_000, 1500E-6, 4250E-6, 0.823, 0.633, pol, diffuse_light=False, sphere_material='B270')
        # print(sim.inputs)
        # sim.create_ZMX()
        # sim.shoot_rays()
        # sim.Close_Zemax()
        sim.Load_parquetfile()
        # sim.AOP()
        # sim.calculate_mus()
        # sim.calculate_ke_theo()
        # sim.calculate_MOPL()
        # sim.calculate_alpha()
        # sim.calculate_mua()
        # sim.calculate_ke_rt()
        # sim.map_stokes_reflectance()
        # sim.map_DOP_reflectance()
        # sim.plot_DOP_radius_reflectance()
        # sim.plot_DOP_transmitance()
        sim.plot_stokes_transmitance(filt_source=True)
        # sim.plot_irradiances(filt=source_radius)
        # sim.plot_MOPL_radius_reflectance()
        # sim.plot_lair_radius_reflectance()
        # sim.export_properties()
        # sim.properties()
        # del sim