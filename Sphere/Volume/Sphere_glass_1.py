# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:15:51 2020

@author: jplan58
"""
import Sphere_Class as Sc
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    for pol in [[1,1,0,90],[1,0,0,0]]:
        sim = Sc.Sphere_Simulation('test_labo_v1', 3000E-6, 8500E-6, 1_000, 100, 0.633, pol, diffuse_light=False, sphere_material='B270')
        # sim.Load_File()
        sim.create_ZMX()
        sim.create_source()
        sim.create_detectors()
        sim.create_medium()
        sim.shoot_rays_stereo()
        sim.shoot_rays()
        sim.Close_Zemax()
        # sim.Load_parquetfile()
        # sim.AOP()
        # sim.calculate_g_theo()
        # sim.calculate_g_rt()
        # sim.calculate_B()
        # sim.calculate_SSA()
        # sim.calculate_density()
        # sim.calculate_mus()
        # sim.calculate_mua()
        # sim.calculate_musp()
        # sim.calculate_MOPL()
        # sim.calculate_neff()
        # sim.calculate_porosity()
        # sim.calculate_alpha()
        # sim.calculate_ke_theo()
        # sim.calculate_ke_rt()
        # sim.map_stokes_reflectance()
        # sim.plot_DOP_radius_reflectance()
        # sim.map_DOP_reflectance()
        # sim.plot_DOP_transmitance()
        # sim.plot_irradiances()
        # sim.plot_MOPL_radius_reflectance()
        # sim.plot_lair_radius_reflectance()
        # sim.properties()
        # sim.export_properties()