# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:15:51 2020

@author: jplan58
"""
import miepython
import numpy as np
import sys

import Monte_Carlo_Class as MC

path_SSP = 'Z:\Sintering\Sphere\SSP'
sys.path.insert(2,path_SSP)
import SSP_Simulation as SSP



if __name__ == '__main__':
    pol_vector = [[1,0,0,0],[1,1,0,90]]
    wlums = [1.1,1.3,1.5]
    g=[]
    B=[]
    for wlum in wlums:
        # Calcul le B et g pour des indices de réfraction
        sim = SSP.simulation('sim_article_B_g_vs_pol_v2', [500E-6], 0E-6, 1_000_000, 1_000_000, wlum, [1,1,0,90], Random_Pol=True, sphere_material='CUSTOM_INDEX.ZTG')
        sim.Create_ZMX()
        sim.create_detector()
        sim.create_source()
        sim.create_object()
        sim.shoot_rays_stereo()
        sim.shoot_rays()
        sim.Close_Zemax()
        sim.Load_parquetfile()
        # sim.plot_stokes_ref_source()
        # sim.plot_DOP_ref_source()
        # sim.get_stokes_ref_source()
        sim.calculate_SSA()
        sim.calculate_B()
        sim.calculate_g()
        # sim.plot_phase_function_stokes()
        # sim.plot_phase_function_DOP()
        # sim.plot_intensities(bins=100)
        # sim.plot_scattering_matrix_mie(bins=100)
        # sim.plot_source_output()
        # sim.properties()
        sim.export_properties()
        g = round(sim.g,2)
        B = round(sim.B,2)
        print(g,B)
        for pol in pol_vector:
            sim = MC.simulation_MC('sim_article_B_g_vs_pol_v2', 100_000, 500E-6, 1415E-6, g, wlum, pol, B=B, diffuse_light=False, sphere_material='CUSTOM_INDEX.ZTG')
            print(sim.wlum,sim.index_real,sim.index_imag)
            sim.create_ZMX()
            sim.shoot_rays()
            sim.Close_Zemax()
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
            sim.plot_stokes_transmitance()
            sim.plot_DOP_transmitance()
            # sim.plot_irradiances(filt=source_radius)
            # sim.plot_MOPL_radius_reflectance()
            # sim.plot_lair_radius_reflectance()
            sim.export_properties()
            # sim.properties()
            # del sim