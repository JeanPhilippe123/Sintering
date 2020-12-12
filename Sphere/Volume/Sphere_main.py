# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:15:51 2020

@author: jplan58
"""
import Sphere_Class as Sc
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    plt.close('all')
    properties=[]
    
    name= 'test3_sphere'
    #Selecting parameters
    radius=[66E-6,88E-6,176E-6,88E-6,88E-6]
    deltas=[287E-6,382.66E-6,765.33E-6,347.7E-6,482.1E-6]
    # radius=[88E-6,176E-6,88E-6,88E-6]
    # deltas=[382.66E-6,765.33E-6,347.7E-6,482.1E-6]
    shapes = zip(radius,deltas)
    wlums=[1.0]
    # wlums=[1.0]
    pol_vector = [[1,1,0,90],[1,0,0,0]]
    properties_predict=[]
    
    #Simulating for choseen parameters
    for shape in shapes:
        for wlum in wlums:
            for pol in pol_vector:
                sim = Sc.Sphere_Simulation(name,shape[0],shape[1], 10_000, 100,wlum,pol,Random_pol=False,diffuse_light=False)
                if not os.path.exists(sim.fileZMX):
                    sim.create_ZMX()
                    sim.create_source()
                    sim.create_detectors()
                    sim.create_snow()
                else:
                    sim.Load_File()
                if not os.path.exists(sim.path_stereo_parquet):                    
                    sim.shoot_rays_stereo()
                sim.shoot_rays()
                sim.Close_Zemax()
                sim.Load_parquetfile()
                sim.AOP()
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
                # sim.plot_DOP_radius_top_detector()
                # sim.map_DOP_reflectance()
                # sim.plot_irradiances()
                # sim.plot_MOPL_radius_reflectance()
                # sim.plot_lair_radius_reflectance()
                sim.properties()
                sim.export_properties()
                del sim
