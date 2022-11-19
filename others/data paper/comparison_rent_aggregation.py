# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:33:41 2021

@author: charl
"""



import pandas as pd
import numpy as np
import copy
import pickle
import scipy as sc
from scipy import optimize
import matplotlib.pyplot as plt
import os
import math

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"

#path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20210430/"
#os.mkdir(path_calibration)

option = {}
option["model_type"] = "calibrated" #schematic
option["validation"] = 0
option["add_residuals"] = True
option["do_calibration"] = False
#List of cities
list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

#Parameters and city characteristics
INTEREST_RATE = 0.05
HOUSEHOLD_SIZE = 1
TIME_LAG = 2
DEPRECIATION_TIME = 100
DURATION = 20
CARBON_TAX = 0
COEFF_URB = 0.62
FIXED_COST_CAR = 0
WALKING_SPEED = 5

if option["validation"] == 1:
    d_beta = {}
    d_b = {}
    d_kappa = {}
    d_Ro = {}
    r2rent = {}
    r2density = {}
    r2size = {}
    d_share_car = {}
    d_share_walking = {}
    d_share_transit = {}
    d_emissions_per_capita = {}
    d_utility = {}
    
for city in np.unique(list_city.City):
    
    print("\n*** " + city + " ***\n")

    ### IMPORT DATA
    
    print("\n** Import data **\n")

    #Import city data
    (country, density, rents_and_size, land_use,
     driving, transit, grille, centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city)        

    plt.scatter(distance_cbd, rents_and_size.medRent, c = "red", s = 2)
    plt.scatter(distance_cbd, rents_and_size.avgRent, c = "blue", s = 2)
    plt.scatter(distance_cbd, rents_and_size.regRent, c = "green", s = 2)
    plt.legend(["median", "avg", "reg"])
    plt.savefig('C:/Users/charl/OneDrive/Bureau/rent_data/' + city + '.png')
    plt.close()