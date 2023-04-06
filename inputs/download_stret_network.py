# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:07:52 2022

@author: charl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:26:26 2021

@author: charl
"""


from inputs.transport import *
from inputs.parameters import *
from outputs.outputs import *
from model.model import *
from inputs.land_use import *
from inputs.geo_data import *
from inputs.data import *
from calibration.validation import *
from calibration.calibration import *
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math
import os


BRT_scenario = 'speed_40_0_12_50_5'

option_ugb = "predict_urba" #density, data_urba, predict_urba. Résultats actuels avec predict_urba

FUEL_EFFICIENCY_DECREASE = 0.963
BASELINE_EFFICIENCY_DECREASE = 0.99
LIFESPAN = 15

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20211124/" #calibration_20211124
#os.mkdir(path_calibration)
path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/street_network/"
#os.mkdir(path_outputs)

option = {}
option["validation"] = 0
option["add_residuals"] = True
option["do_calibration"] = False
policy = 'BRT'

#List of cities
list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

#Parameters and city characteristics
INTEREST_RATE, HOUSEHOLD_SIZE, TIME_LAG, DEPRECIATION_TIME, DURATION, COEFF_URB, FIXED_COST_CAR, WALKING_SPEED, CO2_EMISSIONS_TRANSIT = import_parameters()

#compute_density = pd.DataFrame(index = np.delete(np.unique(list_city.City), 153), columns = ['City', 'data_pop_2015', 'data_land_cover_2015', 'predicted_land_cover_2015', 'predicted_land_cover_2015_corrected', 'ESA_population_2015', 'predicted_population_2015', 'predicted_population_2015_corrected', 'data_pop_2035', 'predicted_population_2035', 'predicted_land_cover_2035', 'predicted_population_2035_corrected', 'predicted_land_cover_2035_corrected'])
#informal_housing = import_informal_housing(list_city, path_folder)
#gdp_capita_ppp = pd.read_excel(path_folder + "gdp_capita_ppp.xlsx")
#gdp_capita_ppp["income"] =, 'data_pop_2035'
#gdp_capita_ppp.income[np.isnan(gdp_capita_ppp.income)] = gdp_capita_ppp.brookings
#gdp_capita_ppp["source"] = ""
#gdp_capita_ppp["source"][np.isnan(gdp_capita_ppp.income)] = "WB"
#gdp_capita_ppp.income[np.isnan(gdp_capita_ppp.income)] = gdp_capita_ppp.world_bank

if option["validation"] == 1:
    (d_beta, d_b, d_kappa, d_Ro, d_selected_cells, d_share_car, d_share_walking, 
     d_share_transit, d_emissions_per_capita, d_utility, d_income, d_corr_density0, 
     d_corr_rent0, d_corr_size0, d_corr_density1, d_corr_rent1, d_corr_size1, 
     d_corr_density2, d_corr_rent2, d_corr_size2, r2_density0, r2_rent0, 
     r2_size0, r2_density1, r2_rent1, r2_size1, r2_density2, r2_rent2, r2_size2, 
     mae_density0, mae_rent0, mae_size0, mae_density1, mae_rent1, mae_size1, 
     mae_density2, mae_rent2, mae_size2, rae_density0, rae_rent0, rae_size0, 
     rae_density1, rae_rent1, rae_size1, rae_density2, rae_rent2, 
     rae_size2) = initialize_dict()

for city in np.delete(np.unique(list_city.City), 153): 
    
    print("\n*** " + city + " ***\n")
    index = 0

    ### IMPORT DATA
    
    print("\n** Import data **\n")

    #Import city data
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    
    #Density, rents and dwelling sizes
    density = density.loc[:,density.columns.str.startswith("density")]
    density = np.array(density).squeeze()
    rent = (rents_and_size.avgRent / conversion_rate) * 12
    size = rents_and_size.medSize
    if (city == 'Mashhad') |(city == 'Isfahan') | (city == 'Tehran'):
        rent = 100 * rent
    if city == "Abidjan":
        rent[size > 100] = np.nan
        size[size > 100] = np.nan
    if (city == "Casablanca") | (city == "Yerevan"):
        rent[size > 100] = np.nan
        size[size > 100] = np.nan
    if city == "Toluca":
        rent[size > 250] = np.nan
        size[size > 250] = np.nan
    if (city == "Manaus")| (city == "Rabat")| (city == "Sao_Paulo"):
        rent[rent > 200] = np.nan
        size[rent > 200] = np.nan
    if (city == "Salvador"):
        rent[rent > 250] = np.nan
        size[rent > 250] = np.nan
    if (city == "Karachi") | (city == "Lahore"):
        rent[size > 200] = np.nan
        size[size > 200] = np.nan
    if city == "Addis_Ababa":
        rent[rent > 400] = rent / 100
        size = size / 10
    size.mask((size > 1000), inplace = True)
    agricultural_rent = import_agricultural_rent(path_folder, country)
    population = np.nansum(density)
    region = pd.read_excel(path_folder + "city_country_region.xlsx").region[pd.read_excel(path_folder + "city_country_region.xlsx").city == city].squeeze()
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()
    coeff_land = (land_use.OpenedToUrb / land_use.TotalArea) * COEFF_URB
    density_to_cover = convert_density_to_urban_footprint(city, path_data, list_city)
    agricultural_rent_2015 = copy.deepcopy(agricultural_rent)
    
    #Import transport data
    fuel_price = import_fuel_price(country, 'gasoline', path_folder) #L/100km
    fuel_consumption = import_fuel_conso(country, path_folder) #dollars/L
    CO2_emissions_car = 2300 * fuel_consumption / 100
    monetary_cost_pt = import_public_transport_cost_data(path_folder, city).squeeze()
        
    #Import street network
    if (policy == 'BRT') | (policy == 'synergy'):
        orig_dist, target_dist, transit_dist, length_network = import_street_network(list_city, city, country, proj, centre, path_folder, path_data)
    else:
        orig_dist = np.nan
        target_dist = np.nan
        transit_dist = np.nan
        
    np.save(path_outputs + city + "_orig_dist.npy", orig_dist)
    np.save(path_outputs + city + "_target_dist.npy", target_dist)
    np.save(path_outputs + city + "_transit_dist.npy", transit_dist)
    np.save(path_outputs + city + "_length_network.npy", length_network)