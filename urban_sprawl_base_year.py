# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 10:07:21 2022

@author: charl
"""

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math
import os

option_ssp = 'SSP2'
BRT_scenario = 'speed_40_0_12_50_5'

#'speed_40_0_12_50_5'
#capital_evolution_25_0_12_50_income
#capital_evolution_25_0_12_15_income

option_ugb = "predict_urba" #density, data_urba, predict_urba. Résultats actuels avec predict_urba

FUEL_EFFICIENCY_DECREASE = 0.98 #0.963
BASELINE_EFFICIENCY_DECREASE = 0.99
LIFESPAN = 15

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20211124/" #calibration_20211124
#os.mkdir(path_calibration)
#path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/" + BRT_scenario + '/'
#path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/sensitivity/kappa_agri_rent/kappa/BAU/"
#path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_resid4_predict_urba/"
path_outputs = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp2_tcost_croissant/"
#os.mkdir(path_outputs)

from calibration.calibration import *
from calibration.validation import *
from inputs.data import *
#from inputs.geo_data import *
from inputs.land_use import *
from model.model import *
from outputs.outputs import *
from inputs.parameters import *
from inputs.transport import *
from scipy.stats import pearsonr


#from inputs.geo_data import *
from density_functions import *


option = {}
option["validation"] = 0
option["add_residuals"] = True
option["do_calibration"] = False
policy = 'None'

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



df = pd.DataFrame(columns = ['City', 'area_ESA', 'area_GHSL', 'area_low_density', 'urban_core'], index = np.delete(np.unique(list_city.City), 153))

DURATION = 85

for city in np.delete(np.unique(list_city.City), 153):
#for city in ["Turku"]:
    
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
    size = rents_and_size.midsize
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
    #informal_housing_city = informal_housing.informal_housing[informal_housing.City == city]
    #rent = (rentemp_capita_ppp.city == city].squeeze() == "WB":
    #    income = income * 1.33
    agricultural_rent = import_agricultural_rent(path_folder, country)
    population = np.nansum(density)
    region = pd.read_excel(path_folder + "city_country_region.xlsx").region[pd.read_excel(path_folder + "city_country_region.xlsx").city == city].squeeze()
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()
    coeff_land = (land_use.OpenedToUrb / land_use.TotalArea) * COEFF_URB
    density_to_cover = convert_density_to_urban_footprint(city, path_data, list_city)
    agricultural_rent_2015 = copy.deepcopy(agricultural_rent)
    
    df.area_ESA.loc[df.index == city] = np.nansum(land_cover_ESACCI.ESACCI190 / 1000000)
    df.area_GHSL.loc[df.index == city] = sum(density > 150)
    df.area_low_density.loc[df.index == city] = sum((density > 150) & (density < 1500))
    df.urban_core.loc[df.index == city] = sum(density > 1500)
    
plt.scatter(df.area_ESA, df.area_GHSL, s = 8, c = "red", edgecolors = "black", linewidth=1)
plt.xlim(0, 16000)
plt.ylim(0, 16000)
plt.plot([0, 16000], [0, 16000], linewidth = 0.5, c = "black")
print(pearsonr(df.area_ESA, df.area_GHSL))

plt.scatter(df.area_ESA, df.urban_core, s = 8, c = "red", edgecolors = "black", linewidth=1)
plt.xlim(0, 8000)
plt.ylim(0, 8000)
plt.plot([0, 8000], [0, 8000], linewidth = 0.5, c = "black")
print(pearsonr(df.area_ESA, df.urban_core))

plt.scatter(df.area_ESA, df.area_low_density, s = 8, c = "red", edgecolors = "black", linewidth=1)
plt.xlim(0, 7000)
plt.ylim(0, 7000)
plt.plot([0, 7000], [0, 7000], linewidth = 0.5, c = "black")
print(pearsonr(df.area_ESA, df.area_low_density))

plt.scatter(df.urban_core, df.area_low_density, s = 8, c = "red", edgecolors = "black", linewidth=1)
print(pearsonr(df.urban_core, df.area_low_density))
    
print(sum(df.area_ESA))
print(sum(df.area_GHSL))

### ANALYSIS

path_ssp1 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp1_20220805/"
path_ssp2 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp2_20220805/"
path_ssp3 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp3_20220805/"
path_ssp4 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp4_20220805/"
path_ssp5 = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp5_20220805/"

for city in list(sample_of_cities.City[sample_of_cities.final_sample == 1]):
    
    
    
    save_density_ssp1 = np.load(path_ssp1 + city + "_density.npy")
    save_density_ssp2 = np.load(path_ssp2 + city + "_density.npy")
    save_density_ssp3 = np.load(path_ssp3 + city + "_density.npy")
    save_density_ssp4 = np.load(path_ssp4 + city + "_density.npy")
    save_density_ssp5 = np.load(path_ssp5 + city + "_density.npy")
    
    urb_ssp1 = np.load(path_ssp1 + city + "_ua.npy")
    urb_ssp2 = np.load(path_ssp2 + city + "_ua.npy")
    urb_ssp3 = np.load(path_ssp3 + city + "_ua.npy")
    urb_ssp4 = np.load(path_ssp4 + city + "_ua.npy")
    urb_ssp5 = np.load(path_ssp5 + city + "_ua.npy")
    
    save_density_ssp1_core = np.where(save_density_ssp1 < 1500, 0, save_density_ssp1)
    save_density_ssp2_core = np.where(save_density_ssp2 < 1500, 0, save_density_ssp2)
    save_density_ssp3_core = np.where(save_density_ssp3 < 1500, 0, save_density_ssp3)
    save_density_ssp4_core = np.where(save_density_ssp4 < 1500, 0, save_density_ssp4)
    save_density_ssp5_core = np.where(save_density_ssp5 < 1500, 0, save_density_ssp5)
    
    save_density_ssp1_low_density = np.where((save_density_ssp1 > 1500) | (save_density_ssp1 < 150), 0, save_density_ssp1)
    save_density_ssp2_low_density = np.where((save_density_ssp2 > 1500) | (save_density_ssp2 < 150), 0, save_density_ssp2)
    save_density_ssp3_low_density = np.where((save_density_ssp3 > 1500) | (save_density_ssp3 < 150), 0, save_density_ssp3)
    save_density_ssp4_low_density = np.where((save_density_ssp4 > 1500) | (save_density_ssp4 < 150), 0, save_density_ssp4)
    save_density_ssp5_low_density = np.where((save_density_ssp5 > 1500) | (save_density_ssp5 < 150), 0, save_density_ssp5)
    
    df_pop_core_ssp1 = np.nansum(save_density_ssp1_core, 1)
    df_pop_core_ssp2 = np.nansum(save_density_ssp2_core, 1)
    df_pop_core_ssp3= np.nansum(save_density_ssp3_core, 1)
    df_pop_core_ssp4 = np.nansum(save_density_ssp4_core, 1)
    df_pop_core_ssp5= np.nansum(save_density_ssp5_core, 1)
    
    df_pop_low_density_ssp1 = np.nansum(save_density_ssp1_low_density, 1)
    df_pop_low_density_ssp2 = np.nansum(save_density_ssp2_low_density, 1)
    df_pop_low_density_ssp3 = np.nansum(save_density_ssp3_low_density, 1)
    df_pop_low_density_ssp4 = np.nansum(save_density_ssp4_low_density, 1)
    df_pop_low_density_ssp5 = np.nansum(save_density_ssp5_low_density, 1)

    plt.figure()
    plt.plot(df_pop_low_density_ssp1 / (df_pop_low_density_ssp1 + df_pop_core_ssp1), label = "SSP1")
    plt.plot(df_pop_low_density_ssp2 / (df_pop_low_density_ssp2 + df_pop_core_ssp2), label = "SSP2")
    plt.plot(df_pop_low_density_ssp3 / (df_pop_low_density_ssp3 + df_pop_core_ssp3), label = "SSP3")
    plt.plot(df_pop_low_density_ssp4 / (df_pop_low_density_ssp4 + df_pop_core_ssp4), label = "SSP4")
    plt.plot(df_pop_low_density_ssp5 / (df_pop_low_density_ssp5 + df_pop_core_ssp5), label = "SSP5")
    plt.legend()
    plt.title(city)
    plt.show()
    
    plt.figure()
    plt.plot(np.nansum(save_density_ssp1 > 150, 1), label = "SSP1")
    plt.plot(np.nansum(save_density_ssp2 > 150, 1), label = "SSP2")
    plt.plot(np.nansum(save_density_ssp3 > 150, 1), label = "SSP3")
    plt.plot(np.nansum(save_density_ssp4 > 150, 1), label = "SSP4")
    plt.plot(np.nansum(save_density_ssp5 > 150, 1), label = "SSP5")
    plt.legend()
    plt.title(city)
    plt.show()
    
    plt.figure()
    plt.plot((urb), label = "urban area (ESA)")
    plt.plot(np.nansum(save_density_ssp2 > 150, 1), label = "urban area (GHSL)")
    plt.plot(np.nansum(save_density_ssp2 > 1500, 1), label = "urban core (GHSL)")
    plt.plot(np.nansum((save_density_ssp2 > 150) & (save_density_ssp2 < 1500), 1), label = "low-density area (GHSL)")
    plt.legend()
    plt.title(city)
    plt.show()
    
### RESULTATS AGGREGES
df_area_esa_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_area_esa_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_area_esa_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_area_esa_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_area_esa_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))


for city in list(sample_of_cities.City[sample_of_cities.final_sample == 1]):
    
    save_density_ssp1 = np.load(path_ssp1 + city + "_density.npy")
    save_density_ssp2 = np.load(path_ssp2 + city + "_density.npy")
    save_density_ssp3 = np.load(path_ssp3 + city + "_density.npy")
    save_density_ssp4 = np.load(path_ssp4 + city + "_density.npy")
    save_density_ssp5 = np.load(path_ssp5 + city + "_density.npy")
    
    urb_ssp1 = np.load(path_ssp1 + city + "_ua.npy")
    urb_ssp2 = np.load(path_ssp2 + city + "_ua.npy")
    urb_ssp3 = np.load(path_ssp3 + city + "_ua.npy")
    urb_ssp4 = np.load(path_ssp4 + city + "_ua.npy")
    urb_ssp5 = np.load(path_ssp5 + city + "_ua.npy")
    
    df_area_esa_ssp1.loc[df_area_esa_ssp1.index == city] = urb_ssp1
    df_area_esa_ssp2.loc[df_area_esa_ssp2.index == city] = urb_ssp2
    df_area_esa_ssp3.loc[df_area_esa_ssp3.index == city] = urb_ssp3
    df_area_esa_ssp4.loc[df_area_esa_ssp4.index == city] = urb_ssp4
    df_area_esa_ssp5.loc[df_area_esa_ssp5.index == city] = urb_ssp5
    
    df_area_GHSL_ssp1.loc[df_area_GHSL_ssp1.index == city] = np.nansum(save_density_ssp1 > 150, 1)
    df_area_GHSL_ssp2.loc[df_area_GHSL_ssp2.index == city] = np.nansum(save_density_ssp2 > 150, 1)
    df_area_GHSL_ssp3.loc[df_area_GHSL_ssp3.index == city] = np.nansum(save_density_ssp3 > 150, 1)
    df_area_GHSL_ssp4.loc[df_area_GHSL_ssp4.index == city] = np.nansum(save_density_ssp4 > 150, 1)
    df_area_GHSL_ssp5.loc[df_area_GHSL_ssp5.index == city] = np.nansum(save_density_ssp5 > 150, 1)
    
    df_core_ssp1.loc[df_core_ssp1.index == city] = np.nansum(save_density_ssp1 > 1500, 1)
    df_core_ssp2.loc[df_core_ssp2.index == city] = np.nansum(save_density_ssp2 > 1500, 1)
    df_core_ssp3.loc[df_core_ssp3.index == city] = np.nansum(save_density_ssp3 > 1500, 1)
    df_core_ssp4.loc[df_core_ssp4.index == city] = np.nansum(save_density_ssp4 > 1500, 1)
    df_core_ssp5.loc[df_core_ssp5.index == city] = np.nansum(save_density_ssp5 > 1500, 1)
    
    df_low_density_ssp1.loc[df_low_density_ssp1.index == city] = np.nansum((save_density_ssp1 > 150) & (save_density_ssp1 < 1500), 1)
    df_low_density_ssp2.loc[df_low_density_ssp2.index == city] = np.nansum((save_density_ssp2 > 150) & (save_density_ssp2 < 1500), 1)
    df_low_density_ssp3.loc[df_low_density_ssp3.index == city] = np.nansum((save_density_ssp3 > 150) & (save_density_ssp3 < 1500), 1)
    df_low_density_ssp4.loc[df_low_density_ssp4.index == city] = np.nansum((save_density_ssp4 > 150) & (save_density_ssp4 < 1500), 1)
    df_low_density_ssp5.loc[df_low_density_ssp5.index == city] = np.nansum((save_density_ssp5 > 150) & (save_density_ssp5 < 1500), 1)

    save_density_ssp1_core = np.where(save_density_ssp1 < 1500, 0, save_density_ssp1)
    save_density_ssp2_core = np.where(save_density_ssp2 < 1500, 0, save_density_ssp2)
    save_density_ssp3_core = np.where(save_density_ssp3 < 1500, 0, save_density_ssp3)
    save_density_ssp4_core = np.where(save_density_ssp4 < 1500, 0, save_density_ssp4)
    save_density_ssp5_core = np.where(save_density_ssp5 < 1500, 0, save_density_ssp5)
    
    save_density_ssp1_low_density = np.where((save_density_ssp1 > 1500) | (save_density_ssp1 < 150), 0, save_density_ssp1)
    save_density_ssp2_low_density = np.where((save_density_ssp2 > 1500) | (save_density_ssp2 < 150), 0, save_density_ssp2)
    save_density_ssp3_low_density = np.where((save_density_ssp3 > 1500) | (save_density_ssp3 < 150), 0, save_density_ssp3)
    save_density_ssp4_low_density = np.where((save_density_ssp4 > 1500) | (save_density_ssp4 < 150), 0, save_density_ssp4)
    save_density_ssp5_low_density = np.where((save_density_ssp5 > 1500) | (save_density_ssp5 < 150), 0, save_density_ssp5)
    
    df_pop_core_ssp1.loc[df_pop_core_ssp1.index == city] = np.nansum(save_density_ssp1_core, 1)
    df_pop_core_ssp2.loc[df_pop_core_ssp2.index == city] = np.nansum(save_density_ssp2_core, 1)
    df_pop_core_ssp3.loc[df_pop_core_ssp3.index == city] = np.nansum(save_density_ssp3_core, 1)
    df_pop_core_ssp4.loc[df_pop_core_ssp4.index == city] = np.nansum(save_density_ssp4_core, 1)
    df_pop_core_ssp5.loc[df_pop_core_ssp5.index == city] = np.nansum(save_density_ssp5_core, 1)
    
    df_pop_low_density_ssp1.loc[df_pop_low_density_ssp1.index == city] = np.nansum(save_density_ssp1_low_density, 1)
    df_pop_low_density_ssp2.loc[df_pop_low_density_ssp2.index == city] = np.nansum(save_density_ssp2_low_density, 1)
    df_pop_low_density_ssp3.loc[df_pop_low_density_ssp3.index == city] = np.nansum(save_density_ssp3_low_density, 1)
    df_pop_low_density_ssp4.loc[df_pop_low_density_ssp4.index == city] = np.nansum(save_density_ssp4_low_density, 1)
    df_pop_low_density_ssp5.loc[df_pop_low_density_ssp5.index == city] = np.nansum(save_density_ssp5_low_density, 1)



plt.plot(np.nansum(df_area_esa_ssp1, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_area_esa_ssp2, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_area_esa_ssp3, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_area_esa_ssp4, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_area_esa_ssp5, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP5")
plt.legend()

font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : True,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.tick_params(top=False, bottom=True, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5)
plt.plot(np.nansum(df_area_GHSL_ssp1, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_area_GHSL_ssp2, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_area_GHSL_ssp3, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_area_GHSL_ssp4, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_area_GHSL_ssp5, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP5")
plt.xlim(0, 35)
plt.ylim(1, 1.13)
plt.xticks(np.arange(0, 36, 5), np.arange(2015, 2051, 5))
plt.yticks(np.arange(1, 1.13, 0.02), np.arange(0, 13, 2))
plt.ylabel("Variation - %")
plt.legend(ncol = 1)


city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0,1,2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')

class_region = pd.read_excel("C:/Users/charl/OneDrive/Bureau/class_region.xlsx")
class_region["id"]=1
class_region.loc[:,["id", "IPCC"]].groupby("IPCC").sum()
class_region.loc[:,["id", "IPCC2"]].groupby("IPCC2").sum()
class_region["IPCC2"] = class_region["IPCC"]
class_region["IPCC2"][(class_region["IPCC"] == "Meso America")|(class_region["IPCC"] == "South America")] = "Latin America and Caribbean"
class_region["IPCC2"][(class_region["IPCC"] == "Eastern Asia")|(class_region["IPCC"] == "India & Sri Lanka")] = "Asia and developing Pacific (APC)"

city_continent = city_continent.merge(class_region, on = "City")
city_continent = city_continent.loc[:,["City","IPCC2"]]
city_continent.columns = ["City", "Continent"]

df_area_esa_ssp1 = df_area_esa_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_esa_ssp2 = df_area_esa_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_esa_ssp3 = df_area_esa_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_esa_ssp4 = df_area_esa_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_esa_ssp5 = df_area_esa_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_esa_ssp1 = df_area_esa_ssp1.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp1.Continent).sum()
aggregate_esa_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_esa_ssp2 = df_area_esa_ssp2.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp2.Continent).sum()
aggregate_esa_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_esa_ssp3 = df_area_esa_ssp3.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp3.Continent).sum()
aggregate_esa_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_esa_ssp4 = df_area_esa_ssp4.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp4.Continent).sum()
aggregate_esa_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_esa_ssp5 = df_area_esa_ssp5.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp5.Continent).sum()
aggregate_esa_ssp5.columns = ['year_2015', 'SSP5_2050']

cmap = plt.cm.get_cmap('plasma')
rgba1 = cmap(0.1)
rgba2 = cmap(0.3)
rgba3 = cmap(0.5)
rgba4 = cmap(0.7)
rgba5 = cmap(0.9)

plt.bar(np.arange(8)-0.45, aggregate_esa_ssp1['year_2015']/aggregate_esa_ssp1['year_2015'], label = "2015", width = 0.15, color = 'grey')
plt.bar(np.arange(8)-0.3, aggregate_esa_ssp1['SSP1_2050']/aggregate_esa_ssp1['year_2015'], label = "2050 - SSP1", width = 0.15, color = rgba1)
plt.bar(np.arange(8)-0.15, aggregate_esa_ssp2['SSP2_2050']/aggregate_esa_ssp2['year_2015'], label = "2050 - SSP2", width = 0.15, color = rgba2)
plt.bar(np.arange(8), aggregate_esa_ssp3['SSP3_2050']/aggregate_esa_ssp3['year_2015'], label = "2050 - SSP3", width = 0.15, color = rgba3)
plt.bar(np.arange(8)+0.15, aggregate_esa_ssp4['SSP4_2050']/aggregate_esa_ssp4['year_2015'], label = "2050 - SSP4", width = 0.15, color = rgba4)
plt.bar(np.arange(8)+0.3, aggregate_esa_ssp5['SSP5_2050']/aggregate_esa_ssp5['year_2015'], label = "2050 - SSP5", width = 0.15, color = rgba5)
#plt.xticks(np.arange(8), aggregate_esa_ssp2.index)
plt.xticks(np.arange(8), ["APC", "Aust", "East. Eur.", "LAC", "ME", "NA", "NWE", "SEE"])
plt.ylim(0, 2.3)
plt.legend(ncol = 3)

df_area_esa_ssp1["id"] = 1
df_area_esa_ssp1.loc[:,["id","Continent"]].groupby("Continent").sum()

df_area_GHSL_ssp1 = df_area_GHSL_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp2 = df_area_GHSL_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp3 = df_area_GHSL_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp4 = df_area_GHSL_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp5 = df_area_GHSL_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_GHSL_ssp1 = df_area_GHSL_ssp1.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp1.Continent).sum()
aggregate_GHSL_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_GHSL_ssp2 = df_area_GHSL_ssp2.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp2.Continent).sum()
aggregate_GHSL_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_GHSL_ssp3 = df_area_GHSL_ssp3.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp3.Continent).sum()
aggregate_GHSL_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_GHSL_ssp4 = df_area_GHSL_ssp4.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp4.Continent).sum()
aggregate_GHSL_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_GHSL_ssp5 = df_area_GHSL_ssp5.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp5.Continent).sum()
aggregate_GHSL_ssp5.columns = ['year_2015', 'SSP5_2050']

cmap = plt.cm.get_cmap('plasma')
rgba1 = cmap(0.1)
rgba2 = cmap(0.3)
rgba3 = cmap(0.5)
rgba4 = cmap(0.7)
rgba5 = cmap(0.9)

font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : False,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5, zorder=0)
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.bar(np.arange(8)-0.45, 100*(aggregate_GHSL_ssp1['SSP1_2050']-aggregate_GHSL_ssp1['year_2015'])/aggregate_GHSL_ssp1['year_2015'], label = "SSP1", width = 0.18, color = rgba1, alpha = 1, zorder=1)
plt.bar(np.arange(8)-0.27, 100*(aggregate_GHSL_ssp2['SSP2_2050']-aggregate_GHSL_ssp2['year_2015'])/aggregate_GHSL_ssp2['year_2015'], label = "SSP2", width = 0.18, color = rgba2, alpha = 1, zorder=2)
plt.bar(np.arange(8)-0.09, 100*(aggregate_GHSL_ssp3['SSP3_2050']-aggregate_GHSL_ssp3['year_2015'])/aggregate_GHSL_ssp3['year_2015'], label = "SSP3", width = 0.18, color = rgba3, alpha = 1, zorder=3)
plt.bar(np.arange(8)+0.09, 100*(aggregate_GHSL_ssp4['SSP4_2050']-aggregate_GHSL_ssp4['year_2015'])/aggregate_GHSL_ssp4['year_2015'], label = "SSP4", width = 0.18, color = rgba4, alpha = 1, zorder=4)
plt.bar(np.arange(8)+0.27, 100*(aggregate_GHSL_ssp5['SSP5_2050']-aggregate_GHSL_ssp5['year_2015'])/aggregate_GHSL_ssp5['year_2015'], label = "SSP5", width = 0.18, color = rgba5, alpha = 1, zorder=5)
#plt.xticks(np.arange(8), ["Asia", "Europe", "North \n America", "Oceania", "South \n America"])
plt.xticks(np.arange(8), ["APC", "Aust", "East. Eur.", "LAC", "ME", "NA", "NWE", "SEE"])
plt.ylabel("Variation - %")
plt.legend(ncol = 2)

font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : True,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5)
plt.plot(np.nansum(df_area_GHSL_ssp1, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_area_GHSL_ssp2, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_area_GHSL_ssp3, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_area_GHSL_ssp4, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_area_GHSL_ssp5, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP5")
plt.xlim(0, 35)
plt.ylim(1, 1.13)
plt.xticks(np.arange(0, 36, 5), np.arange(2015, 2051, 5))
plt.yticks(np.arange(1, 1.13, 0.02), np.arange(0, 13, 2))
plt.ylabel("Variation - %")
plt.legend(ncol = 1)

df_low_density_ssp1 = df_low_density_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp2 = df_low_density_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp3 = df_low_density_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp4 = df_low_density_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp5 = df_low_density_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_low_density_ssp1 = df_low_density_ssp1.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp1.Continent).sum()
aggregate_low_density_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_low_density_ssp2 = df_low_density_ssp2.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp2.Continent).sum()
aggregate_low_density_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_low_density_ssp3 = df_low_density_ssp3.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp3.Continent).sum()
aggregate_low_density_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_low_density_ssp4 = df_low_density_ssp4.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp4.Continent).sum()
aggregate_low_density_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_low_density_ssp5 = df_low_density_ssp5.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp5.Continent).sum()
aggregate_low_density_ssp5.columns = ['year_2015', 'SSP5_2050']


font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : False,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5, zorder=0)
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.bar(np.arange(8)-0.45, 100*aggregate_low_density_ssp1['year_2015']/aggregate_GHSL_ssp1['year_2015'], label = "2015", width = 0.15, color = 'grey', alpha = 1, zorder=1)
plt.bar(np.arange(8)-0.30, 100*aggregate_low_density_ssp1['SSP1_2050']/aggregate_GHSL_ssp1['SSP1_2050'], label = "SSP1", width = 0.15, color = rgba1, alpha = 1, zorder=1)
plt.bar(np.arange(8)-0.15, 100*aggregate_low_density_ssp2['SSP2_2050']/aggregate_GHSL_ssp2['SSP2_2050'], label = "SSP2", width = 0.15, color = rgba2, alpha = 1, zorder=2)
plt.bar(np.arange(8), 100*aggregate_low_density_ssp3['SSP3_2050']/aggregate_GHSL_ssp3['SSP3_2050'], label = "SSP3", width = 0.15, color = rgba3, alpha = 1, zorder=3)
plt.bar(np.arange(8)+0.15, 100*aggregate_low_density_ssp4['SSP4_2050']/aggregate_GHSL_ssp4['SSP4_2050'], label = "SSP4", width = 0.15, color = rgba4, alpha = 1, zorder=4)
plt.bar(np.arange(8)+0.30, 100*aggregate_low_density_ssp5['SSP5_2050']/aggregate_GHSL_ssp5['SSP5_2050'], label = "SSP5", width = 0.15, color = rgba5, alpha = 1, zorder=5)
#plt.xticks(np.arange(5), ["Asia", "Europe", "North \n America", "Oceania", "South \n America"])
plt.xticks(np.arange(8), ["APC", "Aust", "East. Eur.", "LAC", "ME", "NA", "NWE", "SEE"])
plt.ylabel("Share - %")
plt.ylim(0, 150)
plt.legend(ncol = 2)

df_pop_core_ssp1 = df_pop_core_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp2 = df_pop_core_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp3 = df_pop_core_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp4 = df_pop_core_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp5 = df_pop_core_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_pop_core_ssp1 = df_pop_core_ssp1.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp1.Continent).sum()
aggregate_pop_core_ssp1.columns = ['year_2015', 'SSP1_2050']
aggregate_pop_core_ssp2 = df_pop_core_ssp2.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp2.Continent).sum()
aggregate_pop_core_ssp2.columns = ['year_2015', 'SSP2_2050']
aggregate_pop_core_ssp3 = df_pop_core_ssp3.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp3.Continent).sum()
aggregate_pop_core_ssp3.columns = ['year_2015', 'SSP3_2050']
aggregate_pop_core_ssp4 = df_pop_core_ssp4.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp4.Continent).sum()
aggregate_pop_core_ssp4.columns = ['year_2015', 'SSP4_2050']
aggregate_pop_core_ssp5 = df_pop_core_ssp5.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp5.Continent).sum()
aggregate_pop_core_ssp5.columns = ['year_2015', 'SSP5_2050']




df_pop_low_density_ssp1 = df_pop_low_density_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp2 = df_pop_low_density_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp3 = df_pop_low_density_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp4 = df_pop_low_density_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp5 = df_pop_low_density_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_pop_low_density_ssp1 = df_pop_low_density_ssp1.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp1.Continent).sum()
aggregate_pop_low_density_ssp1.columns = ['year_2015', 'SSP1_2050']
aggregate_pop_low_density_ssp2 = df_pop_low_density_ssp2.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp2.Continent).sum()
aggregate_pop_low_density_ssp2.columns = ['year_2015', 'SSP2_2050']
aggregate_pop_low_density_ssp3 = df_pop_low_density_ssp3.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp3.Continent).sum()
aggregate_pop_low_density_ssp3.columns = ['year_2015', 'SSP3_2050']
aggregate_pop_low_density_ssp4 = df_pop_low_density_ssp4.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp4.Continent).sum()
aggregate_pop_low_density_ssp4.columns = ['year_2015', 'SSP4_2050']
aggregate_pop_low_density_ssp5 = df_pop_low_density_ssp5.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp5.Continent).sum()
aggregate_pop_low_density_ssp5.columns = ['year_2015', 'SSP5_2050']


font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : False,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5, zorder=0)
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.bar(np.arange(8)-0.45, 100*aggregate_pop_low_density_ssp1['year_2015']/(aggregate_pop_low_density_ssp1['year_2015'] + aggregate_pop_core_ssp1['year_2015']), label = "2015", width = 0.15, color = 'grey', alpha = 1, zorder=1)
plt.bar(np.arange(8)-0.30, 100*aggregate_pop_low_density_ssp1['SSP1_2050']/(aggregate_pop_low_density_ssp1['SSP1_2050'] + aggregate_pop_core_ssp1['SSP1_2050']), label = "SSP1", width = 0.15, color = rgba1, alpha = 1, zorder=1)
plt.bar(np.arange(8)-0.15, 100*aggregate_pop_low_density_ssp2['SSP2_2050']/(aggregate_pop_low_density_ssp2['SSP2_2050'] + aggregate_pop_core_ssp2['SSP2_2050']), label = "SSP2", width = 0.15, color = rgba2, alpha = 1, zorder=2)
plt.bar(np.arange(8), 100*aggregate_pop_low_density_ssp3['SSP3_2050']/(aggregate_pop_low_density_ssp3['SSP3_2050'] + aggregate_pop_core_ssp3['SSP3_2050']), label = "SSP3", width = 0.15, color = rgba3, alpha = 1, zorder=3)
plt.bar(np.arange(8)+0.15, 100*aggregate_pop_low_density_ssp4['SSP4_2050']/(aggregate_pop_low_density_ssp4['SSP4_2050'] + aggregate_pop_core_ssp4['SSP4_2050']), label = "SSP4", width = 0.15, color = rgba4, alpha = 1, zorder=4)
plt.bar(np.arange(8)+0.30, 100*aggregate_pop_low_density_ssp5['SSP5_2050']/(aggregate_pop_low_density_ssp5['SSP5_2050'] + aggregate_pop_core_ssp5['SSP5_2050']), label = "SSP5", width = 0.15, color = rgba5, alpha = 1, zorder=5)
#plt.xticks(np.arange(5), ["Asia", "Europe", "North \n America", "Oceania", "South \n America"])
plt.xticks(np.arange(8), ["APC", "Aust", "East. Eur.", "LAC", "ME", "NA", "NWE", "SEE"])
plt.ylabel("Share - %")
plt.ylim(0, 150)
plt.legend(ncol = 2)



df_population = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = ['Pop'])
for city in list(sample_of_cities.City[sample_of_cities.final_sample == 1]):   
    save_density_ssp1 = np.load(path_ssp1 + city + "_density.npy")
    df_population.loc[df_population.index == city] = np.nansum(save_density_ssp1, 1)[0]
3df_population["Pop_Q"] = np.where(df_population.Pop.astype(float) < 1000000, "<1M", "")
4f_population["Pop_Q"] = np.where(df_population.Pop.astype(float) > 4000000, ">4M", df_population.Pop_Q)
5f_population["Pop_Q"] = np.where((df_population.Pop.astype(float) > 1000000) & (df_population.Pop.astype(float) < 2000000), ">1M-2M", df_population.Pop_Q)
df_population["Pop_Q"] = np.where((df_population.Pop.astype(float) > 2000000) & (df_population.Pop.astype(float) < 4000000), ">2M-4M", df_population.Pop_Q)

df_area_esa_ssp1 = df_area_esa_ssp1.merge(df_population, left_index = True, right_index = True, how = 'left')
df_area_esa_ssp2 = df_area_esa_ssp2.merge(df_population, left_index = True, right_index = True, how = 'left')
df_area_esa_ssp3 = df_area_esa_ssp3.merge(df_population, left_index = True, right_index = True, how = 'left')
df_area_esa_ssp4 = df_area_esa_ssp4.merge(df_population, left_index = True, right_index = True, how = 'left')
df_area_esa_ssp5 = df_area_esa_ssp5.merge(df_population, left_index = True, right_index = True, how = 'left')

aggregate_esa_ssp1 = df_area_esa_ssp1.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp1.Pop_Q).sum()
aggregate_esa_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_esa_ssp2 = df_area_esa_ssp2.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp2.Pop_Q).sum()
aggregate_esa_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_esa_ssp3 = df_area_esa_ssp3.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp3.Pop_Q).sum()
aggregate_esa_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_esa_ssp4 = df_area_esa_ssp4.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp4.Pop_Q).sum()
aggregate_esa_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_esa_ssp5 = df_area_esa_ssp5.iloc[:,[0,35]].astype(float).groupby(df_area_esa_ssp5.Pop_Q).sum()
aggregate_esa_ssp5.columns = ['year_2015', 'SSP5_2050']

cmap = plt.cm.get_cmap('plasma')
rgba1 = cmap(0.1)
rgba2 = cmap(0.3)
rgba3 = cmap(0.5)
rgba4 = cmap(0.7)
rgba5 = cmap(0.9)

plt.bar(np.arange(4)-0.45, aggregate_esa_ssp1['year_2015']/aggregate_esa_ssp1['year_2015'], label = "2015", width = 0.15, color = 'grey')
plt.bar(np.arange(4)-0.3, aggregate_esa_ssp1['SSP1_2050']/aggregate_esa_ssp1['year_2015'], label = "2050 - SSP1", width = 0.15, color = rgba1)
plt.bar(np.arange(4)-0.15, aggregate_esa_ssp2['SSP2_2050']/aggregate_esa_ssp2['year_2015'], label = "2050 - SSP2", width = 0.15, color = rgba2)
plt.bar(np.arange(4), aggregate_esa_ssp3['SSP3_2050']/aggregate_esa_ssp3['year_2015'], label = "2050 - SSP3", width = 0.15, color = rgba3)
plt.bar(np.arange(4)+0.15, aggregate_esa_ssp4['SSP4_2050']/aggregate_esa_ssp4['year_2015'], label = "2050 - SSP4", width = 0.15, color = rgba4)
plt.bar(np.arange(4)+0.3, aggregate_esa_ssp5['SSP5_2050']/aggregate_esa_ssp5['year_2015'], label = "2050 - SSP5", width = 0.15, color = rgba5)
plt.xticks(np.arange(4), aggregate_esa_ssp2.index)
plt.ylim(0, 1.8)
plt.legend(ncol = 3)

df_area_GHSL_ssp1 = df_area_GHSL_ssp1.merge(df_population, left_index = True, right_index = True, how = 'left')
df_area_GHSL_ssp2 = df_area_GHSL_ssp2.merge(df_population, left_index = True, right_index = True, how = 'left')
df_area_GHSL_ssp3 = df_area_GHSL_ssp3.merge(df_population, left_index = True, right_index = True, how = 'left')
df_area_GHSL_ssp4 = df_area_GHSL_ssp4.merge(df_population, left_index = True, right_index = True, how = 'left')
df_area_GHSL_ssp5 = df_area_GHSL_ssp5.merge(df_population, left_index = True, right_index = True, how = 'left')

aggregate_GHSL_ssp1 = df_area_GHSL_ssp1.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp1.Pop_Q).sum()
aggregate_GHSL_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_GHSL_ssp2 = df_area_GHSL_ssp2.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp2.Pop_Q).sum()
aggregate_GHSL_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_GHSL_ssp3 = df_area_GHSL_ssp3.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp3.Pop_Q).sum()
aggregate_GHSL_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_GHSL_ssp4 = df_area_GHSL_ssp4.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp4.Pop_Q).sum()
aggregate_GHSL_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_GHSL_ssp5 = df_area_GHSL_ssp5.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp5.Pop_Q).sum()
aggregate_GHSL_ssp5.columns = ['year_2015', 'SSP5_2050']

cmap = plt.cm.get_cmap('plasma')
rgba1 = cmap(0.1)
rgba2 = cmap(0.3)
rgba3 = cmap(0.5)
rgba4 = cmap(0.7)
rgba5 = cmap(0.9)

plt.bar(np.arange(4)-0.45, aggregate_GHSL_ssp1['year_2015']/aggregate_GHSL_ssp1['year_2015'], label = "2015", width = 0.15, color = 'grey')
plt.bar(np.arange(4)-0.3, aggregate_GHSL_ssp1['SSP1_2050']/aggregate_GHSL_ssp1['year_2015'], label = "2050 - SSP1", width = 0.15, color = rgba1)
plt.bar(np.arange(4)-0.15, aggregate_GHSL_ssp2['SSP2_2050']/aggregate_GHSL_ssp2['year_2015'], label = "2050 - SSP2", width = 0.15, color = rgba2)
plt.bar(np.arange(4), aggregate_GHSL_ssp3['SSP3_2050']/aggregate_GHSL_ssp3['year_2015'], label = "2050 - SSP3", width = 0.15, color = rgba3)
plt.bar(np.arange(4)+0.15, aggregate_GHSL_ssp4['SSP4_2050']/aggregate_GHSL_ssp4['year_2015'], label = "2050 - SSP4", width = 0.15, color = rgba4)
plt.bar(np.arange(4)+0.3, aggregate_GHSL_ssp5['SSP5_2050']/aggregate_GHSL_ssp5['year_2015'], label = "2050 - SSP5", width = 0.15, color = rgba5)
plt.xticks(np.arange(4), aggregate_GHSL_ssp2.index)
plt.ylim(0, 1.6)
plt.legend(ncol = 3)




#Cities sprawling
sum(df_area_esa_ssp1.iloc[:,35] > df_area_esa_ssp1.iloc[:,0])
sum(df_area_esa_ssp2.iloc[:,35] > df_area_esa_ssp1.iloc[:,0])
sum(df_area_esa_ssp3.iloc[:,35] > df_area_esa_ssp1.iloc[:,0])
sum(df_area_esa_ssp4.iloc[:,35] > df_area_esa_ssp1.iloc[:,0])
sum(df_area_esa_ssp5.iloc[:,35] > df_area_esa_ssp1.iloc[:,0])

sum(df_area_GHSL_ssp1.iloc[:,35] > df_area_GHSL_ssp1.iloc[:,0])
sum(df_area_GHSL_ssp2.iloc[:,35] > df_area_GHSL_ssp2.iloc[:,0])
sum(df_area_GHSL_ssp3.iloc[:,35] > df_area_GHSL_ssp3.iloc[:,0])
sum(df_area_GHSL_ssp4.iloc[:,35] > df_area_GHSL_ssp4.iloc[:,0])
sum(df_area_GHSL_ssp5.iloc[:,35] > df_area_GHSL_ssp5.iloc[:,0])

#Urban core and low-density area sprawling

sum(df_core_ssp1.iloc[:,35] > df_core_ssp1.iloc[:,0])
sum(df_core_ssp2.iloc[:,35] > df_core_ssp2.iloc[:,0])
sum(df_core_ssp3.iloc[:,35] > df_core_ssp3.iloc[:,0])
sum(df_core_ssp4.iloc[:,35] > df_core_ssp4.iloc[:,0])
sum(df_core_ssp5.iloc[:,35] > df_core_ssp5.iloc[:,0])


sum(df_low_density_ssp1.iloc[:,35] > df_low_density_ssp1.iloc[:,0])
sum(df_low_density_ssp2.iloc[:,35] > df_low_density_ssp2.iloc[:,0])
sum(df_low_density_ssp3.iloc[:,35] > df_low_density_ssp3.iloc[:,0])
sum(df_low_density_ssp4.iloc[:,35] > df_low_density_ssp4.iloc[:,0])
sum(df_low_density_ssp5.iloc[:,35] > df_low_density_ssp5.iloc[:,0])

plt.plot(np.nansum(df_core_ssp1, 0)/np.nansum(df_core_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_core_ssp2, 0)/np.nansum(df_core_ssp2, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_core_ssp3, 0)/np.nansum(df_core_ssp3, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_core_ssp4, 0)/np.nansum(df_core_ssp4, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_core_ssp5, 0)/np.nansum(df_core_ssp5, 0)[0], label = "SSP5")
plt.legend()

plt.plot(np.nansum(df_low_density_ssp1, 0)/np.nansum(df_low_density_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_low_density_ssp2, 0)/np.nansum(df_low_density_ssp2, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_low_density_ssp3, 0)/np.nansum(df_low_density_ssp3, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_low_density_ssp4, 0)/np.nansum(df_low_density_ssp4, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_low_density_ssp5, 0)/np.nansum(df_low_density_ssp5, 0)[0], label = "SSP5")
plt.legend()

#Type of city
sum((df_core_ssp1.iloc[:,35] >= df_core_ssp1.iloc[:,0]) & (df_low_density_ssp1.iloc[:,35] >= df_low_density_ssp1.iloc[:,0]))
sum((df_core_ssp1.iloc[:,35] >= df_core_ssp1.iloc[:,0]) & (df_low_density_ssp1.iloc[:,35] < df_low_density_ssp1.iloc[:,0]))
sum((df_core_ssp1.iloc[:,35] < df_core_ssp1.iloc[:,0]) & (df_low_density_ssp1.iloc[:,35] >= df_low_density_ssp1.iloc[:,0]))
sum((df_core_ssp1.iloc[:,35] < df_core_ssp1.iloc[:,0]) & (df_low_density_ssp1.iloc[:,35] < df_low_density_ssp1.iloc[:,0]))

sum((df_core_ssp2.iloc[:,35] >= df_core_ssp2.iloc[:,0]) & (df_low_density_ssp2.iloc[:,35] >= df_low_density_ssp2.iloc[:,0]))
sum((df_core_ssp2.iloc[:,35] >= df_core_ssp2.iloc[:,0]) & (df_low_density_ssp2.iloc[:,35] < df_low_density_ssp2.iloc[:,0]))
sum((df_core_ssp2.iloc[:,35] < df_core_ssp2.iloc[:,0]) & (df_low_density_ssp2.iloc[:,35] >= df_low_density_ssp2.iloc[:,0]))
sum((df_core_ssp2.iloc[:,35] < df_core_ssp2.iloc[:,0]) & (df_low_density_ssp2.iloc[:,35] < df_low_density_ssp2.iloc[:,0]))


df_type_city = pd.DataFrame(index = df_core_ssp2.index, columns = ['City', 'Type'])
df_type_city.loc[((df_core_ssp2.iloc[:,35] >= df_core_ssp2.iloc[:,0]) & (df_low_density_ssp2.iloc[:,35] >= df_low_density_ssp2.iloc[:,0])), 'Type'] = 'Core_increase_lowD_increase'
df_type_city.loc[((df_core_ssp2.iloc[:,35] >= df_core_ssp2.iloc[:,0]) & (df_low_density_ssp2.iloc[:,35] < df_low_density_ssp2.iloc[:,0])), 'Type'] = 'Core_increase_lowD_decrease'
df_type_city.loc[((df_core_ssp2.iloc[:,35] < df_core_ssp2.iloc[:,0]) & (df_low_density_ssp2.iloc[:,35] >= df_low_density_ssp2.iloc[:,0])), 'Type'] = 'Core_decrease_lowD_increase'
df_type_city.loc[((df_core_ssp2.iloc[:,35] < df_core_ssp2.iloc[:,0]) & (df_low_density_ssp2.iloc[:,35] < df_low_density_ssp2.iloc[:,0])), 'Type'] = 'Core_decrease_lowD_decrease'
df_type_city["City"] = df_type_city.index
df_type_city.to_excel('C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/type_city.xlsx')

sum((df_core_ssp3.iloc[:,35] >= df_core_ssp3.iloc[:,0]) & (df_low_density_ssp3.iloc[:,35] >= df_low_density_ssp3.iloc[:,0]))
sum((df_core_ssp3.iloc[:,35] >= df_core_ssp3.iloc[:,0]) & (df_low_density_ssp3.iloc[:,35] < df_low_density_ssp3.iloc[:,0]))
sum((df_core_ssp3.iloc[:,35] < df_core_ssp3.iloc[:,0]) & (df_low_density_ssp3.iloc[:,35] >= df_low_density_ssp3.iloc[:,0]))
sum((df_core_ssp3.iloc[:,35] < df_core_ssp3.iloc[:,0]) & (df_low_density_ssp3.iloc[:,35] < df_low_density_ssp3.iloc[:,0]))

sum((df_core_ssp4.iloc[:,35] >= df_core_ssp4.iloc[:,0]) & (df_low_density_ssp4.iloc[:,35] >= df_low_density_ssp4.iloc[:,0]))
sum((df_core_ssp4.iloc[:,35] >= df_core_ssp4.iloc[:,0]) & (df_low_density_ssp4.iloc[:,35] < df_low_density_ssp4.iloc[:,0]))
sum((df_core_ssp4.iloc[:,35] < df_core_ssp4.iloc[:,0]) & (df_low_density_ssp4.iloc[:,35] >= df_low_density_ssp4.iloc[:,0]))
sum((df_core_ssp4.iloc[:,35] < df_core_ssp4.iloc[:,0]) & (df_low_density_ssp4.iloc[:,35] < df_low_density_ssp4.iloc[:,0]))


sum((df_core_ssp5.iloc[:,35] >= df_core_ssp5.iloc[:,0]) & (df_low_density_ssp5.iloc[:,35] >= df_low_density_ssp5.iloc[:,0]))
sum((df_core_ssp5.iloc[:,35] >= df_core_ssp5.iloc[:,0]) & (df_low_density_ssp5.iloc[:,35] < df_low_density_ssp5.iloc[:,0]))
sum((df_core_ssp5.iloc[:,35] < df_core_ssp5.iloc[:,0]) & (df_low_density_ssp5.iloc[:,35] >= df_low_density_ssp5.iloc[:,0]))
sum((df_core_ssp5.iloc[:,35] < df_core_ssp5.iloc[:,0]) & (df_low_density_ssp5.iloc[:,35] < df_low_density_ssp5.iloc[:,0]))



font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : True,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.tick_params(top=False, bottom=True, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5)
plt.plot(np.nanmean(df_low_density_ssp1/df_area_GHSL_ssp1, 0), label = "SSP1")
plt.plot(np.nanmean(df_low_density_ssp2/df_area_GHSL_ssp2, 0), label = "SSP2")
plt.plot(np.nanmean(df_low_density_ssp3/df_area_GHSL_ssp3, 0), label = "SSP3")
plt.plot(np.nanmean(df_low_density_ssp4/df_area_GHSL_ssp4, 0), label = "SSP4")
plt.plot(np.nanmean(df_low_density_ssp5/df_area_GHSL_ssp5, 0), label = "SSP5")
plt.xlim(0, 35)
plt.ylim(0.80, 0.88)
plt.xticks(np.arange(0, 36, 5), np.arange(2015, 2051, 5))
plt.yticks(np.arange(0.80, 0.88, 0.01), np.arange(80,88,1))
plt.ylabel("Share - %")
plt.legend(ncol = 1)

font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : True,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.tick_params(top=False, bottom=True, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5)
plt.plot(np.nansum(df_low_density_ssp1, 0)/np.nansum(df_area_GHSL_ssp1, 0), label = "SSP1")
plt.plot(np.nansum(df_low_density_ssp2, 0)/np.nansum(df_area_GHSL_ssp2, 0), label = "SSP2")
plt.plot(np.nansum(df_low_density_ssp3, 0)/np.nansum(df_area_GHSL_ssp3, 0), label = "SSP3")
plt.plot(np.nansum(df_low_density_ssp4, 0)/np.nansum(df_area_GHSL_ssp4, 0), label = "SSP4")
plt.plot(np.nansum(df_low_density_ssp5, 0)/np.nansum(df_area_GHSL_ssp5, 0), label = "SSP5")
plt.xlim(0, 35)
plt.ylim(0.78, 0.86)
plt.xticks(np.arange(0, 36, 5), np.arange(2015, 2051, 5))
plt.yticks(np.arange(0.78, 0.86, 0.01), np.arange(78,86,1))
plt.ylabel("Share - %")
plt.legend(ncol = 1)


### PAREIL MAIS EN PROPORTION DE LA POP
plt.plot(np.nansum(df_pop_core_ssp1, 0)/np.nansum(df_pop_core_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_pop_core_ssp2, 0)/np.nansum(df_pop_core_ssp2, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_pop_core_ssp3, 0)/np.nansum(df_pop_core_ssp3, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_pop_core_ssp4, 0)/np.nansum(df_pop_core_ssp4, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_pop_core_ssp5, 0)/np.nansum(df_pop_core_ssp5, 0)[0], label = "SSP5")
plt.legend()

plt.plot(np.nansum(df_pop_low_density_ssp1, 0)/np.nansum(df_pop_low_density_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_pop_low_density_ssp2, 0)/np.nansum(df_pop_low_density_ssp2, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_pop_low_density_ssp3, 0)/np.nansum(df_pop_low_density_ssp3, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_pop_low_density_ssp4, 0)/np.nansum(df_pop_low_density_ssp4, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_pop_low_density_ssp5, 0)/np.nansum(df_pop_low_density_ssp5, 0)[0], label = "SSP5")
plt.legend()

plt.plot(np.nanmean(df_pop_low_density_ssp1/(df_pop_low_density_ssp1 + df_pop_core_ssp1), 0), label = "SSP1")
plt.plot(np.nanmean(df_pop_low_density_ssp2/(df_pop_low_density_ssp2 + df_pop_core_ssp2), 0), label = "SSP2")
plt.plot(np.nanmean(df_pop_low_density_ssp3/(df_pop_low_density_ssp3 + df_pop_core_ssp3), 0), label = "SSP3")
plt.plot(np.nanmean(df_pop_low_density_ssp4/(df_pop_low_density_ssp4 + df_pop_core_ssp4), 0), label = "SSP4")
plt.plot(np.nanmean(df_pop_low_density_ssp5/(df_pop_low_density_ssp5 + df_pop_core_ssp5), 0), label = "SSP5")
plt.legend()

plt.plot(np.nansum(df_pop_low_density_ssp1, 0)/(np.nansum(df_pop_low_density_ssp1, 0) + np.nansum(df_pop_core_ssp1, 0)), label = "SSP1")
plt.plot(np.nansum(df_pop_low_density_ssp2, 0)/(np.nansum(df_pop_low_density_ssp2, 0) + np.nansum(df_pop_core_ssp2, 0)), label = "SSP2")
plt.plot(np.nansum(df_pop_low_density_ssp3, 0)/(np.nansum(df_pop_low_density_ssp3, 0) + np.nansum(df_pop_core_ssp3, 0)), label = "SSP3")
plt.plot(np.nansum(df_pop_low_density_ssp4, 0)/(np.nansum(df_pop_low_density_ssp4, 0) + np.nansum(df_pop_core_ssp4, 0)), label = "SSP4")
plt.plot(np.nansum(df_pop_low_density_ssp5, 0)/(np.nansum(df_pop_low_density_ssp5, 0) + np.nansum(df_pop_core_ssp5, 0)), label = "SSP5")
plt.legend()

font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : True,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.tick_params(top=False, bottom=True, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5)
plt.plot(np.nanmean(df_pop_low_density_ssp1/(df_pop_low_density_ssp1 + df_pop_core_ssp1), 0), label = "SSP1")
plt.plot(np.nanmean(df_pop_low_density_ssp2/(df_pop_low_density_ssp2 + df_pop_core_ssp2), 0), label = "SSP2")
plt.plot(np.nanmean(df_pop_low_density_ssp3/(df_pop_low_density_ssp3 + df_pop_core_ssp3), 0), label = "SSP3")
plt.plot(np.nanmean(df_pop_low_density_ssp4/(df_pop_low_density_ssp4 + df_pop_core_ssp4), 0), label = "SSP4")
plt.plot(np.nanmean(df_pop_low_density_ssp5/(df_pop_low_density_ssp5 + df_pop_core_ssp5), 0), label = "SSP5")
plt.xlim(0, 35)
plt.ylim(0.64, 0.72)
plt.xticks(np.arange(0, 36, 5), np.arange(2015, 2051, 5))
plt.yticks(np.arange(0.64, 0.72, 0.01), np.arange(64,72,1))
plt.ylabel("Share - %")
plt.legend(ncol = 1)

font = {'family' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : True,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.tick_params(top=False, bottom=True, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5)
plt.plot(np.nansum(df_pop_low_density_ssp1, 0)/(np.nansum(df_pop_low_density_ssp1, 0) + np.nansum(df_pop_core_ssp1, 0)), label = "SSP1")
plt.plot(np.nansum(df_pop_low_density_ssp2, 0)/(np.nansum(df_pop_low_density_ssp2, 0) + np.nansum(df_pop_core_ssp2, 0)), label = "SSP2")
plt.plot(np.nansum(df_pop_low_density_ssp3, 0)/(np.nansum(df_pop_low_density_ssp3, 0) + np.nansum(df_pop_core_ssp3, 0)), label = "SSP3")
plt.plot(np.nansum(df_pop_low_density_ssp4, 0)/(np.nansum(df_pop_low_density_ssp4, 0) + np.nansum(df_pop_core_ssp4, 0)), label = "SSP4")
plt.plot(np.nansum(df_pop_low_density_ssp5, 0)/(np.nansum(df_pop_low_density_ssp5, 0) + np.nansum(df_pop_core_ssp5, 0)), label = "SSP5")
plt.xlim(0, 35)
plt.ylim(0.36, 0.46)
plt.xticks(np.arange(0, 36, 5), np.arange(2015, 2051, 5))
plt.yticks(np.arange(0.36, 0.46, 0.02), np.arange(36,47,2))
plt.ylabel("Share - %")
plt.legend(ncol = 1)


df_pop_low_density_share_ssp1 = df_pop_low_density_ssp1/(df_pop_low_density_ssp1 + df_pop_core_ssp1)
df_pop_low_density_share_ssp2 = df_pop_low_density_ssp2/(df_pop_low_density_ssp2 + df_pop_core_ssp2)
df_pop_low_density_share_ssp3 = df_pop_low_density_ssp3/(df_pop_low_density_ssp3 + df_pop_core_ssp3)
df_pop_low_density_share_ssp4 = df_pop_low_density_ssp4/(df_pop_low_density_ssp4 + df_pop_core_ssp4)
df_pop_low_density_share_ssp5 = df_pop_low_density_ssp5/(df_pop_low_density_ssp5 + df_pop_core_ssp5)

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')

df_pop_low_density_share_ssp1 = df_pop_low_density_share_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_share_ssp2 = df_pop_low_density_share_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_share_ssp3 = df_pop_low_density_share_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_share_ssp4 = df_pop_low_density_share_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_share_ssp5 = df_pop_low_density_share_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_share_ssp1 = df_pop_low_density_share_ssp1.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_share_ssp1.Continent).mean()
aggregate_share_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_share_ssp2 = df_pop_low_density_share_ssp2.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_share_ssp2.Continent).mean()
aggregate_share_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_share_ssp3 = df_pop_low_density_share_ssp3.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_share_ssp3.Continent).mean()
aggregate_share_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_share_ssp4 = df_pop_low_density_share_ssp4.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_share_ssp4.Continent).mean()
aggregate_share_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_share_ssp5 = df_pop_low_density_share_ssp5.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_share_ssp5.Continent).mean()
aggregate_share_ssp5.columns = ['year_2015', 'SSP5_2050']

cmap = plt.cm.get_cmap('plasma')
rgba1 = cmap(0.1)
rgba2 = cmap(0.3)
rgba3 = cmap(0.5)
rgba4 = cmap(0.7)
rgba5 = cmap(0.9)

plt.bar(np.arange(5)-0.45, aggregate_share_ssp1['year_2015'], label = "2015", width = 0.15, color = 'grey')
plt.bar(np.arange(5)-0.3, aggregate_share_ssp1['SSP1_2050'], label = "2050 - SSP1", width = 0.15, color = rgba1)
plt.bar(np.arange(5)-0.15, aggregate_share_ssp2['SSP2_2050'], label = "2050 - SSP2", width = 0.15, color = rgba2)
plt.bar(np.arange(5), aggregate_share_ssp3['SSP3_2050'], label = "2050 - SSP3", width = 0.15, color = rgba3)
plt.bar(np.arange(5)+0.15, aggregate_share_ssp4['SSP4_2050'], label = "2050 - SSP4", width = 0.15, color = rgba4)
plt.bar(np.arange(5)+0.3, aggregate_share_ssp5['SSP5_2050'], label = "2050 - SSP5", width = 0.15, color = rgba5)
plt.xticks(np.arange(5), aggregate_share_ssp5.index)
plt.ylim(0, 1.3)
plt.legend(ncol = 3)


### POLICY

df_area_esa_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_esa_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_esa_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_esa_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_esa_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_area_esa_ssp1_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_esa_ssp2_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_esa_ssp3_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_esa_ssp4_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_esa_ssp5_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_area_GHSL_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_area_GHSL_ssp1_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp2_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp3_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp4_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_area_GHSL_ssp5_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_core_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_low_density_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_core_ssp1_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp2_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp3_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp4_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_core_ssp5_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_low_density_ssp1_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp2_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp3_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp4_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_low_density_ssp5_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_pop_core_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_pop_low_density_ssp1 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp2 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp3 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp4 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp5 = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_pop_core_ssp1_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp2_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp3_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp4_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_core_ssp5_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))

df_pop_low_density_ssp1_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp2_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp3_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp4_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))
df_pop_low_density_ssp5_ct = pd.DataFrame(index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]), columns = np.arange(2015, 2051))


path_ssp1_ct = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp1_ct/"
path_ssp2_ct = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp2_ct/"
path_ssp3_ct = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp3_ct/"
path_ssp4_ct = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp4_ct/"
path_ssp5_ct = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ssp5_ct/"


for city in list(sample_of_cities.City[sample_of_cities.final_sample == 1]):
    
    urb_ssp1 = np.load(path_ssp1 + city + "_ua.npy")
    urb_ssp2 = np.load(path_ssp2 + city + "_ua.npy")
    urb_ssp3 = np.load(path_ssp3 + city + "_ua.npy")
    urb_ssp4 = np.load(path_ssp4 + city + "_ua.npy")
    urb_ssp5 = np.load(path_ssp5 + city + "_ua.npy")
    
    df_area_esa_ssp1.loc[df_area_esa_ssp1.index == city] = urb_ssp1
    df_area_esa_ssp2.loc[df_area_esa_ssp2.index == city] = urb_ssp2
    df_area_esa_ssp3.loc[df_area_esa_ssp3.index == city] = urb_ssp3
    df_area_esa_ssp4.loc[df_area_esa_ssp4.index == city] = urb_ssp4
    df_area_esa_ssp5.loc[df_area_esa_ssp5.index == city] = urb_ssp5
    
    urb_ssp1_ct = np.load(path_ssp1_ct + city + "_ua.npy")
    urb_ssp2_ct = np.load(path_ssp2_ct + city + "_ua.npy")
    urb_ssp3_ct = np.load(path_ssp3_ct + city + "_ua.npy")
    urb_ssp4_ct = np.load(path_ssp4_ct + city + "_ua.npy")
    urb_ssp5_ct = np.load(path_ssp5_ct + city + "_ua.npy")
    
    df_area_esa_ssp1_ct.loc[df_area_esa_ssp1_ct.index == city] = urb_ssp1_ct
    df_area_esa_ssp2_ct.loc[df_area_esa_ssp2_ct.index == city] = urb_ssp2_ct
    df_area_esa_ssp3_ct.loc[df_area_esa_ssp3_ct.index == city] = urb_ssp3_ct
    df_area_esa_ssp4_ct.loc[df_area_esa_ssp4_ct.index == city] = urb_ssp4_ct
    df_area_esa_ssp5_ct.loc[df_area_esa_ssp5_ct.index == city] = urb_ssp5_ct
    
    save_density_ssp1 = np.load(path_ssp1 + city + "_density.npy")
    save_density_ssp2 = np.load(path_ssp2 + city + "_density.npy")
    save_density_ssp3 = np.load(path_ssp3 + city + "_density.npy")
    save_density_ssp4 = np.load(path_ssp4 + city + "_density.npy")
    save_density_ssp5 = np.load(path_ssp5 + city + "_density.npy")
    
    df_area_GHSL_ssp1.loc[df_area_GHSL_ssp1.index == city] = np.nansum(save_density_ssp1 > 150, 1)
    df_area_GHSL_ssp2.loc[df_area_GHSL_ssp2.index == city] = np.nansum(save_density_ssp2 > 150, 1)
    df_area_GHSL_ssp3.loc[df_area_GHSL_ssp3.index == city] = np.nansum(save_density_ssp3 > 150, 1)
    df_area_GHSL_ssp4.loc[df_area_GHSL_ssp4.index == city] = np.nansum(save_density_ssp4 > 150, 1)
    df_area_GHSL_ssp5.loc[df_area_GHSL_ssp5.index == city] = np.nansum(save_density_ssp5 > 150, 1)
    
    save_density_ssp1_ct = np.load(path_ssp1_ct + city + "_density.npy")
    save_density_ssp2_ct = np.load(path_ssp2_ct + city + "_density.npy")
    save_density_ssp3_ct = np.load(path_ssp3_ct + city + "_density.npy")
    save_density_ssp4_ct = np.load(path_ssp4_ct + city + "_density.npy")
    save_density_ssp5_ct = np.load(path_ssp5_ct + city + "_density.npy")
    
    df_area_GHSL_ssp1_ct.loc[df_area_GHSL_ssp1_ct.index == city] = np.nansum(save_density_ssp1_ct > 150, 1)
    df_area_GHSL_ssp2_ct.loc[df_area_GHSL_ssp2_ct.index == city] = np.nansum(save_density_ssp2_ct > 150, 1)
    df_area_GHSL_ssp3_ct.loc[df_area_GHSL_ssp3_ct.index == city] = np.nansum(save_density_ssp3_ct > 150, 1)
    df_area_GHSL_ssp4_ct.loc[df_area_GHSL_ssp4_ct.index == city] = np.nansum(save_density_ssp4_ct > 150, 1)
    df_area_GHSL_ssp5_ct.loc[df_area_GHSL_ssp5_ct.index == city] = np.nansum(save_density_ssp5_ct > 150, 1)
    
    df_core_ssp1.loc[df_core_ssp1.index == city] = np.nansum(save_density_ssp1 > 1500, 1)
    df_core_ssp2.loc[df_core_ssp2.index == city] = np.nansum(save_density_ssp2 > 1500, 1)
    df_core_ssp3.loc[df_core_ssp3.index == city] = np.nansum(save_density_ssp3 > 1500, 1)
    df_core_ssp4.loc[df_core_ssp4.index == city] = np.nansum(save_density_ssp4 > 1500, 1)
    df_core_ssp5.loc[df_core_ssp5.index == city] = np.nansum(save_density_ssp5 > 1500, 1)
    
    df_low_density_ssp1.loc[df_low_density_ssp1.index == city] = np.nansum((save_density_ssp1 > 150) & (save_density_ssp1 < 1500), 1)
    df_low_density_ssp2.loc[df_low_density_ssp2.index == city] = np.nansum((save_density_ssp2 > 150) & (save_density_ssp2 < 1500), 1)
    df_low_density_ssp3.loc[df_low_density_ssp3.index == city] = np.nansum((save_density_ssp3 > 150) & (save_density_ssp3 < 1500), 1)
    df_low_density_ssp4.loc[df_low_density_ssp4.index == city] = np.nansum((save_density_ssp4 > 150) & (save_density_ssp4 < 1500), 1)
    df_low_density_ssp5.loc[df_low_density_ssp5.index == city] = np.nansum((save_density_ssp5 > 150) & (save_density_ssp5 < 1500), 1)

    df_core_ssp1_ct.loc[df_core_ssp1_ct.index == city] = np.nansum(save_density_ssp1_ct > 1500, 1)
    df_core_ssp2_ct.loc[df_core_ssp2_ct.index == city] = np.nansum(save_density_ssp2_ct > 1500, 1)
    df_core_ssp3_ct.loc[df_core_ssp3_ct.index == city] = np.nansum(save_density_ssp3_ct > 1500, 1)
    df_core_ssp4_ct.loc[df_core_ssp4_ct.index == city] = np.nansum(save_density_ssp4_ct > 1500, 1)
    df_core_ssp5_ct.loc[df_core_ssp5_ct.index == city] = np.nansum(save_density_ssp5_ct > 1500, 1)
    
    df_low_density_ssp1_ct.loc[df_low_density_ssp1_ct.index == city] = np.nansum((save_density_ssp1_ct > 150) & (save_density_ssp1_ct < 1500), 1)
    df_low_density_ssp2_ct.loc[df_low_density_ssp2_ct.index == city] = np.nansum((save_density_ssp2_ct > 150) & (save_density_ssp2_ct < 1500), 1)
    df_low_density_ssp3_ct.loc[df_low_density_ssp3_ct.index == city] = np.nansum((save_density_ssp3_ct > 150) & (save_density_ssp3_ct < 1500), 1)
    df_low_density_ssp4_ct.loc[df_low_density_ssp4_ct.index == city] = np.nansum((save_density_ssp4_ct > 150) & (save_density_ssp4_ct < 1500), 1)
    df_low_density_ssp5_ct.loc[df_low_density_ssp5_ct.index == city] = np.nansum((save_density_ssp5_ct > 150) & (save_density_ssp5_ct < 1500), 1)

    save_density_ssp1_core = np.where(save_density_ssp1 < 1500, 0, save_density_ssp1)
    save_density_ssp2_core = np.where(save_density_ssp2 < 1500, 0, save_density_ssp2)
    save_density_ssp3_core = np.where(save_density_ssp3 < 1500, 0, save_density_ssp3)
    save_density_ssp4_core = np.where(save_density_ssp4 < 1500, 0, save_density_ssp4)
    save_density_ssp5_core = np.where(save_density_ssp5 < 1500, 0, save_density_ssp5)
    
    save_density_ssp1_low_density = np.where((save_density_ssp1 > 1500) | (save_density_ssp1 < 150), 0, save_density_ssp1)
    save_density_ssp2_low_density = np.where((save_density_ssp2 > 1500) | (save_density_ssp2 < 150), 0, save_density_ssp2)
    save_density_ssp3_low_density = np.where((save_density_ssp3 > 1500) | (save_density_ssp3 < 150), 0, save_density_ssp3)
    save_density_ssp4_low_density = np.where((save_density_ssp4 > 1500) | (save_density_ssp4 < 150), 0, save_density_ssp4)
    save_density_ssp5_low_density = np.where((save_density_ssp5 > 1500) | (save_density_ssp5 < 150), 0, save_density_ssp5)
    
    df_pop_core_ssp1.loc[df_pop_core_ssp1.index == city] = np.nansum(save_density_ssp1_core, 1)
    df_pop_core_ssp2.loc[df_pop_core_ssp2.index == city] = np.nansum(save_density_ssp2_core, 1)
    df_pop_core_ssp3.loc[df_pop_core_ssp3.index == city] = np.nansum(save_density_ssp3_core, 1)
    df_pop_core_ssp4.loc[df_pop_core_ssp4.index == city] = np.nansum(save_density_ssp4_core, 1)
    df_pop_core_ssp5.loc[df_pop_core_ssp5.index == city] = np.nansum(save_density_ssp5_core, 1)
    
    df_pop_low_density_ssp1.loc[df_pop_low_density_ssp1.index == city] = np.nansum(save_density_ssp1_low_density, 1)
    df_pop_low_density_ssp2.loc[df_pop_low_density_ssp2.index == city] = np.nansum(save_density_ssp2_low_density, 1)
    df_pop_low_density_ssp3.loc[df_pop_low_density_ssp3.index == city] = np.nansum(save_density_ssp3_low_density, 1)
    df_pop_low_density_ssp4.loc[df_pop_low_density_ssp4.index == city] = np.nansum(save_density_ssp4_low_density, 1)
    df_pop_low_density_ssp5.loc[df_pop_low_density_ssp5.index == city] = np.nansum(save_density_ssp5_low_density, 1)

    save_density_ssp1_core_ct = np.where(save_density_ssp1_ct < 1500, 0, save_density_ssp1_ct)
    save_density_ssp2_core_ct = np.where(save_density_ssp2_ct < 1500, 0, save_density_ssp2_ct)
    save_density_ssp3_core_ct = np.where(save_density_ssp3_ct < 1500, 0, save_density_ssp3_ct)
    save_density_ssp4_core_ct = np.where(save_density_ssp4_ct < 1500, 0, save_density_ssp4_ct)
    save_density_ssp5_core_ct = np.where(save_density_ssp5_ct < 1500, 0, save_density_ssp5_ct)
    
    save_density_ssp1_low_density_ct = np.where((save_density_ssp1_ct > 1500) | (save_density_ssp1_ct < 150), 0, save_density_ssp1_ct)
    save_density_ssp2_low_density_ct = np.where((save_density_ssp2_ct > 1500) | (save_density_ssp2_ct < 150), 0, save_density_ssp2_ct)
    save_density_ssp3_low_density_ct = np.where((save_density_ssp3_ct > 1500) | (save_density_ssp3_ct < 150), 0, save_density_ssp3_ct)
    save_density_ssp4_low_density_ct = np.where((save_density_ssp4_ct > 1500) | (save_density_ssp4_ct < 150), 0, save_density_ssp4_ct)
    save_density_ssp5_low_density_ct = np.where((save_density_ssp5_ct > 1500) | (save_density_ssp5_ct < 150), 0, save_density_ssp5_ct)
    
    df_pop_core_ssp1_ct.loc[df_pop_core_ssp1_ct.index == city] = np.nansum(save_density_ssp1_core_ct, 1)
    df_pop_core_ssp2_ct.loc[df_pop_core_ssp2_ct.index == city] = np.nansum(save_density_ssp2_core_ct, 1)
    df_pop_core_ssp3_ct.loc[df_pop_core_ssp3_ct.index == city] = np.nansum(save_density_ssp3_core_ct, 1)
    df_pop_core_ssp4_ct.loc[df_pop_core_ssp4_ct.index == city] = np.nansum(save_density_ssp4_core_ct, 1)
    df_pop_core_ssp5_ct.loc[df_pop_core_ssp5_ct.index == city] = np.nansum(save_density_ssp5_core_ct, 1)
    
    df_pop_low_density_ssp1_ct.loc[df_pop_low_density_ssp1_ct.index == city] = np.nansum(save_density_ssp1_low_density_ct, 1)
    df_pop_low_density_ssp2_ct.loc[df_pop_low_density_ssp2_ct.index == city] = np.nansum(save_density_ssp2_low_density_ct, 1)
    df_pop_low_density_ssp3_ct.loc[df_pop_low_density_ssp3_ct.index == city] = np.nansum(save_density_ssp3_low_density_ct, 1)
    df_pop_low_density_ssp4_ct.loc[df_pop_low_density_ssp4_ct.index == city] = np.nansum(save_density_ssp4_low_density_ct, 1)
    df_pop_low_density_ssp5_ct.loc[df_pop_low_density_ssp5_ct.index == city] = np.nansum(save_density_ssp5_low_density_ct, 1)


city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')


city_continent = city_continent.merge(class_region, on = "City")
city_continent = city_continent.loc[:,["City","IPCC2"]]
city_continent.columns = ["City", "Continent"]

df_area_GHSL_ssp1 = df_area_GHSL_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp2 = df_area_GHSL_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp3 = df_area_GHSL_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp4 = df_area_GHSL_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp5 = df_area_GHSL_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_GHSL_ssp1 = df_area_GHSL_ssp1.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp1.Continent).sum()
aggregate_GHSL_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_GHSL_ssp2 = df_area_GHSL_ssp2.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp2.Continent).sum()
aggregate_GHSL_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_GHSL_ssp3 = df_area_GHSL_ssp3.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp3.Continent).sum()
aggregate_GHSL_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_GHSL_ssp4 = df_area_GHSL_ssp4.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp4.Continent).sum()
aggregate_GHSL_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_GHSL_ssp5 = df_area_GHSL_ssp5.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp5.Continent).sum()
aggregate_GHSL_ssp5.columns = ['year_2015', 'SSP5_2050']

df_area_GHSL_ssp1_ct = df_area_GHSL_ssp1_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp2_ct = df_area_GHSL_ssp2_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp3_ct = df_area_GHSL_ssp3_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp4_ct = df_area_GHSL_ssp4_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_area_GHSL_ssp5_ct = df_area_GHSL_ssp5_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_GHSL_ssp1_ct = df_area_GHSL_ssp1_ct.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp1_ct.Continent).sum()
aggregate_GHSL_ssp1_ct.columns = ['year_2015', 'SSP1_2050_ct']

aggregate_GHSL_ssp2_ct = df_area_GHSL_ssp2_ct.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp2_ct.Continent).sum()
aggregate_GHSL_ssp2_ct.columns = ['year_2015', 'SSP2_2050_ct']

aggregate_GHSL_ssp3_ct = df_area_GHSL_ssp3_ct.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp3_ct.Continent).sum()
aggregate_GHSL_ssp3_ct.columns = ['year_2015', 'SSP3_2050_ct']

aggregate_GHSL_ssp4_ct = df_area_GHSL_ssp4_ct.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp4_ct.Continent).sum()
aggregate_GHSL_ssp4_ct.columns = ['year_2015', 'SSP4_2050_ct']

aggregate_GHSL_ssp5_ct = df_area_GHSL_ssp5_ct.iloc[:,[0,35]].astype(float).groupby(df_area_GHSL_ssp5_ct.Continent).sum()
aggregate_GHSL_ssp5_ct.columns = ['year_2015', 'SSP5_2050_ct']




df_core_ssp1 = df_core_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_core_ssp2 = df_core_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_core_ssp3 = df_core_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_core_ssp4 = df_core_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_core_ssp5 = df_core_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_core_ssp1 = df_core_ssp1.iloc[:,[0,35]].astype(float).groupby(df_core_ssp1.Continent).sum()
aggregate_core_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_core_ssp2 = df_core_ssp2.iloc[:,[0,35]].astype(float).groupby(df_core_ssp2.Continent).sum()
aggregate_core_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_core_ssp3 = df_core_ssp3.iloc[:,[0,35]].astype(float).groupby(df_core_ssp3.Continent).sum()
aggregate_core_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_core_ssp4 = df_core_ssp4.iloc[:,[0,35]].astype(float).groupby(df_core_ssp4.Continent).sum()
aggregate_core_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_core_ssp5 = df_core_ssp5.iloc[:,[0,35]].astype(float).groupby(df_core_ssp5.Continent).sum()
aggregate_core_ssp5.columns = ['year_2015', 'SSP5_2050']

df_core_ssp1_ct = df_core_ssp1_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_core_ssp2_ct = df_core_ssp2_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_core_ssp3_ct = df_core_ssp3_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_core_ssp4_ct = df_core_ssp4_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_core_ssp5_ct = df_core_ssp5_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_core_ssp1_ct = df_core_ssp1_ct.iloc[:,[0,35]].astype(float).groupby(df_core_ssp1_ct.Continent).sum()
aggregate_core_ssp1_ct.columns = ['year_2015', 'SSP1_2050_ct']

aggregate_core_ssp2_ct = df_core_ssp2_ct.iloc[:,[0,35]].astype(float).groupby(df_core_ssp2_ct.Continent).sum()
aggregate_core_ssp2_ct.columns = ['year_2015', 'SSP2_2050_ct']

aggregate_core_ssp3_ct = df_core_ssp3_ct.iloc[:,[0,35]].astype(float).groupby(df_core_ssp3_ct.Continent).sum()
aggregate_core_ssp3_ct.columns = ['year_2015', 'SSP3_2050_ct']

aggregate_core_ssp4_ct = df_core_ssp4_ct.iloc[:,[0,35]].astype(float).groupby(df_core_ssp4_ct.Continent).sum()
aggregate_core_ssp4_ct.columns = ['year_2015', 'SSP4_2050_ct']

aggregate_core_ssp5_ct = df_core_ssp5_ct.iloc[:,[0,35]].astype(float).groupby(df_core_ssp5_ct.Continent).sum()
aggregate_core_ssp5_ct.columns = ['year_2015', 'SSP5_2050_ct']


df_low_density_ssp1 = df_low_density_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp2 = df_low_density_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp3 = df_low_density_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp4 = df_low_density_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp5 = df_low_density_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_low_density_ssp1 = df_low_density_ssp1.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp1.Continent).sum()
aggregate_low_density_ssp1.columns = ['year_2015', 'SSP1_2050']

aggregate_low_density_ssp2 = df_low_density_ssp2.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp2.Continent).sum()
aggregate_low_density_ssp2.columns = ['year_2015', 'SSP2_2050']

aggregate_low_density_ssp3 = df_low_density_ssp3.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp3.Continent).sum()
aggregate_low_density_ssp3.columns = ['year_2015', 'SSP3_2050']

aggregate_low_density_ssp4 = df_low_density_ssp4.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp4.Continent).sum()
aggregate_low_density_ssp4.columns = ['year_2015', 'SSP4_2050']

aggregate_low_density_ssp5 = df_low_density_ssp5.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp5.Continent).sum()
aggregate_low_density_ssp5.columns = ['year_2015', 'SSP5_2050']

df_low_density_ssp1_ct = df_low_density_ssp1_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp2_ct = df_low_density_ssp2_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp3_ct = df_low_density_ssp3_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp4_ct = df_low_density_ssp4_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_low_density_ssp5_ct = df_low_density_ssp5_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_low_density_ssp1_ct = df_low_density_ssp1_ct.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp1_ct.Continent).sum()
aggregate_low_density_ssp1_ct.columns = ['year_2015', 'SSP1_2050_ct']

aggregate_low_density_ssp2_ct = df_low_density_ssp2_ct.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp2_ct.Continent).sum()
aggregate_low_density_ssp2_ct.columns = ['year_2015', 'SSP2_2050_ct']

aggregate_low_density_ssp3_ct = df_low_density_ssp3_ct.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp3_ct.Continent).sum()
aggregate_low_density_ssp3_ct.columns = ['year_2015', 'SSP3_2050_ct']

aggregate_low_density_ssp4_ct = df_low_density_ssp4_ct.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp4_ct.Continent).sum()
aggregate_low_density_ssp4_ct.columns = ['year_2015', 'SSP4_2050_ct']

aggregate_low_density_ssp5_ct = df_low_density_ssp5_ct.iloc[:,[0,35]].astype(float).groupby(df_low_density_ssp5_ct.Continent).sum()
aggregate_low_density_ssp5_ct.columns = ['year_2015', 'SSP5_2050_ct']

df_pop_core_ssp1 = df_pop_core_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp2 = df_pop_core_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp3 = df_pop_core_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp4 = df_pop_core_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp5 = df_pop_core_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_pop_core_ssp1 = df_pop_core_ssp1.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp1.Continent).sum()
aggregate_pop_core_ssp1.columns = ['year_2015', 'SSP1_2050']
aggregate_pop_core_ssp2 = df_pop_core_ssp2.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp2.Continent).sum()
aggregate_pop_core_ssp2.columns = ['year_2015', 'SSP2_2050']
aggregate_pop_core_ssp3 = df_pop_core_ssp3.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp3.Continent).sum()
aggregate_pop_core_ssp3.columns = ['year_2015', 'SSP3_2050']
aggregate_pop_core_ssp4 = df_pop_core_ssp4.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp4.Continent).sum()
aggregate_pop_core_ssp4.columns = ['year_2015', 'SSP4_2050']
aggregate_pop_core_ssp5 = df_pop_core_ssp5.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp5.Continent).sum()
aggregate_pop_core_ssp5.columns = ['year_2015', 'SSP5_2050']

df_pop_core_ssp1_ct = df_pop_core_ssp1_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp2_ct = df_pop_core_ssp2_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp3_ct = df_pop_core_ssp3_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp4_ct = df_pop_core_ssp4_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_core_ssp5_ct = df_pop_core_ssp5_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_pop_core_ssp1_ct = df_pop_core_ssp1_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp1_ct.Continent).sum()
aggregate_pop_core_ssp1_ct.columns = ['year_2015', 'SSP1_2050_ct']
aggregate_pop_core_ssp2_ct = df_pop_core_ssp2_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp2_ct.Continent).sum()
aggregate_pop_core_ssp2_ct.columns = ['year_2015', 'SSP2_2050_ct']
aggregate_pop_core_ssp3_ct = df_pop_core_ssp3_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp3_ct.Continent).sum()
aggregate_pop_core_ssp3_ct.columns = ['year_2015', 'SSP3_2050_ct']
aggregate_pop_core_ssp4_ct = df_pop_core_ssp4_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp4_ct.Continent).sum()
aggregate_pop_core_ssp4_ct.columns = ['year_2015', 'SSP4_2050_ct']
aggregate_pop_core_ssp5_ct = df_pop_core_ssp5_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_core_ssp5_ct.Continent).sum()
aggregate_pop_core_ssp5_ct.columns = ['year_2015', 'SSP5_2050_ct']


df_pop_low_density_ssp1 = df_pop_low_density_ssp1.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp2 = df_pop_low_density_ssp2.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp3 = df_pop_low_density_ssp3.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp4 = df_pop_low_density_ssp4.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp5 = df_pop_low_density_ssp5.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_pop_low_density_ssp1 = df_pop_low_density_ssp1.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp1.Continent).sum()
aggregate_pop_low_density_ssp1.columns = ['year_2015', 'SSP1_2050']
aggregate_pop_low_density_ssp2 = df_pop_low_density_ssp2.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp2.Continent).sum()
aggregate_pop_low_density_ssp2.columns = ['year_2015', 'SSP2_2050']
aggregate_pop_low_density_ssp3 = df_pop_low_density_ssp3.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp3.Continent).sum()
aggregate_pop_low_density_ssp3.columns = ['year_2015', 'SSP3_2050']
aggregate_pop_low_density_ssp4 = df_pop_low_density_ssp4.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp4.Continent).sum()
aggregate_pop_low_density_ssp4.columns = ['year_2015', 'SSP4_2050']
aggregate_pop_low_density_ssp5 = df_pop_low_density_ssp5.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp5.Continent).sum()
aggregate_pop_low_density_ssp5.columns = ['year_2015', 'SSP5_2050']

df_pop_low_density_ssp1_ct = df_pop_low_density_ssp1_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp2_ct = df_pop_low_density_ssp2_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp3_ct = df_pop_low_density_ssp3_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp4_ct = df_pop_low_density_ssp4_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')
df_pop_low_density_ssp5_ct = df_pop_low_density_ssp5_ct.merge(city_continent, left_index = True, right_on = "City", how = 'left')

aggregate_pop_low_density_ssp1_ct = df_pop_low_density_ssp1_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp1_ct.Continent).sum()
aggregate_pop_low_density_ssp1_ct.columns = ['year_2015', 'SSP1_2050_ct']
aggregate_pop_low_density_ssp2_ct = df_pop_low_density_ssp2_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp2_ct.Continent).sum()
aggregate_pop_low_density_ssp2_ct.columns = ['year_2015', 'SSP2_2050_ct']
aggregate_pop_low_density_ssp3_ct = df_pop_low_density_ssp3_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp3_ct.Continent).sum()
aggregate_pop_low_density_ssp3_ct.columns = ['year_2015', 'SSP3_2050_ct']
aggregate_pop_low_density_ssp4_ct = df_pop_low_density_ssp4_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp4_ct.Continent).sum()
aggregate_pop_low_density_ssp4_ct.columns = ['year_2015', 'SSP4_2050_ct']
aggregate_pop_low_density_ssp5_ct = df_pop_low_density_ssp5_ct.iloc[:,[0,35]].astype(float).groupby(df_pop_low_density_ssp5_ct.Continent).sum()
aggregate_pop_low_density_ssp5_ct.columns = ['year_2015', 'SSP5_2050_ct']







cmap = plt.cm.get_cmap('plasma')
rgba1 = cmap(0.1)
rgba2 = cmap(0.3)
rgba3 = cmap(0.5)
rgba4 = cmap(0.7)
rgba5 = cmap(0.9)

plt.bar(np.arange(5)-0.3, 100 * (aggregate_GHSL_ssp1_ct['SSP1_2050_ct'] - aggregate_GHSL_ssp1['SSP1_2050'] )/aggregate_GHSL_ssp1['SSP1_2050'], label = "2050 - SSP1", width = 0.15, color = rgba1)
plt.bar(np.arange(5)-0.15, 100 * (aggregate_GHSL_ssp2_ct['SSP2_2050_ct'] - aggregate_GHSL_ssp2['SSP2_2050'] )/aggregate_GHSL_ssp2['SSP2_2050'], label = "2050 - SSP2", width = 0.15, color = rgba2)
plt.bar(np.arange(5), 100 * (aggregate_GHSL_ssp3_ct['SSP3_2050_ct'] - aggregate_GHSL_ssp3['SSP3_2050'] )/aggregate_GHSL_ssp3['SSP3_2050'], label = "2050 - SSP3", width = 0.15, color = rgba3)
plt.bar(np.arange(5)+0.15, 100 * (aggregate_GHSL_ssp4_ct['SSP4_2050_ct'] - aggregate_GHSL_ssp4['SSP4_2050'] )/aggregate_GHSL_ssp4['SSP4_2050'], label = "2050 - SSP4", width = 0.15, color = rgba4)
plt.bar(np.arange(5)+0.3, 100 * (aggregate_GHSL_ssp5_ct['SSP5_2050_ct'] - aggregate_GHSL_ssp5['SSP5_2050'] )/aggregate_GHSL_ssp5['SSP5_2050'], label = "2050 - SSP5", width = 0.15, color = rgba5)
plt.xticks(np.arange(5), aggregate_GHSL_ssp1_ct.index)
plt.ylim(-1,0)
plt.legend(ncol = 3)

plt.bar(np.arange(5)-0.3, 100 * (aggregate_core_ssp1_ct['SSP1_2050_ct'] - aggregate_core_ssp1['SSP1_2050'] )/aggregate_core_ssp1['SSP1_2050'], label = "2050 - SSP1", width = 0.15, color = rgba1)
plt.bar(np.arange(5)-0.15, 100 * (aggregate_core_ssp2_ct['SSP2_2050_ct'] - aggregate_core_ssp2['SSP2_2050'] )/aggregate_core_ssp2['SSP2_2050'], label = "2050 - SSP2", width = 0.15, color = rgba2)
plt.bar(np.arange(5), 100 * (aggregate_core_ssp3_ct['SSP3_2050_ct'] - aggregate_core_ssp3['SSP3_2050'] )/aggregate_core_ssp3['SSP3_2050'], label = "2050 - SSP3", width = 0.15, color = rgba3)
plt.bar(np.arange(5)+0.15, 100 * (aggregate_core_ssp4_ct['SSP4_2050_ct'] - aggregate_core_ssp4['SSP4_2050'] )/aggregate_core_ssp4['SSP4_2050'], label = "2050 - SSP4", width = 0.15, color = rgba4)
plt.bar(np.arange(5)+0.3, 100 * (aggregate_core_ssp5_ct['SSP5_2050_ct'] - aggregate_core_ssp5['SSP5_2050'] )/aggregate_core_ssp5['SSP5_2050'], label = "2050 - SSP5", width = 0.15, color = rgba5)
plt.xticks(np.arange(5), aggregate_core_ssp1_ct.index)
plt.ylim(-2, 14)
plt.legend(ncol = 3)

plt.bar(np.arange(5)-0.3, 100 * (aggregate_low_density_ssp1_ct['SSP1_2050_ct'] - aggregate_low_density_ssp1['SSP1_2050'] )/aggregate_low_density_ssp1['SSP1_2050'], label = "2050 - SSP1", width = 0.15, color = rgba1)
plt.bar(np.arange(5)-0.15, 100 * (aggregate_low_density_ssp2_ct['SSP2_2050_ct'] - aggregate_low_density_ssp2['SSP2_2050'] )/aggregate_low_density_ssp2['SSP2_2050'], label = "2050 - SSP2", width = 0.15, color = rgba2)
plt.bar(np.arange(5), 100 * (aggregate_low_density_ssp3_ct['SSP3_2050_ct'] - aggregate_low_density_ssp3['SSP3_2050'] )/aggregate_low_density_ssp3['SSP3_2050'], label = "2050 - SSP3", width = 0.15, color = rgba3)
plt.bar(np.arange(5)+0.15, 100 * (aggregate_low_density_ssp4_ct['SSP4_2050_ct'] - aggregate_low_density_ssp4['SSP4_2050'] )/aggregate_low_density_ssp4['SSP4_2050'], label = "2050 - SSP4", width = 0.15, color = rgba4)
plt.bar(np.arange(5)+0.3, 100 * (aggregate_low_density_ssp5_ct['SSP5_2050_ct'] - aggregate_low_density_ssp5['SSP5_2050'] )/aggregate_low_density_ssp5['SSP5_2050'], label = "2050 - SSP5", width = 0.15, color = rgba5)
plt.xticks(np.arange(5), aggregate_low_density_ssp1_ct.index)
plt.ylim(-1,1)
plt.legend(ncol = 3)

plt.rcParams["figure.autolayout"] = True
s = {"axes.spines.left"   : False,
    "axes.spines.bottom" : False,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False}
plt.rcParams.update(s)
plt.rc('axes',edgecolor='black')
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=True, labelbottom=True)
plt.grid(axis = 'y', color = 'grey', linewidth = 0.5, alpha = 0.5)
plt.bar(np.arange(8)-0.3, 100 * (aggregate_GHSL_ssp2_ct['SSP2_2050_ct'] - aggregate_GHSL_ssp2['SSP2_2050'] )/aggregate_GHSL_ssp2['SSP2_2050'], label = "Urban areas", width = 0.15, color = rgba1)
plt.bar(np.arange(8)-0.15, 100 * (aggregate_core_ssp2_ct['SSP2_2050_ct'] - aggregate_core_ssp2['SSP2_2050'] )/aggregate_core_ssp2['SSP2_2050'], label = "Core areas", width = 0.15, color = rgba2)
plt.bar(np.arange(8), 100 * (aggregate_pop_core_ssp2_ct['SSP2_2050_ct'] - aggregate_pop_core_ssp2['SSP2_2050'] )/aggregate_pop_core_ssp2['SSP2_2050'], label = "Core areas populations", width = 0.15, color = rgba3)
plt.bar(np.arange(8)+0.15, 100 * (aggregate_low_density_ssp2_ct['SSP2_2050_ct'] - aggregate_low_density_ssp2['SSP2_2050'] )/aggregate_low_density_ssp2['SSP2_2050'], label = "Low-density areas", width = 0.15, color = rgba4)
plt.bar(np.arange(8)+0.3, 100 * (aggregate_pop_low_density_ssp2_ct['SSP2_2050_ct'] - aggregate_pop_low_density_ssp2['SSP2_2050'] )/aggregate_pop_low_density_ssp2['SSP2_2050'], label = "Low-density areas population", width = 0.15, color = rgba5)
#plt.xticks(np.arange(5), ["Asia", "Europe", "North \n America", "Oceania", "South \n America"])
plt.xticks(np.arange(8), ["APC", "Aust", "East. Eur.", "LAC", "ME", "NA", "NWE", "SEE"])
plt.ylabel("Variation - %")
plt.legend(ncol = 1)






    
cmap = plt.cm.get_cmap('plasma')
rgba1 = cmap(0.1)
rgba2 = cmap(0.3)
rgba3 = cmap(0.5)
rgba4 = cmap(0.7)
rgba5 = cmap(0.9)

plt.plot(np.nansum(df_area_esa_ssp1, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_area_esa_ssp2, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_area_esa_ssp3, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_area_esa_ssp4, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_area_esa_ssp5, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP5")
plt.plot(np.nansum(df_area_esa_ssp1, 0)/np.nansum(df_area_esa_ssp1, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_area_esa_ssp1_ct, 0)/np.nansum(df_area_esa_ssp1_ct, 0)[0], label = "SSP1")
plt.plot(np.nansum(df_area_esa_ssp2_ct, 0)/np.nansum(df_area_esa_ssp1_ct, 0)[0], label = "SSP2")
plt.plot(np.nansum(df_area_esa_ssp3_ct, 0)/np.nansum(df_area_esa_ssp1_ct, 0)[0], label = "SSP3")
plt.plot(np.nansum(df_area_esa_ssp4_ct, 0)/np.nansum(df_area_esa_ssp1_ct, 0)[0], label = "SSP4")
plt.plot(np.nansum(df_area_esa_ssp5_ct, 0)/np.nansum(df_area_esa_ssp1_ct, 0)[0], label = "SSP5")
plt.legend()

plt.plot(np.nansum(df_area_GHSL_ssp1, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP1", color = rgba1)
plt.plot(np.nansum(df_area_GHSL_ssp2, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP2", color = rgba2)
plt.plot(np.nansum(df_area_GHSL_ssp3, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP3", color = rgba3)
plt.plot(np.nansum(df_area_GHSL_ssp4, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP4", color = rgba4)
plt.plot(np.nansum(df_area_GHSL_ssp5, 0)/np.nansum(df_area_GHSL_ssp1, 0)[0], label = "SSP5", color = rgba5)
plt.plot(np.nansum(df_area_GHSL_ssp1_ct, 0)/np.nansum(df_area_GHSL_ssp1_ct, 0)[0], color = rgba1)
plt.plot(np.nansum(df_area_GHSL_ssp2_ct, 0)/np.nansum(df_area_GHSL_ssp1_ct, 0)[0], color = rgba2)
plt.plot(np.nansum(df_area_GHSL_ssp3_ct, 0)/np.nansum(df_area_GHSL_ssp1_ct, 0)[0], color = rgba3)
plt.plot(np.nansum(df_area_GHSL_ssp4_ct, 0)/np.nansum(df_area_GHSL_ssp1_ct, 0)[0], color = rgba4)
plt.plot(np.nansum(df_area_GHSL_ssp5_ct, 0)/np.nansum(df_area_GHSL_ssp1_ct, 0)[0], color = rgba5)
plt.legend()



