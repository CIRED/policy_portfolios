# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:31:35 2021

@author: charl
"""




import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math
import os

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20211025/"
#os.mkdir(path_calibration)


from calibration.calibration import *
from calibration.validation import *
from inputs.data import *
from inputs.land_use import *
from model.model import *
from outputs.outputs import *
from inputs.parameters import *
from inputs.transport import *

INTEREST_RATE, HOUSEHOLD_SIZE, TIME_LAG, DEPRECIATION_TIME, DURATION, COEFF_URB, FIXED_COST_CAR, WALKING_SPEED, BRT_SPEED, COST_PER_KM_BRT, CO2_EMISSIONS_TRANSIT = import_parameters()
policy =  'None'
index = 0
list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

rows = []
orig_dist = np.nan
target_dist = np.nan
transit_dist = np.nan
for city in np.unique(list_city.City)[154:192]:
    
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    
    #Density, rents and dwelling sizes
    density = density.loc[:,density.columns.str.startswith("density")],
    density = np.array(density).squeeze()
    rent = (rents_and_size.avgRent / conversion_rate) * 12
    if city == "Addis_Ababa":
        rent[rent > 50000] = rent / 100
        rent[rent > 500] = rent / 100
    if (city == 'Yerevan'):
        rent[rent > 1000] = np.nan
    if (city == 'Isfahan') | (city == 'Tehran'):
        rent = 100 * rent
    size = rents_and_size.medSize
    size[size > 1000] = np.nan
    
    #Land-use
    #coeff_land = (land_use.OpenedToUrb / land_use.TotalArea) * COEFF_URB
    #density_to_cover = convert_density_to_urban_footprint(city, path_data, list_city)
    
    #Other inputs
    population = np.nansum(density)
    region = pd.read_excel(path_folder + "city_country_region.xlsx").region[pd.read_excel(path_folder + "city_country_region.xlsx").city == city].squeeze()
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()
    agricultural_rent = import_agricultural_rent(path_folder, country)
    
    #Import transport data
    fuel_price = import_fuel_price(country, 'gasoline', path_folder) #L/100km
    fuel_consumption = import_fuel_conso(country, path_folder) #dollars/L
    CO2_emissions_car = 2300 * fuel_consumption / 100
    monetary_cost_pt = import_public_transport_cost_data(path_folder, city).squeeze()
        
        
    #Import scenarios
    imaclim = pd.read_excel(path_folder + "Charlotte_ResultsIncomePriceAutoEmissionsAuto_V1.xlsx", sheet_name = 'Extraction_Baseline')
    population_growth = import_city_scenarios(city, country, path_folder)
    if isinstance(population_growth["2015-2020"], pd.Series) == True:
        population_growth = import_country_scenarios(country, path_folder)
    income_growth = imaclim[imaclim.Region == region][imaclim.Variable == "Index_income"].squeeze()
     
    #if option["model_type"] == "schematic":
    #    natural_constraints = import_profile(distance_cbd, coeff_land, grille, 'out_of_grid') #grid_only        
    #    grille = GridSimulation()
    #    grille.create_grid(100)
    #    coeff_land = np.ones(len(grille.distance_centre))
    #   for i in range(len(natural_constraints)):
    #        coeff_land[(grille.distance_centre > i) & (grille.distance_centre > i + 1)] = natural_constraints[i]
    #    BETA = 0.3
    #    B = 0.64
    #    kappa = 2.014 * (pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == 'Paris'].squeeze()** B) / (income ** B)
    #    amenities = np.ones(len(grille.distance_centre))
        
    ### TRANSPORT MODELLING
    
    print("\n** Transport modelling **\n")
    
    prix_transport, mode_choice = transport_modeling(driving, transit, income, fuel_price, fuel_consumption, FIXED_COST_CAR, monetary_cost_pt, distance_cbd, WALKING_SPEED, policy, index, city, grille, centre, orig_dist, target_dist, transit_dist, BRT_SPEED)
    
    grad_density = LinearRegression().fit(np.array(distance_cbd[np.log(density)> 0]).reshape(-1, 1), np.log(density)[np.log(density)> 0]).coef_[0]
    
    #Income
    #data_gdp = pd.read_csv(path_folder + "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2163510.csv", header =2) 
    #data_gdp["Country Name"][data_gdp["Country Name"] == "Cote d\'Ivoire"] = "Ivory_Coast"
    #data_gdp["Country Name"][data_gdp["Country Name"] == "United States"] = "USA"
    #data_gdp["Country Name"][data_gdp["Country Name"] == "New Zealand"] = "New_Zealand"
    #data_gdp["Country Name"][data_gdp["Country Name"] == "United Kingdom"] = "UK"
    #data_gdp["Country Name"][data_gdp["Country Name"] == "South Africa"] = "South_Africa"
    #data_gdp["Country Name"][data_gdp["Country Name"] == "Russian Federation"] = "Russia"
    #data_gdp["Country Name"][data_gdp["Country Name"] == "Hong Kong SAR, China"] = "Hong_Kong"
    #data_gdp["Country Name"][data_gdp["Country Name"] == "Iran, Islamic Rep."] = "Iran"
    #data_gdp["Country Name"][data_gdp["Country Name"] == "Czech Republic"] = "Czech_Republic"
    #income = data_gdp["2018"][data_gdp["Country Name"] == country].squeeze() 
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()
    
    speed_driving = np.nansum(((driving.Distance / 1000) / (driving.Duration / 3600)) * density / np.nansum(density))
    speed_transit = np.nansum(((transit.Distance / 1000) / (transit.Duration / 3600)) * density / np.nansum(density))  
    cost_p = monetary_cost_pt
    cost_c = fuel_price * fuel_consumption / 100
    modal_share_car = sum(density[mode_choice == 0]) / sum(density)
    
    substitution_potential = np.nansum(density[(transit.Duration < 1.4 * driving.Duration) & (~np.isnan(transit.Duration))]) / np.nansum(density)
     
    #Beta, b, kappa
    BETA = np.array(np.load(path_calibration + "beta.npy", allow_pickle = True), ndmin = 1)[0][city]
    B = np.array(np.load(path_calibration + "b.npy", allow_pickle = True), ndmin = 1)[0][city]
    kappa = np.array(np.load(path_calibration + "kappa.npy", allow_pickle = True), ndmin = 1)[0][city]
    
    #Nat const.
    #coeff_land = 1 - (land_use.OpenedToUrb / land_use.TotalArea)
    #natural_constraints = import_profile(distance_cbd, coeff_land, grille, 'out_of_grid') #grid_only        
    #avg_cons = np.nanmean(natural_constraints)
    #avg_cons2 = np.nanmean(coeff_land)
    #weights = (math.pi * ((np.arange(0, int(max(distance_cbd))) + 1) ** 2 )) - (math.pi * ((np.arange(0, int(max(distance_cbd)))) ** 2 ))
    #avg_cons3 = np.nansum(natural_constraints * weights) / np.nansum(weights)
    #gradient_cons = float(sm.OLS(natural_constraints, pd.DataFrame({"X": np.arange(0, int(max(distance_cbd))), "intercept": np.ones((int(max(distance_cbd)))).squeeze()})).fit().params['X'])
    
    #Variation income
    income_variation = income_growth[2050] / income_growth[2015]
    population_variation = ((1 + (population_growth['2015-2020'] / 100)) ** 5) * ((1 + (population_growth['2020-2025'] / 100)) ** 5) * ((1 + (population_growth['2025-2030'] / 100)) ** 5) * ((1 + (population_growth['2030-2035'] / 100)) ** 20)
    
    rows.append([city, population, income, cost_p, cost_c, agricultural_rent, BETA, B, kappa, income_variation, population_variation, substitution_potential, grad_density])
    
#city_characteristics = pd.DataFrame(rows, columns = ["city", "population", "income", "speed_driving", "speed_transit", "cost_p", "fuel_price", "fuel_conso", "modal_share_car", "Ra", "BETA", "B", "kappa", "avg_cons", "avg_cons2", "avg_cons3", "gradient_cons", "income_variation", "population_variation"])
city_characteristics = pd.DataFrame(rows, columns = ["city", "population", "income", "cost_p", "cost_c", "agricultural_rent", "BETA", "B", "kappa", "income_variation", "population_variation", "substitution_potential", "grad_density"])
city_characteristics.modal_share_other_sources[city_characteristics.modal_share_other_sources > 1] = city_characteristics.modal_share_other_sources / 100
city_characteristics.to_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics_20211027.xlsx")

city_characteristics = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics.xlsx")

subset_usa = city_characteristics[(city_characteristics.city == 'New_York') | (city_characteristics.city == 'Atlanta')|(city_characteristics.city == 'Portland')|(city_characteristics.city == 'San_Fransisco')|(city_characteristics.city == 'Washington_DC')]
describe_subset_usa = subset_usa.describe()

subset_sa = city_characteristics[(city_characteristics.city == 'Surabaya') | (city_characteristics.city == 'Singapore')|(city_characteristics.city == 'Bangkok')]
describe_subset_sa = subset_sa.describe()

subset_china = city_characteristics[(city_characteristics.city == 'Beijing') | (city_characteristics.city == 'Jinan')|(city_characteristics.city == 'Shanghai')|(city_characteristics.city == 'Wuhan')|(city_characteristics.city == 'Zhengzhou')]
describe_subset_china = subset_china.describe()


city_characteristics.describe()
city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
city_characteristics = city_characteristics.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

city_characteristics[city_characteristics.Continent == 'Oceania'].mean()