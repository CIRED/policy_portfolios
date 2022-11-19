# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:08:23 2021

@author: charl
"""

#### COBENEFITS ANALYSIS ####

# IMPORT DATA

import pandas as pd
import numpy as np
import copy
import pickle
import scipy as sc
from scipy import optimize
import matplotlib.pyplot as plt
import os


path_no_policy_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20210519/"
path_greeenbelt_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/greenbelt_20210519/"
path_carbon_tax_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbontax_20210519/"
path_transport_speed_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/ptspeed_20210519/"
path_fuelefficiency_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuelefficiency_20210519/"

path_no_policy_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20210819/"
path_greeenbelt_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/UGB_20210819/"
path_carbon_tax_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbontax_20210819/"
path_transport_speed_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/transit_speed_20210819/"
path_fuelefficiency_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuel_efficiency_20210819/"
path_basicinfra = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/basic_infra_20210819/"

#def death_avoided_per_year(density, mode_choice, distance_cbd, WALKING_SPEED, mortality_rate, pop):
#    pop = sum(density[mode_choice == 2])
#    avg_distance = np.nansum(distance_cbd[mode_choice == 2] * density[mode_choice == 2]) / pop
#    reference_volume = 168
#    RR = 0.89
#    return (60 * (avg_distance / WALKING_SPEED) / reference_volume) * (1 - RR) * mortality_rate * pop

def death_avoided_per_year(density, mode_choice, distance_cbd, WALKING_SPEED, mortality_rate):
    pop = sum(density[mode_choice == 2])
    avg_distance = np.nansum(distance_cbd[mode_choice == 2] * density[mode_choice == 2]) / pop
    reference_volume = 168
    RR = 0.89
    return (10 * 60 * (avg_distance / WALKING_SPEED) / reference_volume) * (1 - RR) * mortality_rate * (pop / sum(density))


### BAU
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
air_pollution_2015 = {}
air_pollution_2035 = {}
residential_emissions_2035 = {}
injuries_2035 = {}
housing_affordability_2015 = {}
housing_affordability_2035 = {}
death_avoided_2035 = {}

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_no_policy_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_no_policy_simulation + city + "_utility.npy")
    dwelling_size = np.load(path_no_policy_simulation + city + "_dwelling_size.npy")
    density = np.load(path_no_policy_simulation + city + "_density.npy")
    distance = np.load(path_no_policy_simulation + city + "_distance.npy")
    modal_shares = np.load(path_no_policy_simulation + city + "_modal_shares.npy")
    min_dist = np.argmin(distance)
    housing_affordability_2015[city] = dwelling_size[0][min_dist]
    housing_affordability_2035[city] = dwelling_size[20][min_dist]
    death_avoided_2035[city] = death_avoided_per_year(density[20], modal_shares[20], distance, 5, 1)
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    air_pollution_2015[city] = np.load(path_no_policy_simulation + city + "_emissions.npy")[0]/sum(density[0] > 400)
    air_pollution_2035[city] = np.load(path_no_policy_simulation + city + "_emissions.npy")[20]/sum(density[20] > 400)
    residential_emissions_2035[city] = np.nansum(dwelling_size[20] * density[20]) / np.nansum(dwelling_size[0] * density[0])
    injuries_2035[city] = sum(density[20] * (modal_shares[20] == 0) * distance) / sum(density[0] * (modal_shares[0] == 0) * distance)
    
BAU = pd.DataFrame()
BAU["city"] = emission_2015.keys()
BAU["emissions_2035"] = np.array(list(emission_2035.values())) # / np.array(list(emission_2015.values()))
BAU["utility_2035"] = np.array(list(utility_2035.values())) # / np.array(list(utility_2015.values()))
BAU["residential_emissions_2035"] = residential_emissions_2035.values()
BAU["injuries_2035"] = injuries_2035.values()
BAU["air_pollution_2035"] = np.array(list(air_pollution_2035.values())) #/ np.array(list(air_pollution_2015.values()))
BAU["housing_affordability_2035"] = np.array(list(housing_affordability_2035.values())) #/ np.array(list(housing_affordability_2015.values()))
BAU["death_avoided_2035"] = np.array(list(death_avoided_2035.values()))


### GREENBELT
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
air_pollution_2015 = {}
air_pollution_2035 = {}
residential_emissions_2035 = {}
injuries_2035 = {}
housing_affordability_2015 = {}
housing_affordability_2035 = {}
death_avoided_2035 = {}

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_greeenbelt_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_greeenbelt_simulation + city + "_utility.npy")
    dwelling_size = np.load(path_greeenbelt_simulation + city + "_dwelling_size.npy")
    density = np.load(path_greeenbelt_simulation + city + "_density.npy")
    distance = np.load(path_greeenbelt_simulation + city + "_distance.npy")
    min_dist = np.argmin(distance)
    housing_affordability_2015[city] = dwelling_size[0][min_dist]
    housing_affordability_2035[city] = dwelling_size[20][min_dist]
    modal_shares = np.load(path_greeenbelt_simulation + city + "_modal_shares.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    death_avoided_2035[city] = death_avoided_per_year(density[20], modal_shares[20], distance, 5, 1)
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    air_pollution_2015[city] = np.load(path_greeenbelt_simulation + city + "_emissions.npy")[0]/sum(density[0] > 400)
    air_pollution_2035[city] = np.load(path_greeenbelt_simulation + city + "_emissions.npy")[20]/sum(density[20] > 400)
    residential_emissions_2035[city] = np.nansum(dwelling_size[20] * density[20]) / np.nansum(dwelling_size[0] * density[0])
    injuries_2035[city] = sum(density[20] * (modal_shares[20] == 0) * distance) / sum(density[0] * (modal_shares[0] == 0) * distance)
    
greenbelt = pd.DataFrame()
greenbelt["city"] = emission_2015.keys()
greenbelt["emissions_2035"] = np.array(list(emission_2035.values())) # / np.array(list(emission_2015.values()))
greenbelt["utility_2035"] = np.array(list(utility_2035.values())) #/ np.array(list(utility_2015.values()))
greenbelt["residential_emissions_2035"] = residential_emissions_2035.values()
greenbelt["injuries_2035"] = injuries_2035.values()
greenbelt["air_pollution_2035"] = np.array(list(air_pollution_2035.values())) #/ np.array(list(air_pollution_2015.values()))
greenbelt["housing_affordability_2035"] = np.array(list(housing_affordability_2035.values())) #/ np.array(list(housing_affordability_2015.values()))
greenbelt["death_avoided_2035"] = np.array(list(death_avoided_2035.values()))

### CARBON TAX
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
air_pollution_2015 = {}
air_pollution_2035 = {}
residential_emissions_2035 = {}
injuries_2035 = {}
housing_affordability_2015 = {}
housing_affordability_2035 = {}
death_avoided_2035 = {}

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_carbon_tax_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_carbon_tax_simulation + city + "_utility.npy")
    distance = np.load(path_carbon_tax_simulation + city + "_distance.npy")
    dwelling_size = np.load(path_carbon_tax_simulation + city + "_dwelling_size.npy")
    density = np.load(path_carbon_tax_simulation + city + "_density.npy")
    modal_shares = np.load(path_carbon_tax_simulation + city + "_modal_shares.npy")
    min_dist = np.argmin(distance)
    housing_affordability_2015[city] = dwelling_size[0][min_dist]
    housing_affordability_2035[city] = dwelling_size[20][min_dist]
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    death_avoided_2035[city] = death_avoided_per_year(density[20], modal_shares[20], distance, 5, 1)
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    air_pollution_2015[city] = np.load(path_carbon_tax_simulation + city + "_emissions.npy")[0]/sum(density[0] > 400)
    air_pollution_2035[city] = np.load(path_carbon_tax_simulation + city + "_emissions.npy")[20]/sum(density[20] > 400)
    residential_emissions_2035[city] = np.nansum(dwelling_size[20] * density[20]) / np.nansum(dwelling_size[0] * density[0])
    injuries_2035[city] = sum(density[20] * (modal_shares[20] == 0) * distance) / sum(density[0] * (modal_shares[0] == 0) * distance)
    
carbontax = pd.DataFrame()
carbontax["city"] = emission_2015.keys()
carbontax["emissions_2035"] = np.array(list(emission_2035.values()))# / np.array(list(emission_2015.values()))
carbontax["utility_2035"] = np.array(list(utility_2035.values())) #/ np.array(list(utility_2015.values()))
carbontax["residential_emissions_2035"] = residential_emissions_2035.values()
carbontax["injuries_2035"] = injuries_2035.values()
carbontax["air_pollution_2035"] = np.array(list(air_pollution_2035.values())) #/ np.array(list(air_pollution_2015.values()))
carbontax["housing_affordability_2035"] = np.array(list(housing_affordability_2035.values())) #/ np.array(list(housing_affordability_2015.values()))
carbontax["death_avoided_2035"] = np.array(list(death_avoided_2035.values()))

### PT SPEED
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
air_pollution_2015 = {}
air_pollution_2035 = {}
residential_emissions_2035 = {}
injuries_2035 = {}
housing_affordability_2015 = {}
housing_affordability_2035 = {}
death_avoided_2035 = {}

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_transport_speed_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_transport_speed_simulation + city + "_utility.npy")
    distance = np.load(path_transport_speed_simulation + city + "_distance.npy")
    dwelling_size = np.load(path_transport_speed_simulation + city + "_dwelling_size.npy")
    density = np.load(path_transport_speed_simulation + city + "_density.npy")
    modal_shares = np.load(path_transport_speed_simulation + city + "_modal_shares.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    min_dist = np.argmin(distance)
    housing_affordability_2015[city] = dwelling_size[0][min_dist]
    housing_affordability_2035[city] = dwelling_size[20][min_dist]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    death_avoided_2035[city] = death_avoided_per_year(density[20], modal_shares[20], distance, 5, 1)
    air_pollution_2015[city] = np.load(path_transport_speed_simulation + city + "_emissions.npy")[0]/sum(density[0] > 400)
    air_pollution_2035[city] = np.load(path_transport_speed_simulation + city + "_emissions.npy")[20]/sum(density[20] > 400)
    residential_emissions_2035[city] = np.nansum(dwelling_size[20] * density[20]) / np.nansum(dwelling_size[0] * density[0])
    injuries_2035[city] = sum(density[20] * (modal_shares[20] == 0) * distance) / sum(density[0] * (modal_shares[0] == 0) * distance)
    
ptspeed = pd.DataFrame()
ptspeed["city"] = emission_2015.keys()
ptspeed["emissions_2035"] = np.array(list(emission_2035.values())) #/ np.array(list(emission_2015.values()))
ptspeed["utility_2035"] = np.array(list(utility_2035.values())) #/ np.array(list(utility_2015.values()))
ptspeed["residential_emissions_2035"] = residential_emissions_2035.values()
ptspeed["injuries_2035"] = injuries_2035.values()
ptspeed["air_pollution_2035"] = np.array(list(air_pollution_2035.values())) #/ np.array(list(air_pollution_2015.values()))
ptspeed["housing_affordability_2035"] = np.array(list(housing_affordability_2035.values())) #/ np.array(list(housing_affordability_2015.values()))
ptspeed["death_avoided_2035"] = np.array(list(death_avoided_2035.values()))

### FUEL EFFICIENCY
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
air_pollution_2015 = {}
air_pollution_2035 = {}
residential_emissions_2035 = {}
injuries_2035 = {}
housing_affordability_2015 = {}
housing_affordability_2035 = {}
death_avoided_2035 = {}

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_fuelefficiency_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_fuelefficiency_simulation + city + "_utility.npy")
    distance = np.load(path_fuelefficiency_simulation + city + "_distance.npy")
    dwelling_size = np.load(path_fuelefficiency_simulation + city + "_dwelling_size.npy")
    density = np.load(path_fuelefficiency_simulation + city + "_density.npy")
    modal_shares = np.load(path_fuelefficiency_simulation + city + "_modal_shares.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    death_avoided_2035[city] = death_avoided_per_year(density[20], modal_shares[20], distance, 5, 1)
    min_dist = np.argmin(distance)
    housing_affordability_2015[city] = dwelling_size[0][min_dist]
    housing_affordability_2035[city] = dwelling_size[20][min_dist]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    air_pollution_2015[city] = np.load(path_fuelefficiency_simulation + city + "_emissions.npy")[0]/sum(density[0] > 400)
    air_pollution_2035[city] = np.load(path_fuelefficiency_simulation + city + "_emissions.npy")[20]/sum(density[20] > 400)
    residential_emissions_2035[city] = np.nansum(dwelling_size[20] * density[20]) / np.nansum(dwelling_size[0] * density[0])
    injuries_2035[city] = sum(density[20] * (modal_shares[20] == 0) * distance) / sum(density[0] * (modal_shares[0] == 0) * distance)
    
fuelefficiency = pd.DataFrame()
fuelefficiency["city"] = emission_2015.keys()
fuelefficiency["emissions_2035"] = np.array(list(emission_2035.values())) #/ np.array(list(emission_2015.values()))
fuelefficiency["utility_2035"] = np.array(list(utility_2035.values())) #/ np.array(list(utility_2015.values()))
fuelefficiency["residential_emissions_2035"] = residential_emissions_2035.values()
fuelefficiency["injuries_2035"] = injuries_2035.values()
fuelefficiency["air_pollution_2035"] = np.array(list(air_pollution_2035.values())) #/ np.array(list(air_pollution_2015.values()))
fuelefficiency["housing_affordability_2035"] = np.array(list(housing_affordability_2035.values())) #/ np.array(list(housing_affordability_2015.values()))
fuelefficiency["death_avoided_2035"] = np.array(list(death_avoided_2035.values()))

### BASIC INFRA
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
air_pollution_2015 = {}
air_pollution_2035 = {}
residential_emissions_2035 = {}
injuries_2035 = {}
housing_affordability_2015 = {}
housing_affordability_2035 = {}
death_avoided_2035 = {}

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_basicinfra + city + "_emissions_per_capita.npy")
    utility = np.load(path_basicinfra + city + "_utility.npy")
    distance = np.load(path_basicinfra + city + "_distance.npy")
    dwelling_size = np.load(path_basicinfra + city + "_dwelling_size.npy")
    density = np.load(path_basicinfra + city + "_density.npy")
    modal_shares = np.load(path_basicinfra + city + "_modal_shares.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    death_avoided_2035[city] = death_avoided_per_year(density[20], modal_shares[20], distance, 5, 1)
    min_dist = np.argmin(distance)
    housing_affordability_2015[city] = dwelling_size[0][min_dist]
    housing_affordability_2035[city] = dwelling_size[20][min_dist]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    air_pollution_2015[city] = np.load(path_basicinfra + city + "_emissions.npy")[0]/sum(density[0] > 400)
    air_pollution_2035[city] = np.load(path_basicinfra + city + "_emissions.npy")[20]/sum(density[20] > 400)
    residential_emissions_2035[city] = np.nansum(dwelling_size[20] * density[20]) / np.nansum(dwelling_size[0] * density[0])
    injuries_2035[city] = sum(density[20] * (modal_shares[20] == 0) * distance) / sum(density[0] * (modal_shares[0] == 0) * distance)
    
basic_infra = pd.DataFrame()
basic_infra["city"] = emission_2015.keys()
basic_infra["emissions_2035"] = np.array(list(emission_2035.values())) #/ np.array(list(emission_2015.values()))
basic_infra["utility_2035"] = np.array(list(utility_2035.values())) #/ np.array(list(utility_2015.values()))
basic_infra["residential_emissions_2035"] = residential_emissions_2035.values()
basic_infra["injuries_2035"] = injuries_2035.values()
basic_infra["air_pollution_2035"] = np.array(list(air_pollution_2035.values())) #/ np.array(list(air_pollution_2015.values()))
basic_infra["housing_affordability_2035"] = np.array(list(housing_affordability_2035.values())) #/ np.array(list(housing_affordability_2015.values()))
basic_infra["death_avoided_2035"] = np.array(list(death_avoided_2035.values()))



# Simple dataviz with 1 city
for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    df_city = BAU[BAU.city == city]
    df_city = df_city.append(greenbelt[greenbelt.city == city])
    df_city = df_city.append(carbontax[carbontax.city == city])
    df_city = df_city.append(ptspeed[ptspeed.city == city])
    df_city = df_city.append(fuelefficiency[fuelefficiency.city == city])
    df_city.index = ["BAU", "greenbelt", "carbon_tax", "public_transport_speed", "fuel_efficiency"]
    df_city.iloc[1:5, 1:8] = 100 * (df_city.iloc[1:5, 1:8] - df_city.iloc[0, 1:8]) / df_city.iloc[0, 1:8]
    df_city = df_city.iloc[:,[0, 1, 3, 4, 5, 6, 7]]
    df_city.iloc[:,[1, 2, 3, 4]] = - df_city.iloc[:,[1, 2, 3, 4]]
    
    categories=list(df_city)[1:]
    N = len(categories)
    values2=df_city.loc["greenbelt"].drop('city').values.flatten().tolist()
    values2 += values2[:1]
    values3=df_city.loc["carbon_tax"].drop('city').values.flatten().tolist()
    values3 += values3[:1]
    values4=df_city.loc["public_transport_speed"].drop('city').values.flatten().tolist()
    values4 += values4[:1]
    values5=df_city.loc["fuel_efficiency"].drop('city').values.flatten().tolist()
    values5 += values5[:1]
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], ["Emissions", "Residential \n emissions", "Injuries", "Air pollution", "Rent pressure", "Mortality"], color='grey', size=40)
    ax.set_rlabel_position(0)
    plt.yticks(np.arange(-20, 21, 20), labels = ["20", "0", "-20"], color="grey", size=30)
    ax.plot(angles, values2, linewidth=5, linestyle='solid', label = "greenbelt")
    #ax.fill(angles, values2, 'b', alpha=0.1)
    ax.plot(angles, values3, linewidth=5, linestyle='solid', label = "carbon_tax")
    #ax.fill(angles, values3, 'b', alpha=0.1)
    ax.plot(angles, values4, linewidth=5, linestyle='solid', label = "public_transport_speed")
    #ax.fill(angles, values4, 'b', alpha=0.1)
    ax.plot(angles, values5, linewidth=5, linestyle='solid', label = "fuel_efficiency")
    #ax.fill(angles, values5, 'b', alpha=0.1)
    plt.legend(loc = "upper right")
    plt.savefig("C:/Users/charl/OneDrive/Bureau/cobenefits_by_city/" + city + ".png")
    plt.close()
    
# Average gain per continent
city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')

greenbelt = greenbelt.merge(city_continent, left_on = "city", right_on = "City", how = 'left')
carbontax = carbontax.merge(city_continent, left_on = "city", right_on = "City", how = 'left')
ptspeed = ptspeed.merge(city_continent, left_on = "city", right_on = "City", how = 'left')
fuelefficiency = fuelefficiency.merge(city_continent, left_on = "city", right_on = "City", how = 'left')
basic_infra = basic_infra.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

greenbelt_by_continent = (100 * (greenbelt.iloc[:, [1, 3, 4, 5, 6, 7]] - BAU.iloc[:, [1, 3, 4, 5, 6, 7]]) / BAU.iloc[:, [1, 3, 4, 5, 6, 7]])
greenbelt_to_plot = pd.DataFrame(columns = ['emissions_2035', 'residential_emissions_2035', 'injuries_2035', 'air_pollution_2035', "housing affordability", "death avoided"], index = np.unique(greenbelt.Continent))
for continent in np.unique(greenbelt.Continent):
    greenbelt_subset = greenbelt_by_continent[greenbelt.Continent == continent]
    greenbelt_to_plot.loc[continent] = (np.nanmedian(greenbelt_subset, axis = 0))

greenbelt_to_plot.plot.bar()
plt.ylabel("Variation from BAU scenario (%)")
plt.legend(bbox_to_anchor=(-0.60,0.5), loc="lower left", labels = ["Transport emissions", "Res. emissions", "Injuries", "Pollution", "Hous. affordability", "Death avoided \nfrom active \ntransportation modes"])

carbontax_by_continent = (100 * (carbontax.iloc[:, [1, 3, 4, 5, 6, 7]] - BAU.iloc[:, [1, 3, 4, 5, 6, 7]]) / BAU.iloc[:, [1, 3, 4, 5, 6, 7]])
carbontax_to_plot = pd.DataFrame(columns = ['emissions_2035', 'residential_emissions_2035', 'injuries_2035', 'air_pollution_2035', "housing affordability", "death avoided"], index = np.unique(greenbelt.Continent))
for continent in np.unique(carbontax.Continent):
    carbontax_subset = carbontax_by_continent[carbontax.Continent == continent]
    carbontax_to_plot.loc[continent] = (np.nanmean(carbontax_subset, axis = 0))

carbontax_to_plot.plot.bar()
plt.ylabel("Variation from BAU scenario (%)")
plt.legend(bbox_to_anchor=(-0.55,0.5), loc="lower left", labels = ["Emissions", "Res. emissions", "Injuries", "Pollution", "Hous. affordability", "Death avoided"])


ptspeed_by_continent = (100 * (ptspeed.iloc[:, [1, 3, 4, 5, 6, 7]] - BAU.iloc[:, [1, 3, 4, 5, 6, 7]]) / BAU.iloc[:, [1, 3, 4, 5, 6, 7]])
ptspeed_to_plot = pd.DataFrame(columns = ['emissions_2035', 'residential_emissions_2035', 'injuries_2035', 'air_pollution_2035', "housing affordability", "death avoided"], index = np.unique(greenbelt.Continent))
for continent in np.unique(ptspeed.Continent):
    ptspeed_subset = ptspeed_by_continent[ptspeed.Continent == continent]
    ptspeed_to_plot.loc[continent] = (np.nanmean(ptspeed_subset, axis = 0))

ptspeed_to_plot.plot.bar()
plt.ylabel("Variation from BAU scenario (%)")
plt.legend(bbox_to_anchor=(-0.55,0.5), loc="lower left", labels = ["Emissions", "Res. emissions", "Injuries", "Pollution", "Hous. affordability", "Death avoided"])


fuelefficiency_by_continent = (100 * (fuelefficiency.iloc[:, [1, 3, 4, 5, 6, 7]] - BAU.iloc[:, [1, 3, 4, 5, 6, 7]]) / BAU.iloc[:, [1, 3, 4, 5, 6, 7]])
fuelefficiency_to_plot = pd.DataFrame(columns = ['emissions_2035', 'residential_emissions_2035', 'injuries_2035', 'air_pollution_2035', "housing affordability", "death avoided"], index = np.unique(greenbelt.Continent))
for continent in np.unique(fuelefficiency.Continent):
    fuelefficiency_subset = fuelefficiency_by_continent[fuelefficiency.Continent == continent]
    fuelefficiency_to_plot.loc[continent] = (np.nanmean(fuelefficiency_subset, axis = 0))

fuelefficiency_to_plot.plot.bar()
plt.ylabel("Variation from BAU scenario (%)")
plt.legend(bbox_to_anchor=(-0.55,0.5), loc="lower left", labels = ["Emissions", "Res. emissions", "Injuries", "Pollution", "Hous. affordability", "Death avoided"])

basic_infra_by_continent = (100 * (basic_infra.iloc[:, [1, 3, 4, 5, 6, 7]] - BAU.iloc[:, [1, 3, 4, 5, 6, 7]]) / BAU.iloc[:, [1, 3, 4, 5, 6, 7]])
basic_infra_to_plot = pd.DataFrame(columns = ['emissions_2035', 'residential_emissions_2035', 'injuries_2035', 'air_pollution_2035', "housing affordability", "death avoided"], index = np.unique(basic_infra.Continent))
for continent in np.unique(basic_infra.Continent):
    basic_infra_subset = basic_infra_by_continent[basic_infra.Continent == continent]
    basic_infra_to_plot.loc[continent] = (np.nanmean(basic_infra_subset, axis = 0))

basic_infra_to_plot.plot.bar()
plt.ylabel("Variation from BAU scenario (%)")
plt.legend(bbox_to_anchor=(-0.55,0.5), loc="lower left", labels = ["Emissions", "Res. emissions", "Injuries", "Pollution", "Hous. affordability", "Death avoided"])


(100 * (carbontax.iloc[:, 1:6] - BAU.iloc[:, 1:6]) / BAU.iloc[:, 1:6]).groupby(carbontax.Continent).nanmedian().plot.bar()
plt.legend(bbox_to_anchor=(-0.5,0.5), loc="lower left", labels = ["Emissions", "Utility", "Res. emissions", "Injuries", "Pollution"])
(100 * (ptspeed.iloc[:, 1:6] - BAU.iloc[:, 1:6]) / BAU.iloc[:, 1:6]).groupby(ptspeed.Continent).median().plot.bar()
plt.legend(bbox_to_anchor=(-0.5,0.5), loc="lower left", labels = ["Emissions", "Utility", "Res. emissions", "Injuries", "Pollution"])
(100 * (fuelefficiency.iloc[:, 1:6] - BAU.iloc[:, 1:6]) / BAU.iloc[:, 1:6]).groupby(fuelefficiency.Continent).median().plot.bar()
plt.legend(bbox_to_anchor=(-0.5,0.5), loc="lower left", labels = ["Emissions", "Utility", "Res. emissions", "Injuries", "Pollution"])


