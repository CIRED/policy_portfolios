# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:40:16 2021

@author: charl
"""

### IMPORT PACKAGES AND FUNCTIONS

import pandas as pd
import numpy as np
import copy
import pickle
import scipy as sc
import matplotlib.lines as mlines
from scipy import optimize
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from stargazer.stargazer import Stargazer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from functions import *

### IMPORT RESULTS

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"

path_no_policy_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_limited_residuals/"
path_greeenbelt_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/UGB_limited_residuals/"
path_carbon_tax_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbontax_limited_residuals/"
path_transport_speed_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/transit_speed_limited_residuals/"
path_fuelefficiency_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuel_efficiency_limited_residuals/"
path_basicinfra = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/basic_infra_limited_residuals"
#path_propertytax_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/property_tax_limited_residuals/"

path_no_policy_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_noresiduals/"
path_greeenbelt_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/UGB_no_residuals/"
path_carbon_tax_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbontax_no_residuals/"
path_transport_speed_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/transit_speed_no_residuals/"
path_fuelefficiency_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuel_efficiency_no_residuals/"
path_basicinfra = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/basic_infra_no_residuals"

path_no_policy_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/2050/BAU/"
path_greeenbelt_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/2050/property_tax/"
path_carbon_tax_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/2050/carbontax/"
path_transport_speed_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/2050/transit_speed/"
path_fuelefficiency_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/2050/fuel_efficiency/"
path_basicinfra = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/2050/basic_infra"

path_no_policy_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20210819/"
path_greeenbelt_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/UGB_20210819/"
path_carbon_tax_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbontax_20210819/"
path_transport_speed_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/transit_speed_20210819/"
path_fuelefficiency_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuel_efficiency_20210819/"
path_basicinfra = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/basic_infra_20210819"


path_BRT = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BRT_20210901"
path_no_policy_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20210819/"
path_greeenbelt_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/UGB_20210819/"
path_carbon_tax_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbon_tax_20210901/"
path_fuelefficiency_simulation = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuel_efficiency_20210819/"

DURATION = 20

### IMPORT BAU

emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}

list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_no_policy_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_no_policy_simulation + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[DURATION]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[DURATION]
    
no_policy = pd.DataFrame()
no_policy["city"] = emission_2015.keys()
no_policy["emissions_2015"] = emission_2015.values()
no_policy["emissions_2035"] = emission_2035.values()
no_policy["utility_2015"] = utility_2015.values()
no_policy["utility_2035"] = utility_2035.values()

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
no_policy = no_policy.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 40})
colors = list(no_policy['Continent'].unique())
for i in range(0 , len(colors)):
    data = no_policy.loc[no_policy['Continent'] == colors[i]]
    plt.scatter(data.utility_2015, data.emissions_2015, color=data.Continent.map(color_tab), label=colors[i], s = 200)
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.size': 25})
plt.legend()
plt.xlabel("Utility")
plt.ylabel("Transport emissions per capita per year (gCO2)")
plt.show()

no_policy["emissions_increase"] = (no_policy.emissions_2035 - no_policy.emissions_2015) * 100 / no_policy.emissions_2015
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 40})
sns.distplot(no_policy["emissions_increase"], kde = False)
plt.legend()
plt.ylabel("Number of citiess")
plt.xlabel("Variation in emissions per capita (%)")
plt.xticks([0, 50, 100, 150, 200,250, 300 ])

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 40})
plt.scatter(no_policy["emissions_increase"], no_policy["Continent"],  s = 200)
plt.xticks([0, 50, 100, 150, 200,250, 300])

### CARBON TAX

emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_carbon_tax_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_carbon_tax_simulation + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[DURATION]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[DURATION]
    
carbon_tax = pd.DataFrame()
carbon_tax["city"] = emission_2015.keys()
carbon_tax["emissions_2015"] = emission_2015.values()
carbon_tax["emissions_2035"] = emission_2035.values()
carbon_tax["utility_2015"] = utility_2015.values()
carbon_tax["utility_2035"] = utility_2035.values()

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
carbon_tax = carbon_tax.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
carbon_tax = carbon_tax.merge(baseline, left_on = "city", right_on = "city", how = 'left')
carbon_tax["emissions_reduction_2015"] = carbon_tax.emissions_2015 / carbon_tax.emissions_2015_BAU
carbon_tax["utility_reduction_2015"] = carbon_tax.utility_2015 / carbon_tax.utility_2015_BAU
carbon_tax["emissions_reduction_2035"] = carbon_tax.emissions_2035 / carbon_tax.emissions_2035_BAU
carbon_tax["utility_reduction_2035"] = carbon_tax.utility_2035 / carbon_tax.utility_2035_BAU

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.scatter((carbon_tax.utility_reduction_2035 - 1) * 100, (carbon_tax.emissions_reduction_2035 - 1) * 100, s = 200, c= carbon_tax['Continent'].map(color_tab))
red_na = mlines.Line2D([], [], color='red', marker= '.', linestyle='None', markersize=20, label='North_America')
green_europe = mlines.Line2D([], [], color='green', marker= '.', linestyle='None', markersize=20, label='Europe')
blue_asia = mlines.Line2D([], [], color='blue', marker= '.', linestyle='None', markersize=20, label='Asia')
yellow_oceania = mlines.Line2D([], [], color='yellow', marker= '.', linestyle='None', markersize=20, label='Oceania')
brown_sa = mlines.Line2D([], [], color='brown', marker= '.', linestyle='None', markersize=20, label='South_America')
orange_afr = mlines.Line2D([], [], color='orange', marker= '.', linestyle='None', markersize=20, label='Africa')
plt.xlabel("Utility reduction compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.ylim(-8,0)
plt.xlim(-4,0)
plt.legend(loc = 'lower left', handles=[red_na, green_europe, blue_asia, yellow_oceania, brown_sa, orange_afr])

carbon_tax.city[carbon_tax.utility_reduction_2035 > 1]
carbon_tax.city[carbon_tax.emissions_reduction_2035 > 1]

### BASIC INFRA

emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_basicinfra + '/' + city + "_emissions_per_capita.npy")
    utility = np.load(path_basicinfra + '/'  + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[DURATION]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[DURATION]
    
basic_infra = pd.DataFrame()
basic_infra["city"] = emission_2015.keys()
basic_infra["emissions_2015"] = emission_2015.values()
basic_infra["emissions_2035"] = emission_2035.values()
basic_infra["utility_2015"] = utility_2015.values()
basic_infra["utility_2035"] = utility_2035.values()

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
basic_infra = basic_infra.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
basic_infra = basic_infra.merge(baseline, left_on = "city", right_on = "city", how = 'left')
basic_infra["emissions_reduction_2015"] = basic_infra.emissions_2015 / basic_infra.emissions_2015_BAU
basic_infra["utility_reduction_2015"] = basic_infra.utility_2015 / basic_infra.utility_2015_BAU
basic_infra["emissions_reduction_2035"] = basic_infra.emissions_2035 / basic_infra.emissions_2035_BAU
basic_infra["utility_reduction_2035"] = basic_infra.utility_2035 / basic_infra.utility_2035_BAU

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.scatter((basic_infra.utility_reduction_2035 - 1) * 100, (basic_infra.emissions_reduction_2035 - 1) * 100, s = 200, c= basic_infra['Continent'].map(color_tab))
red_na = mlines.Line2D([], [], color='red', marker= '.', linestyle='None', markersize=20, label='North_America')
green_europe = mlines.Line2D([], [], color='green', marker= '.', linestyle='None', markersize=20, label='Europe')
blue_asia = mlines.Line2D([], [], color='blue', marker= '.', linestyle='None', markersize=20, label='Asia')
yellow_oceania = mlines.Line2D([], [], color='yellow', marker= '.', linestyle='None', markersize=20, label='Oceania')
brown_sa = mlines.Line2D([], [], color='brown', marker= '.', linestyle='None', markersize=20, label='South_America')
orange_afr = mlines.Line2D([], [], color='orange', marker= '.', linestyle='None', markersize=20, label='Africa')
plt.ylim(None, 0)
plt.xlabel("Utility reduction compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.legend(loc = 'lower left', handles=[red_na, green_europe, blue_asia, yellow_oceania, brown_sa, orange_afr])

### BRT

emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_BRT + '/' + city + "_emissions_per_capita.npy")
    utility = np.load(path_BRT + '/'  + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[DURATION]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[DURATION]
    
BRT = pd.DataFrame()
BRT["city"] = emission_2015.keys()
BRT["emissions_2015"] = emission_2015.values()
BRT["emissions_2035"] = emission_2035.values()
BRT["utility_2015"] = utility_2015.values()
BRT["utility_2035"] = utility_2035.values()

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
BRT = BRT.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
BRT = BRT.merge(baseline, left_on = "city", right_on = "city", how = 'left')
BRT["emissions_reduction_2015"] = BRT.emissions_2015 / BRT.emissions_2015_BAU
BRT["utility_reduction_2015"] = BRT.utility_2015 / BRT.utility_2015_BAU
BRT["emissions_reduction_2035"] = BRT.emissions_2035 / BRT.emissions_2035_BAU
BRT["utility_reduction_2035"] = BRT.utility_2035 / BRT.utility_2035_BAU

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.scatter((BRT.utility_reduction_2035 - 1) * 100, (BRT.emissions_reduction_2035 - 1) * 100, s = 200, c= BRT['Continent'].map(color_tab))
red_na = mlines.Line2D([], [], color='red', marker= '.', linestyle='None', markersize=20, label='North_America')
green_europe = mlines.Line2D([], [], color='green', marker= '.', linestyle='None', markersize=20, label='Europe')
blue_asia = mlines.Line2D([], [], color='blue', marker= '.', linestyle='None', markersize=20, label='Asia')
yellow_oceania = mlines.Line2D([], [], color='yellow', marker= '.', linestyle='None', markersize=20, label='Oceania')
brown_sa = mlines.Line2D([], [], color='brown', marker= '.', linestyle='None', markersize=20, label='South_America')
orange_afr = mlines.Line2D([], [], color='orange', marker= '.', linestyle='None', markersize=20, label='Africa')
plt.xlabel("Utility reduction compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.ylim(-30, 0)
plt.xlim(0, 1.5)
plt.legend(loc = 'lower left', handles=[red_na, green_europe, blue_asia, yellow_oceania, brown_sa, orange_afr])


### GREENBELT

emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_greeenbelt_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_greeenbelt_simulation + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[DURATION]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[DURATION]
    
greenbelt = pd.DataFrame()
greenbelt["city"] = emission_2015.keys()
greenbelt["emissions_2015"] = emission_2015.values()
greenbelt["emissions_2035"] = emission_2035.values()
greenbelt["utility_2015"] = utility_2015.values()
greenbelt["utility_2035"] = utility_2035.values()

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
greenbelt = greenbelt.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
greenbelt = greenbelt.merge(baseline, left_on = "city", right_on = "city", how = 'left')
greenbelt["emissions_reduction_2015"] = greenbelt.emissions_2015 / greenbelt.emissions_2015_BAU
greenbelt["utility_reduction_2015"] = greenbelt.utility_2015 / greenbelt.utility_2015_BAU
greenbelt["emissions_reduction_2035"] = greenbelt.emissions_2035 / greenbelt.emissions_2035_BAU
greenbelt["utility_reduction_2035"] = greenbelt.utility_2035 / greenbelt.utility_2035_BAU

greenbelt["utility_increase_BAU"] = greenbelt.utility_2035_BAU / greenbelt.utility_2015_BAU
greenbelt["utility_increase"] = greenbelt.utility_2035 / greenbelt.utility_2015

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.figure(figsize=(20, 20))
plt.scatter((greenbelt.utility_reduction_2035 - 1) * 100, (greenbelt.emissions_reduction_2035 - 1) * 100, s = 200, c= greenbelt['Continent'].map(color_tab))
red_na = mlines.Line2D([], [], color='red', marker= '.', linestyle='None', markersize=20, label='North_America')
green_europe = mlines.Line2D([], [], color='green', marker= '.', linestyle='None', markersize=20, label='Europe')
blue_asia = mlines.Line2D([], [], color='blue', marker= '.', linestyle='None', markersize=20, label='Asia')
yellow_oceania = mlines.Line2D([], [], color='yellow', marker= '.', linestyle='None', markersize=20, label='Oceania')
brown_sa = mlines.Line2D([], [], color='brown', marker= '.', linestyle='None', markersize=20, label='South_America')
orange_afr = mlines.Line2D([], [], color='orange', marker= '.', linestyle='None', markersize=20, label='Africa')
plt.xlabel("Utility variation compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.legend(loc = 'lower left', bbox_to_anchor=(0, 0), handles=[red_na, green_europe, blue_asia, yellow_oceania, brown_sa, orange_afr])

### PUBLIC TRANSPORT SPEED

emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_transport_speed_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_transport_speed_simulation + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[DURATION]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[DURATION]
    
pt_speed = pd.DataFrame()
pt_speed["city"] = emission_2015.keys()
pt_speed["emissions_2015"] = emission_2015.values()
pt_speed["emissions_2035"] = emission_2035.values()
pt_speed["utility_2015"] = utility_2015.values()
pt_speed["utility_2035"] = utility_2035.values()

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
pt_speed = pt_speed.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
pt_speed = pt_speed.merge(baseline, left_on = "city", right_on = "city", how = 'left')
pt_speed["emissions_reduction_2015"] = pt_speed.emissions_2015 / pt_speed.emissions_2015_BAU
pt_speed["utility_reduction_2015"] = pt_speed.utility_2015 / pt_speed.utility_2015_BAU
pt_speed["emissions_reduction_2035"] = pt_speed.emissions_2035 / pt_speed.emissions_2035_BAU
pt_speed["utility_reduction_2035"] = pt_speed.utility_2035 / pt_speed.utility_2035_BAU

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.scatter((pt_speed.utility_reduction_2035 - 1) * 100, (pt_speed.emissions_reduction_2035 - 1) * 100, s = 200, c= pt_speed['Continent'].map(color_tab))
red_na = mlines.Line2D([], [], color='red', marker= '.', linestyle='None', markersize=20, label='North_America')
green_europe = mlines.Line2D([], [], color='green', marker= '.', linestyle='None', markersize=20, label='Europe')
blue_asia = mlines.Line2D([], [], color='blue', marker= '.', linestyle='None', markersize=20, label='Asia')
yellow_oceania = mlines.Line2D([], [], color='yellow', marker= '.', linestyle='None', markersize=20, label='Oceania')
brown_sa = mlines.Line2D([], [], color='brown', marker= '.', linestyle='None', markersize=20, label='South_America')
orange_afr = mlines.Line2D([], [], color='orange', marker= '.', linestyle='None', markersize=20, label='Africa')
plt.xlabel("Utility variation compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.legend(loc = 'lower left', bbox_to_anchor=(0.6, 0.6), handles=[red_na, green_europe, blue_asia, yellow_oceania, brown_sa, orange_afr])

### FUEL EFFICIENCY

#create the dataset
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for city in np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78):
    emissions_per_capita = np.load(path_fuelefficiency_simulation + city + "_emissions_per_capita.npy")
    utility = np.load(path_fuelefficiency_simulation + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[DURATION]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[DURATION]
    
fuel_efficiency = pd.DataFrame()
fuel_efficiency["city"] = emission_2015.keys()
fuel_efficiency["emissions_2015"] = emission_2015.values()
fuel_efficiency["emissions_2035"] = emission_2035.values()
fuel_efficiency["utility_2015"] = utility_2015.values()
fuel_efficiency["utility_2035"] = utility_2035.values()

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
fuel_efficiency = fuel_efficiency.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
fuel_efficiency = fuel_efficiency.merge(baseline, left_on = "city", right_on = "city", how = 'left')
fuel_efficiency["emissions_reduction_2015"] = fuel_efficiency.emissions_2015 / fuel_efficiency.emissions_2015_BAU
fuel_efficiency["utility_reduction_2015"] = fuel_efficiency.utility_2015 / fuel_efficiency.utility_2015_BAU
fuel_efficiency["emissions_reduction_2035"] = fuel_efficiency.emissions_2035 / fuel_efficiency.emissions_2035_BAU
fuel_efficiency["utility_reduction_2035"] = fuel_efficiency.utility_2035 / fuel_efficiency.utility_2035_BAU

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.scatter((fuel_efficiency.utility_reduction_2035 - 1) * 100, (fuel_efficiency.emissions_reduction_2035 - 1) * 100, s = 200, c= fuel_efficiency['Continent'].map(color_tab))
red_na = mlines.Line2D([], [], color='red', marker= '.', linestyle='None', markersize=20, label='North_America')
green_europe = mlines.Line2D([], [], color='green', marker= '.', linestyle='None', markersize=20, label='Europe')
blue_asia = mlines.Line2D([], [], color='blue', marker= '.', linestyle='None', markersize=20, label='Asia')
yellow_oceania = mlines.Line2D([], [], color='yellow', marker= '.', linestyle='None', markersize=20, label='Oceania')
brown_sa = mlines.Line2D([], [], color='brown', marker= '.', linestyle='None', markersize=20, label='South_America')
orange_afr = mlines.Line2D([], [], color='orange', marker= '.', linestyle='None', markersize=20, label='Africa')
plt.xlabel("Utility variation compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.legend(handles=[red_na, green_europe, blue_asia, yellow_oceania, brown_sa, orange_afr])
plt.legend(loc = 'lower left', bbox_to_anchor=(0.6, 0.6), handles=[red_na, green_europe, blue_asia, yellow_oceania, brown_sa, orange_afr])


#### COST EFFECTIVENESS

#on veut
# -emissions reduction < 1
# utility reduction < 1 pour CT et GB et > 1 FE et BRT

carbon_tax["cost_effectiveness"] = carbon_tax.emissions_reduction_2035 / carbon_tax.utility_reduction_2035
fuel_efficiency["cost_effectiveness"] = fuel_efficiency.emissions_reduction_2035 / fuel_efficiency.utility_reduction_2035
greenbelt["cost_effectiveness"] = greenbelt.emissions_reduction_2035 / greenbelt.utility_reduction_2035
BRT["cost_effectiveness"] = BRT.emissions_reduction_2035 / BRT.utility_reduction_2035

np.nanmax(carbon_tax["cost_effectiveness"])
np.nanmin(carbon_tax["cost_effectiveness"])
np.nanmax(fuel_efficiency["cost_effectiveness"])
np.nanmin(fuel_efficiency["cost_effectiveness"])
np.nanmax(greenbelt["cost_effectiveness"])
np.nanmin(greenbelt["cost_effectiveness"])
np.nanmax(BRT["cost_effectiveness"])
np.nanmin(BRT["cost_effectiveness"])

#carbon_tax["cost_effectiveness"][carbon_tax["cost_effectiveness"] < 0] = np.nan
#fuel_efficiency["cost_effectiveness"][fuel_efficiency["cost_effectiveness"] < 0] = np.nan
#greenbelt["cost_effectiveness"][greenbelt["cost_effectiveness"] < 0] = np.nan

#pt_speed["cost_effectiveness"] = 1 / pt_speed["cost_effectiveness"]#
#fuel_efficiency["cost_effectiveness"] = 1 / fuel_efficiency["cost_effectiveness"]
#basic_infra["cost_effectiveness"] = 1 / basic_infra["cost_effectiveness"]

#pt_speed["cost_effectiveness"][np.isnan(pt_speed.cost_effectiveness)] = 0
#basic_infra["cost_effectiveness"][np.isnan(basic_infra.cost_effectiveness)] = 0

opportunity = carbon_tax.loc[:, ["cost_effectiveness"]]
opportunity["carbon_tax"] = carbon_tax["cost_effectiveness"]
opportunity["BRT"] = BRT["cost_effectiveness"]
opportunity["fuel_efficiency"] = fuel_efficiency["cost_effectiveness"]
opportunity["greenbelt"] = greenbelt["cost_effectiveness"]
#opportunity["basic_infra"] = basic_infra["cost_effectiveness"]
opportunity["city"] = emission_2015.keys()
opportunity.to_excel('C:/Users/charl/OneDrive/Bureau/opportunity_20210906.xlsx')


#carbon_tax["cost_effectiveness2"] = (carbon_tax.emissions_2035_BAU - carbon_tax.emissions_2035) / (carbon_tax.utility_2035_BAU - carbon_tax.utility_2035)

#cost_eff = carbon_tax.loc[:, ["city", "cost_effectiveness1", "cost_effectiveness2"]]
#cost_eff.to_excel('C:/Users/charl/OneDrive/Bureau/cost_eff.xlsx')


'''
### OPPORTUNITY STATS BY CONTINENT

opportunity = pd.read_excel('C:/Users/charl/OneDrive/Bureau/opportunity_limited_residuals.xlsx')

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
opportunity = opportunity.merge(city_continent, left_on = "city", right_on = "City", how = 'left')
opportunity["total"] = 1

by_continent = opportunity.groupby('Continent').sum()

plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.bar(by_continent.index, 100 * by_continent.dummy_basic_infra / by_continent.total)
plt.xticks(ticks = ['Africa', 'Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], labels = ['Africa', 'Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel('Proportion of cities (%)')

plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.bar(by_continent.index, 100 * by_continent.dummy_fuel_efficiency / by_continent.total)
plt.xticks(ticks = ['Africa', 'Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], labels = ['Africa', 'Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel('Proportion of cities (%)')

plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.bar(by_continent.index, 100 * by_continent.dummy_greenbelt / by_continent.total)
plt.xticks(ticks = ['Africa', 'Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], labels = ['Africa', 'Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel('Proportion of cities (%)')

plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.bar(by_continent.index, 100 * by_continent.dummy_carbon_tax / by_continent.total)
plt.xticks(ticks = ['Africa', 'Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], labels = ['Africa', 'Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel('Proportion of cities (%)')

plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.bar(by_continent.index, 100 * by_continent.dummy_pt_speed / by_continent.total)
plt.xticks(ticks = ['Africa', 'Asia', 'Europe', 'North_America', 'Oceania', 'South_America'], labels = ['Africa', 'Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
plt.ylabel('Proportion of cities (%)')


city_country = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_country = city_country.iloc[:, [0, 1]]
city_country = city_country.drop_duplicates(subset = "City")
city_country = city_country.sort_values('City')
opportunity = opportunity.merge(city_country, left_on = "city", right_on = "City", how = 'left')
opportunity["total"] = 1

opportunity = opportunity.loc[opportunity.Continent == "Asia",]
by_country = opportunity.groupby('Country').sum()

plt.rcParams.update({'font.size': 50})
plt.figure(figsize=(50, 50))
plt.bar(by_country.index, 100 * by_country.dummy_basic_infra / by_country.total)
plt.rcParams.update({'font.size': 50})
plt.figure(figsize=(50, 50))
plt.bar(by_country.index, 100 * by_country.dummy_fuel_efficiency / by_country.total)
plt.rcParams.update({'font.size': 50})
plt.figure(figsize=(50, 50))
plt.bar(by_country.index, 100 * by_country.dummy_greenbelt / by_country.total)
plt.rcParams.update({'font.size': 50})
plt.figure(figsize=(50, 50))
plt.bar(by_country.index, 100 * by_country.dummy_carbon_tax / by_country.total)
plt.rcParams.update({'font.size': 50})
plt.figure(figsize=(50, 50))
plt.bar(by_country.index, 100 * by_country.dummy_pt_speed / by_country.total)
'''


### REGRESSION

#df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics.xlsx")
df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics_20210810.xlsx")
carbon_tax = carbon_tax.merge(df, on = "city")
pt_speed = pt_speed.merge(df, on = "city")
fuel_efficiency = fuel_efficiency.merge(df, on = "city")
greenbelt = greenbelt.merge(df, on = "city")
basic_infra = basic_infra.merge(df, on = "city")

carbon_tax.emissions_reduction_2035 = (carbon_tax.emissions_reduction_2035 - 1) * 100
pt_speed.emissions_reduction_2035 = (pt_speed.emissions_reduction_2035 - 1) * 100
fuel_efficiency.emissions_reduction_2035 = (fuel_efficiency.emissions_reduction_2035 - 1) * 100
greenbelt.emissions_reduction_2035 = (greenbelt.emissions_reduction_2035 - 1) * 100
basic_infra.emissions_reduction_2035 = (basic_infra.emissions_reduction_2035 - 1) * 100

carbon_tax.utility_reduction_2035 = (carbon_tax.utility_reduction_2035 - 1) * 100
pt_speed.utility_reduction_2035 = (pt_speed.utility_reduction_2035 - 1) * 100
fuel_efficiency.utility_reduction_2035 = (fuel_efficiency.utility_reduction_2035 - 1) * 100
greenbelt.utility_reduction_2035 = (greenbelt.utility_reduction_2035 - 1) * 100
basic_infra.utility_reduction_2035 = (basic_infra.utility_reduction_2035 - 1) * 100


greenbelt["dummy_greenbelt"] = ((greenbelt.emissions_reduction_2035 < np.nanmedian(greenbelt.emissions_reduction_2035)) & (greenbelt.utility_reduction_2035 > np.nanmedian(greenbelt.utility_reduction_2035)))
pt_speed["dummy_pt_speed"] = ((pt_speed.emissions_reduction_2035 < np.nanmedian(pt_speed.emissions_reduction_2035)) & (pt_speed.utility_reduction_2035 > np.nanmedian(pt_speed.utility_reduction_2035)))
carbon_tax["dummy_carbon_tax"] = ((carbon_tax.emissions_reduction_2035 < np.nanmedian(carbon_tax.emissions_reduction_2035)) & (carbon_tax.utility_reduction_2035 > np.nanmedian(carbon_tax.utility_reduction_2035)))
fuel_efficiency["dummy_fuel_efficiency"] = ((fuel_efficiency.emissions_reduction_2035 < np.nanmedian(fuel_efficiency.emissions_reduction_2035)) & (fuel_efficiency.utility_reduction_2035 > np.nanmedian(fuel_efficiency.utility_reduction_2035)))
basic_infra["dummy_basic_infra"] = ((basic_infra.emissions_reduction_2035 < np.nanmedian(basic_infra.emissions_reduction_2035)) & (basic_infra.utility_reduction_2035 > np.nanmedian(basic_infra.utility_reduction_2035)))

greenbelt["dummy_greenbelt"] = np.multiply(greenbelt["dummy_greenbelt"], 1)
pt_speed["dummy_pt_speed"] = np.multiply(pt_speed["dummy_pt_speed"], 1)
carbon_tax["dummy_carbon_tax"] = np.multiply(carbon_tax["dummy_carbon_tax"], 1)
fuel_efficiency["dummy_fuel_efficiency"] = np.multiply(fuel_efficiency["dummy_fuel_efficiency"], 1)
basic_infra["dummy_basic_infra"] = np.multiply(basic_infra["dummy_basic_infra"], 1)

df = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics_20210810.xlsx")
opportunity = opportunity.merge(df, on = "city")

regr1  = DecisionTreeClassifier(max_depth=2)
res_reg1 = regr1.fit((carbon_tax.iloc[:, [16, 17, 22, 27, 30, 31, 35, 37, 38]]), carbon_tax.dummy_carbon_tax)
plt.figure(figsize=(12,12))
tree.plot_tree(res_reg1, fontsize=10, feature_names = carbon_tax.columns[[16, 17, 22, 27, 30, 31, 35, 37, 38]], class_names = ["No", "Yes"], impurity = False) 

regr1  = DecisionTreeClassifier(max_depth=2)
res_reg1 = regr1.fit((pt_speed.iloc[:, [16, 17, 22, 27, 30, 31, 35, 37, 38]]), pt_speed.dummy_pt_speed)
plt.figure(figsize=(12,12))
tree.plot_tree(res_reg1, fontsize=10, feature_names = pt_speed.columns[[16, 17, 22, 27, 30, 31, 35, 37, 38]], class_names = ["No", "Yes"], impurity = False) 

regr1  = DecisionTreeClassifier(max_depth=2)
res_reg1 = regr1.fit((greenbelt.iloc[:, [18, 19, 24, 29, 32, 33, 37, 38, 40]]), greenbelt.dummy_greenbelt)
plt.figure(figsize=(12,12))
tree.plot_tree(res_reg1, fontsize=10, feature_names = greenbelt.columns[[18, 19, 24, 29, 32, 33, 37, 38, 40]], class_names = ["No", "Yes"], impurity = False) 

regr1  = DecisionTreeClassifier(max_depth=2)
res_reg1 = regr1.fit((basic_infra.iloc[:, [16, 17, 22, 27, 30, 31, 35, 37, 38]]), basic_infra.dummy_basic_infra)
plt.figure(figsize=(12,12))
tree.plot_tree(res_reg1, fontsize=10, feature_names = basic_infra.columns[[16, 17, 22, 27, 30, 31, 35, 37, 38]], class_names = ["No", "Yes"], impurity = False) 

regr1  = DecisionTreeClassifier(max_depth=2)
res_reg1 = regr1.fit((fuel_efficiency.iloc[:, [16, 17, 22, 27, 30, 31, 35, 37, 38]]), fuel_efficiency.dummy_fuel_efficiency)
plt.figure(figsize=(12,12))
tree.plot_tree(res_reg1, fontsize=10, feature_names = fuel_efficiency.columns[[16, 17, 22, 27, 30, 31, 35, 37, 38]], class_names = ["No", "Yes"], impurity = False) 


carbon_tax_emissions = ols("dummy_carbon_tax ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential + avg_cons2 + grad_density", data=carbon_tax).fit()
basic_infra_emissions = ols("dummy_basic_infra ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential + avg_cons2 + grad_density", data=basic_infra).fit()
pt_speed_emissions = ols("dummy_pt_speed ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential + avg_cons2 + grad_density", data=pt_speed).fit()
fuel_efficiency_emissions = ols("dummy_fuel_efficiency ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential+ avg_cons2  + grad_density", data=fuel_efficiency).fit()
greenbelt_emissions = ols("dummy_greenbelt ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential + avg_cons2 + grad_density", data=greenbelt).fit()

print(Stargazer([carbon_tax_emissions, pt_speed_emissions, greenbelt_emissions, basic_infra_emissions, fuel_efficiency_emissions]).render_latex())

carbon_tax_emissions = ols("carbon_tax ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential + avg_cons2 + grad_density", data=opportunity).fit().summary()
brt_emissions = ols("BRT ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential + avg_cons2 + grad_density", data=opportunity).fit().summary()
fuel_efficiency_emissions = ols("fuel_efficiency ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential+ avg_cons2 + grad_density", data=opportunity).fit().summary()
greenbelt_emissions = ols("greenbelt ~ population + income + cost_p + BETA + B + income_variation + population_variation + substitution_potential + avg_cons2 + grad_density", data=opportunity).fit().summary()

opportunity = opportunity.loc[:, ['city', 'BRT']].sort_values('BRT')




opportunity = greenbelt.loc[:, ["dummy_greenbelt"]]
opportunity["dummy_pt_speed"] = pt_speed["dummy_pt_speed"]
opportunity["dummy_carbon_tax"] = carbon_tax["dummy_carbon_tax"]
opportunity["dummy_fuel_efficiency"] = fuel_efficiency["dummy_fuel_efficiency"]
opportunity["dummy_basic_infra"] = basic_infra["dummy_basic_infra"]
opportunity["nb_pol"] = opportunity["dummy_greenbelt"] + opportunity["dummy_pt_speed"]  + opportunity["dummy_carbon_tax"]  + opportunity["dummy_fuel_efficiency"]  + opportunity["dummy_basic_infra"]
opportunity["city"] = emission_2015.keys()
opportunity.to_excel('C:/Users/charl/OneDrive/Bureau/opportunity_20210819.xlsx')















#carbon_tax["robust_cities"] = df.robust_cities
#pt_speed["robust_cities"] = df.robust_cities
#fuel_efficiency["robust_cities"] = df.robust_cities
#greenbelt["robust_cities"] = df.robust_cities
#basic_infra["robust_cities"] = df.robust_cities

#basic_infra = basic_infra[basic_infra.robust_cities == 1]
#greenbelt = greenbelt[greenbelt.robust_cities == 1]
#fuel_efficiency = fuel_efficiency[fuel_efficiency.robust_cities == 1]
#pt_speed = pt_speed[pt_speed.robust_cities == 1]
#carbon_tax = carbon_tax[carbon_tax.robust_cities == 1]

ss = StandardScaler()
carbon_tax = carbon_tax.select_dtypes(include=[np.number])
names = carbon_tax.columns
carbon_tax = ss.fit_transform(carbon_tax)
carbon_tax = pd.DataFrame(carbon_tax, columns = names)

pt_speed = pt_speed.select_dtypes(include=[np.number])
names = pt_speed.columns
pt_speed = ss.fit_transform(pt_speed)
pt_speed = pd.DataFrame(pt_speed, columns = names)

fuel_efficiency = fuel_efficiency.select_dtypes(include=[np.number])
names = fuel_efficiency.columns
fuel_efficiency = ss.fit_transform(fuel_efficiency)
fuel_efficiency = pd.DataFrame(fuel_efficiency, columns = names)

greenbelt = greenbelt.select_dtypes(include=[np.number])
names = greenbelt.columns
greenbelt = ss.fit_transform(greenbelt)
greenbelt = pd.DataFrame(greenbelt, columns = names)

basic_infra = basic_infra.select_dtypes(include=[np.number])
names = basic_infra.columns
basic_infra = ss.fit_transform(basic_infra)
basic_infra = pd.DataFrame(basic_infra, columns = names)

carbon_tax_emissions = ols("emissions_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=carbon_tax).fit()
basic_infra_emissions = ols("emissions_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=basic_infra).fit()
pt_speed_emissions = ols("emissions_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=pt_speed).fit()
greenbelt_emissions = ols("emissions_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B +kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=greenbelt).fit()
fuel_efficiency_emissions = ols("emissions_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=fuel_efficiency).fit()
print(Stargazer([carbon_tax_emissions, pt_speed_emissions, greenbelt_emissions, basic_infra_emissions, fuel_efficiency_emissions]).render_latex())

carbon_tax_emissions = ols("emissions_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density -1", data=carbon_tax).fit()
pt_speed_emissions = ols("emissions_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density -1", data=pt_speed).fit()
basic_infra_emissions = ols("emissions_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density  -1", data=basic_infra).fit()
greenbelt_emissions = ols("emissions_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density   -1", data=greenbelt).fit()
fuel_efficiency_emissions = ols("emissions_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density -1", data=fuel_efficiency).fit()
print(Stargazer([carbon_tax_emissions, pt_speed_emissions, greenbelt_emissions, basic_infra_emissions, fuel_efficiency_emissions]).render_latex())

carbon_tax_utility = ols("utility_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density  -1", data=carbon_tax).fit()
pt_speed_utility = ols("utility_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density  -1", data=pt_speed).fit()
basic_infra_utility = ols("utility_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density   -1", data=basic_infra).fit()
greenbelt_utility = ols("utility_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density  -1", data=greenbelt).fit()
fuel_efficiency_utility = ols("utility_reduction_2035 ~ income + population + Ra + income_variation + population_variation + substitution_potential + avg_cons3 + avg_density2 + grad_density  -1", data=fuel_efficiency).fit()
print(Stargazer([carbon_tax_utility, pt_speed_utility, greenbelt_utility, basic_infra_utility, fuel_efficiency_utility]).render_latex())





df = np.empty((5, 24))
from scipy.stats import pearsonr
df[0] = np.corrcoef(carbon_tax.utility_reduction_2035, np.transpose(carbon_tax.iloc[:, 14:37]))[0]
df[1] = np.corrcoef(pt_speed.utility_reduction_2035, np.transpose(pt_speed.iloc[:, 14:37]))[0]
df[2] = np.corrcoef(basic_infra.utility_reduction_2035, np.transpose(basic_infra.iloc[:, 14:37]))[0]
df[3] = np.corrcoef(greenbelt.utility_reduction_2035, np.transpose(greenbelt.iloc[:, 16:39]))[0]
df[4] = np.corrcoef(fuel_efficiency.utility_reduction_2035, np.transpose(fuel_efficiency.iloc[:, 14:37]))[0]

df = pd.DataFrame(df)
df = df.transpose().drop(0)
df.columns = ["carbon_tax", "pt_speed", "basic_infra", "greenbelt", "fuel_efficiency"]
df.index = carbon_tax.iloc[:, 14:37].columns
df = df.drop(["cost_p", "fuel_price", "fuel_consumption", "modal_share_car", "BETA", "B", "kappa", "avg_cons", "avg_cons2", "gradient_cons", "pop_2035", "inc_2035", "modal_share_other_sources", "avg_density"])
df.round(2).to_excel("C:/Users/charl/OneDrive/Bureau/correlations_utility.xlsx")

for i in [14, 15, 20, 26, 28, 29, 33, 35, 36]:
    print(pearsonr(greenbelt.utility_reduction_2035, greenbelt.iloc[:, i+2]))

df = np.empty((5, 24))
from scipy.stats import pearsonr
df[0] = np.corrcoef(carbon_tax.emissions_reduction_2035, np.transpose(carbon_tax.iloc[:, 14:37]))[0]
df[1] = np.corrcoef(pt_speed.emissions_reduction_2035, np.transpose(pt_speed.iloc[:, 14:37]))[0]
df[2] = np.corrcoef(basic_infra.emissions_reduction_2035, np.transpose(basic_infra.iloc[:, 14:37]))[0]
df[3] = np.corrcoef(greenbelt.emissions_reduction_2035, np.transpose(greenbelt.iloc[:, 16:39]))[0]
df[4] = np.corrcoef(fuel_efficiency.emissions_reduction_2035, np.transpose(fuel_efficiency.iloc[:, 14:37]))[0]

df = pd.DataFrame(df)
df = df.transpose().drop(0)
df.columns = ["carbon_tax", "pt_speed", "basic_infra", "greenbelt", "fuel_efficiency"]
df.index = carbon_tax.iloc[:, 14:37].columns
df = df.drop(["cost_p", "fuel_price", "fuel_consumption", "modal_share_car", "BETA", "B", "kappa", "avg_cons", "avg_cons2", "gradient_cons", "pop_2035", "inc_2035", "modal_share_other_sources", "avg_density"])
df.round(2).to_excel("C:/Users/charl/OneDrive/Bureau/correlations_emissions.xlsx")

for i in [14, 15, 20, 26, 28, 29, 33, 35, 36]:
    print(pearsonr(greenbelt.emissions_reduction_2035, greenbelt.iloc[:, i + 2]))

 
carbon_tax_utility = ols("utility_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=carbon_tax).fit()
pt_speed_utility = ols("utility_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=pt_speed).fit()
fuel_efficiency_utility = ols("utility_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=fuel_efficiency).fit()
greenbelt_utility = ols("utility_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=greenbelt).fit()
basic_infra_utility = ols("utility_reduction_2035 ~ income + population + speed_driving + speed_transit + cost_p + fuel_price + fuel_conso + Ra + BETA + B + kappa + avg_cons3 + gradient_cons + income_variation + population_variation-1", data=basic_infra).fit()

for i in np.arange(0,10):
    x = [carbon_tax_emissions, basic_infra_emissions, pt_speed_emissions, greenbelt_emissions, fuel_efficiency_emissions, carbon_tax_utility, basic_infra_utility, pt_speed_utility, greenbelt_utility, fuel_efficiency_utility][i]
    name = ["carbon_tax_emissions", "basic_infra_emissions", "pt_speed_emissions", "greenbelt_emissions", "fuel_efficiency_emissions", "carbon_tax_utility", "basic_infra_utility", "pt_speed_utility", "greenbelt_utility", "fuel_efficiency_utility"][i]
    err_series = x.params - x.conf_int()[0]
    coef_df = pd.DataFrame({'coef': x.params.values,
                            'err': err_series.values,
                            'varname': ['Income', 'Population', 'Driving speed', 'Public trans. speed',
                                        'Public trans. cost', 'Fuel price', 'Fuel conso', 'Agri. rent', r'$\beta$', 'b', r'$\kappa$',
                                        'Avg nat. constraints', 'Nat. cons. gradient', 'Income growth',
                                        'Population growth']
                            })
    plt.rcParams.update({'font.size': 60})
    plt.figure(figsize=(20, 20))
    plt.barh(coef_df.sort_values(by = ["coef"])["varname"], -coef_df.sort_values(by = ["coef"])["coef"], xerr = coef_df.sort_values(by = ["coef"])["err"])
    plt.savefig("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/characteristic_comparison_graphs/" + name + ".png", bbox_inches="tight")
    plt.close()
    
    
    
carbon_tax_ut = pd.DataFrame(carbon_tax_utility.params, columns = ["value_uti"]).set_index([['Income', 'Population', 'Driving speed', 'Public trans. speed',
       'Public trans. cost', 'Fuel price', 'Fuel conso', 'Agri. rent', r'$\beta$', 'b', r'$\kappa$',
       'Avg nat. constraints', 'Nat. cons. gradient', 'Income growth',
       'Population growth']])

carbon_tax_df = carbon_tax_df.merge(carbon_tax_ut,left_index = True, right_index = True)

plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(20, 20))
plt.barh(np.arange(len(carbon_tax_df.index)) - 0.2, -carbon_tax_df.sort_values(by = ["value"])["value"], 0.4, label = "Efficiency")
plt.barh(np.arange(len(carbon_tax_df.index)) + 0.2, carbon_tax_df.sort_values(by = ["value"])["value_uti"], 0.4, label = "Welfare")
plt.yticks(range(len(carbon_tax_df.index)), carbon_tax_df.sort_values(by = ["value"]).index)
plt.legend()

print(Stargazer([carbon_tax_utility, pt_speed_utility, greenbelt_utility, basic_infra_utility, fuel_efficiency_utility]).render_latex())
