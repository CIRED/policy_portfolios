# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:23:01 2021

@author: charl
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from inputs.parameters import *


#### COBENEFITS
def compute_air_pollution(density, mode_choice, distance_cbd, Country, path_folder, year, Region, imaclim, policy, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN):
    #values_petrol = {
    #    "EURO_0": 5.9,
   #     "EURO_1": 1.7,
  #      "EURO_2": 0.9,
  #      "EURO_3": 0.3,
  #      "EURO_4": 0.3,
  #      "EURO_5": 0.3
  #      } #cts/vkm, euro 2000
  #  
  #  values_diesel = {
  #      "EURO_0": 13.8,
  #      "EURO_1": 4.8,
  #      "EURO_2": 4.0,
  ####      "EURO_3": 3.1,
  #      "EURO_4": 1.7,
  #      "EURO_5": 0.7
  #      } #cts/vkm, euro 2000
  #  
  #  marginal_cost_petrol = sum(list(values_petrol.values()) * np.array([0.09, 0.12, 0.22, 0.268, 0.137, 0.165])) / 100 #https://www.eea.europa.eu/data-and-maps/indicators/proportion-of-vehicle-fleet-meeting/proportion-of-vehicle-fleet-meeting-6
  #  marginal_cost_diesel = sum(list(values_diesel.values()) * np.array([0.022, 0.036, 0.123, 0.329, 0.23, 0.26])) / 100 #https://www.eea.europa.eu/data-and-maps/indicators/proportion-of-vehicle-fleet-meeting/proportion-of-vehicle-fleet-meeting-6
  #  
  #  diesel_share = pd.read_csv(path_folder + 'dieselisation-of-diesel-cars-in-4.csv')
  #  diesel_share["MS:text"][diesel_share["MS:text"] == "Czech Republic"] = 'Czech_Republic'
  #  diesel_share["MS:text"][diesel_share["MS:text"] == "United Kingdom"] = 'UK'
    
  #  if Country in list(diesel_share["MS:text"]):
  #      diesel_cars = diesel_share["2017:number"][diesel_share["MS:text"] == Country].squeeze() / 100
  #  else:
  #      diesel_cars = diesel_share["2017:number"][diesel_share["MS:text"] == 'EU28'].squeeze() / 100
    
  #  marginal_cost_ref = (diesel_cars * marginal_cost_diesel) + ((1-diesel_cars) * marginal_cost_petrol) #https://www.eea.europa.eu/data-and-maps/daviz/dieselisation-of-diesel-cars-in-4#tab-chart_1
    marginal_cost_ref = 0.0273
    gdp_per_capita = import_gdp_per_capita(path_folder, Country, "2015")
    gdp_per_capita_ref = import_gdp_per_capita(path_folder, "Germany", "2015")
    
    ref_inc_growth = imaclim[(imaclim.Region == "EUR") & (imaclim.Variable == "Index_income")].squeeze()
    #ref_inc_growth = ref_inc_growth.append(pd.Series(ref_inc_growth[2002] * ((ref_inc_growth[2002]/ref_inc_growth[2003]) ** 2), index = [2000]))
    
    conversion_to_ppa = import_conversion_to_ppa(path_folder, "Germany", "2015")
    
    marginal_cost = marginal_cost_ref * (ref_inc_growth[2015] / ref_inc_growth[2008]) * (1/conversion_to_ppa) * (gdp_per_capita / gdp_per_capita_ref)
    
    income_growth_year = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[year]
    income_growth_2015 = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[2015]
    marginal_cost = marginal_cost * (income_growth_year / income_growth_2015)
    
    total_vkm = np.nansum(density[mode_choice == 0] * distance_cbd[mode_choice == 0]) * 365 * 2
    
    air_pollution = marginal_cost * total_vkm
    
    if ((policy == "fuel_efficiency") & (year > 2019)) |((policy == "synergy") & (year > 2019))|((policy == "all") & (year > 2019)):
        #air_pollution = air_pollution * (BASELINE_EFFICIENCY_DECREASE ** (5)) * (FUEL_EFFICIENCY_DECREASE ** (year - 2020))
        #fuel_consumption = (((LIFESPAN - 1)/LIFESPAN) + ((1/LIFESPAN) * (BASELINE_EFFICIENCY_DECREASE ** (5 - index + 15)) * (FUEL_EFFICIENCY_DECREASE ** (index - 5)))) * fuel_consumption
        for y in range(2015, year):
            if y < 2020:
                air_pollution = air_pollution * (((LIFESPAN - 1)/LIFESPAN) + ((BASELINE_EFFICIENCY_DECREASE ** LIFESPAN) /LIFESPAN))
                print("a")
            else:
                air_pollution = (((LIFESPAN - 1)/LIFESPAN) + ((1/LIFESPAN) * (BASELINE_EFFICIENCY_DECREASE ** (2020 - y + 15)) * (FUEL_EFFICIENCY_DECREASE ** (y - 2020)))) * air_pollution
                print("b")
    else:
        #air_pollution = air_pollution * (BASELINE_EFFICIENCY_DECREASE ** (year - 2015))
        #fuel_consumption = (((LIFESPAN - 1)/LIFESPAN) + ((BASELINE_EFFICIENCY_DECREASE ** LIFESPAN) /LIFESPAN)) * fuel_consumption
        air_pollution = air_pollution * ((((LIFESPAN - 1)/LIFESPAN) + ((BASELINE_EFFICIENCY_DECREASE ** LIFESPAN) /LIFESPAN)) ** (year - 2015))
    return air_pollution

def compute_air_pollution_all_welfare_increasing(density, mode_choice, distance_cbd, Country, path_folder, year, Region, imaclim, policy_fe, FUEL_EFFICIENCY_DECREASE, BASELINE_EFFICIENCY_DECREASE, LIFESPAN):
    #values_petrol = {
    #    "EURO_0": 5.9,
   #     "EURO_1": 1.7,
  #      "EURO_2": 0.9,
  #      "EURO_3": 0.3,
  #      "EURO_4": 0.3,
  #      "EURO_5": 0.3
  #      } #cts/vkm, euro 2000
  #  
  #  values_diesel = {
  #      "EURO_0": 13.8,
  #      "EURO_1": 4.8,
  #      "EURO_2": 4.0,
  ####      "EURO_3": 3.1,
  #      "EURO_4": 1.7,
  #      "EURO_5": 0.7
  #      } #cts/vkm, euro 2000
  #  
  #  marginal_cost_petrol = sum(list(values_petrol.values()) * np.array([0.09, 0.12, 0.22, 0.268, 0.137, 0.165])) / 100 #https://www.eea.europa.eu/data-and-maps/indicators/proportion-of-vehicle-fleet-meeting/proportion-of-vehicle-fleet-meeting-6
  #  marginal_cost_diesel = sum(list(values_diesel.values()) * np.array([0.022, 0.036, 0.123, 0.329, 0.23, 0.26])) / 100 #https://www.eea.europa.eu/data-and-maps/indicators/proportion-of-vehicle-fleet-meeting/proportion-of-vehicle-fleet-meeting-6
  #  
  #  diesel_share = pd.read_csv(path_folder + 'dieselisation-of-diesel-cars-in-4.csv')
  #  diesel_share["MS:text"][diesel_share["MS:text"] == "Czech Republic"] = 'Czech_Republic'
  #  diesel_share["MS:text"][diesel_share["MS:text"] == "United Kingdom"] = 'UK'
    
  #  if Country in list(diesel_share["MS:text"]):
  #      diesel_cars = diesel_share["2017:number"][diesel_share["MS:text"] == Country].squeeze() / 100
  #  else:
  #      diesel_cars = diesel_share["2017:number"][diesel_share["MS:text"] == 'EU28'].squeeze() / 100
    
  #  marginal_cost_ref = (diesel_cars * marginal_cost_diesel) + ((1-diesel_cars) * marginal_cost_petrol) #https://www.eea.europa.eu/data-and-maps/daviz/dieselisation-of-diesel-cars-in-4#tab-chart_1
    marginal_cost_ref = 0.0273
    gdp_per_capita = import_gdp_per_capita(path_folder, Country, "2015")
    gdp_per_capita_ref = import_gdp_per_capita(path_folder, "Germany", "2015")
    
    ref_inc_growth = imaclim[(imaclim.Region == "EUR") & (imaclim.Variable == "Index_income")].squeeze()
    #ref_inc_growth = ref_inc_growth.append(pd.Series(ref_inc_growth[2002] * ((ref_inc_growth[2002]/ref_inc_growth[2003]) ** 2), index = [2000]))
    
    conversion_to_ppa = import_conversion_to_ppa(path_folder, "Germany", "2015")
    
    marginal_cost = marginal_cost_ref * (ref_inc_growth[2015] / ref_inc_growth[2008]) * (1/conversion_to_ppa) * (gdp_per_capita / gdp_per_capita_ref)
    
    income_growth_year = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[year]
    income_growth_2015 = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[2015]
    marginal_cost = marginal_cost * (income_growth_year / income_growth_2015)
    
    total_vkm = np.nansum(density[mode_choice == 0] * distance_cbd[mode_choice == 0]) * 365 * 2
    
    air_pollution = marginal_cost * total_vkm
    
    if ((policy_fe == True) & (year > 2019)):
        #air_pollution = air_pollution * (BASELINE_EFFICIENCY_DECREASE ** (5)) * (FUEL_EFFICIENCY_DECREASE ** (year - 2020))
        #fuel_consumption = (((LIFESPAN - 1)/LIFESPAN) + ((1/LIFESPAN) * (BASELINE_EFFICIENCY_DECREASE ** (5 - index + 15)) * (FUEL_EFFICIENCY_DECREASE ** (index - 5)))) * fuel_consumption
        for y in range(2015, year):
            if y < 2020:
                air_pollution = air_pollution * (((LIFESPAN - 1)/LIFESPAN) + ((BASELINE_EFFICIENCY_DECREASE ** LIFESPAN) /LIFESPAN))
                print("a")
            else:
                air_pollution = (((LIFESPAN - 1)/LIFESPAN) + ((1/LIFESPAN) * (BASELINE_EFFICIENCY_DECREASE ** (2020 - y + 15)) * (FUEL_EFFICIENCY_DECREASE ** (y - 2020)))) * air_pollution
                print("b")
    else:
        #air_pollution = air_pollution * (BASELINE_EFFICIENCY_DECREASE ** (year - 2015))
        #fuel_consumption = (((LIFESPAN - 1)/LIFESPAN) + ((BASELINE_EFFICIENCY_DECREASE ** LIFESPAN) /LIFESPAN)) * fuel_consumption
        air_pollution = air_pollution * ((((LIFESPAN - 1)/LIFESPAN) + ((BASELINE_EFFICIENCY_DECREASE ** LIFESPAN) /LIFESPAN)) ** (year - 2015))
    return air_pollution


def compute_active_modes(density, mode_choice, distance_cbd, WALKING_SPEED, Country, path_folder, Region, year, imaclim): 
    mortality_rate, VSL = import_heat_values(Country, path_folder)
    RR = 0.89
    reference_volume = 168
    conversion_to_ppa = import_conversion_to_ppa(path_folder, "Germany", "2015")
    
    income_growth_year = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[year]
    income_growth_2015 = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[2015]
    
    VSL = VSL * (income_growth_year / income_growth_2015) * (1 / conversion_to_ppa)

    non_active_pop = np.nansum(density[mode_choice == 0]) + np.nansum(density[mode_choice == 1])
    active_pop = np.nansum(density[mode_choice == 2])
    walking_per_week = 60 * (10 * (np.nansum(distance_cbd[mode_choice == 2] * density[mode_choice == 2]) / active_pop) /  WALKING_SPEED)
    mortality_rate_non_active_pop = mortality_rate
    mortality_rate_active_pop = mortality_rate_non_active_pop * (1 - ((1-RR) * (walking_per_week / reference_volume)))
    active_modes = VSL * (((active_pop + non_active_pop) * mortality_rate_non_active_pop) - ((mortality_rate_non_active_pop * non_active_pop) + (mortality_rate_active_pop * active_pop)))
    if active_pop == 0:
        active_modes = 0
    return active_modes


def compute_noise(density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder):
    
    gdp_per_capita = import_gdp_per_capita(path_folder, Country, "2015")
    gdp_per_capita_ref = import_gdp_per_capita(path_folder, "Germany", "2015")
    
    ref_inc_growth = imaclim[(imaclim.Region == "EUR")&(imaclim.Variable == "Index_income")].squeeze()
    #ref_inc_growth = ref_inc_growth.append(pd.Series(ref_inc_growth[2002] * ((ref_inc_growth[2002]/ref_inc_growth[2003]) ** 2), index = [2000]))
    
    conversion_to_ppa = import_conversion_to_ppa(path_folder, "Germany", "2015")
    #marginal_cost_noise_ref = 0.0076 #Germany, EURO 2000
    marginal_cost_noise_ref = 0.009 #Germany, EURO 2000
    
    marginal_cost_noise = marginal_cost_noise_ref * (ref_inc_growth[2015] / ref_inc_growth[2008]) * (1/conversion_to_ppa) * (gdp_per_capita / gdp_per_capita_ref)
    
    total_vkm = np.nansum(density[mode_choice == 0] * distance_cbd[mode_choice == 0]) * 365 * 2
    
    income_growth_year = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[year]
    income_growth_2015 = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[2015]
    marginal_cost_noise = marginal_cost_noise * (income_growth_year / income_growth_2015)
    
    
    noise = marginal_cost_noise * total_vkm
    
    return noise

def compute_car_accidents(density, mode_choice, distance_cbd, Country, Region, year, imaclim, path_folder):
    values_maibach = {
        "Austria": 8.67,
        "Belgium": 8.01,
        "Bulgaria": 5.37,
        "Switzerland": 3.13,
        "Czech_Republic": 13.46,
        "Germany": 5.43,
        "Denmark": 6.31,
        "Estonia": 8.98,
        "Spain": 4.98,
        "Finland": 7.44,
        "France": 5.88,
        "Greece": 22.37,
        "Hungary": 14.26,
        "Ireland": 9.76,
        "Italy": 7.57,
        "Lithuania": 8.68,
        "Luxembourg": 11.60,
        "Latvia": 19.78,
        "Netherlands": 3.21,
        "Norway": 6.55,
        "Poland": 8.02,
        "Portugal": 5.51,
        "Romania": 4.69,
        "Sweden": 5.44,
        "Slovenia": 12.98,
        "Slovakia": 9.30,
        "UK": 3.59
        } #ct/vkm, euros 2000
    
    #if Country in values_maibach.keys():
     #   ref_inc_growth = imaclim[(imaclim.Region == "EUR")&(imaclim.Variable == "Index_income")].squeeze()
        
     #   conversion_to_ppa = import_conversion_to_ppa(path_folder, Country, "2015")
        
     #   marginal_cost = (values_maibach[Country] / 100) * (ref_inc_growth[2015] / ref_inc_growth[2008]) * (1/conversion_to_ppa)
    #else:
    marginal_cost_ref = (values_maibach["Germany"] / 100)
        
    gdp_per_capita = import_gdp_per_capita(path_folder, Country, "2015")
    gdp_per_capita_ref = import_gdp_per_capita(path_folder, "Germany", "2015")
    
    ref_inc_growth = imaclim[imaclim.Region == "EUR"][imaclim.Variable == "Index_income"].squeeze()
    
    conversion_to_ppa = import_conversion_to_ppa(path_folder, "Germany", "2015")
        
    marginal_cost = marginal_cost_ref * (ref_inc_growth[2015] / ref_inc_growth[2008]) * (1/conversion_to_ppa) * (gdp_per_capita / gdp_per_capita_ref)
        
    income_growth_year = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[year]
    income_growth_2015 = imaclim[(imaclim.Region == Region)&(imaclim.Variable == "Index_income")].squeeze()[2015]
    marginal_cost = marginal_cost * (income_growth_year / income_growth_2015)
    
    total_vkm = np.nansum(density[mode_choice == 0] * distance_cbd[mode_choice == 0]) * 365 * 2
    car_accidents = marginal_cost * total_vkm
    
    return car_accidents


def import_heat_values(Country, path_folder):
    
    heat = pd.read_excel(path_folder + "heat.xlsx")
    data_gdp = pd.read_csv(path_folder + "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2163510.csv", header =2) 
    data_gdp.loc[data_gdp["Country Name"] == "Cote d\'Ivoire", "Country Name"] = "Ivory_Coast"
    data_gdp.loc[data_gdp["Country Name"] == "United States", "Country Name"] = "USA"
    data_gdp.loc[data_gdp["Country Name"] == "New Zealand", "Country Name"] = "New_Zealand"
    data_gdp.loc[data_gdp["Country Name"] == "United Kingdom", "Country Name"] = "UK"
    data_gdp.loc[data_gdp["Country Name"] == "South Africa", "Country Name"] = "South_Africa"
    data_gdp.loc[data_gdp["Country Name"] == "Russian Federation", "Country Name"] = "Russia"
    data_gdp.loc[data_gdp["Country Name"] == "Hong Kong SAR, China", "Country Name"] = "Hong_Kong"
    data_gdp.loc[data_gdp["Country Name"] == "Iran, Islamic Rep.", "Country Name"] = "Iran"
    data_gdp.loc[data_gdp["Country Name"] == "Czech Republic", "Country Name"] = "Czech_Republic"

    mortality_rate = heat.mortality_rate[heat.Country == Country].squeeze()
    VSL = heat.VSL_2015[heat.Country == Country].squeeze()
    if np.isnan(VSL):
        heat_without_nan = heat[~np.isnan(heat.VSL_2015)]   
        heat_without_nan = heat_without_nan.merge(data_gdp, left_on = "Country", right_on = "Country Name")
    

        y = np.array(heat_without_nan.VSL_2015).reshape(-1, 1)
        X = pd.DataFrame({'X': heat_without_nan["2018"], 'intercept': np.ones(len(y)).squeeze()})
        ols = sm.OLS(y, X)
        reg1 = ols.fit()
    
        VSL = (reg1.params['X'] * data_gdp["2018"][data_gdp["Country Name"] == Country].squeeze()) + reg1.params['intercept']
    
        y = np.array(heat_without_nan.mortality_rate).reshape(-1, 1)
        X = pd.DataFrame({'X': heat_without_nan["2018"], 'intercept': np.ones(len(y)).squeeze()})
        ols = sm.OLS(y, X)
        reg1 = ols.fit()
        
        mortality_rate = (reg1.params['X'] * data_gdp["2018"][data_gdp["Country Name"] == Country].squeeze()) + reg1.params['intercept']
    return mortality_rate, VSL

    