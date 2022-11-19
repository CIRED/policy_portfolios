# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:31:46 2021

@author: charl
"""

import pandas as pd
import numpy as np

### IMPORT AND UPDATE PARAMETERS

def import_parameters():
    INTEREST_RATE = 0.05
    HOUSEHOLD_SIZE = 1
    TIME_LAG = 2
    DEPRECIATION_TIME = 100
    DURATION = 20
    COEFF_URB = 0.62
    FIXED_COST_CAR = 0
    WALKING_SPEED = 5
    CO2_EMISSIONS_TRANSIT = 15 #gCO2/pkm
    return INTEREST_RATE, HOUSEHOLD_SIZE, TIME_LAG, DEPRECIATION_TIME, DURATION, COEFF_URB, FIXED_COST_CAR, WALKING_SPEED, CO2_EMISSIONS_TRANSIT

def import_BRT_parameters(BRT_scenario):
    if BRT_scenario == 'baseline_25_0_12_50_5':
        BRT_SPEED = 25
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 12230000
        option_capital_cost_evolution = '50years_5percent'
    elif BRT_scenario == 'speed_5_0_12_50_5':
        BRT_SPEED = 5
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 12230000
        option_capital_cost_evolution = '50years_5percent'
    elif BRT_scenario == 'speed_40_0_12_50_5':
        BRT_SPEED = 40
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 12230000
        option_capital_cost_evolution = '50years_5percent'
    elif BRT_scenario == 'speed_40_0_0_50_5':
        BRT_SPEED = 40
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 0
        option_capital_cost_evolution = '50years_5percent'
    elif BRT_scenario == 'speed_90_0_12_50_5':
        BRT_SPEED = 90
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 12230000
        option_capital_cost_evolution = '50years_5percent'
    elif BRT_scenario == 'operating_costs_25_333_12_50_5':
        BRT_SPEED = 25
        BRT_OPERATING_COST = 3330000
        BRT_CAPITAL_COST = 12230000
        option_capital_cost_evolution = '50years_5percent'
    elif BRT_scenario == 'capital_costs_25_0_01_50_5':
        BRT_SPEED = 25
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 100000
        option_capital_cost_evolution = '50years_5percent'
    elif BRT_scenario == 'capital_costs_25_0_40_50_5':
        BRT_SPEED = 25
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 40000000
        option_capital_cost_evolution = '50years_5percent'
    elif BRT_scenario == 'capital_evolution_25_0_12_15_5':
        BRT_SPEED = 25
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 12230000
        option_capital_cost_evolution = '15years_5percent'
    elif BRT_scenario == 'capital_evolution_25_0_12_50_income':
        BRT_SPEED = 25
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 12230000
        option_capital_cost_evolution = 'prop_income_50'
    elif BRT_scenario == 'capital_evolution_25_0_12_15_income':
        BRT_SPEED = 25
        BRT_OPERATING_COST = 0
        BRT_CAPITAL_COST = 12230000
        option_capital_cost_evolution = 'prop_income_15'
    return BRT_SPEED, BRT_OPERATING_COST, BRT_CAPITAL_COST, option_capital_cost_evolution

def import_conversion_to_ppa(path_folder, country, year):
    conversion_to_ppa = pd.read_csv(path_folder + 'API_PA.NUS.PPP_DS2_en_csv_v2_2165956.csv', header = 2)
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "Cote d\'Ivoire", "Country Name"] = "Ivory_Coast"
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "United States", "Country Name"] = "USA"
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "New Zealand", "Country Name"] = "New_Zealand"
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "United Kingdom", "Country Name"] = "UK"
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "South Africa", "Country Name"] = "South_Africa"
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "Russian Federation", "Country Name"] = "Russia"
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "Hong Kong SAR, China", "Country Name"] = "Hong_Kong"
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "Iran, Islamic Rep.", "Country Name"] = "Iran"
    conversion_to_ppa.loc[conversion_to_ppa["Country Name"] == "Czech Republic", "Country Name"] = "Czech_Republic"
    conversion_rate = conversion_to_ppa[year][conversion_to_ppa["Country Name"] == country].iloc[0]
    return conversion_rate

def import_gdp_per_capita(path_folder, country, year, source = None):
    if source == 'PP':
        data_gdp = pd.read_csv(path_folder + "GDP/income_data/API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_988619/API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_988619.csv", header =2) 
    else:
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
    data_gdp.loc[data_gdp["Country Name"] == "Iran", "2018"] = data_gdp.loc[data_gdp["Country Name"] == "Iran", "2017"]
    gdp_per_capita = data_gdp[year][data_gdp["Country Name"] == country].squeeze()
    return gdp_per_capita

def import_gini_index(path_folder):
    gini = pd.read_csv(path_folder + "API_SI.POV.GINI_DS2_en_csv_v2_2252167.csv", header = 2)
    gini.loc[gini["Country Name"] == "Cote d\'Ivoire", "Country Name"] = "Ivory_Coast"
    gini.loc[gini["Country Name"] == "United States", "Country Name"] = "USA"
    gini.loc[gini["Country Name"] == "New Zealand", "Country Name"] = "New_Zealand"
    gini.loc[gini["Country Name"] == "United Kingdom", "Country Name"] = "UK"
    gini.loc[gini["Country Name"] == "South Africa", "Country Name"] = "South_Africa"
    gini.loc[gini["Country Name"] == "Russian Federation", "Country Name"] = "Russia"
    gini.loc[gini["Country Name"] == "Hong Kong SAR, China", "Country Name"] = "Hong_Kong"
    gini.loc[gini["Country Name"] == "Iran, Islamic Rep.", "Country Name"] = "Iran"
    gini.loc[gini["Country Name"] == "Czech Republic", "Country Name"] = "Czech_Republic"
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2018"]
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2017"]
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2016"]
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2015"]
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2014"]
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2013"]
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2012"]
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2011"]
    gini.loc[np.isnan(gini["2019"]), "2019"] = gini["2010"]
    gini = gini[["Country Name", "2019"]]
    gini.columns = ["Country", "gini"]
    return gini

def import_country_scenarios(country, path_data):
    """ Import World Urbanization Prospects scenarios.
    
    Urban opulation growth rate at the country scale.
    To be used when data at the city scale are not available.
    """
    
    country = country.replace('_', ' ')
    
    
    scenario_growth_rate = pd.read_excel(path_data + 'WUP2018-F06-Urban_Growth_Rate.xls', 
                                         skiprows = 14, 
                                         header = 1)
    
    scenario_growth_rate = scenario_growth_rate.rename(
        columns = {
            'Unnamed: 1' : 'country', 
            'Unnamed: 17': '2015-2020', 
            'Unnamed: 18': '2020-2025', 
            'Unnamed: 19': '2025-2030', 
            'Unnamed: 20': '2030-2035'})
    
    growth_rate = {
        "2015-2020" : scenario_growth_rate.loc[(scenario_growth_rate.country.str.find(country) != -1), '2015-2020'].squeeze(),
        "2020-2025" : scenario_growth_rate.loc[(scenario_growth_rate.country.str.find(country) != -1), '2020-2025'].squeeze(),
        "2025-2030" : scenario_growth_rate.loc[(scenario_growth_rate.country.str.find(country) != -1), '2025-2030'].squeeze(),
        "2030-2035" : scenario_growth_rate.loc[(scenario_growth_rate.country.str.find(country) != -1), '2030-2035'].squeeze()}
       
    return growth_rate

def import_city_scenarios(city, country, path_data):
    """ Import World Urbanization Prospects scenarios.
    
    Population growth rate at the city scale.
    """
    
    city = city.replace('_', ' ')
    city = city.replace('Ahmedabad', 'Ahmadabad')
    city = city.replace('Belem', 'Belém')
    city = city.replace('Bogota', 'Bogot')
    city = city.replace('Brasilia', 'Bras')
    city = city.replace('Brussels', 'Brussel')
    city = city.replace('Wroclaw', 'Wroc')
    city = city.replace('Valparaiso', 'Valpar')
    city = city.replace('Ulan Bator', 'Ulaanbaatar')
    city = city.replace('St Petersburg', 'Petersburg')
    city = city.replace('Sfax', 'Safaqis')
    city = city.replace('Seville', 'Sevilla')
    city = city.replace('Sao Paulo', 'Paulo')
    city = city.replace('Poznan', 'Pozna')
    city = city.replace('Porto Alegre', 'Alegre')
    city = city.replace('Nuremberg', 'Nurenberg')
    city = city.replace('Medellin', 'Medell')
    city = city.replace('Washington DC', 'Washington')
    city = city.replace('San Fransisco', 'San Francisco')
    city = city.replace('Rostov on Don', 'Rostov')
    city = city.replace('Nizhny Novgorod', 'Novgorod')
    city = city.replace('Mar del Plata', 'Mar Del Plata')
    city = city.replace('Malmo', 'Malm')
    city = city.replace('Lodz', 'Łódź')
    city = city.replace('Leeds', 'West Yorkshire')
    city = city.replace('Jinan', "Ji'nan")
    city = city.replace('Isfahan', 'Esfahan')
    city = city.replace('Hanover', 'Hannover')
    city = city.replace('Gothenburg', 'teborg')
    city = city.replace('Goiania', 'nia')
    city = city.replace('Ghent', 'Gent')
    city = city.replace('Geneva', 'Genève')
    city = city.replace('Fez', 'Fès')
    city = city.replace('Cluj Napoca', 'Cluj-Napoca')
    city = city.replace('Cordoba', 'rdoba')
    city = city.replace('Concepcion', 'Concepc')
    country = country.replace('_', ' ')
    country = country.replace('UK', 'United Kingdom')
    country = country.replace('Russia', 'Russian Federation')
    country = country.replace('USA', 'United States of America')
    country = country.replace('Czech Republic', 'Czechia')
    country = country.replace('Ivory Coast', 'Ivoire')
    
    scenario_growth_rate = pd.read_excel(path_data + 'WUP2018-F14-Growth_Rate_Cities.xls', 
                                         skiprows = 15, 
                                         header = 1)
    
    scenario_growth_rate = scenario_growth_rate.rename(
        columns={
            "Urban Agglomeration" : 'city', 
            "Country or area" : 'country'})
    
    growth_rate = {
        "2015-2020" : scenario_growth_rate.loc[(scenario_growth_rate.city.str.find(city) != -1) & (scenario_growth_rate.country.str.find(country) != -1), '2015-2020'].squeeze(),
        "2020-2025" : scenario_growth_rate.loc[(scenario_growth_rate.city.str.find(city) != -1) & (scenario_growth_rate.country.str.find(country) != -1), '2020-2025'].squeeze(),
        "2025-2030" : scenario_growth_rate.loc[(scenario_growth_rate.city.str.find(city) != -1) & (scenario_growth_rate.country.str.find(country) != -1), '2025-2030'].squeeze(),
        "2030-2035" : scenario_growth_rate.loc[(scenario_growth_rate.city.str.find(city) != -1) & (scenario_growth_rate.country.str.find(country) != -1), '2030-2035'].squeeze()}
    

    return growth_rate

def import_agricultural_rent(path_folder, country):
    data_gdp = pd.read_csv(path_folder + "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2163564.csv", header = 4) 
    data_gdp.loc[data_gdp["Country Name"] == "Cote d\'Ivoire", "Country Name"] = "Ivory_Coast"
    data_gdp.loc[data_gdp["Country Name"] == "United States", "Country Name"] = "USA"
    data_gdp.loc[data_gdp["Country Name"] == "New Zealand", "Country Name"] = "New_Zealand"
    data_gdp.loc[data_gdp["Country Name"] == "United Kingdom", "Country Name"] = "UK"
    data_gdp.loc[data_gdp["Country Name"] == "South Africa", "Country Name"] = "South_Africa"
    data_gdp.loc[data_gdp["Country Name"] == "Russian Federation", "Country Name"] = "Russia"
    data_gdp.loc[data_gdp["Country Name"] == "Hong Kong SAR, China", "Country Name"] = "Hong_Kong"
    data_gdp.loc[data_gdp["Country Name"] == "Iran, Islamic Rep.", "Country Name"] = "Iran"
    data_gdp.loc[data_gdp["Country Name"] == "Czech Republic", "Country Name"] = "Czech_Republic"
    gdp = data_gdp["2018"][data_gdp["Country Name"] == country]
    
    data_share_gdp = pd.read_csv(path_folder + "FAOSTAT_GPP_share.csv", header = 0) 
    data_share_gdp.loc[data_share_gdp["Area"] == "Côte d\'Ivoire", "Area"] = "Ivory_Coast"
    data_share_gdp.loc[data_share_gdp["Area"] == "United States of America", "Area"] = "USA"
    data_share_gdp.loc[data_share_gdp["Area"] == "New Zealand", "Area"] = "New_Zealand"
    data_share_gdp.loc[data_share_gdp["Area"] == "United Kingdom of Great Britain and Northern Ireland", "Area"] = "UK"
    data_share_gdp.loc[data_share_gdp["Area"] == "South Africa", "Area"] = "South_Africa"
    data_share_gdp.loc[data_share_gdp["Area"] == "Russian Federation", "Area"] = "Russia"
    data_share_gdp.loc[data_share_gdp["Area"] == "China, Hong Kong SAR", "Area"] = "Hong_Kong"
    data_share_gdp.loc[data_share_gdp["Area"] == "Iran (Islamic Republic of)", "Area"] = "Iran"
    data_share_gdp.loc[data_share_gdp["Area"] == "Czechia", "Area"] = "Czech_Republic"
    share_gdp = data_share_gdp["Value"][(data_share_gdp["Area"] == country) & (data_share_gdp["Year"] == 2018) & (data_share_gdp["Item"] == "Value Added (Agriculture, Forestry and Fishing)")]
    
    agricultural_gdp = (share_gdp.squeeze()/100) * gdp.squeeze()
    
    data_surface = pd.read_csv(path_folder + "FAOSTAT_data_4-13-2021.csv", header = 0) 
    data_surface.loc[data_surface["Area"] == "Côte d\'Ivoire", "Area"] = "Ivory_Coast"
    data_surface.loc[data_surface["Area"] == "United States of America", "Area"] = "USA"
    data_surface.loc[data_surface["Area"] == "New Zealand", "Area"] = "New_Zealand"
    data_surface.loc[data_surface["Area"] == "United Kingdom of Great Britain and Northern Ireland", "Area"] = "UK"
    data_surface.loc[data_surface["Area"] == "South Africa", "Area"] = "South_Africa"
    data_surface.loc[data_surface["Area"] == "Russian Federation", "Area"] = "Russia"
    data_surface.loc[data_surface["Area"] == "China, Hong Kong SAR", "Area"] = "Hong_Kong"
    data_surface.loc[data_surface["Area"] == "Iran (Islamic Republic of)", "Area"] = "Iran"
    data_surface.loc[data_surface["Area"] == "Czechia", "Area"] = "Czech_Republic"
    surface = data_surface["Value"][(data_surface["Area"] == country) & (data_surface["Year"] == 2018)] * 10**7
    
    agricultural_rent = agricultural_gdp / surface.squeeze()
    return agricultural_rent


def update_kappa(kappa, income_lag, income, b):
    #return kappa * ((income / income_lag) ** 2)
    return kappa * ((income / income_lag) ** (-b))

def update_population(population, population_growth, index):
    if index < 5:
        population = population * (1 + (population_growth["2015-2020"] / 100))
    elif index < 10:
        population = population * (1 + (population_growth["2020-2025"] / 100))
    elif index < 15:
        population = population * (1 + (population_growth["2025-2030"] / 100))
    elif index > 14:
        population = population * (1 + (population_growth["2030-2035"] / 100))
    return population

def import_informal_housing(list_city, path_folder):
    #Informal housing
    informal_housing = pd.read_csv(path_folder + "API_EN.POP.SLUM.UR.ZS_DS2_en_csv_v2_2257750.csv", header = 2)
    informal_housing["Country Name"][informal_housing["Country Name"] == "Cote d\'Ivoire"] = "Ivory_Coast"
    informal_housing["Country Name"][informal_housing["Country Name"] == "United States"] = "USA"
    informal_housing["Country Name"][informal_housing["Country Name"] == "New Zealand"] = "New_Zealand"
    informal_housing["Country Name"][informal_housing["Country Name"] == "United Kingdom"] = "UK"
    informal_housing["Country Name"][informal_housing["Country Name"] == "South Africa"] = "South_Africa"
    informal_housing["Country Name"][informal_housing["Country Name"] == "Russian Federation"] = "Russia"
    informal_housing["Country Name"][informal_housing["Country Name"] == "Hong Kong SAR, China"] = "Hong_Kong"
    informal_housing["Country Name"][informal_housing["Country Name"] == "Iran, Islamic Rep."] = "Iran"
    informal_housing["Country Name"][informal_housing["Country Name"] == "Czech Republic"] = "Czech_Republic"
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2018"]
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2017"]
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2016"]
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2015"]
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2014"]
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2013"]
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2012"]
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2011"]
    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2010"]
    informal_housing = informal_housing[["Country Name", "2019"]]
    informal_housing.columns = ["Country", "informal_housing"]
    informal_housing.informal_housing[np.isnan(informal_housing.informal_housing)] = 0
    informal_housing = list_city.loc[:, ['City', 'Country']].merge(informal_housing, on = "Country")
    return informal_housing
