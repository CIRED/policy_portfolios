# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:34:00 2022

@author: charl
"""

import pandas as pd
import numpy as np

def import_data_urbanized_area_reg(list_city):
    
    d_urbanized_area = {}
    d_population = {}
    d_income = {}
    d_land_price = {}
    d_commuting_price = {}
    d_commuting_time = {}
    d_polycentricity = {}
    
    for city in np.unique(list_city.City):
        country = list_city.Country[list_city.City == city].iloc[0]
        proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))

        #urbanized area
        land_use = pd.read_csv(path_data + 'Data/' + country + '/' + city + 
                               '/Land_Cover/grid_ESACCI_LandCover_2015_' + 
                               str.upper(city) + '_' + proj +'.csv')
        d_urbanized_area[city] = np.nansum(land_use.ESACCI190) / 1000000

        #Population
        density = pd.read_csv(path_data + 'Data/' + country + '/' + city +
                              '/Population_Density/grille_GHSL_density_2015_' +
                              str.upper(city) + '.txt', sep = '\s+|,', engine='python')
        density = density.loc[:,density.columns.str.startswith("density")],
        density = np.array(density).squeeze()
        d_population[city] = np.nansum(density)

        #Incomes
        data_gdp = pd.read_csv(path_folder + "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2163510.csv", header =2) 
        data_gdp["Country Name"][data_gdp["Country Name"] == "Cote d\'Ivoire"] = "Ivory_Coast"
        data_gdp["Country Name"][data_gdp["Country Name"] == "United States"] = "USA"
        data_gdp["Country Name"][data_gdp["Country Name"] == "New Zealand"] = "New_Zealand"
        data_gdp["Country Name"][data_gdp["Country Name"] == "United Kingdom"] = "UK"
        data_gdp["Country Name"][data_gdp["Country Name"] == "South Africa"] = "South_Africa"
        data_gdp["Country Name"][data_gdp["Country Name"] == "Russian Federation"] = "Russia"
        data_gdp["Country Name"][data_gdp["Country Name"] == "Hong Kong SAR, China"] = "Hong_Kong"
        data_gdp["Country Name"][data_gdp["Country Name"] == "Iran, Islamic Rep."] = "Iran"            
        data_gdp["Country Name"][data_gdp["Country Name"] == "Czech Republic"] = "Czech_Republic"
        d_income[city] = data_gdp["2018"][data_gdp["Country Name"] == country].squeeze()
    
        #Land prices
        d_land_price[city] = import_agricultural_rent(path_folder, country)

        #Commuting times
        transport_source = list_city.TransportSource[list_city.City == city].iloc[0]
        hour = list_city.RushHour[list_city.City == city].iloc[0]
        driving = pd.read_csv(path_data + 'Data/' +country + '/' + city + 
                              '/Transport/interpDrivingTimes' + transport_source + '_' 
                              + city + '_' + hour + '_' + proj +'.csv')
    
        transit = pd.read_csv(path_data + 'Data/' +country + '/' + city + 
                              '/Transport/interpTransitTimesGoogle_'+ city + '_' + 
                              hour + '_' + proj + '.csv')
    
        speed_driving = (driving.Distance / 1000) / (driving.Duration / 3600)
        speed_transit = (transit.Distance / 1000) / (transit.Duration / 3600)
        max_speed = np.fmax(speed_driving, speed_transit)
        d_commuting_time[city] = np.nansum(max_speed * density / np.nansum(density))
    
        #Modal share/Access to public transport ? Or average speed to go to the city center?
        #Diesel prices
        fuel_price = import_fuel_price(country, 'gasoline', path_folder) #L/100km
        #fuel_consumption = import_fuel_conso(country, path_folder) #dollars/L
        d_commuting_price[city] = fuel_price # * fuel_consumption / 100
    
        #Polycentricity index
        polycentricity = pd.read_excel(path_data + 'Article/Data_Criterias/CBD_Criterias_Table.ods', engine="odf")
        polycentricity = polycentricity.iloc[:, [2, 16]]
        polycentricity.columns = ['city', 'polycentricity_index']
        polycentricity.city[polycentricity.city == "Addis Ababa"] = "Addis_Ababa"
        polycentricity.city[polycentricity.city == "Belo Horizonte"] = "Belo_Horizonte"
        polycentricity.city[polycentricity.city == "Buenos Aires"] = "Buenos_Aires"
        polycentricity.city[polycentricity.city == "Cape Town"] = "Cape_Town"
        polycentricity.city[polycentricity.city == "Chiang Mai"] = "Chiang_Mai"
        polycentricity.city[polycentricity.city == "Cluj-Napoca"] = "Cluj_Napoca"
        polycentricity.city[polycentricity.city == "Frankfurt am Main"] = "Frankfurt_am_Main"
        polycentricity.city[polycentricity.city == "Goiânia"] = "Goiania"
        polycentricity.city[polycentricity.city == "Hong Kong"] = "Hong_Kong"
        polycentricity.city[polycentricity.city == "Los Angeles"] = "Los_Angeles"
        polycentricity.city[polycentricity.city == "Malmö"] = "Malmo"
        polycentricity.city[polycentricity.city == "Mar del Plata"] = "Mar_del_Plata"
        polycentricity.city[polycentricity.city == "Mexico City"] = "Mexico_City"
        polycentricity.city[polycentricity.city == "New York"] = "New_York"
        polycentricity.city[polycentricity.city == "Nizhny Novgorod"] = "Nizhny_Novgorod"
        polycentricity.city[polycentricity.city == "Porto Alegre"] = "Porto_Alegre"
        polycentricity.city[polycentricity.city == "Rio de Janeiro"] = "Rio_de_Janeiro"
        polycentricity.city[polycentricity.city == "Rostov-on-Don"] = "Rostov_on_Don"
        polycentricity.city[polycentricity.city == "San Diego"] = "San_Diego"
        polycentricity.city[polycentricity.city == "San Fransisco"] = "San_Fransisco"
        polycentricity.city[polycentricity.city == "Sao Paulo"] = "Sao_Paulo"
        polycentricity.city[polycentricity.city == "St Petersburg"] = "St_Petersburg"
        polycentricity.city[polycentricity.city == "The Hague"] = "The_Hague"
        polycentricity.city[polycentricity.city == "Ulan Bator"] = "Ulan_Bator"
        polycentricity.city[polycentricity.city == "Washington DC"] = "Washington_DC"
        polycentricity.city[polycentricity.city == "Zürich"] = "Zurich"
        d_polycentricity[city] = polycentricity.polycentricity_index[polycentricity.city == city].squeeze()
    
    df = pd.DataFrame()
    df["city"] = d_urbanized_area.keys()
    df["urbanized_area"] = d_urbanized_area.values()
    df["population"] = d_population.values()
    df["income"] = d_income.values()
    df["land_prices"] = d_land_price.values()
    df["commuting_price"] = d_commuting_price.values()
    df["commuting_time"] = d_commuting_time.values()
    df["polycentricity"] = d_polycentricity.values()
    
    df.to_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/dataset_model_density.xlsx")

    return df


def load_ssp(option_bdd):
    
    ssp = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ISO_code.xlsx")

    #POPULATION AND URBANIZATION RATE
    ssp_population = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ssp_population.xlsx")
    ssp_population = ssp_population.iloc[:, [1, 2, 6, 13, 23]]
    ssp_population.columns = ['Scenario', 'ISO', 'total_pop_2015', 'total_pop_2050', 'total_pop_2100']

    ssp_urban_share = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ssp_urbanization.xlsx")
    ssp_urban_share = ssp_urban_share.iloc[:, [1, 2, 6, 13, 23]]
    ssp_urban_share.columns = ['Scenario', 'ISO', 'urban_share_2015', 'urban_share_2050', 'urban_share_2100']

    ssp_population = ssp_population.merge(ssp_urban_share, on = ['ISO', 'Scenario'])
    
    ssp_population["urban_pop_2015"] = 1000000 * ssp_population["total_pop_2015"] * ssp_population["urban_share_2015"] / 100
    ssp_population["urban_pop_2050"] = 1000000 * ssp_population["total_pop_2050"] * ssp_population["urban_share_2050"] / 100
    ssp_population["urban_pop_2100"] = 1000000 * ssp_population["total_pop_2100"] * ssp_population["urban_share_2100"] / 100
    
    ssp_population["pop_growth_rate_2050"] = ssp_population["urban_pop_2050"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2100"] = ssp_population["urban_pop_2100"] / ssp_population["urban_pop_2015"]

    ssp_population = ssp_population.loc[:, ["Scenario", "ISO", "pop_growth_rate_2050", "pop_growth_rate_2100"]].iloc[0:965]
    ssp_population_2050 = ssp_population.pivot(index='ISO', columns='Scenario', values="pop_growth_rate_2050")
    ssp_population_2050.columns = ["pop_growth_rate_2050_SSP1", "pop_growth_rate_2050_SSP2", "pop_growth_rate_2050_SSP3", "pop_growth_rate_2050_SSP4", "pop_growth_rate_2050_SSP5"]
    ssp_population_2100 = ssp_population.pivot(index='ISO', columns='Scenario', values="pop_growth_rate_2100")
    ssp_population_2100.columns = ["pop_growth_rate_2100_SSP1", "pop_growth_rate_2100_SSP2", "pop_growth_rate_2100_SSP3", "pop_growth_rate_2100_SSP4", "pop_growth_rate_2100_SSP5"]
    
    ssp = ssp.merge(ssp_population_2050, left_on = "ISO", right_index = True)
    ssp = ssp.merge(ssp_population_2100, left_on = "ISO", right_index = True)
    
    #GDP
    if option_bdd == 'OECD':
        ssp_income = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ssp_gdp_oecd.xlsx")
    
    ssp_income = ssp_income.iloc[:, [1, 2, 6, 13, 23]]
    ssp_income.columns = ['Scenario', 'ISO', 'income_2015', 'income_2050', 'income_2100']

    ssp_income["income_growth_rate_2050"] = ssp_income["income_2050"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2100"] = ssp_income["income_2100"] / ssp_income["income_2015"]

    ssp_income = ssp_income.loc[:, ["Scenario", "ISO", "income_growth_rate_2050", "income_growth_rate_2100"]].iloc[0:920]
    ssp_income_2050 = ssp_income.pivot(index='ISO', columns='Scenario', values="income_growth_rate_2050")
    ssp_income_2050.columns = ["income_growth_rate_2050_SSP1", "income_growth_rate_2050_SSP2", "income_growth_rate_2050_SSP3", "income_growth_rate_2050_SSP4", "income_growth_rate_2050_SSP5"]
    ssp_income_2100 = ssp_income.pivot(index='ISO', columns='Scenario', values="income_growth_rate_2100")
    ssp_income_2100.columns = ["income_growth_rate_2100_SSP1", "income_growth_rate_2100_SSP2", "income_growth_rate_2100_SSP3", "income_growth_rate_2100_SSP4", "income_growth_rate_2100_SSP5"]
    
    ssp = ssp.merge(ssp_income_2050, left_on = "ISO", right_index = True)
    ssp = ssp.merge(ssp_income_2100, left_on = "ISO", right_index = True)
    
    return ssp

#def import_informal_housing():
#    informal_housing = pd.read_csv(path_folder + "API_EN.POP.SLUM.UR.ZS_DS2_en_csv_v2_2257750.csv", header = 2)
#    informal_housing["Country Name"][informal_housing["Country Name"] == "Cote d\'Ivoire"] = "Ivory_Coast"
#    informal_housing["Country Name"][informal_housing["Country Name"] == "United States"] = "USA"
#    informal_housing["Country Name"][informal_housing["Country Name"] == "New Zealand"] = "New_Zealand"
#    informal_housing["Country Name"][informal_housing["Country Name"] == "United Kingdom"] = "UK"
#    informal_housing["Country Name"][informal_housing["Country Name"] == "South Africa"] = "South_Africa"
#    informal_housing["Country Name"][informal_housing["Country Name"] == "Russian Federation"] = "Russia"
#    informal_housing["Country Name"][informal_housing["Country Name"] == "Hong Kong SAR, China"] = "Hong_Kong"
#    informal_housing["Country Name"][informal_housing["Country Name"] == "Iran, Islamic Rep."] = "Iran"
#    informal_housing["Country Name"][informal_housing["Country Name"] == "Czech Republic"] = "Czech_Republic"
#    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2018"]
#    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2017"]
#    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2016"]
#    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2015"]
#    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2014"]
#    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2013"]
#    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2012"]
#    informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2011"]
 #   informal_housing["2019"][np.isnan(informal_housing["2019"])] = informal_housing["2010"]
#    informal_housing = informal_housing[["Country Name", "2019"]]
#    informal_housing.columns = ["Country", "informal_housing"]
#    informal_housing.informal_housing[np.isnan(informal_housing.informal_housing)] = 0
#    return informal_housing

def load_ssp_nedum(option_bdd, option_ssp, city):
    
    ssp = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ISO_code.xlsx")
    
    #POPULATION AND URBANIZATION RATE
    ssp_population = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ssp_population.xlsx")
    ssp_population = ssp_population.loc[ssp_population.Scenario == option_ssp, :].iloc[:, [2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    ssp_population.columns = ['ISO', 'total_pop_2015', 'total_pop_2020', 'total_pop_2025', 'total_pop_2030', 'total_pop_2035', 'total_pop_2040', 'total_pop_2045', 'total_pop_2050', 'total_pop_2055', 'total_pop_2060', 'total_pop_2065', 'total_pop_2070', 'total_pop_2075', 'total_pop_2080', 'total_pop_2085', 'total_pop_2090', 'total_pop_2095', 'total_pop_2100']

    ssp_urban_share = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ssp_urbanization.xlsx")
    ssp_urban_share = ssp_urban_share.loc[ssp_urban_share.Scenario == option_ssp, :].iloc[:, [2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    ssp_urban_share.columns = ['ISO', 'urban_share_2015', 'urban_share_2020', 'urban_share_2025', 'urban_share_2030', 'urban_share_2035', 'urban_share_2040', 'urban_share_2045', 'urban_share_2050', 'urban_share_2055', 'urban_share_2060', 'urban_share_2065', 'urban_share_2070', 'urban_share_2075', 'urban_share_2080', 'urban_share_2085', 'urban_share_2090', 'urban_share_2095', 'urban_share_2100']

    ssp_population = ssp_population.merge(ssp_urban_share, on = ['ISO'])
    
    ssp_population["urban_pop_2015"] = 1000000 * ssp_population["total_pop_2015"] * ssp_population["urban_share_2015"] / 100
    ssp_population["urban_pop_2020"] = 1000000 * ssp_population["total_pop_2020"] * ssp_population["urban_share_2020"] / 100
    ssp_population["urban_pop_2025"] = 1000000 * ssp_population["total_pop_2025"] * ssp_population["urban_share_2025"] / 100
    ssp_population["urban_pop_2030"] = 1000000 * ssp_population["total_pop_2030"] * ssp_population["urban_share_2030"] / 100
    ssp_population["urban_pop_2035"] = 1000000 * ssp_population["total_pop_2035"] * ssp_population["urban_share_2035"] / 100
    ssp_population["urban_pop_2040"] = 1000000 * ssp_population["total_pop_2040"] * ssp_population["urban_share_2040"] / 100
    ssp_population["urban_pop_2045"] = 1000000 * ssp_population["total_pop_2045"] * ssp_population["urban_share_2045"] / 100
    ssp_population["urban_pop_2050"] = 1000000 * ssp_population["total_pop_2050"] * ssp_population["urban_share_2050"] / 100
    ssp_population["urban_pop_2055"] = 1000000 * ssp_population["total_pop_2055"] * ssp_population["urban_share_2055"] / 100
    ssp_population["urban_pop_2060"] = 1000000 * ssp_population["total_pop_2060"] * ssp_population["urban_share_2060"] / 100
    ssp_population["urban_pop_2065"] = 1000000 * ssp_population["total_pop_2065"] * ssp_population["urban_share_2065"] / 100
    ssp_population["urban_pop_2070"] = 1000000 * ssp_population["total_pop_2070"] * ssp_population["urban_share_2070"] / 100
    ssp_population["urban_pop_2075"] = 1000000 * ssp_population["total_pop_2075"] * ssp_population["urban_share_2075"] / 100
    ssp_population["urban_pop_2080"] = 1000000 * ssp_population["total_pop_2080"] * ssp_population["urban_share_2080"] / 100
    ssp_population["urban_pop_2085"] = 1000000 * ssp_population["total_pop_2085"] * ssp_population["urban_share_2085"] / 100
    ssp_population["urban_pop_2090"] = 1000000 * ssp_population["total_pop_2090"] * ssp_population["urban_share_2090"] / 100
    ssp_population["urban_pop_2095"] = 1000000 * ssp_population["total_pop_2095"] * ssp_population["urban_share_2095"] / 100
    ssp_population["urban_pop_2100"] = 1000000 * ssp_population["total_pop_2100"] * ssp_population["urban_share_2100"] / 100
    
    ssp_population["pop_growth_rate_2015"] = ssp_population["urban_pop_2015"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2020"] = ssp_population["urban_pop_2020"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2025"] = ssp_population["urban_pop_2025"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2030"] = ssp_population["urban_pop_2030"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2035"] = ssp_population["urban_pop_2035"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2040"] = ssp_population["urban_pop_2040"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2045"] = ssp_population["urban_pop_2045"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2050"] = ssp_population["urban_pop_2050"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2055"] = ssp_population["urban_pop_2055"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2060"] = ssp_population["urban_pop_2060"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2065"] = ssp_population["urban_pop_2065"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2070"] = ssp_population["urban_pop_2070"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2075"] = ssp_population["urban_pop_2075"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2080"] = ssp_population["urban_pop_2080"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2085"] = ssp_population["urban_pop_2085"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2090"] = ssp_population["urban_pop_2090"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2095"] = ssp_population["urban_pop_2095"] / ssp_population["urban_pop_2015"]
    ssp_population["pop_growth_rate_2100"] = ssp_population["urban_pop_2100"] / ssp_population["urban_pop_2015"]

    ssp_population = ssp_population.loc[:, ["ISO", "pop_growth_rate_2015", "pop_growth_rate_2020", "pop_growth_rate_2025", "pop_growth_rate_2030", "pop_growth_rate_2035", "pop_growth_rate_2040", "pop_growth_rate_2045", "pop_growth_rate_2050", "pop_growth_rate_2055", "pop_growth_rate_2060", "pop_growth_rate_2065", "pop_growth_rate_2070", "pop_growth_rate_2075", "pop_growth_rate_2080", "pop_growth_rate_2085", "pop_growth_rate_2090", "pop_growth_rate_2095", "pop_growth_rate_2100"]].iloc[0:965]
    
    #GDP
    if option_bdd == 'OECD':
        ssp_income = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl/ssp_gdp_oecd.xlsx")
    
    ssp_income = ssp_income.loc[ssp_income.Scenario == option_ssp, :].iloc[:, [2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    ssp_income.columns = ['ISO', 'income_2015', 'income_2020', 'income_2025', 'income_2030', 'income_2035', 'income_2040', 'income_2045', 'income_2050', 'income_2055', 'income_2060', 'income_2065', 'income_2070', 'income_2075', 'income_2080', 'income_2085', 'income_2090', 'income_2095', 'income_2100']
    
    ssp_income["income_growth_rate_2015"] = ssp_income["income_2015"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2020"] = ssp_income["income_2020"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2025"] = ssp_income["income_2025"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2030"] = ssp_income["income_2030"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2035"] = ssp_income["income_2035"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2040"] = ssp_income["income_2040"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2045"] = ssp_income["income_2045"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2050"] = ssp_income["income_2050"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2055"] = ssp_income["income_2055"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2060"] = ssp_income["income_2060"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2065"] = ssp_income["income_2065"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2070"] = ssp_income["income_2070"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2075"] = ssp_income["income_2075"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2080"] = ssp_income["income_2080"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2085"] = ssp_income["income_2085"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2090"] = ssp_income["income_2090"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2095"] = ssp_income["income_2095"] / ssp_income["income_2015"]
    ssp_income["income_growth_rate_2100"] = ssp_income["income_2100"] / ssp_income["income_2015"]

    ssp_income = ssp_income.loc[:, ["ISO", "income_growth_rate_2015", "income_growth_rate_2020", "income_growth_rate_2025", "income_growth_rate_2030", "income_growth_rate_2035", "income_growth_rate_2040", "income_growth_rate_2045", "income_growth_rate_2050", "income_growth_rate_2055", "income_growth_rate_2060", "income_growth_rate_2065", "income_growth_rate_2070", "income_growth_rate_2075", "income_growth_rate_2080", "income_growth_rate_2085", "income_growth_rate_2090", "income_growth_rate_2095", "income_growth_rate_2100"]].iloc[0:920]
    #ssp_income_2050 = ssp_income.pivot(index='ISO', columns='Scenario', values="income_growth_rate_2050")
    #ssp_income_2050.columns = ["income_growth_rate_2050_SSP1", "income_growth_rate_2050_SSP2", "income_growth_rate_2050_SSP3", "income_growth_rate_2050_SSP4", "income_growth_rate_2050_SSP5"]
    #ssp_income_2100 = ssp_income.pivot(index='ISO', columns='Scenario', values="income_growth_rate_2100")
    #ssp_income_2100.columns = ["income_growth_rate_2100_SSP1", "income_growth_rate_2100_SSP2", "income_growth_rate_2100_SSP3", "income_growth_rate_2100_SSP4", "income_growth_rate_2100_SSP5"]
    
    ssp = ssp.merge(ssp_population, on = "ISO")
    ssp = ssp.merge(ssp_income, on = "ISO")
    
    ssp = ssp.loc[ssp.City == city, :].iloc[:, 3:39]
    
    return ssp