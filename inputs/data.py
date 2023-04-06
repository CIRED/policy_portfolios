# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:41:24 2021

@author: charl
"""

import pandas as pd
import numpy as np

from inputs.parameters import *

def list_of_cities_and_databases(path_data_quentin,name):
    """ Import a list of cities for which data are available.

    Import the characteristics of the real estate, density and transport
    databases for each city.
    Select the most recent real estate database.
    Exclude Sales databases.
    Select Baidu transport database for chinese cities.
    """

    list_city = pd.read_csv(path_data_quentin + 'CityDatabases/'+name+'.csv')   
    list_city = list_city[['City', 'Country', 'GridEPSG', 'TransportSource', 
                           'RushHour', 'TransactionType', 'TransactionSource', 
                           'TransactionMonth', 'TransactionYear']]    
    list_city = list_city[list_city.TransactionType.eq('Rent')]
    #list_city = list_city[list_city.TransactionType.eq('Sale')]

    for city in np.unique(list_city.City):        
        most_recent_data = max(list_city.TransactionYear[list_city.City == city])
        i = list_city[((list_city.City == city) & 
                       (list_city.TransactionYear < most_recent_data))].index
        list_city = list_city.drop(i)
    
    for city in list_city.City[list_city.TransportSource == 'Baidu']:
        i = list_city[((list_city.City == city) & 
                       (list_city.TransportSource == 'Google'))].index
        list_city = list_city.drop(i)

    return list_city


def import_data(list_city, paths_data, city, path_folder):
    """ Import all data for a given city """


    path_data_city=paths_data+"Data/"


    country = list_city.Country[list_city.City == city].iloc[0]
    proj = str(int(list_city.GridEPSG[list_city.City == city].iloc[0]))
    transport_source = list_city.TransportSource[list_city.City == city].iloc[0]
    hour = list_city.RushHour[list_city.City == city].iloc[0]
    source_ici = list_city.TransactionSource[list_city.City == city].iloc[0]
    year = str(int(list_city.TransactionYear[list_city.City == city].iloc[0]))
    month = list_city.TransactionMonth[list_city.City == city].iloc[0]

    #Import data
    density = pd.read_csv(path_data_city + country + '/' + city +
                          '/Population_Density/grille_GHSL_density_2015_' +
                          str.upper(city) + '.txt', sep = '\s+|,', engine='python')

    rents_and_size = pd.read_csv(path_data_city + country + '/' + city + 
                                 '/Real_Estate/GridData/griddedRent_' + source_ici + 
                                 '_' + month + year + '_' + str.upper(city) + '_boxplotOutliers.csv')


    land_use = pd.read_csv(path_data_city + country + '/' + city + 
                           '/Land_Cover/gridUrb_ESACCI_LandCover_2015_' + 
                           str.upper(city) + '_' + proj +'.csv')
    
    land_cover_ESACCI = pd.read_csv(path_data_city + country + '/' + city + 
                           '/Land_Cover/grid_ESACCI_LandCover_2015_' + 
                           str.upper(city) + '_' + proj +'.csv')
            
    conversion_rate = import_conversion_to_ppa(path_folder, country, "2019")
    
    driving = pd.read_csv(path_data_city + country + '/' + city + 
                          '/Transport/interpDrivingTimes' + transport_source + '_' 
                          + city + '_' + hour + '_' + proj +'.csv')

    transit = pd.read_csv(path_data_city + country + '/' + city + 
                          '/Transport/interpTransitTimes' + transport_source + '_' + city + '_' + 
                          hour + '_' + proj + '.csv')

    grille = pd.read_csv(path_data_city + country + '/' + city + '/Grid/grille_' + 
                         str.upper(city) + '_finale.csv')
    
    centre = pd.read_csv(path_data_city + country + '/' + city + '/Grid/Centre_' 
                         + str.upper(city) + '_final.csv').to_numpy()

    if city == 'Prague' or city == 'Tianjin' or city == 'Paris':
        distance_cbd = ((grille.XCOORD / 1000 - centre[0, 0] / 1000) ** 2 + 
                        (grille.YCOORD / 1000 - centre[0, 1] / 1000) ** 2) ** 0.5
    elif city == 'Buenos_Aires' or city == 'Yerevan':
        distance_cbd = ((grille.XCOORD / 1000 - centre[0, 3] / 1000) ** 2 + 
                        (grille.YCOORD / 1000 - centre[0, 4] / 1000) ** 2) ** 0.5
    else:
        distance_cbd = ((grille.XCOORD / 1000 - centre[0, 1] / 1000) ** 2 + 
                        (grille.YCOORD / 1000 - centre[0, 2] / 1000) ** 2) ** 0.5

    print("Données pour " + city + " chargées")

    return country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, centre, distance_cbd, conversion_rate
