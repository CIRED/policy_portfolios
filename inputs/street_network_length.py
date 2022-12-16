# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt 
import shapely
import osmnx as ox

from functions import *

def import_street_network(city, paths_data, list_city):
    
    path_data_city=paths_data+"Data/"
    country = list_city.Country[list_city.City == city].iloc[0]
    crs = (int(list_city.GridEPSG[list_city.City == city].iloc[0]))
    
    path_fig = "C:/Users/charl/OneDrive/Bureau/BRT/primary/fig/" + city + ".png"
    path_shp = "C:/Users/charl/OneDrive/Bureau/BRT/primary/graph/" + city + ".graphml"
    
    #import boundaries
    boundary = gpd.read_file("C:/Users/charl/OneDrive/Bureau/City_dataStudy/Data/" + country +"/" + city + "/Grid/grille_" + city.upper() + "_finale_" + str(crs) + ".shp")
    boundary_4326 = boundary.to_crs(4326)
    
    # download the graph
    city_streets = ox.graph_from_polygon(boundary_4326.unary_union,
                                         simplify=True, 
                                         network_type='drive',
                                         #custom_filter='["highway"~"motorway|trunk"]')
                                         #custom_filter='["highway"~"motorway|trunk|primary"]')
                                         #custom_filter='["highway"~"motorway|trunk|primary|secondary"]')
                                         custom_filter='["highway"~"primary"]')
                                         #custom_filter='["busway"]')
    #city_streets.to_crs(crs)                              
    ox.save_graphml(city_streets, path_shp)
    
    city_streets = ox.graph_to_gdfs(city_streets, 
                                    nodes=False, 
                                    edges=True,
                                    node_geometry=False, 
                                    fill_edge_geometry=True)
    
    
    #Back to local crs
    city_streets = city_streets.to_crs(crs)
    
    #Figure
    fig, ax = plt.subplots(figsize=(10,10))
    boundary.boundary.plot(ax=ax, alpha = 0.1)
    city_streets.plot(ax=ax, color='black',alpha=1)
    plt.savefig(path_fig)
    plt.close()
    
    length_network = sum(city_streets.geometry.length)
     
    return length_network


path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
list_city = list_of_cities_and_databases(path_data,'cityDatabase')
length = {}

for city in np.unique(list_city.City):
    length[city] = import_street_network(city, path_data, list_city)


    