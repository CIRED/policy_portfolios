# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:35:30 2022

@author: charl
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx
import shapely

from inputs.parameters import *

def import_street_network(list_city, city, country, proj, centre, path_folder, path_data):
    boundary = gpd.read_file(path_data + 'Data/' + country +"/" + city + "/Grid/grille_" + city.upper() + "_finale_" + str(proj) + ".shp")
    G2 = ox.load_graphml(path_folder + "BRT/primary/graph/" + city + ".graphml")
    if ((city == 'Buenos_Aires') | (city == 'Yerevan')):
        target_xy = (centre[0, 4], centre[0, 3])
        
    elif city == 'Prague' or city == 'Tianjin' or city == 'Paris':
        target_xy = (centre[0, 1], centre[0, 0])
        
    else:
        target_xy = (centre[:, 2][0], centre[:, 1][0]) #2500000, 250000 (y, x)
        
    graph_proj = ox.project_graph(G2, to_crs='epsg:' + proj)
    Gs = ox.utils_graph.get_largest_component(graph_proj, strongly=True)
    
    city_streets = ox.graph_to_gdfs(Gs, 
                                    nodes=False, 
                                    edges=True,
                                    node_geometry=False, 
                                    fill_edge_geometry=True)
    
    length_network = sum(city_streets.geometry.length)
    #np.save(path_outputs + city + "_length_network.npy", length_network)
    
    nodes_proj, edges_proj = ox.graph_to_gdfs(Gs, nodes=True, edges=True)
    orig_dist = np.empty(len(boundary))
    target_dist = np.empty(len(boundary))
    transit_dist = np.empty(len(boundary))
    for cell in np.arange(len(boundary)):
        try:
            print(cell)
            orig_xy = (boundary.iloc[cell].YCOORD, boundary.iloc[cell].XCOORD)
            orig_node = ox.get_nearest_node(Gs, orig_xy, method='euclidean')
            target_node = ox.get_nearest_node(Gs, target_xy, method='euclidean')
            route = nx.shortest_path(G=Gs, source=orig_node, target=target_node, weight='length')
            #fig, ax = ox.plot_graph_route(Gs, route) #, origin_point=orig_xy, destination_point=target_xy)
            #plt.tight_layout()
            route_nodes = nodes_proj.loc[route]
            route_line = shapely.geometry.LineString(list(route_nodes.geometry.values))
            route_geom = gpd.GeoDataFrame(crs=edges_proj.crs)
            route_geom['geometry'] = None
            route_geom['osmids'] = None
            route_geom.loc[0, 'geometry'] = route_line
            route_geom['length_m'] = route_geom.length


            orig_dist[cell] = np.sqrt(((orig_xy[1] - graph_proj.nodes[orig_node]['x']) ** 2 ) + ((orig_xy[0]- graph_proj.nodes[orig_node]['y']) ** 2))
            target_dist[cell] = np.sqrt(((target_xy[1] - graph_proj.nodes[target_node]['x']) ** 2 ) + ((target_xy[0]- graph_proj.nodes[target_node]['y']) ** 2))
            transit_dist[cell] = route_geom['length_m']
        except:
            orig_dist[cell] = np.inf
            target_dist[cell] = np.inf
            transit_dist[cell] = np.inf
            pass
    
    return orig_dist, target_dist, transit_dist, length_network