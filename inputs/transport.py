# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:19:25 2021

@author: charl
"""

import pandas as pd
import numpy as np
import copy
import math

###### TRANSPORT

def import_fuel_price(country, type_fuel, path_data):
    if type_fuel == 'gasoline':
        data = pd.read_excel(path_data + "API_EP.PMP.SGAS.CD_DS2_en_excel_v2_2057373.xls", sheet_name = 'Data', header = 3, index_col = 0)
    if type_fuel == 'diesel':
        data = pd.read_excel(path_data + "API_EP.PMP.DESL.CD_DS2_en_excel_v2_2055850.xls", sheet_name = 'Data', header = 3, index_col = 0)

    for i in range(len(data)):
        if np.isnan(data["2016"][i]):
            data["2016"].iloc[i] = copy.deepcopy(data["2014"][i])
        
        if np.isnan(data["2016"][i]):
            data["2016"].iloc[i] = copy.deepcopy(data["2012"][i])
        
    data.rename(index={'United States':'USA'},inplace=True)
    #data.rename(index={'Hong Kong SAR, China':'China, Hong Kong SAR'},inplace=True)
    data.rename(index={"Cote d'Ivoire":"Ivory_Coast"},inplace=True)
    data.rename(index={"Czech Republic":"Czech_Republic"},inplace=True)
    data.rename(index={"New Zealand":"New_Zealand"},inplace=True)
    data.rename(index={"United Kingdom":"UK"},inplace=True)
    
    data.rename(index={'South Africa':'South_Africa'},inplace=True)
    data.rename(index={'Russian Federation':'Russia'},inplace=True)
    data.rename(index={'Hong Kong SAR, China':'Hong_Kong'},inplace=True)
    data.rename(index={'Iran, Islamic Rep.':'Iran'},inplace=True)
 

    price_fuel = data["2016"][country]
    
    return price_fuel

def import_public_transport_cost_data(path_folder, city):
    public_transport_price = pd.read_excel(path_folder + 'transport_price_data.ods', engine="odf")
    public_transport_price = public_transport_price.drop(0)
    public_transport_price.Numbep = public_transport_price["Numbep"].astype(float)
    public_transport_price.Worldatlas = public_transport_price["Worldatlas"].astype(float)
    public_transport_price.Kiwi = public_transport_price["Kiwi"].astype(float)
    public_transport_price.loc[np.isnan(public_transport_price.Numbep), 'Numbep'] = (public_transport_price.Worldatlas[np.isnan(public_transport_price.Numbep)] + public_transport_price.Kiwi[np.isnan(public_transport_price.Numbep)]) / 2
    public_transport_price.loc[public_transport_price.country == 'Argentina', 'Numbep'] = public_transport_price.Numbep[public_transport_price.city == "Buenos_Aires"]
    public_transport_price.loc[np.isnan(public_transport_price.Numbep), 'Numbep'] = 0
    return public_transport_price.Numbep[public_transport_price.city == city]

def import_fuel_conso(country, path_data):
    fuel_consumption_data = pd.read_excel(path_data + "GFEIAnnexC.xlsx", sheet_name = "Average fuel consumption", header = 2, index_col = 0)
    fuel_consumption_data.columns = fuel_consumption_data.columns.map(str)
    fuel_consumption_data.rename(index={'United States of America':'USA'},inplace=True)
    fuel_consumption_data.rename(index={'Czech Republic':'Czechia'},inplace=True)
    fuel_consumption_data.rename(index={'Russian Federation':'Russia'},inplace=True)
    fuel_consumption_data.rename(index={'South Africa':'South_Africa'},inplace=True)
    fuel_consumption_data.rename(index={'United Kingdom':'UK'},inplace=True)


    for i in range(len(fuel_consumption_data)):
        if np.isnan(fuel_consumption_data["2017"][i]):
            fuel_consumption_data["2017"].iloc[i] = copy.deepcopy(fuel_consumption_data["2015"][i])
    
        if np.isnan(fuel_consumption_data["2017"][i]):
            fuel_consumption_data["2017"].iloc[i] = copy.deepcopy(fuel_consumption_data["2013"][i])

    fuel_consumption = np.nanmedian(fuel_consumption_data["2017"])    
    try:
        fuel_consumption = copy.deepcopy(fuel_consumption_data["2017"][country])

    except KeyError:
        print('Not in the database - we take the median')
        
    return fuel_consumption



def transport_modeling(driving, transit, income, fuel_price, fuel_consumption, FIXED_COST_CAR, monetary_cost_pt, distance_cbd, WALKING_SPEED, policy, index, city, grille, centre, orig_dist, target_dist, transit_dist, BRT_SPEED):
    prix_driving = driving.Duration * income / (3600 * 24) / 365 + driving.Distance * fuel_price * fuel_consumption / 100000 + FIXED_COST_CAR
    if ((index > 4) & (policy == 'transit_speed')):
        prix_transit = ((transit.Duration  / 1.2) * income / (3600 * 24) / 365) + monetary_cost_pt
    elif ((policy == 'BRT') & (index > 4)) | ((policy == 'synergy') & (index > 4)):#| ((policy == 'all') & (index > 4)):
        prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
        prix_BRT = (((((orig_dist + target_dist) / (WALKING_SPEED * 1000)) + ((transit_dist / 1000) / BRT_SPEED)) * (income / (24 * 365))) + monetary_cost_pt).squeeze()
        prix_transit = np.nanmin(np.vstack((prix_transit, prix_BRT)), axis = 0)
    elif ((policy == 'basic_infra') & (index > 4))| ((policy == 'all') & (index > 4)):
        if city == 'Prague' or city == 'Tianjin' or city == 'Paris':
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][0]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][1]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][0]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][1]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
        elif city == 'Buenos_Aires' or city == 'Yerevan':
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][3]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][4]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][3]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][4]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
        else:
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][1]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][2]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][1]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][2]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)

    else:
        prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
    prix_walking = ((distance_cbd) / WALKING_SPEED) * (income / (24 * 365))
    prix_walking[distance_cbd > 8] = math.inf
    tous_prix=np.vstack((prix_driving,prix_transit, prix_walking))
    prix_transport=np.amin(tous_prix, axis=0)
    prix_transport[np.isnan(prix_transit)]=np.amin(np.vstack((prix_driving[np.isnan(prix_transit)], prix_walking[np.isnan(prix_transit)])), axis = 0)
    prix_transport=prix_transport*2*365
    prix_transport=pd.Series(prix_transport)
    mode_choice=np.argmin(tous_prix, axis=0)
    mode_choice[np.isnan(prix_transit) & (prix_driving < prix_walking)] = 0
    mode_choice[np.isnan(prix_transit) & (prix_driving > prix_walking)] = 2
    return prix_transport, mode_choice, prix_driving, prix_transit, prix_walking

def transport_modeling_bis(driving, transit, income, fuel_price, fuel_consumption, FIXED_COST_CAR, monetary_cost_pt, distance_cbd, WALKING_SPEED, policy, index, city, grille, centre, orig_dist, target_dist, transit_dist, BRT_SPEED, alpha_cong, beta_cong, Q):
    #duration_with_cong = 3600 * ((driving.Distance / 1000) / np.maximum((alpha_cong - (beta_cong * Q)), 10))
    duration_with_cong = 3600 * ((driving.Distance / 1000) / (alpha_cong - (beta_cong * Q)))
    prix_driving = duration_with_cong * income / (3600 * 24) / 365 + driving.Distance * fuel_price * fuel_consumption / 100000 + FIXED_COST_CAR
    if ((index > 4) & (policy == 'transit_speed')):
        prix_transit = ((transit.Duration  / 1.2) * income / (3600 * 24) / 365) + monetary_cost_pt
    elif ((policy == 'BRT') & (index > 4)) | ((policy == 'synergy') & (index > 4)):#| ((policy == 'all') & (index > 4)):
        prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
        prix_BRT = (((((orig_dist + target_dist) / (WALKING_SPEED * 1000)) + ((transit_dist / 1000) / BRT_SPEED)) * (income / (24 * 365))) + monetary_cost_pt).squeeze()
        prix_transit = np.nanmin(np.vstack((prix_transit, prix_BRT)), axis = 0)
    elif ((policy == 'basic_infra') & (index > 4))| ((policy == 'all') & (index > 4)):
        if city == 'Prague' or city == 'Tianjin' or city == 'Paris':
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][0]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][1]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][0]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][1]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
        elif city == 'Buenos_Aires' or city == 'Yerevan':
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][3]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][4]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][3]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][4]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
        else:
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][1]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][2]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][1]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][2]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)

    else:
        prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
    prix_walking = ((distance_cbd) / WALKING_SPEED) * (income / (24 * 365))
    prix_walking[distance_cbd > 8] = math.inf
    tous_prix=np.vstack((prix_driving,prix_transit, prix_walking))
    prix_transport=np.amin(tous_prix, axis=0)
    prix_transport[np.isnan(prix_transit)]=np.amin(np.vstack((prix_driving[np.isnan(prix_transit)], prix_walking[np.isnan(prix_transit)])), axis = 0)
    prix_transport=prix_transport*2*365
    prix_transport=pd.Series(prix_transport)
    mode_choice=np.argmin(tous_prix, axis=0)
    mode_choice[np.isnan(prix_transit) & (prix_driving < prix_walking)] = 0
    mode_choice[np.isnan(prix_transit) & (prix_driving > prix_walking)] = 2
    return prix_transport, mode_choice, prix_driving, prix_transit, prix_walking


def transport_modeling_all_welfare_increasing(driving, transit, income, fuel_price, fuel_consumption, FIXED_COST_CAR, monetary_cost_pt, distance_cbd, WALKING_SPEED, policy_brt, index, city, grille, centre, orig_dist, target_dist, transit_dist, BRT_SPEED):
    prix_driving = driving.Duration * income / (3600 * 24) / 365 + driving.Distance * fuel_price * fuel_consumption / 100000 + FIXED_COST_CAR
    if ((policy_brt == True) & (index > 4)):
        #prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
        #prix_BRT = (((((orig_dist + target_dist) / (WALKING_SPEED * 1000)) + ((transit_dist / 1000) / BRT_SPEED)) * (income / (24 * 365))) + monetary_cost_pt).squeeze()
        #prix_transit = np.nanmin(np.vstack((prix_transit, prix_BRT)), axis = 0)
        if city == 'Prague' or city == 'Tianjin' or city == 'Paris':
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][0]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][1]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][0]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][1]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
        elif city == 'Buenos_Aires' or city == 'Yerevan':
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][3]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][4]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][3]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][4]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
        else:
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][1]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][2]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][1]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][2]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
    else:
        prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
    prix_walking = ((distance_cbd) / WALKING_SPEED) * (income / (24 * 365))
    prix_walking[distance_cbd > 8] = math.inf
    tous_prix=np.vstack((prix_driving,prix_transit, prix_walking))
    prix_transport=np.amin(tous_prix, axis=0)
    prix_transport[np.isnan(prix_transit)]=np.amin(np.vstack((prix_driving[np.isnan(prix_transit)], prix_walking[np.isnan(prix_transit)])), axis = 0)
    prix_transport=prix_transport*2*365
    prix_transport=pd.Series(prix_transport)
    mode_choice=np.argmin(tous_prix, axis=0)
    mode_choice[np.isnan(prix_transit) & (prix_driving < prix_walking)] = 0
    mode_choice[np.isnan(prix_transit) & (prix_driving > prix_walking)] = 2
    return prix_transport, mode_choice

def transport_modeling_all_welfare_increasing_bis(driving, transit, income, fuel_price, fuel_consumption, FIXED_COST_CAR, monetary_cost_pt, distance_cbd, WALKING_SPEED, policy_brt, index, city, grille, centre, orig_dist, target_dist, transit_dist, BRT_SPEED, alpha_cong, beta_cong, Q):
    #prix_driving = driving.Duration * income / (3600 * 24) / 365 + driving.Distance * fuel_price * fuel_consumption / 100000 + FIXED_COST_CAR
    duration_with_cong = 3600 * ((driving.Distance / 1000) / (alpha_cong - (beta_cong * Q)))
    prix_driving = duration_with_cong * income / (3600 * 24) / 365 + driving.Distance * fuel_price * fuel_consumption / 100000 + FIXED_COST_CAR
    
    if ((policy_brt == True) & (index > 4)):
        #prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
        #prix_BRT = (((((orig_dist + target_dist) / (WALKING_SPEED * 1000)) + ((transit_dist / 1000) / BRT_SPEED)) * (income / (24 * 365))) + monetary_cost_pt).squeeze()
        #prix_transit = np.nanmin(np.vstack((prix_transit, prix_BRT)), axis = 0)
        if city == 'Prague' or city == 'Tianjin' or city == 'Paris':
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][0]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][1]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][0]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][1]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
        elif city == 'Buenos_Aires' or city == 'Yerevan':
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][3]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][4]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][3]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][4]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
        else:
            prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
            price_north_axis = ((((np.abs(grille.XCOORD - centre[0][1]) / 1000) / WALKING_SPEED) + ((np.abs(grille.YCOORD - centre[0][2]) / 1000) / 25)) * (income / (24 * 365))) + monetary_cost_pt
            price_east_axis = ((((np.abs(grille.XCOORD - centre[0][1]) / 1000) / 25) + ((np.abs(grille.YCOORD - centre[0][2]) / 1000) / WALKING_SPEED)) * (income / (24 * 365))) + monetary_cost_pt
            prix_transit = np.nanmin(np.vstack((prix_transit, price_north_axis, price_east_axis)), axis = 0)
    else:
        prix_transit = ((transit.Duration) * income / (3600 * 24) / 365) + monetary_cost_pt
    prix_walking = ((distance_cbd) / WALKING_SPEED) * (income / (24 * 365))
    prix_walking[distance_cbd > 8] = math.inf
    tous_prix=np.vstack((prix_driving,prix_transit, prix_walking))
    prix_transport=np.amin(tous_prix, axis=0)
    prix_transport[np.isnan(prix_transit)]=np.amin(np.vstack((prix_driving[np.isnan(prix_transit)], prix_walking[np.isnan(prix_transit)])), axis = 0)
    prix_transport=prix_transport*2*365
    prix_transport=pd.Series(prix_transport)
    mode_choice=np.argmin(tous_prix, axis=0)
    mode_choice[np.isnan(prix_transit) & (prix_driving < prix_walking)] = 0
    mode_choice[np.isnan(prix_transit) & (prix_driving > prix_walking)] = 2
    return prix_transport, mode_choice, prix_driving, prix_transit, prix_walking


