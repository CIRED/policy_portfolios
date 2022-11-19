# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:53:35 2021

@author: charl
"""

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math
import os
import scipy.stats
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import seaborn as sns

from calibration.calibration import *
from calibration.validation import *
from inputs.data import *
from inputs.land_use import *
from model.model import *
from outputs.outputs import *
from inputs.parameters import *
from inputs.transport import *

path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"

path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/calibration_20211124/"
#path_BAU = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20211124/"
path_BAU = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_resid4_20220329/"

list_city = list_of_cities_and_databases(path_data,'cityDatabase')#cityDatabase_intermediaire

### STEP 1: SAMPLE OF CITIES

sample_of_cities = pd.DataFrame(columns = ['City', 'criterion1', 'criterion2', 'criterion3', 'final_sample'], index = np.unique(list_city.City))
sample_of_cities.City = sample_of_cities.index

#Criterion 1: selected cells

selected_cells = np.load(path_calibration + "d_selected_cells.npy", allow_pickle = True)
selected_cells = np.array(selected_cells, ndmin = 1)[0]

for city in list(np.delete(sample_of_cities.index, 153)):
    if (selected_cells[city] > 1):
        sample_of_cities.loc[city, "criterion1"] = 1
    elif (selected_cells[city] == 1):
        sample_of_cities.loc[city, "criterion1"] = 0
        
print("Number of cities excluded because of criterion 1:", sum(sample_of_cities.criterion1 == 0))

#Criterion 2: housing budget exceeds income

def weighted_percentile(data, percents, weights=None):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()*100
    y=np.interp(percents, p, d)
    return y

#median_share_housing = pd.DataFrame(columns = ['City', 'median_share_housing', 'median_share_housing_corrected'], index = list(sample_of_cities.index))
#informal_housing = import_informal_housing(list_city, path_folder)

#gdp_capita_ppp = pd.read_excel(path_folder + "gdp_capita_ppp.xlsx")
#gdp_capita_ppp["income"] = gdp_capita_ppp.oecd
#gdp_capita_ppp.income[np.isnan(gdp_capita_ppp.income)] = gdp_capita_ppp.brookings
#gdp_capita_ppp["source"] = ""
#gdp_capita_ppp["source"][np.isnan(gdp_capita_ppp.income)] = "WB"
#gdp_capita_ppp.income[np.isnan(gdp_capita_ppp.income)] = gdp_capita_ppp.world_bank

for city in list(sample_of_cities.index):
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    density = density.loc[:,density.columns.str.startswith("density")],
    density = np.array(density).squeeze()
    rent = (rents_and_size.avgRent / conversion_rate) * 12
    size = rents_and_size.medSize
    if (city == 'Mashhad') |(city == 'Isfahan') | (city == 'Tehran'):
        rent = 100 * rent
    if city == "Abidjan":
        rent[size > 100] = np.nan
        size[size > 100] = np.nan
    if (city == "Casablanca") | (city == "Yerevan"):
        rent[size > 100] = np.nan
        size[size > 100] = np.nan
    if city == "Toluca":
        rent[size > 250] = np.nan
        size[size > 250] = np.nan
    if (city == "Manaus")| (city == "Rabat")| (city == "Sao_Paulo"):
        rent[rent > 200] = np.nan
        size[rent > 200] = np.nan
    if (city == "Salvador"):
        rent[rent > 250] = np.nan
        size[rent > 250] = np.nan
    if (city == "Karachi") | (city == "Lahore"):
        rent[size > 200] = np.nan
        size[size > 200] = np.nan
    if city == "Addis_Ababa":
        rent[rent > 400] = rent / 100
        size = size / 10
    size[size > 1000] = np.nan
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()   
    #income = gdp_capita_ppp.income[gdp_capita_ppp.city == city].squeeze()
    #if gdp_capita_ppp.source[gdp_capita_ppp.city == city].squeeze() == "WB":
    #    income = income * 1.33
    #income = income_all[income_all.city == city]['2015'].squeeze()
    #informal_housing_city = informal_housing.informal_housing[informal_housing.City == city]
    share_housing = rent * size / income
    #share_housing_corrected = (rent * (1 - (informal_housing_city.squeeze() / 100))) * size / income
    #median_share_housing["City"][median_share_housing.index == city] = city
    #median_share_housing["median_share_housing"][median_share_housing.index == city] = weighted_percentile(np.array(share_housing)[~np.isnan(density) & ~np.isnan(share_housing)], 50, weights=density[~np.isnan(density) & ~np.isnan(share_housing)])
    #median_share_housing["median_share_housing_corrected"][median_share_housing.index == city] = weighted_percentile(np.array(share_housing_corrected)[~np.isnan(density) & ~np.isnan(share_housing_corrected)], 50, weights=density[~np.isnan(density) & ~np.isnan(share_housing_corrected)])
    if weighted_percentile(np.array(share_housing)[~np.isnan(density) & ~np.isnan(share_housing)], 80, weights=density[~np.isnan(density) & ~np.isnan(share_housing)]) > 1:
        sample_of_cities.loc[city, ['criterion2']] = 0
    else:
        sample_of_cities.loc[city, ['criterion2']] = 1

#median_share_housing.median_share_housing.astype(float).describe()
#median_share_housing.median_share_housing_corrected.astype(float).describe()
#np.nansum(median_share_housing.median_share_housing > 1)
#np.nansum(median_share_housing.median_share_housing_corrected > 1)

#median_share_housing.to_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/median_share_housing.xlsx")
#median_share_housing_v2 = median_share_housing.sort_values(['median_share_housing'])
#plt.figure(figsize = (10, 80))
#plt.rcParams.update({'font.size': 20})
#plt.ylim(0, 192)
#plt.scatter(median_share_housing_v2.median_share_housing, median_share_housing_v2.City)
#plt.axvline(1)
#plt.xscale('log')
#plt.xlabel("Median housing budget / income ratio. Vertical line: ratio = 1")

#median_share_housing_v3 = median_share_housing.sort_values(['median_share_housing_corrected'])
#plt.figure(figsize = (10, 80))
#plt.rcParams.update({'font.size': 20})
#plt.ylim(0, 192)
#plt.scatter(median_share_housing_v3.median_share_housing_corrected, median_share_housing_v3.City)
#plt.axvline(1)
#plt.xscale('log')
#plt.xlabel("Median housing budget / income ratio. Vertical line: ratio = 1")

#median_share_housing = median_share_housing.merge(informal_housing.loc[:, ["City", "informal_housing"]], on = "City")

#scipy.stats.pearsonr(median_share_housing.median_share_housing, median_share_housing.informal_housing)
#scipy.stats.pearsonr(median_share_housing.median_share_housing > 0.5, median_share_housing.informal_housing)
#scipy.stats.pearsonr(median_share_housing.median_share_housing > 0.8, median_share_housing.informal_housing)
#scipy.stats.pearsonr(median_share_housing.median_share_housing > 1, median_share_housing.informal_housing)
#modeleReg=LinearRegression()
#modeleReg.fit(np.array(median_share_housing["median_share_housing"] > 0.8).reshape(-1, 1), median_share_housing.informal_housing)
#modeleReg.score(np.array(median_share_housing["median_share_housing"] > 0.8).reshape(-1, 1), median_share_housing.informal_housing)   

#income_all = pd.read_excel(path_folder + "income.xlsx")
#income_all = income_all.merge(list_city, left_on = "city", right_on = "City")

#Incomes
#data_gdp = pd.read_csv(path_folder + "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2163510.csv", header =2) 
#data_gdp["Country Name"][data_gdp["Country Name"] == "Cote d\'Ivoire"] = "Ivory_Coast"
#data_gdp["Country Name"][data_gdp["Country Name"] == "United States"] = "USA"
#data_gdp["Country Name"][data_gdp["Country Name"] == "New Zealand"] = "New_Zealand"
#data_gdp["Country Name"][data_gdp["Country Name"] == "United Kingdom"] = "UK"
#data_gdp["Country Name"][data_gdp["Country Name"] == "South Africa"] = "South_Africa"
##data_gdp["Country Name"][data_gdp["Country Name"] == "Russian Federation"] = "Russia"
#data_gdp["Country Name"][data_gdp["Country Name"] == "Hong Kong SAR, China"] = "Hong_Kong"
#data_gdp["Country Name"][data_gdp["Country Name"] == "Iran, Islamic Rep."] = "Iran"
#data_gdp["Country Name"][data_gdp["Country Name"] == "Czech Republic"] = "Czech_Republic"
#data_gdp = data_gdp["2018"]

#income_all = income_all.merge(data_gdp, left_on = "Country", right_on = "Country Name")
#income_all["ratio_2019"] = income_all.income / income_all["2019"]
#income_all["ratio_2018"] = income_all.income / income_all["2018"]
#income_all["ratio_2015"] = income_all.income / income_all["2015"]
##income_all["ratio_2018"] = income_all.income / income_all["2018"]
#income_all.ratio.describe()

print("Number of cities excluded because of criterion 2:", sum(sample_of_cities.criterion2 == 0))

#### CODE TEMPORAIRE

#sample_of_cities.final_sample[(sample_of_cities.criterion1 == 0) | (sample_of_cities.criterion2 == 0)] = 0
#sample_of_cities.final_sample[(sample_of_cities.criterion1 == 1) & (sample_of_cities.criterion2 == 1)] = 1

#print("Number of cities kept in the end:", sum(sample_of_cities.final_sample == 1))

### Criterion 4

sample_of_cities["criterion4"] = 1

#for city in sample_of_cities.index:
#    if city in ['Barcelona', 'Bern', 'Bilbao', 'Lisbon', 'New_York', 'Prague', 'Shanghai', 'Singapore', 'Stockholm', 'Zurich']:
#        sample_of_cities["criterion4"][sample_of_cities.City == city] = 0

#for city in sample_of_cities.index:
#    if city in ['Amsterdam', 'Basel', 'Beijing', 'Edinburgh', 'Munich', 'Nuremberg', 'The_Hague', 'Valencia', 'Wellington']:
#        sample_of_cities["criterion4"][sample_of_cities.City == city] = 0

#Bern, Cordoba, Cracow, Glasgow, Izmir, Johannesburg, Leeds, Monterrey, New_York, San_Fransisco, Singapore, Sofia, Stockholm, Zurich
#Basel, Bern, Bilbao, Chicago, Cordoba, Cracow, Curitiba, Glasgow, Izmir, Johannesburg, Liverpool, Malmo, Monterrey, Munich, New_York, Nottingham, Nuremberg, San_Fransisco, Seattle, Singapore, Sofia, Stockholm, Valencia, Zurich


for city in sample_of_cities.index:
    if city in ['Basel', 'Bern', 'Bilbao', 'Chicago', 'Cordoba', 'Cracow', 'Curitiba', 'Glasgow', 'Izmir', 'Johannesburg', 'Leeds', 'Liverpool', 'Malmo', 'Monterrey', 'Munich', 'New_York', 'Nottingham', 'Nuremberg', 'Salvador', 'San_Fransisco', 'Seattle', 'Singapore', 'Sofia', 'Stockholm', 'Valencia', 'Zurich']:
        sample_of_cities["criterion4"][sample_of_cities.City == city] = 0

print("Number of cities excluded because of criterion 4:", sum(sample_of_cities.criterion4 == 0))


#Criterion 3: bad fit on density and rents

d_corr_density_scells2 = np.load(path_calibration + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_calibration + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]

for city in list(sample_of_cities.index[sample_of_cities.criterion1 == 1 ]):
    if (d_corr_density_scells2[city][0] < 0) | (d_corr_rent_scells2[city][0] < 0):
        sample_of_cities.loc[city, "criterion3"] = 0
    else:
        sample_of_cities.loc[city, "criterion3"] = 1
        
print("Number of cities excluded because of criterion 3:", sum((sample_of_cities.criterion3 == 0) & (sample_of_cities.criterion2 == 1)& (sample_of_cities.criterion4 == 1)))

#Total

#sample_of_cities["criterion4"] = 1

#for city in sample_of_cities.index:
#    if city in ['Barcelona', 'Bern', 'Bilbao', 'Lisbon', 'New_York', 'Prague', 'Shanghai', 'Singapore', 'Stockholm', 'Zurich']:
#        sample_of_cities["criterion4"][sample_of_cities.City == city] = 0

#for city in sample_of_cities.index:
#    if city in ['Amsterdam', 'Basel', 'Beijing', 'Edinburgh', 'Munich', 'Nuremberg', 'The_Hague', 'Valencia', 'Wellington']:
#        sample_of_cities["criterion4"][sample_of_cities.City == city] = 0

#Bern, Cordoba, Cracow, Glasgow, Izmir, Johannesburg, Leeds, Monterrey, New_York, San_Fransisco, Singapore, Sofia, Stockholm, Zurich
#Basel, Bern, Bilbao, Chicago, Cordoba, Cracow, Curitiba, Glasgow, Izmir, Johannesburg, Liverpool, Malmo, Monterrey, Munich, New_York, Nottingham, Nuremberg, San_Fransisco, Seattle, Singapore, Sofia, Stockholm, Valencia, Zurich


#for city in sample_of_cities.index:
 #   if city in ['Basel', 'Bern', 'Bilbao', 'Chicago', 'Cordoba', 'Cracow', 'Curitiba', 'Glasgow', 'Izmir', 'Johannesburg', 'Leeds', 'Liverpool', 'Malmo', 'Monterrey', 'Munich', 'New_York', 'Nottingham', 'Nuremberg', 'San_Fransisco', 'Seattle', 'Singapore', 'Sofia', 'Stockholm', 'Valencia', 'Zurich']:
 #       sample_of_cities["criterion4"][sample_of_cities.City == city] = 0

#print("Number of cities excluded because of criterion 4:", sum(sample_of_cities.criterion4 == 0))


sample_of_cities.final_sample[(sample_of_cities.criterion1 == 0) | (sample_of_cities.criterion2 == 0) | (sample_of_cities.criterion3 == 0) | (sample_of_cities.criterion4 == 0)] = 0
sample_of_cities.final_sample[(sample_of_cities.criterion1 == 1) & (sample_of_cities.criterion2 == 1) & (sample_of_cities.criterion3 == 1) & (sample_of_cities.criterion4 == 1)] = 1

#sample_of_cities.to_excel("C:/Users/charl/OneDrive/Bureau/sample_of_cities_20220121.xlsx")

print("Number of cities kept in the end:", sum(sample_of_cities.final_sample == 1))

whole_sample = pd.read_csv("C:/Users/charl/OneDrive/Bureau/City_dataStudy/CityDatabases/City_List.csv")
whole_sample = whole_sample.merge(sample_of_cities.loc[:, ["City", "final_sample"]], on ="City", how = "outer")
whole_sample["final_sample"] = whole_sample["final_sample"].astype(float)
whole_sample.loc[np.isnan(whole_sample.final_sample), "final_sample"] = 0

gini = import_gini_index(path_folder)

whole_sample = whole_sample.merge(gini, left_on = "Country", right_on = "Country", how = "left")

np.nanmean(whole_sample.gini[whole_sample.final_sample == 1]) #36.01
np.nanmean(whole_sample.gini[whole_sample.final_sample == 0]) #38.92

np.nanmedian(whole_sample.gini[whole_sample.final_sample == 1]) #34.7
np.nanmedian(whole_sample.gini[whole_sample.final_sample == 0]) #38.5

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

whole_sample = whole_sample.merge(informal_housing, left_on = "Country", right_on = "Country", how = "left")
whole_sample.informal_housing[np.isnan(whole_sample.informal_housing)] = 0


np.nanmean(whole_sample.informal_housing[whole_sample.final_sample == 1]) #4.72
np.nanmean(whole_sample.informal_housing[whole_sample.final_sample == 0]) #20.51

np.nanmedian(whole_sample.informal_housing[whole_sample.final_sample == 1]) #0.0
np.nanmedian(whole_sample.informal_housing[whole_sample.final_sample == 0]) #16

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
data_gdp = data_gdp.loc[:,["2018", "Country Name"]]
data_gdp.columns = ["GDP", "Country"]

whole_sample = whole_sample.merge(data_gdp, left_on = "Country", right_on = "Country", how = "left")


np.nanmean(whole_sample.GDP[whole_sample.final_sample == 1]) #33528.91
np.nanmean(whole_sample.GDP[whole_sample.final_sample == 0]) #18573.87

np.nanmedian(whole_sample.GDP[whole_sample.final_sample == 1]) #34615.76
np.nanmedian(whole_sample.GDP[whole_sample.final_sample == 0]) #7956.63


#city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
#city_continent = city_continent.iloc[:, [0, 2]]
#city_continent = city_continent.drop_duplicates(subset = "City")
#city_continent = city_continent.sort_values('City')
#sample_of_cities = sample_of_cities.merge(city_continent, on = "City", how = 'left')

#sample_of_cities["final_sample"] = pd.to_numeric(sample_of_cities["final_sample"])
#sample_of_cities.loc[:,["final_sample", "Continent"]].groupby("Continent").sum()

### STEP 2: VALIDATION

#Calibrated parameters

calibrated_parameters = pd.DataFrame(columns = ['City', 'beta', 'b', 'kappa', 'Ro', 'sample_cities'], index = sample_of_cities.City[sample_of_cities.final_sample == 1])
calibrated_parameters.City = calibrated_parameters.index

beta = np.load(path_calibration + "beta.npy", allow_pickle = True)
beta = np.array(beta, ndmin = 1)[0]
b = np.load(path_calibration + "b.npy", allow_pickle = True)
b = np.array(b, ndmin = 1)[0]
kappa = np.load(path_calibration + "kappa.npy", allow_pickle = True)
kappa = np.array(kappa, ndmin = 1)[0]
Ro = np.load(path_calibration + "Ro.npy", allow_pickle = True)
Ro = np.array(Ro, ndmin = 1)[0]

for city in calibrated_parameters.index:
    calibrated_parameters.loc[calibrated_parameters.City == city, 'beta'] = beta[city]
    calibrated_parameters.loc[calibrated_parameters.City == city, 'b'] = b[city]
    calibrated_parameters.loc[calibrated_parameters.City == city, 'kappa'] = kappa[city]
    calibrated_parameters.loc[calibrated_parameters.City == city, 'Ro'] = Ro[city]

for param in ['beta', 'b', 'kappa', 'Ro']:
    plt.figure(figsize = (15, 10))
    plt.rcParams.update({'font.size': 60})
    plt.hist(calibrated_parameters[param])
    plt.ylabel("Number of cities")

calibrated_parameters.City[calibrated_parameters.beta > 0.9] #21
calibrated_parameters.City[calibrated_parameters.Ro > 600] #HONG Kong, Riga
#Validation modal shares

modal_shares_data = pd.DataFrame(columns = ['City', 'Simul_car', 'Simul_walking', 'Simul_transit'], index = list(sample_of_cities.City[(sample_of_cities.final_sample == 1)]))# | (sample_of_cities.criterion4 == 0)]))
modal_shares_data.City = modal_shares_data.index

for city in modal_shares_data.index:
    density = np.load(path_BAU + city + "_density.npy")[0]
    modal_share = np.load(path_BAU + city + "_modal_shares.npy")[0]
    modal_shares_data.Simul_car[modal_shares_data.City == city] = np.nansum(density * (modal_share == 0)) / np.nansum(density)
    modal_shares_data.Simul_transit[modal_shares_data.City == city] = np.nansum(density * (modal_share == 1)) / np.nansum(density)
    modal_shares_data.Simul_walking[modal_shares_data.City == city] = np.nansum(density * (modal_share == 2)) / np.nansum(density)

wiki = import_modal_shares_wiki(path_folder).iloc[:, 0:5]
wiki.columns = ['City', 'Wiki_walking', 'Wiki_cycling', 'Wiki_transit', 'Wiki_car']

epomm = pd.read_excel(path_folder + 'modal_shares_data.xlsx')
epomm = epomm.loc[:, ['city', 'walk', 'bike', 'public_transport', 'car']]
epomm.columns = ['City', 'Epomm_walking', 'Epomm_cycling', 'Epomm_transit', 'Epomm_car']

c40 = pd.read_excel(path_folder + 'modal_shares_c40.xlsx', header = 1)
c40.columns = ['City', 'c40_car', 'c40_transit', 'c40_walking']

studylib = pd.read_excel(path_folder + 'modal_shares_c40.xlsx', header = 1, sheet_name = 'studylib').iloc[:,0:4]
studylib.columns = ['City', 'studylib_car', 'studylib_transit', 'studylib_walking']

deloitte = pd.read_excel(path_folder + 'modal_shares_c40.xlsx', header = 1, sheet_name = 'Deloitte')
deloitte.columns = ['City', 'deloitte_car', 'deloitte_transit', 'deloitte_walking']

modal_shares_data = modal_shares_data.merge(wiki, on = 'City', how = 'left')
modal_shares_data = modal_shares_data.merge(epomm, on = 'City', how = 'left')
modal_shares_data = modal_shares_data.merge(c40, on = 'City', how = 'left')
modal_shares_data = modal_shares_data.merge(studylib, on = 'City', how = 'left')
modal_shares_data = modal_shares_data.merge(deloitte, on = 'City', how = 'left')

#total: Deloitte, puis C40, puis EPOMM, puis wiki, puis studylib
modal_shares_data["Total_car"] = modal_shares_data["deloitte_car"]
modal_shares_data["Total_car"][np.isnan(modal_shares_data["Total_car"])] = modal_shares_data["c40_car"]
modal_shares_data["Total_car"][np.isnan(modal_shares_data["Total_car"])] = modal_shares_data["Epomm_car"]
#modal_shares_data["Total_car"][np.isnan(modal_shares_data["Total_car"])] = modal_shares_data["Wiki_car"].astype(float) * 100
#modal_shares_data["Total_car"][np.isnan(modal_shares_data["Total_car"])] = modal_shares_data["studylib_car"]

modal_shares_data["Total_transit"] = modal_shares_data["deloitte_transit"]
modal_shares_data["Total_transit"][np.isnan(modal_shares_data["Total_transit"])] = modal_shares_data["c40_transit"]
modal_shares_data["Total_transit"][np.isnan(modal_shares_data["Total_transit"])] = modal_shares_data["Epomm_transit"]
#modal_shares_data["Total_transit"][np.isnan(modal_shares_data["Total_transit"])] = modal_shares_data["Wiki_transit"].astype(float)* 100
#modal_shares_data["Total_transit"][np.isnan(modal_shares_data["Total_transit"])] = modal_shares_data["studylib_transit"]

modal_shares_data["Total_walking"] = modal_shares_data["deloitte_walking"]
modal_shares_data["Total_walking"][np.isnan(modal_shares_data["Total_walking"])] = modal_shares_data["c40_walking"]
modal_shares_data["Total_walking"][np.isnan(modal_shares_data["Total_walking"])] = modal_shares_data["Epomm_walking"] + modal_shares_data["Epomm_cycling"]
#modal_shares_data["Total_walking"][np.isnan(modal_shares_data["Total_walking"])] = modal_shares_data["Wiki_walking"].astype(float) + modal_shares_data["Wiki_cycling"].astype(float)* 100
#modal_shares_data["Total_walking"][np.isnan(modal_shares_data["Total_walking"])] = modal_shares_data["studylib_walking"]

#### Private cars

#epomm
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Epomm_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (EPOMM data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Epomm_car"])[i], np.array(modal_shares_data.Simul_car)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array(modal_shares_data["Epomm_car"].astype(float)[~np.isnan(modal_shares_data["Epomm_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["Epomm_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Epomm_car"]).astype(float))))

#wiki
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Wiki_car"] * 100, modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (Wikipedia data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Wiki_car"] * 100)[i], np.array(modal_shares_data.Simul_car)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array(modal_shares_data["Wiki_car"].astype(float)[~np.isnan(modal_shares_data["Wiki_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["Wiki_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Wiki_car"]).astype(float))))

#c40
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["c40_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (C40 data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["c40_car"].astype(float)[~np.isnan(modal_shares_data["c40_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["c40_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["c40_car"]).astype(float))))

#deloitte
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["deloitte_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (Deloitte data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["deloitte_car"].astype(float)[~np.isnan(modal_shares_data["deloitte_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["deloitte_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["deloitte_car"]).astype(float))))

#studylib
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["studylib_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (Studylib data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["studylib_car"].astype(float)[~np.isnan(modal_shares_data["studylib_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["studylib_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["studylib_car"]).astype(float))))

#Total
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Total_car"], modal_shares_data.Simul_car * 100, s = 200)
plt.xlabel("Modal share of private car (Studylib data - %)", size = 20)
plt.ylabel("Modal share of private car (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["Total_car"].astype(float)[~np.isnan(modal_shares_data["Total_car"].astype(float))]), np.array(100 * modal_shares_data.Simul_car)[~np.isnan(modal_shares_data["Total_car"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Total_car"]).astype(float))))

#### Transit

#epomm
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Epomm_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (EPOMM data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Epomm_transit"])[i], np.array(modal_shares_data.Simul_transit)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array(modal_shares_data["Epomm_transit"].astype(float)[~np.isnan(modal_shares_data["Epomm_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["Epomm_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Epomm_transit"]).astype(float))))

#wiki
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Wiki_transit"] * 100, modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (Wikipedia data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Wiki_transit"])[i] * 100, np.array(modal_shares_data.Simul_transit)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array(modal_shares_data["Wiki_transit"].astype(float)[~np.isnan(modal_shares_data["Wiki_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["Wiki_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Wiki_transit"]).astype(float))))

#c40
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["c40_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (c40 data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["c40_transit"].astype(float)[~np.isnan(modal_shares_data["c40_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["c40_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["c40_transit"]).astype(float))))

#deloitte
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["deloitte_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (Deloitte data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["deloitte_transit"].astype(float)[~np.isnan(modal_shares_data["deloitte_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["deloitte_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["deloitte_transit"]).astype(float))))

#studylib
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["studylib_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (Studylib data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["studylib_transit"].astype(float)[~np.isnan(modal_shares_data["studylib_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["studylib_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["studylib_transit"]).astype(float))))

#Total
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Total_transit"], modal_shares_data.Simul_transit * 100, s = 200)
plt.xlabel("Modal share of public transport (Studylib data - %)", size = 20)
plt.ylabel("Modal share of public transport (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["Total_transit"].astype(float)[~np.isnan(modal_shares_data["Total_transit"].astype(float))]), np.array(100 * modal_shares_data.Simul_transit)[~np.isnan(modal_shares_data["Total_transit"].astype(float))]))
print(sum(~np.isnan((modal_shares_data["Total_transit"]).astype(float))))


#### Active modes

#epomm
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (EPOMM data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"])[i], np.array(modal_shares_data.Simul_walking)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array((modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"]).astype(float)[~np.isnan((modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["Epomm_cycling"] + modal_shares_data["Epomm_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["Epomm_walking"]).astype(float))))

#wiki
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter((modal_shares_data["Wiki_cycling"] + modal_shares_data["Wiki_walking"]) * 100, modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (Wikipedia data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
for i in np.arange(len(modal_shares_data.index)):
    plt.annotate(np.array(modal_shares_data.City)[i], (np.array(modal_shares_data["Wiki_cycling"] + modal_shares_data["Wiki_walking"])[i] * 100, np.array(modal_shares_data.Simul_walking)[i] * 100), size = 20)
print(sc.stats.pearsonr(np.array((modal_shares_data["Wiki_cycling"] + modal_shares_data["Wiki_walking"]).astype(float)[~np.isnan((modal_shares_data["Wiki_cycling"] + modal_shares_data["Wiki_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["Wiki_cycling"] + modal_shares_data["Wiki_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["Wiki_cycling"]).astype(float))))

#c40
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["c40_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (Wikipedia data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["c40_walking"].astype(float)[~np.isnan((modal_shares_data["c40_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["c40_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["c40_walking"]).astype(float))))

#deloitte
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["deloitte_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (Wikipedia data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["deloitte_walking"].astype(float)[~np.isnan((modal_shares_data["deloitte_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["deloitte_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["deloitte_walking"]).astype(float))))

#c40
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["studylib_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (Wikipedia data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["studylib_walking"].astype(float)[~np.isnan((modal_shares_data["studylib_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["studylib_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["studylib_walking"]).astype(float))))

#Total
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(modal_shares_data["Total_walking"], modal_shares_data.Simul_walking * 100, s = 200)
plt.xlabel("Modal share of active modes (Wikipedia data - %)", size = 20)
plt.ylabel("Modal share of active modes (Simulations - %)", size = 20)
plt.xlim(0, 100)
plt.ylim(0, 100)
print(sc.stats.pearsonr(np.array(modal_shares_data["Total_walking"].astype(float)[~np.isnan((modal_shares_data["Total_walking"]).astype(float))]), np.array(100 * modal_shares_data.Simul_walking)[~np.isnan((modal_shares_data["Total_walking"]).astype(float))]))
print(sum(~np.isnan((modal_shares_data["Total_walking"]).astype(float))))


modal_shares_data = modal_shares_data.merge(sample_of_cities.loc[:, ['City']]) #, 'criterion4']])






print(modal_shares_data.City[(modal_shares_data["Total_transit"].astype(float) > 30) & (100 * modal_shares_data.Simul_transit < 15)])
#Bern, Cordoba, Cracow, Glasgow, Izmir, Johannesburg, Leeds, Monterrey, New_York, San_Fransisco, Singapore, Sofia, Stockholm, Zurich

print(modal_shares_data.City[(modal_shares_data["Total_transit"].astype(float) > 20) & (100 * modal_shares_data.Simul_transit < 10)])
#Basel, Bern, Bilbao, Chicago, Cordoba, Cracow, Curitiba, Glasgow, Izmir, Johannesburg, Liverpool, Malmo, Monterrey, Munich, New_York, Nottingham, Nuremberg, San_Fransisco, Seattle, Singapore, Sofia, Stockholm, Valencia, Zurich


#### A FAIRE
#### 1. Exclure le nouveau set de villes
#### 2. Refaire les résultats
#### 3. Modal shares: 3 graphes + Corr coeff pour Deloitte, C40, EPOMM en annexxe. Seulement les corr coeff dans le corps du texte.


#Validation emissions

df_emissions = pd.DataFrame(columns = ['City', 'Simul_emissions'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df_emissions.City = df_emissions.index

for city in df_emissions.index:
    df_emissions.Simul_emissions[df_emissions.City == city] = np.load(path_BAU + city + "_emissions_per_capita.npy")[0]

felix_data = pd.read_excel(path_folder + "emissions_databases/datapaper felix/DATA/D_FINAL.xlsx", header = 0)
felix_data_for_comparison = felix_data.loc[:, ["City name", "Scope-1 GHG emissions [tCO2 or tCO2-eq]", "Scope-2 (CDP) [tCO2-eq]", "Population (CDP)"]]

felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Addis Ababa"] = "Addis_Ababa"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Belo Horizonte"] = "Belo_Horizonte"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Bogotá"] = "Bogota"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Brasília"] = "Brasilia"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Buenos Aires"] = "Buenos_Aires"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Cape Town"] = "Cape_Town"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Goiânia"] = "Goiania"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Greater London"] = "London"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Hong Kong"] = "Hong_Kong"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Jinan, Shandong"] = "Jinan"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Los Angeles"] = "Los_Angeles"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Mexico City"] = "Mexico_City"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "New York City"] = "New_York"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Porto Alegre"] = "Porto_Alegre"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Rio de Janeiro"] = "Rio_de_Janeiro"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "San Francisco"] = "San_Fransisco"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "San Diego"] = "San_Diego"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Sao Paulo"] = "Sao_Paulo"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Toluca de Lerdo"] = "Toluca"
felix_data_for_comparison["City name"][felix_data_for_comparison["City name"] == "Zürich"] = "Zurich"

felix_data_for_comparison["felix_scope1"] = felix_data_for_comparison["Scope-1 GHG emissions [tCO2 or tCO2-eq]"] / felix_data_for_comparison["Population (CDP)"]
felix_data_for_comparison["felix_scope2"] = felix_data_for_comparison["Scope-2 (CDP) [tCO2-eq]"] / felix_data_for_comparison["Population (CDP)"]
felix_data_for_comparison = felix_data_for_comparison.loc[:, ['City name', "felix_scope1", "felix_scope2"]]

plt.scatter(felix_data_for_comparison["felix_scope1"], felix_data_for_comparison["felix_scope2"])
print(sc.stats.pearsonr(np.array(felix_data_for_comparison["felix_scope1"].astype(float)[~np.isnan((felix_data_for_comparison["felix_scope2"]).astype(float)) & ~np.isnan((felix_data_for_comparison["felix_scope1"]).astype(float))]), np.array(felix_data_for_comparison.felix_scope2)[~np.isnan((felix_data_for_comparison["felix_scope2"]).astype(float)) & ~np.isnan((felix_data_for_comparison["felix_scope1"]).astype(float))]))

erl_data = pd.read_excel(path_folder + "emissions_databases/ERL/ERL_13_064041_SD_Moran Spatial Footprints Data Appendix.xlsx", header = 0, sheet_name = 'S.2.3a - Top 500 Cities')
erl_data = erl_data.iloc[:, [0, 2]]
erl_data.columns = ['city', 'erl_emissions']

erl_data.city[erl_data.city == 'Ahmadabad'] = 'Ahmedabad'
erl_data.city[erl_data.city == 'Antwerpen'] = 'Antwerp'
erl_data.city[erl_data.city == 'Belo_Horizonte'] = 'Belo_Horizonte'
erl_data.city[erl_data.city == 'Buenos Aires'] = 'Buenos_Aires'
erl_data.city[erl_data.city == 'Cape Town'] = 'Cape_Town'
erl_data.city[erl_data.city == 'Esfahan'] = 'Isfahan'
erl_data.city[erl_data.city == 'Frankfurt'] = 'Frankfurt_am_Main'
erl_data.city[erl_data.city == 'Hannover'] = 'Hanover'
erl_data.city[erl_data.city == 'Hong Kong'] = 'Hong_Kong'
erl_data.city[erl_data.city == 'Kazan\''] = 'Kazan'
erl_data.city[erl_data.city == 'Los Angeles'] = 'Los_Angeles'
erl_data.city[erl_data.city == 'Mexico City'] = 'Mexico_City'
erl_data.city[erl_data.city == 'New Delhi'] = 'Delhi'
erl_data.city[erl_data.city == 'New York'] = 'New_York'
erl_data.city[erl_data.city == 'Porto Alegre'] = 'Porto_Alegre'
erl_data.city[erl_data.city == 'Rio de Janeiro'] = 'Rio_de_Janeiro'
erl_data.city[erl_data.city == 'Rostov-on-Don'] = 'Rostov_on_Don'
erl_data.city[erl_data.city == 'Saint Petersburg'] = 'St_Petersburg'
erl_data.city[erl_data.city == 'San Diego'] = 'San_Diego'
erl_data.city[erl_data.city == 'Sao Paulo'] = 'Sao_Paulo'
erl_data.city[erl_data.city == 'Washington D.C.'] = 'Washington_DC'
erl_data.erl_emissions[erl_data.city == 'Valencia'] = 5.7

erl_data2 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/emissions_databases/ERL/ERL_13_064041_SD_Moran Spatial Footprints Data Appendix.xlsx", header = 0, sheet_name = 'S.2.3b - Top 500 Per Capita')
erl_data2 = erl_data2.iloc[:, [0, 5]]
erl_data2.columns = ['city', 'erl_emissions2']

com_label = pd.read_excel(path_folder + "emissions_databases/CoM/GlobalCoM.xlsx", header = 0, sheet_name = 'Table 3')
com_label = com_label.iloc[:, 0:2]
com_label.columns = ['GCoM_ID', 'city']

com_label.city[com_label.city == 'Grand Lyon'] = 'Lyon'
com_label.city[com_label.city == 'Lisboa'] = 'Lisbon'
com_label.city[com_label.city == 'Malmö'] = 'Malmo'
com_label.city[com_label.city == 'NANTES METROPOLE'] = 'Nantes'
com_label.city[com_label.city == 'München'] = 'Munich'
com_label.city[com_label.city == 'Málaga'] = 'Malaga'
com_label.city[com_label.city == 'Genève'] = 'Geneva'
com_label.city[com_label.city == 'Genova'] = 'Genoa'
com_label.city[com_label.city == 'Zürich'] = 'Zurich'
com_label.city[com_label.city == 'BORDEAUX METROPOLE'] = 'Bordeaux'
com_label.city[com_label.city == 'Nice Côte d\'Azur'] = 'Nice'
com_label.city[com_label.city == 'Gent'] = 'Ghent'
com_label.city[com_label.city == 'Izmir Metropolitan Municipality'] = 'Izmir'
com_label.city[com_label.city == 'Bremen'] = 'X'
com_label.columns = ['GCoM_ID', 'city']

com_data = pd.read_excel(path_folder + "emissions_databases/CoM/GlobalCoM.xlsx", header = 0, sheet_name = 'Table 2')
com_data["com_emissions_per_capita"] = com_data.emissions / com_data.population_in_the_inventory_year
com_data = com_data.merge(com_label, on = 'GCoM_ID', how = 'left')
com_data = com_data[com_data.emission_inventory_sector == 'Transportation']
com_data = com_data[com_data.type_of_emission_inventory == 'baseline_emission_inventory']
direct_emissions_data = com_data[com_data.type_of_emissions == 'direct_emissions']
indirect_emissions_data = com_data[com_data.type_of_emissions == 'indirect_emissions']
direct_emissions_data = direct_emissions_data.iloc[:, 12:14]
indirect_emissions_data = indirect_emissions_data.iloc[:, 12:14]
direct_emissions_data.columns = ['com_direct_emissions', 'city']
indirect_emissions_data.columns = ['com_indirect_emissions', 'city']

df_emissions = df_emissions.merge(felix_data_for_comparison, left_on = 'City', right_on = 'City name', how = 'left')
df_emissions = df_emissions.merge(erl_data, left_on = 'City', right_on = 'city', how = 'left')
df_emissions = df_emissions.merge(erl_data2, left_on = 'City', right_on = 'city', how = 'left')
df_emissions = df_emissions.merge(direct_emissions_data, left_on = 'City', right_on = 'city', how = 'left')
df_emissions = df_emissions.merge(indirect_emissions_data, left_on = 'City', right_on = 'city', how = 'left')

df_emissions.erl_emissions[np.isnan(df_emissions.erl_emissions)] = df_emissions.erl_emissions2[np.isnan(df_emissions.erl_emissions)]

df_emissions.loc[df_emissions.felix_scope1>20, 'felix_scope1'] = np.nan

for var in ['felix_scope1', 'felix_scope2', 'erl_emissions', 'erl_emissions2', 'com_direct_emissions','com_indirect_emissions']:
    plt.figure(figsize = (15, 10))
    plt.rcParams.update({'font.size': 20})
    plt.scatter(df_emissions['Simul_emissions'], df_emissions[var], s = 200)
    plt.xlabel("Simulated emissions per capita", size = 20)
    plt.ylabel("Data", size = 20)
    print(sc.stats.pearsonr(np.array(df_emissions[var].astype(float)[~np.isnan(df_emissions[var].astype(float))]), np.array(df_emissions.Simul_emissions)[~np.isnan(df_emissions[var].astype(float))]))
    print(len((np.array(df_emissions[var].astype(float)[~np.isnan(df_emissions[var].astype(float))]))))

#df_emissions = df_emissions.drop(108)

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.scatter(df_emissions['Simul_emissions'], df_emissions['felix_scope1'], s = 200)
#plt.scatter(df_emissions['Simul_emissions'], df_emissions['felix_scope2'], s = 200)
#plt.scatter(df_emissions['felix_scope1'], df_emissions['felix_scope2'], s = 200)

print(sc.stats.pearsonr(np.array(df_emissions["felix_scope1"].astype(float)[~np.isnan((df_emissions["felix_scope2"]).astype(float)) & ~np.isnan((df_emissions["felix_scope1"]).astype(float))]), np.array(df_emissions.felix_scope2)[~np.isnan((df_emissions["felix_scope2"]).astype(float)) & ~np.isnan((df_emissions["felix_scope1"]).astype(float))]))
print(sc.stats.pearsonr(np.array(df_emissions["Simul_emissions"].astype(float)[~np.isnan((df_emissions["felix_scope2"]).astype(float)) & ~np.isnan((df_emissions["felix_scope2"]).astype(float))]), np.array(df_emissions.felix_scope2)[~np.isnan((df_emissions["felix_scope2"]).astype(float)) & ~np.isnan((df_emissions["felix_scope2"]).astype(float))]))
print(sc.stats.pearsonr(np.array(df_emissions["Simul_emissions"].astype(float)[~np.isnan((df_emissions["felix_scope1"]).astype(float)) & ~np.isnan((df_emissions["felix_scope1"]).astype(float))]), np.array(df_emissions.felix_scope1)[~np.isnan((df_emissions["felix_scope1"]).astype(float)) & ~np.isnan((df_emissions["felix_scope1"]).astype(float))]))

sum(~np.isnan((df_emissions["felix_scope2"]).astype(float)))
sum(~np.isnan((df_emissions["felix_scope1"]).astype(float)))

#Validation (coeff corr, R2, RAE, MAE)


r2density_scells2 = np.load(path_calibration + "r2density_scells2.npy", allow_pickle = True)
r2density_scells2 = np.array(r2density_scells2, ndmin = 1)[0]
r2rent_scells2 = np.load(path_calibration + "r2rent_scells2.npy", allow_pickle = True)
r2rent_scells2 = np.array(r2rent_scells2, ndmin = 1)[0]
r2size_scells2 = np.load(path_calibration + "r2size_scells2.npy", allow_pickle = True)
r2size_scells2 = np.array(r2size_scells2, ndmin = 1)[0]

d_corr_density_scells2 = np.load(path_calibration + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_calibration + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]
d_corr_size_scells2 = np.load(path_calibration + "d_corr_size_scells2.npy", allow_pickle = True)
d_corr_size_scells2 = np.array(d_corr_size_scells2, ndmin = 1)[0]

mae_density_scells2 = np.load(path_calibration + "mae_density_scells2.npy", allow_pickle = True)
mae_density_scells2 = np.array(mae_density_scells2, ndmin = 1)[0]
mae_rent_scells2 = np.load(path_calibration + "mae_rent_scells2.npy", allow_pickle = True)
mae_rent_scells2 = np.array(mae_rent_scells2, ndmin = 1)[0]
mae_size_scells2 = np.load(path_calibration + "mae_size_scells2.npy", allow_pickle = True)
mae_size_scells2 = np.array(mae_size_scells2, ndmin = 1)[0]

rae_density_scells2 = np.load(path_calibration + "rae_density_scells2.npy", allow_pickle = True)
rae_density_scells2 = np.array(rae_density_scells2, ndmin = 1)[0]
rae_rent_scells2 = np.load(path_calibration + "rae_rent_scells2.npy", allow_pickle = True)
rae_rent_scells2 = np.array(rae_rent_scells2, ndmin = 1)[0]
rae_size_scells2 = np.load(path_calibration + "rae_size_scells2.npy", allow_pickle = True)
rae_size_scells2 = np.array(rae_size_scells2, ndmin = 1)[0]

validation = pd.DataFrame()
validation["City"] = np.array(list(r2density_scells2.keys()))
validation["r2density"] = np.array(list(r2density_scells2.values()))
validation["r2rent"] = np.array(list(r2rent_scells2.values()))
validation["r2size"] = np.array(list(r2size_scells2.values()))
validation["corrrdensity"] = np.array(list(d_corr_density_scells2.values()))[:, 0]
validation["corrrent"] = np.array(list(d_corr_rent_scells2.values()))[:, 0]
validation["corrsize"] = np.array(list(d_corr_size_scells2.values()))[:, 0]
validation["mae_density"] = np.array(list(mae_density_scells2.values()))
validation["mae_rent"] = np.array(list(mae_rent_scells2.values()))
validation["mae_size"] = np.array(list(mae_size_scells2.values()))
validation["rae_density"] = np.array(list(rae_density_scells2.values()))
validation["rae_rent"] = np.array(list(rae_rent_scells2.values()))
validation["rae_size"] = np.array(list(rae_size_scells2.values()))
validation = validation.merge(sample_of_cities.loc[:, ['City', 'final_sample']], on = 'City')
              
table_validation = validation[validation.final_sample == 1].describe()

### STEP 3: BAU SCENARIO

BAU_scenario = pd.DataFrame(columns = ['City', 'emissions_2015', 'welfare_without_cobenefits_2015', 'welfare_with_cobenefits_2015', 'welfare_without_cobenefits_2020', 'welfare_with_cobenefits_2020', 'welfare_without_cobenefits_2025', 'welfare_with_cobenefits_2025', 'welfare_without_cobenefits_2030', 'welfare_with_cobenefits_2030', 'emissions_2035', 'welfare_without_cobenefits_2035', 'welfare_with_cobenefits_2035', 'population_2015', 'population_2035', 'avg_utility_2015', 'avg_utility_2035', 'density_2015', 'density_2035'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
BAU_scenario.City = BAU_scenario.index

for city in BAU_scenario.index:
    BAU_scenario.emissions_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_emissions_per_capita.npy")[0]
    BAU_scenario.welfare_without_cobenefits_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[0]
    BAU_scenario.welfare_with_cobenefits_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[0]
    BAU_scenario.emissions_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_emissions_per_capita.npy")[20]
    BAU_scenario.welfare_without_cobenefits_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[20]
    BAU_scenario.welfare_with_cobenefits_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[20]
    BAU_scenario.welfare_without_cobenefits_2030[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[15]
    BAU_scenario.welfare_with_cobenefits_2030[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[15]
    BAU_scenario.welfare_without_cobenefits_2025[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[10]
    BAU_scenario.welfare_with_cobenefits_2025[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[10]
    BAU_scenario.welfare_without_cobenefits_2020[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare.npy")[5]
    BAU_scenario.welfare_with_cobenefits_2020[BAU_scenario.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[5]
    BAU_scenario.population_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_population.npy")[0]
    BAU_scenario.population_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_population.npy")[20]
    BAU_scenario.avg_utility_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_avg_utility.npy")[0]
    BAU_scenario.avg_utility_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_avg_utility.npy")[20]
    #BAU_scenario.density_2015[BAU_scenario.City == city] = np.load(path_BAU + city + "_density.npy")[0]
    #BAU_scenario.density_2035[BAU_scenario.City == city] = np.load(path_BAU + city + "_density.npy")[20]
    
city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
BAU_scenario = BAU_scenario.merge(city_continent, on = "City", how = 'left')
color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}

BAU_scenario["change_cobenefits_2035"] = 100 * (BAU_scenario.welfare_with_cobenefits_2035 - BAU_scenario.welfare_without_cobenefits_2035) / BAU_scenario.welfare_without_cobenefits_2035
BAU_scenario["change_cobenefits_2015"] = 100 * (BAU_scenario.welfare_with_cobenefits_2015 - BAU_scenario.welfare_without_cobenefits_2015) / BAU_scenario.welfare_without_cobenefits_2015

plt.hist(BAU_scenario.change_cobenefits_2015)
plt.hist(BAU_scenario.change_cobenefits_2035)
sum(BAU_scenario.change_cobenefits_2015 < -20)
sum(BAU_scenario.change_cobenefits_2035 < -20)

BAU_scenario.City[BAU_scenario.change_cobenefits_2015 > 20]
BAU_scenario.City[BAU_scenario.change_cobenefits_2035 > 20]

#Emissions and welfare, BAU, year 0

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(BAU_scenario['Continent'].unique())
for i in range(0 , len(colors)):
    data = BAU_scenario.loc[BAU_scenario['Continent'] == colors[i]]
    plt.scatter(data.avg_utility_2015, data.emissions_2015, color=data.Continent.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Avg utility 2015", size = 20)
plt.ylabel("Emissions per capita", size = 20)
plt.legend()

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(BAU_scenario['Continent'].unique())
for i in range(0 , len(colors)):
    data = BAU_scenario.loc[BAU_scenario['Continent'] == colors[i]]
    plt.scatter(data.welfare_without_cobenefits_2015 / data.population_2015, data.emissions_2015, color=data.Continent.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Welfare without cobenefits 2015", size = 20)
plt.ylabel("Emissions per capita", size = 20)
plt.legend()

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(BAU_scenario['Continent'].unique())
for i in range(0 , len(colors)):
    data = BAU_scenario.loc[BAU_scenario['Continent'] == colors[i]]
    plt.scatter(data.welfare_with_cobenefits_2015 / data.population_2015, data.emissions_2015, color=data.Continent.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Welfare with cobenefits 2015", size = 20)
plt.ylabel("Emissions per capita", size = 20)
plt.legend()

#Emissions and welfare, BAU, year 20

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(BAU_scenario['Continent'].unique())
for i in range(0 , len(colors)):
    data = BAU_scenario.loc[BAU_scenario['Continent'] == colors[i]]
    plt.scatter(data.avg_utility_2035, data.emissions_2035, color=data.Continent.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Avg utility 2035", size = 20)
plt.ylabel("Emissions per capita", size = 20)
plt.legend()

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(BAU_scenario['Continent'].unique())
for i in range(0 , len(colors)):
    data = BAU_scenario.loc[BAU_scenario['Continent'] == colors[i]]
    plt.scatter(data.welfare_without_cobenefits_2035 / data.population_2035, data.emissions_2035, color=data.Continent.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Welfare without cobenefits 2035", size = 20)
plt.ylabel("Emissions per capita", size = 20)
plt.legend()

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(BAU_scenario['Continent'].unique())
for i in range(0 , len(colors)):
    data = BAU_scenario.loc[BAU_scenario['Continent'] == colors[i]]
    plt.scatter(data.welfare_with_cobenefits_2035 / data.population_2035, data.emissions_2035, color=data.Continent.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Welfare with cobenefits 2035", size = 20)
plt.ylabel("Emissions per capita", size = 20)
plt.legend()

#Variations emissions (and welfare) BAU

BAU_scenario["var_emissions"] = 100 *(BAU_scenario["emissions_2035"] - BAU_scenario["emissions_2015"]) / BAU_scenario["emissions_2015"]



plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(BAU_scenario.var_emissions[BAU_scenario.var_emissions < 100], color=['#fc8d62'])#("Set2"))
plt.xlabel('Emissions per capita variation (%)', size=14)
plt.ylabel('Number of cities', size=14)
#plt.title('Number of Accidents By Severity', size=18, color='#4f4e4e')
#plt.yticks([], [])
#plt.xticks([], [])
#handles, labels = ax.get_legend_handles_labels()
#♠ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0, 0.35), fontsize = 14, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
plt.savefig('bau1.png')


plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 40})
plt.scatter(100 * (BAU_scenario["emissions_2035"] - BAU_scenario["emissions_2015"]) / BAU_scenario["emissions_2015"], BAU_scenario["Continent"],  s = 200)

plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(BAU_scenario.var_emissions[BAU_scenario.var_emissions < 100], BAU_scenario["Continent"][BAU_scenario.var_emissions < 100], color=['#fc8d62'], size = np.log(np.log(list(np.array(BAU_scenario.population_2015[BAU_scenario.var_emissions < 100])))))#("Set2"))
plt.xlabel('Emissions per capita variation (%)', size=14)
plt.ylabel('', size=14)
#plt.title('Number of Accidents By Severity', size=18, color='#4f4e4e')
#plt.yticks([], [])
#plt.xticks([], [])
plt.legend([],[], frameon=False)
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
plt.savefig('bau2.png')


plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 40})
plt.hist(100 * (BAU_scenario["avg_utility_2035"] - BAU_scenario["avg_utility_2015"]) / BAU_scenario["avg_utility_2015"], bins = 20)
plt.legend()
plt.ylabel("Number of citiess")
plt.xlabel("Variation in utility (%)")

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 40})
plt.scatter(100 * (BAU_scenario["avg_utility_2035"] - BAU_scenario["avg_utility_2015"]) / BAU_scenario["avg_utility_2015"], BAU_scenario["Continent"],  s = 200)

#Welfare with and without cobenefits and 2035

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.hist(100 * (BAU_scenario.welfare_with_cobenefits_2015 - BAU_scenario.welfare_without_cobenefits_2015) / BAU_scenario.welfare_without_cobenefits_2015)

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
plt.hist(100 * (BAU_scenario.welfare_with_cobenefits_2035 - BAU_scenario.welfare_without_cobenefits_2035) / BAU_scenario.welfare_without_cobenefits_2035)

#BAU scenario: density ?

### STEP 4: Results

#Cities where the welfare increases/decreases because of the policies? Evolution in time?

#path_carbon_tax = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbon_tax_20211124_v2/'
path_carbon_tax = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbon_tax_20220322_1percent_resid6/'
path_fuel_efficiency = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuel_efficiency_20220322_1percent_resid6/'
path_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/UGB_resid4_data_urba/'
path_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BRT_20220324_1percent_resid6_baseline_25_0_12_50_5/'
path_synergy = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/synergy_BRT_CT/'

path_policy = path_UGB

df = pd.DataFrame(columns = ['City', 'emissions_2015', 'welfare_without_cobenefits_2015', 'welfare_with_cobenefits_2015', 'welfare_without_cobenefits_2020', 'welfare_with_cobenefits_2020', 'welfare_without_cobenefits_2025', 'welfare_with_cobenefits_2025', 'welfare_without_cobenefits_2030', 'welfare_with_cobenefits_2030', 'emissions_2035', 'welfare_without_cobenefits_2035', 'welfare_with_cobenefits_2035', 'population_2015', 'population_2035', 'avg_utility_2015', 'avg_utility_2035', 'length_network'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df.City = df.index

for city in df.index:
    df.emissions_2015[df.City == city] = np.load(path_policy + city + "_emissions_per_capita.npy")[0]
    df.welfare_without_cobenefits_2015[df.City == city] = np.load(path_policy + city + "_total_welfare.npy")[0]
    df.welfare_with_cobenefits_2015[df.City == city] = np.load(path_policy + city + "_total_welfare_with_cobenefits.npy")[0]
    df.emissions_2035[df.City == city] = np.load(path_policy + city + "_emissions_per_capita.npy")[20]
    df.welfare_without_cobenefits_2035[df.City == city] = np.load(path_policy + city + "_total_welfare.npy")[20]
    df.welfare_with_cobenefits_2035[df.City == city] = np.load(path_policy + city + "_total_welfare_with_cobenefits.npy")[20]
    df.welfare_without_cobenefits_2030[df.City == city] = np.load(path_policy + city + "_total_welfare.npy")[15]
    df.welfare_with_cobenefits_2030[df.City == city] = np.load(path_policy + city + "_total_welfare_with_cobenefits.npy")[15]
    df.welfare_without_cobenefits_2025[df.City == city] = np.load(path_policy + city + "_total_welfare.npy")[10]
    df.welfare_with_cobenefits_2025[df.City == city] = np.load(path_policy + city + "_total_welfare_with_cobenefits.npy")[10]
    df.welfare_without_cobenefits_2020[df.City == city] = np.load(path_policy + city + "_total_welfare.npy")[5]
    df.welfare_with_cobenefits_2020[df.City == city] = np.load(path_policy + city + "_total_welfare_with_cobenefits.npy")[5]
    df.population_2015[df.City == city] = np.load(path_policy + city + "_population.npy")[0]
    df.population_2035[df.City == city] = np.load(path_policy + city + "_population.npy")[20]
    df.avg_utility_2015[df.City == city] = np.load(path_policy + city + "_avg_utility.npy")[0]
    df.avg_utility_2035[df.City == city] = np.load(path_policy + city + "_avg_utility.npy")[20]
    #df.density_2015[df.City == city] = np.load(path_policy + city + "_density.npy")[0]
    #df.density_2035[df.City == city] = np.load(path_policy + city + "_density.npy")[20]
    df.length_network[df.City == city] = np.load('C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/capital_costs_25_0_01_50_5/' + city + "_length_network.npy").item()
    
BAU_scenario = BAU_scenario.add_suffix('_BAU')
df = df.merge(BAU_scenario, left_on = 'City', right_on = 'City_BAU')

(df.length_network / 1000).astype(float).describe()

print(sum(df.avg_utility_2035 > df.avg_utility_2035_BAU))
print(sum(df.avg_utility_2035 < df.avg_utility_2035_BAU))

print(sum(df.welfare_without_cobenefits_2035 > df.welfare_without_cobenefits_2035_BAU))
print(sum(df.welfare_without_cobenefits_2035 < df.welfare_without_cobenefits_2035_BAU))

print(sum(df.welfare_with_cobenefits_2035 > df.welfare_with_cobenefits_2035_BAU))
print(sum(df.welfare_with_cobenefits_2035 < df.welfare_with_cobenefits_2035_BAU))

#Evolution in time?

print(sum(df.welfare_without_cobenefits_2020 > df.welfare_without_cobenefits_2020_BAU))
print(sum(df.welfare_with_cobenefits_2020 > df.welfare_with_cobenefits_2020_BAU))

print(sum(df.welfare_without_cobenefits_2025 > df.welfare_without_cobenefits_2025_BAU))
print(sum(df.welfare_with_cobenefits_2025 > df.welfare_with_cobenefits_2025_BAU))

print(sum(df.welfare_without_cobenefits_2030 > df.welfare_without_cobenefits_2030_BAU))
print(sum(df.welfare_with_cobenefits_2030 > df.welfare_with_cobenefits_2030_BAU))

print(sum(df.welfare_without_cobenefits_2035 > df.welfare_without_cobenefits_2035_BAU))
print(sum(df.welfare_with_cobenefits_2035 > df.welfare_with_cobenefits_2035_BAU))

print(df.City[(df.welfare_with_cobenefits_2035 < df.welfare_with_cobenefits_2035_BAU)])
print(df.City[(df.welfare_without_cobenefits_2035 < df.welfare_without_cobenefits_2035_BAU)])


welfare_increase = pd.DataFrame(columns = ['City', 'Welfare_increase'], index = df.index)
welfare_increase.City = df.City
welfare_increase['Welfare_increase'] = (df.welfare_with_cobenefits_2035 > df.welfare_with_cobenefits_2035_BAU)
welfare_increase.to_excel("C:/Users/charl/OneDrive/Bureau/welfare_increase_brt.xlsx")
#Welfare with and without cobenefits in 2015, 2035?

print(sum(df.welfare_with_cobenefits_2035 > df.welfare_without_cobenefits_2035))
print(sum(df.welfare_with_cobenefits_2035 < df.welfare_without_cobenefits_2035))

print(sum(df.welfare_with_cobenefits_2035_BAU > df.welfare_without_cobenefits_2035_BAU))
print(sum(df.welfare_with_cobenefits_2035_BAU < df.welfare_without_cobenefits_2035_BAU))

#Variation emissions and welfare for each pol? Cost-effectiveness?

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(df['Continent_BAU'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent_BAU'] == colors[i]]
    plt.scatter((data.welfare_with_cobenefits_2035 / data.welfare_with_cobenefits_2035_BAU), data.emissions_2035 / data.emissions_2035_BAU, color=data.Continent_BAU.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Welfare variation (with cobenefits)", size = 20)
plt.ylabel("Emissions variation", size = 20)
plt.legend()

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(df['Continent_BAU'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent_BAU'] == colors[i]]
    plt.scatter((data.welfare_without_cobenefits_2035 / data.welfare_without_cobenefits_2035_BAU), data.emissions_2035 / data.emissions_2035_BAU, color=data.Continent_BAU.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Welfare variation (without cobenefits)", size = 20)
plt.ylabel("Emissions variation", size = 20)
plt.legend()

df["cost_effectiveness_without_cobenefits"] = (df.emissions_2035 / df.emissions_2035_BAU) / (df.welfare_without_cobenefits_2035 / df.welfare_without_cobenefits_2035_BAU)
df["cost_effectiveness_with_cobenefits"] = (df.emissions_2035 / df.emissions_2035_BAU) / (df.welfare_with_cobenefits_2035 / df.welfare_with_cobenefits_2035_BAU)

df.loc[:, ["City", "cost_effectiveness_without_cobenefits", "cost_effectiveness_with_cobenefits"]].to_excel("C:/Users/charl/OneDrive/Bureau/BRT_resid4.xlsx")


city_quartiles = copy.deepcopy(df)

city_quartiles["var"] = "Q0"
city_quartiles["var"][city_quartiles["cost_effectiveness_with_cobenefits"] <  np.quantile(city_quartiles["cost_effectiveness_with_cobenefits"], 0.25)] = "Q1"
city_quartiles["var"][(city_quartiles["cost_effectiveness_with_cobenefits"] >=  np.quantile(city_quartiles["cost_effectiveness_with_cobenefits"], 0.25)) &  (city_quartiles["cost_effectiveness_with_cobenefits"] <  np.quantile(city_quartiles["cost_effectiveness_with_cobenefits"], 0.5))] = "Q2"
city_quartiles["var"][(city_quartiles["cost_effectiveness_with_cobenefits"] >=  np.quantile(city_quartiles["cost_effectiveness_with_cobenefits"], 0.5)) &  (city_quartiles["cost_effectiveness_with_cobenefits"] <  np.quantile(city_quartiles["cost_effectiveness_with_cobenefits"], 0.75))] = "Q3"
city_quartiles["var"][city_quartiles["cost_effectiveness_with_cobenefits"] >=  np.quantile(city_quartiles["cost_effectiveness_with_cobenefits"], 0.75)] = "Q4"

#### Regressions

city_characteristics2 = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics_20211027.xlsx")
city_characteristics = pd.read_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/city_characteristics_20210810.xlsx")
df = df.loc[:, ["City", "cost_effectiveness_without_cobenefits", "cost_effectiveness_with_cobenefits", "length_network"]].merge(city_characteristics, left_on = "City", right_on = 'city')
df["cost_effectiveness_without_cobenefits"] = df["cost_effectiveness_without_cobenefits"].astype(float)
df["cost_effectiveness_with_cobenefits"] = df["cost_effectiveness_with_cobenefits"].astype(float)
df["length_network"] = df["length_network"].astype(float)
df["length_network2"] = df["length_network"] *df["length_network"]

df = df.merge(city_quartiles.loc[:, ["City", "var"]], on = "City")
df["quartiles"] = copy.deepcopy(df["var"])

df = df.merge(city_characteristics2.loc[:, ["city", "agricultural_rent"]], on = "city")

df = df.merge(city_characteristics.loc[:, ["city", "pop_2035", "inc_2035"]], on = "city")

compute_density = pd.read_excel("C:/Users/charl/OneDrive/Bureau/Urban sprawl and density/scenarios_densities_20211115.xlsx")
compute_density["data_density_2015"] = 0.01 * compute_density.data_pop_2015 / compute_density.ESA_land_cover_2015
df = df.merge(compute_density, on = 'City')


#(df["length_network"] / 1000).describe()
#df["length_network"] = np.nan

#for city in df.City:
#    try:
#        df.length_network[df.City == city] = np.load('C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BRT_202109018_operatingcosts/' + city + "_length_network.npy").squeeze()
#    except:
#        pass
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

df["length_network2"] = df.length_network * df.length_network
df["log_population"] = np.log(df.population)
df["log_income"] = np.log(df.income)
df["log_pop2035"] = np.log(df.pop_2035)
df["log_inc2035"] = np.log(df.inc_2035)
df["log_agri_rent"] = np.log(df.agricultural_rent)
df["log_length"] = np.log(df.length_network)

#Other city characteristics (natural constraints,...)

#R2 rents and density
d_corr_density_scells2 = np.load(path_calibration + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_calibration + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]
r2density_scells2 = np.load(path_calibration + "r2density_scells2.npy", allow_pickle = True)
r2density_scells2 = np.array(r2density_scells2, ndmin = 1)[0]
r2rent_scells2 = np.load(path_calibration + "r2rent_scells2.npy", allow_pickle = True)
r2rent_scells2 = np.array(r2rent_scells2, ndmin = 1)[0]

df["corr_rent"] = np.array(list(d_corr_rent_scells2.values()))[sample_of_cities.loc[sample_of_cities.City != 'Sfax', :].final_sample == 1, 0]
df["corr_density"] = np.array(list(d_corr_density_scells2.values()))[sample_of_cities.loc[sample_of_cities.City != 'Sfax', :].final_sample == 1, 0]
df["r2_rent"] = np.array(list(r2rent_scells2.values()))[sample_of_cities.loc[sample_of_cities.City != 'Sfax', :].final_sample == 1]
df["r2_density"] = np.array(list(r2density_scells2.values()))[sample_of_cities.loc[sample_of_cities.City != 'Sfax', :].final_sample == 1]

df = df.loc[df.City != "Thessaloniki", :]
reg1 = ols("cost_effectiveness_with_cobenefits ~ log_population + population_variation + log_income + income_variation + substitution_potential + grad_density + log_agri_rent", data=df).fit(cov_type='HC3')
reg1.summary()



#### DECISION TREE

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import export_graphviz

regr_1 = DecisionTreeRegressor(min_samples_leaf = 10)
#regr_1 = DecisionTreeClassifier(max_depth=3)
regr_1.fit(np.array([df.log_population, df.population_variation, df.log_income, df.income_variation, df.substitution_potential, df.grad_density, df.log_agri_rent, df.corr_density, df.avg_cons2, df.gradient_cons]).transpose(), df.cost_effectiveness_without_cobenefits) #df.quartiles)
fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(regr_1, 
                   feature_names=["log_population", "population_variation", "log_income", "income_variation", "substitution_potential", "grad_density", "log_agricultural_rent", "corr_density", "avg_cons2", "gradient_cons"],
                   filled=True, fontsize=10)

regr_1 = DecisionTreeClassifier(criterion = 'gini',  min_samples_leaf = 10)
regr_1.fit(np.array([df.log_population, df.population_variation, df.log_income, df.income_variation, df.substitution_potential, df.grad_density, df.log_agri_rent, df.corr_density, df.avg_cons2, df.gradient_cons]).transpose(), df.quartiles) #df.quartiles)
fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(regr_1, 
                   feature_names=["log_population", "population_variation", "log_income", "income_variation", "substitution_potential", "grad_density", "log_agricultural_rent", "corr_density", "avg_cons2", "gradient_cons"],
                   filled=True, fontsize=10)


#### Tuning

X = np.array([df.log_population, df.population_variation, df.log_income, df.income_variation, df.substitution_potential, df.grad_density, df.log_agri_rent, df.corr_density, df.avg_cons2, df.gradient_cons, df.log_length]).transpose()
y = df['quartiles']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state= 355)

names_features = ["log_population", "population_variation", "log_income", "income_variation", "substitution_potential", "grad_density", "log_agricultural_rent", "corr_density", "avg_cons2", "gradient_cons", 'log_length']

clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)

fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(clf, 
                   feature_names= names_features,
                   filled=True, fontsize=10)

clf.score(x_train,y_train)
py_pred = clf.predict(x_test)
clf.score(x_test,y_test)
accuracy_score(y_test, py_pred)

grid_param = {'criterion': ['gini', 'entropy'],
              'max_depth' : range(2,30,1),
              'min_samples_leaf' : range(1,10,1),
              'min_samples_split': range(2,10,1),
              'splitter' : ['best', 'random']}

grid_search = GridSearchCV(estimator=clf,param_grid=grid_param,cv=5,n_jobs =-1)

grid_search.fit(x_train,y_train)

best_parameters = grid_search.best_params_
print(best_parameters)
grid_search.best_score_

clf = DecisionTreeClassifier(criterion = 'gini', max_depth =25, min_samples_leaf= 1, min_samples_split= 3, splitter ='random')

clf.fit(x_train,y_train)
clf.score(x_test,y_test)

fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(clf, 
                   feature_names= names_features,
                   filled=True, fontsize=10, max_depth = 3)


### CORR COEFF
X_cor = np.array([df.cost_effectiveness_with_cobenefits, df.log_population, df.population_variation, df.log_income, df.income_variation, df.substitution_potential, df.grad_density, df.log_agri_rent, df.corr_density, df.avg_cons2, df.gradient_cons, df.log_length]).transpose()
names_features = ["cost_effectiveness_with_cobenefits", "log_population", "population_variation", "log_income", "income_variation", "substitution_potential", "grad_density", "log_agricultural_rent", "corr_density", "avg_cons2", "gradient_cons", 'log_length']


corr_df = pd.DataFrame(pd.DataFrame(X_cor).corr())
corr_df.columns= names_features
corr_df.index= names_features

corr_df.to_excel("correlations.xlsx")


######
fuel_tax_results = reg1
fuel_efficiency_results = reg1
brt_results = reg1
ugb_results = reg1

print(fuel_tax_results.summary().as_latex())

print(fuel_efficiency_results.summary().as_latex())

print(ugb_results.summary().as_latex())
from stargazer.stargazer import Stargazer
print(Stargazer([reg1]).render_latex())
### Agggregated impact of each policy

# sum total emissions

path_carbon_tax = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/carbon_tax_20220322_1percent_resid6/'
path_fuel_efficiency = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/fuel_efficiency_20220322_1percent_resid6/'
path_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/UGB_resid4_data_urba/'
path_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BRT_20220324_1percent_resid4_capital_evolution_25_0_12_15_income/'
path_synergy = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/synergy_BRT_CT/'

df = pd.DataFrame(columns = ['City', 'emissions_2035_BAU', 'welfare_2035_BAU', 'emissions_2035_BRT', 'welfare_2035_BRT', 'emissions_2035_FE', 'welfare_2035_FE', 'emissions_2035_UGB', 'welfare_2035_UGB', 'emissions_2035_CT', 'welfare_2035_CT', 'emissions_2035_S', 'welfare_2035_S', 'population_2035'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df.City = df.index

#df = df.drop(index = ['Jinan', 'Tianjin', 'Wuhan', 'Zhengzhou'])

for city in df.index:
    df.population_2035[df.City == city] = np.load(path_BAU + city + "_population.npy")[20]
    df.emissions_2035_BAU[df.City == city] = np.load(path_BAU + city + "_emissions.npy")[20]
    df.welfare_2035_BAU[df.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[20]
    df.emissions_2035_UGB[df.City == city] = np.load(path_UGB + city + "_emissions.npy")[20]
    df.welfare_2035_UGB[df.City == city] = np.load(path_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    df.emissions_2035_BRT[df.City == city] = np.load(path_BRT + city + "_emissions.npy")[20]
    df.welfare_2035_BRT[df.City == city] = np.load(path_BRT + city + "_total_welfare_with_cobenefits.npy")[20]
    df.emissions_2035_FE[df.City == city] = np.load(path_fuel_efficiency + city + "_emissions.npy")[20]
    df.welfare_2035_FE[df.City == city] = np.load(path_fuel_efficiency + city + "_total_welfare_with_cobenefits.npy")[20]
    df.emissions_2035_CT[df.City == city] = np.load(path_carbon_tax + city + "_emissions.npy")[20]
    df.welfare_2035_CT[df.City == city] = np.load(path_carbon_tax + city + "_total_welfare_with_cobenefits.npy")[20]  
    df.population_2035[df.City == city] = np.load(path_BAU + city + "_population.npy")[20]
    df.emissions_2035_S[df.City == city] = np.load(path_synergy + city + "_emissions.npy")[20]
    df.welfare_2035_S[df.City == city] = np.load(path_synergy + city + "_total_welfare_with_cobenefits.npy")[20]
    
emissions_BAU = np.nansum(df.emissions_2035_BAU)
sum_emissions = np.array([np.nansum(df.emissions_2035_CT), np.nansum(df.emissions_2035_BRT), np.nansum(df.emissions_2035_FE), np.nansum(df.emissions_2035_UGB)]) #, np.nansum(df.emissions_2035_S)])
sum_emissions = 100 * (sum_emissions - emissions_BAU) / emissions_BAU
      
data = pd.DataFrame(columns = ['policy', 'Aggregated emissions'])
data['policy'] = ["Carbon tax", "BRT", "Fuel efficiency", "Greenbelt"]
data['emissions'] = sum_emissions





# save your chart as an image
plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(data=data, x="policy", y="emissions", palette=("Set2"))
plt.xlabel('')
plt.ylabel('')
#plt.title('Number of Accidents By Severity', size=18, color='#4f4e4e')
ax.tick_params(axis='both', labelsize=14, color='#4f4e4e')
plt.yticks([], [])
#plt.text(x=1, y=48, s='Most accidents were low severity', 
#                 color='#4f4e4e', fontsize=12, horizontalalignment='center')
plt.text(x=0, y=-1.7, s=str(round(data.emissions[0], 1))+ "%", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=-1.7, s=str(round(data.emissions[1], 1))+ "%", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=-1.7, s=str(round(data.emissions[2], 1))+ "%", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=-1.7, s=str(round(data.emissions[3], 1))+ "%", 
                 color='white', fontsize=18, horizontalalignment='center')
sns.despine(left=True, bottom = True, top = False)
plt.savefig('emissions.png')

##### Quartiles
df = df.merge(city_quartiles[["City", "var"]], on = "City")
df["emissions_2035_CT"] = pd.to_numeric(df["emissions_2035_UGB"])
df["emissions_2035_BAU"] = pd.to_numeric(df["emissions_2035_BAU"])
sum_emissions = (df.loc[:, ["var", "emissions_2035_CT", "emissions_2035_BAU"]]).groupby("var").sum()
sum_emissions = 100 * (sum_emissions["emissions_2035_CT"] - sum_emissions["emissions_2035_BAU"]) / sum_emissions["emissions_2035_BAU"]
      
df.welfare_2035_UGB = 100 * (df.welfare_2035_UGB - df.welfare_2035_BAU) / df.welfare_2035_BAU
sum_welfare = np.array([np.nansum(df.welfare_2035_UGB.loc[df["var"] == "Q1"] * df.population_2035.loc[df["var"] == "Q1"]) / np.nansum(df.population_2035.loc[df["var"] == "Q1"]),
                        np.nansum(df.welfare_2035_UGB.loc[df["var"] == "Q2"] * df.population_2035.loc[df["var"] == "Q2"]) / np.nansum(df.population_2035.loc[df["var"] == "Q2"]),
                        np.nansum(df.welfare_2035_UGB.loc[df["var"] == "Q3"] * df.population_2035.loc[df["var"] == "Q3"]) / np.nansum(df.population_2035.loc[df["var"] == "Q3"]),
                        np.nansum(df.welfare_2035_UGB.loc[df["var"] == "Q4"] * df.population_2035.loc[df["var"] == "Q4"]) / np.nansum(df.population_2035.loc[df["var"] == "Q4"])])


# save your chart as an image
tidy = pd.DataFrame(columns = ["Quartile", "Variable", "Value"])
tidy["Quartile"] = np.array(["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"])
tidy["Variable"] = np.array(["Emissions", "Emissions", "Emissions", "Emissions", "Welfare", "Welfare", "Welfare", "Welfare"])
tidy["Value"] = np.concatenate([np.array(sum_emissions), sum_welfare])
plt.rcParams['figure.dpi'] = 360
plt.rc('font', weight='normal')
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,2))
sns.barplot(data=tidy, x="Quartile", y="Value", hue = "Variable", palette=['#fc8d62', '#66c2a5'])#("Set2"))
plt.xlabel('')
plt.ylabel('')
#plt.title('Number of Accidents By Severity', size=18, color='#4f4e4e')
plt.yticks([], [])
#plt.xticks([], [])
#plt.ylim(-38, 17)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.45, 0.35), fontsize = 10, loc=2, borderaxespad=0.)#,  prop = {'weight':'bold'})
ax.tick_params(axis = 'both', labelsize=10, color='#4f4e4e')
sns.despine(left=True, top = True)
#ax.spines["bottom"].set_position(("data", 0))
#plt.text(x=1, y=48, s='Most accidents were low severity', 
#                 color='#4f4e4e', fontsize=12, horizontalalignment='center')
#plt.text(x=0, y= data["Aggregated emissions"][3]-2, s = data["policy"][0], 
 #                color='#4f4e4e', fontsize=14, horizontalalignment='center')
#plt.text(x=1, y= data["Aggregated emissions"][3]-2, s = data["policy"][1], 
#                 color='#4f4e4e', fontsize=14, horizontalalignment='center')
#plt.text(x=2, y= data["Aggregated emissions"][3]-2, s = data["policy"][2], 
#                 color='#4f4e4e', fontsize=14, horizontalalignment='center')
#plt.text(x=3, y= data["Aggregated emissions"][3]-2, s = data["policy"][3], 
#                 color='#4f4e4e', fontsize=14, horizontalalignment='center')
plt.text(x=0.2, y=2.5, s='+' +str(round(sum_welfare[0], 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.2, y=-5.8, s=str(round(sum_welfare[1], 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.2, y=-5.8, s=str(round(sum_welfare[2], 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=3.2, y=-5.8, s=str(round(sum_welfare[3], 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=-0.2, y=-5.8, s=str(round(sum_emissions[0], 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=0.8, y=-5.8, s=str(round(sum_emissions[1], 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.8, y=-5.8, s=str(round(sum_emissions[2], 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.8, y=-5.8, s=str(round(sum_emissions[3], 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')



df.welfare_2035_BRT = 100 * (df.welfare_2035_BRT - df.welfare_2035_BAU) / df.welfare_2035_BAU
df.welfare_2035_UGB = 100 * (df.welfare_2035_UGB - df.welfare_2035_BAU) / df.welfare_2035_BAU
df.welfare_2035_FE = 100 * (df.welfare_2035_FE - df.welfare_2035_BAU) / df.welfare_2035_BAU
df.welfare_2035_S = 100 * (df.welfare_2035_S - df.welfare_2035_BAU) / df.welfare_2035_BAU
df.welfare_2035_CT = 100 * (df.welfare_2035_CT - df.welfare_2035_BAU) / df.welfare_2035_BAU
    
data['Aggregated welfare'] = [np.nansum(df.welfare_2035_CT * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_BRT * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_FE * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_UGB * df.population_2035) / np.nansum(df.population_2035)]
 
# save your chart as an image
plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(data=data, x="policy", y="welfare", palette=("Set2"))
plt.xlabel('')
plt.ylabel('')
#plt.title('Number of Accidents By Severity', size=18, color='#4f4e4e')
plt.yticks([], [])
#plt.text(x=1, y=48, s='Most accidents were low severity', 
#                 color='#4f4e4e', fontsize=12, horizontalalignment='center')
plt.text(x=0, y=1, s='+' +str(round(data.welfare[0], 1))+ "%", 
                 color='black', fontsize=18, horizontalalignment='center')
plt.text(x=1, y=1, s='+' +str(round(data.welfare[1], 1))+ "%", 
                 color='black', fontsize=18, horizontalalignment='center')
plt.text(x=2, y=1, s='+' +str(round(data.welfare[2], 1))+ "%", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=-1.7, s=str(round(data.welfare[3], 1))+ "%", 
                 color='white', fontsize=18, horizontalalignment='center')
plt.text(x=3, y=1, s='Greenbelt', color='#4f4e4e', fontsize=14, horizontalalignment='center')
sns.despine(left=True, top = True)
ax.spines["bottom"].set_position(("data", 0))
#ax.set_xticks(size=18, color='#4f4e4e')
ax.tick_params(axis='both', labelsize=14, color='#4f4e4e')
plt.savefig('welfare.png');

data = pd.DataFrame(columns = ['policy', 'Aggregated emissions'])
data['policy'] = ["Carbon tax", "BRT", "Fuel efficiency", "Greenbelt"]
data['Aggregated emissions'] = sum_emissions
data['Aggregated welfare'] = [np.nansum(df.welfare_2035_CT * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_BRT * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_FE * df.population_2035) / np.nansum(df.population_2035), np.nansum(df.welfare_2035_UGB * df.population_2035) / np.nansum(df.population_2035)]

# save your chart as an image
tidy = data.melt(id_vars='policy').rename(columns=str.title)
plt.rcParams['figure.dpi'] = 360
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(data=tidy, x="Policy", y="Value", hue = "Variable", palette=['#fc8d62', '#66c2a5'])#("Set2"))
plt.xlabel('')
plt.ylabel('')
#plt.title('Number of Accidents By Severity', size=18, color='#4f4e4e')
plt.yticks([], [])
#plt.xticks([], [])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0, 0.35), fontsize = 14, loc=2, borderaxespad=0.)
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
sns.despine(left=True, top = True)
#ax.spines["bottom"].set_position(("data", 0))
#plt.text(x=1, y=48, s='Most accidents were low severity', 
#                 color='#4f4e4e', fontsize=12, horizontalalignment='center')
#plt.text(x=0, y= data["Aggregated emissions"][3]-2, s = data["policy"][0], 
 #                color='#4f4e4e', fontsize=14, horizontalalignment='center')
#plt.text(x=1, y= data["Aggregated emissions"][3]-2, s = data["policy"][1], 
#                 color='#4f4e4e', fontsize=14, horizontalalignment='center')
#plt.text(x=2, y= data["Aggregated emissions"][3]-2, s = data["policy"][2], 
#                 color='#4f4e4e', fontsize=14, horizontalalignment='center')
#plt.text(x=3, y= data["Aggregated emissions"][3]-2, s = data["policy"][3], 
#                 color='#4f4e4e', fontsize=14, horizontalalignment='center')
plt.text(x=0.2, y=1, s='+' +str(round(data["Aggregated welfare"][0], 1))+ "%", 
                 color='black', fontsize=14, horizontalalignment='center')
plt.text(x=1.2, y=1, s='+' +str(round(data["Aggregated welfare"][1], 1))+ "%", 
                 color='black', fontsize=14, horizontalalignment='center')
plt.text(x=2.2, y=1, s='+' +str(round(data["Aggregated welfare"][2], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=3.2, y=-2.3, s=str(round(data["Aggregated welfare"][3], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=-0.2, y=-2.3, s=str(round(data["Aggregated emissions"][0], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=0.8, y=-2.3, s=str(round(data["Aggregated emissions"][1], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=1.8, y=-2.3, s=str(round(data["Aggregated emissions"][2], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.text(x=2.8, y=-2.3, s=str(round(data["Aggregated emissions"][3], 1))+ "%", 
                 color='white', fontsize=14, horizontalalignment='center')
plt.savefig('welfare_and_emissions.png');

df_modified = df
df_modified.emissions_2035_BRT[df_modified.welfare_2035_BRT < df_modified.welfare_2035_BAU] = df_modified.emissions_2035_BAU
df_modified.emissions_2035_CT[df_modified.welfare_2035_CT < df_modified.welfare_2035_BAU] = df_modified.emissions_2035_BAU
df_modified.emissions_2035_FE[df_modified.welfare_2035_FE < df_modified.welfare_2035_BAU] = df_modified.emissions_2035_BAU
df_modified.emissions_2035_UGB[df_modified.welfare_2035_UGB < df_modified.welfare_2035_BAU] = df_modified.emissions_2035_BAU


emissions_BAU = np.nansum(df_modified.emissions_2035_BAU)
sum_emissions = np.array([np.nansum(df_modified.emissions_2035_CT), np.nansum(df_modified.emissions_2035_BRT), np.nansum(df_modified.emissions_2035_FE), np.nansum(df_modified.emissions_2035_UGB)])
sum_emissions = 100 * (sum_emissions - emissions_BAU) / emissions_BAU
  
plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})   
plt.bar(np.arange(4), sum_emissions)
plt.xticks(np.arange(4), labels = ["CT", "BRT", "FE", "UGB"])
           
#### ANALYSIS POLICY BY POLICY: CARBON TAX

df_carbon_tax = pd.DataFrame(columns = ['City', 'avg_dist_city_center', 'modal_share_cars', 'avg_dist_city_center_BAU', 'modal_share_cars_BAU'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df_carbon_tax.City = df_carbon_tax.index

path_carbon_tax = path_BRT

for city in df_carbon_tax.index:
    density2035 = np.load(path_carbon_tax + city + "_density.npy")[20]
    modes2035 = np.load(path_carbon_tax + city + "_modal_shares.npy")[20]
    distance = np.load(path_carbon_tax + city + "_distance.npy")
    df_carbon_tax.avg_dist_city_center[df_carbon_tax.City == city] = np.nansum(density2035 * distance) / np.nansum(density2035)
    df_carbon_tax.modal_share_cars[df_carbon_tax.City == city] = 100 * np.nansum(density2035[modes2035 == 0]) / np.nansum(density2035)
    density2035_BAU = np.load(path_BAU + city + "_density.npy")[20]
    modes2035_BAU = np.load(path_BAU + city + "_modal_shares.npy")[20]
    distance_BAU = np.load(path_BAU + city + "_distance.npy")
    df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.City == city] = np.nansum(density2035_BAU * distance_BAU) / np.nansum(density2035_BAU)
    df_carbon_tax.modal_share_cars_BAU[df_carbon_tax.City == city] = 100 * np.nansum(density2035_BAU[modes2035_BAU == 0]) / np.nansum(density2035_BAU)

city_continent = pd.read_csv(path_data + "CityDatabases/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
df_carbon_tax = df_carbon_tax.merge(city_continent, on = "City", how = 'left')

df_carbon_tax.avg_dist_city_center = 100 * (df_carbon_tax.avg_dist_city_center - df_carbon_tax.avg_dist_city_center_BAU) / df_carbon_tax.avg_dist_city_center_BAU
df_carbon_tax.modal_share_cars = 100 * (df_carbon_tax.modal_share_cars - df_carbon_tax.modal_share_cars_BAU) / df_carbon_tax.modal_share_cars_BAU

df_carbon_tax.modal_share_cars.astype(float).describe()
df_carbon_tax.avg_dist_city_center.astype(float).describe()

colors = ['#fc8d62', '#fc8d62', '#fc8d62']
colors1 = dict(color=colors[0])
colors2 = dict(color=colors[1])
colors3 = dict(color=colors[2])

fig, ax = plt.subplots(figsize = (15, 10))
#bp1 = ax.boxplot([df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Africa'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Asia'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Europe'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'North_America'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Oceania'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'South_America']], positions = np.arange(1, 7), widths = 0.2, boxprops = colors1, medianprops=colors1, whiskerprops=colors1, capprops=colors1, flierprops = colors1)
bp2 = ax.boxplot([df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Africa'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Asia'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Europe'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'North_America'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Oceania'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'South_America']], positions = np.arange(1, 7), boxprops = colors2, medianprops=colors2, whiskerprops=colors2, capprops=colors2, flierprops = colors2)
plt.xticks(ticks = np.arange(1, 7), labels = ['Africa', 'Asia', 'Europe', 'North_America', 'Oceania', 'South_America'])
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['BAU', 'TAX'], loc='upper right')
plt.title("Fuel efficiency - Distance to city center")

fig, ax = plt.subplots(figsize = (6, 4))
# Remove top and right border
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
plt.ylabel("Variations (%)", fontsize = 14)
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0.25, alpha=0.5)
bp1 = ax.boxplot([df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Asia'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Europe'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'North_America'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Oceania'], df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'South_America']], positions = np.arange(1, 6), boxprops = dict(linewidth=2, color="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False)
bp2 = plt.scatter(y = df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Asia'], x = np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Asia'])), facecolors='none', edgecolors='black')
bp3 = plt.scatter(y = df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Europe'], x = 2 * np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Europe'])), facecolors='none', edgecolors='black')
bp4 = plt.scatter(y = df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'North_America'], x = 3 * np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'North_America'])), facecolors='none', edgecolors='black')
bp5 = plt.scatter(y = df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Oceania'], x = 4 * np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Oceania'])), facecolors='none', edgecolors='black')
bp6 = plt.scatter(y = df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'South_America'], x = 5 * np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'South_America'])), facecolors='none', edgecolors='black')

#bp2 = ax.boxplot([df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Africa'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Asia'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Europe'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'North_America'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Oceania'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'South_America']], positions = np.arange(1, 7), boxprops = colors2, medianprops=colors2, whiskerprops=colors2, capprops=colors2, flierprops = colors2)
plt.xticks(ticks = np.arange(1, 6), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['BAU', 'TAX'], loc='upper right')
#plt.title("Fuel efficiency - Modal shares")
plt.savefig("dist_center")

fig, ax = plt.subplots(figsize = (6, 4))
# Remove top and right border
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
#ax.xaxis.set_ticks_position('none')
plt.ylabel("Variations (%)", fontsize = 14)
ax.tick_params(axis = 'both', labelsize=14, color='#4f4e4e')
ax.grid(color='grey', axis='x', linestyle='-', linewidth=0, alpha=0.15)
bp1 = ax.boxplot([df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Asia'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Europe'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'North_America'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Oceania'], df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'South_America']], positions = np.arange(1, 6), boxprops = dict(linewidth=2, color="#fc8d62"), medianprops = dict(linewidth=2, color="#fc8d62"), whiskerprops = dict(linewidth=2, color="#fc8d62"), capprops = dict(linewidth=2, color="#fc8d62"), flierprops = colors3, showfliers=False)
bp2 = plt.scatter(y = df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Asia'], x = np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Asia'])), facecolors='none', edgecolors='black')
bp3 = plt.scatter(y = df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Europe'], x = 2 * np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Europe'])), facecolors='none', edgecolors='black')
bp4 = plt.scatter(y = df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'North_America'], x = 3 * np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'North_America'])), facecolors='none', edgecolors='black')
bp5 = plt.scatter(y = df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'Oceania'], x = 4 * np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'Oceania'])), facecolors='none', edgecolors='black')
bp6 = plt.scatter(y = df_carbon_tax.modal_share_cars[df_carbon_tax.Continent == 'South_America'], x = 5 * np.ones(len(df_carbon_tax.avg_dist_city_center[df_carbon_tax.Continent == 'South_America'])), facecolors='none', edgecolors='black')
plt.xticks(ticks = np.arange(1, 6), labels = ['Asia', 'Europe', 'North \n America', 'Oceania', 'South \n America'])
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['BAU', 'TAX'], loc='upper right')
#plt.title("Fuel efficiency - Modal shares")
plt.savefig("mod_share")

# Visualize petal length distribution for all species
fig, ax = plt.subplots(figsize=(12, 7))
# Remove top and right border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# Remove y-axis tick marks
ax.yaxis.set_ticks_position('none')
# Add major gridlines in the y-axis
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
# Set plot title
ax.set_title('Distribution of petal length by species')
# Set species names as labels for the boxplot
dataset = [setosa_petal_length, versicolor_petal_length, virginica_petal_length]
labels = iris_df['species_name'].unique()
ax.boxplot(dataset, labels=labels)
plt.show()

#### FE and cobenefits

cobenefits_BAU = pd.DataFrame(columns = ['city', 'air_pollution', 'active_modes', 'noise', 'car_accidents', 'emissions'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
cobenefits_BAU_but_EV = pd.DataFrame(columns = ['city', 'air_pollution', 'active_modes', 'noise', 'car_accidents', 'emissions'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
cobenefits_FE = pd.DataFrame(columns = ['city', 'air_pollution', 'active_modes', 'noise', 'car_accidents', 'emissions'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
cobenefits_FE_but_IV = pd.DataFrame(columns = ['city', 'air_pollution', 'active_modes', 'noise', 'car_accidents', 'emissions'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))

year = 2035
imaclim = pd.read_excel(path_folder + "Charlotte_ResultsIncomePriceAutoEmissionsAuto_V1.xlsx", sheet_name = 'Extraction_Baseline')
WALKING_SPEED = 5
CO2_EMISSIONS_TRANSIT = 15

cobenefits_BAU.city = cobenefits_BAU.index

cobenefits_BAU_but_EV.city = cobenefits_BAU_but_EV.index

cobenefits_FE.city = cobenefits_FE.index
  
cobenefits_FE_but_IV.city = cobenefits_FE_but_IV.index


for city in cobenefits_BAU.index:
    density2035_BAU = np.load(path_BAU + city + "_density.npy")[20]
    modes2035_BAU = np.load(path_BAU + city + "_modal_shares.npy")[20]
    density2035_EV = np.load(path_fuel_efficiency + city + "_density.npy")[20]
    modes2035_EV = np.load(path_fuel_efficiency + city + "_modal_shares.npy")[20]
    distance = np.load(path_BAU + city + "_distance.npy")
    population = np.load(path_BAU + city + "_population.npy")[20]
    Region = pd.read_excel(path_folder + "city_country_region.xlsx").region[pd.read_excel(path_folder + "city_country_region.xlsx").city == city].squeeze()
    Country = list_city.Country[list_city.City == city].squeeze()
    fuel_consumption = import_fuel_conso(Country, path_folder) #dollars/L
    CO2_emissions_car_EV = 2300 * fuel_consumption * (0.963 ** 16) / 100
    CO2_emissions_car_BAU = 2300 * fuel_consumption / 100
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    
    cobenefits_BAU.air_pollution[cobenefits_BAU.city == city] = compute_air_pollution(density2035_BAU, modes2035_BAU, distance, Country, path_folder, year, Region, imaclim, 'None')
    cobenefits_BAU.active_modes[cobenefits_BAU.city == city] = compute_active_modes(density2035_BAU, modes2035_BAU, distance, WALKING_SPEED, Country, path_folder, Region, year, imaclim)
    cobenefits_BAU.noise[cobenefits_BAU.city == city] = compute_noise(density2035_BAU, modes2035_BAU, distance, Country, Region, year, imaclim, path_folder)
    cobenefits_BAU.car_accidents[cobenefits_BAU.city == city] = compute_car_accidents(density2035_BAU, modes2035_BAU, distance, Country, Region, year, imaclim, path_folder)
    cobenefits_BAU.emissions[cobenefits_BAU.city == city] = compute_emissions(CO2_emissions_car_BAU, CO2_EMISSIONS_TRANSIT, density2035_BAU, modes2035_BAU, driving.Distance / 1000, transit.Distance / 1000) / population
    
    cobenefits_FE.air_pollution[cobenefits_BAU.city == city] = compute_air_pollution(density2035_EV, modes2035_EV, distance, Country, path_folder, year, Region, imaclim, 'fuel_efficiency')
    cobenefits_FE.active_modes[cobenefits_BAU.city == city] = compute_active_modes(density2035_EV, modes2035_EV, distance, WALKING_SPEED, Country, path_folder, Region, year, imaclim)
    cobenefits_FE.noise[cobenefits_BAU.city == city] = compute_noise(density2035_EV, modes2035_EV, distance, Country, Region, year, imaclim, path_folder)
    cobenefits_FE.car_accidents[cobenefits_BAU.city == city] = compute_car_accidents(density2035_EV, modes2035_EV, distance, Country, Region, year, imaclim, path_folder)
    cobenefits_FE.emissions[cobenefits_FE.city == city] = compute_emissions(CO2_emissions_car_EV, CO2_EMISSIONS_TRANSIT, density2035_EV, modes2035_EV, driving.Distance / 1000, transit.Distance / 1000) / population
    
    cobenefits_BAU_but_EV.air_pollution[cobenefits_BAU.city == city] = compute_air_pollution(density2035_BAU, modes2035_BAU, distance, Country, path_folder, year, Region, imaclim, 'fuel_efficiency')
    cobenefits_BAU_but_EV.active_modes[cobenefits_BAU.city == city] = compute_active_modes(density2035_BAU, modes2035_BAU, distance, WALKING_SPEED, Country, path_folder, Region, year, imaclim)
    cobenefits_BAU_but_EV.noise[cobenefits_BAU.city == city] = compute_noise(density2035_BAU, modes2035_BAU, distance, Country, Region, year, imaclim, path_folder)
    cobenefits_BAU_but_EV.car_accidents[cobenefits_BAU.city == city] = compute_car_accidents(density2035_BAU, modes2035_BAU, distance, Country, Region, year, imaclim, path_folder)
    cobenefits_BAU_but_EV.emissions[cobenefits_BAU_but_EV.city == city] = compute_emissions(CO2_emissions_car_EV, CO2_EMISSIONS_TRANSIT, density2035_BAU, modes2035_BAU, driving.Distance / 1000, transit.Distance / 1000) / population
    
    cobenefits_FE_but_IV.air_pollution[cobenefits_BAU.city == city] = compute_air_pollution(density2035_EV, modes2035_EV, distance, Country, path_folder, year, Region, imaclim, 'None')
    cobenefits_FE_but_IV.active_modes[cobenefits_BAU.city == city] = compute_active_modes(density2035_EV, modes2035_EV, distance, WALKING_SPEED, Country, path_folder, Region, year, imaclim)
    cobenefits_FE_but_IV.noise[cobenefits_BAU.city == city] = compute_noise(density2035_EV, modes2035_EV, distance, Country, Region, year, imaclim, path_folder)
    cobenefits_FE_but_IV.car_accidents[cobenefits_BAU.city == city] = compute_car_accidents(density2035_EV, modes2035_EV, distance, Country, Region, year, imaclim, path_folder)
    cobenefits_FE_but_IV.emissions[cobenefits_FE_but_IV.city == city] = compute_emissions(CO2_emissions_car_BAU, CO2_EMISSIONS_TRANSIT, density2035_EV, modes2035_EV, driving.Distance / 1000, transit.Distance / 1000) / population
    

colors = ['blue', 'red', '#D94D1A']
colors1 = dict(color=colors[0])
colors2 = dict(color=colors[1])

fig, ax = plt.subplots(figsize = (15, 10))
#bp1 = ax.boxplot([df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Africa'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Asia'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Europe'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'North_America'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Oceania'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'South_America']], positions = np.arange(1, 7), widths = 0.2, boxprops = colors1, medianprops=colors1, whiskerprops=colors1, capprops=colors1, flierprops = colors1)
bp2 = ax.boxplot([100 * (cobenefits_BAU_but_EV.emissions - cobenefits_BAU.emissions) / cobenefits_BAU.emissions, 100 * (cobenefits_FE.emissions - cobenefits_BAU.emissions) / cobenefits_BAU.emissions], positions = np.arange(1, 3), boxprops = colors2, medianprops=colors2, whiskerprops=colors2, capprops=colors2, flierprops = colors2)
plt.xticks(ticks = np.arange(1, 3), labels = ['EV with modal shares and locations as in BAU', 'EV'])
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['BAU', 'TAX'], loc='upper right')
plt.title("Emissions")

fig, ax = plt.subplots(figsize = (15, 10))
#bp1 = ax.boxplot([df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Africa'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Asia'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Europe'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'North_America'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Oceania'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'South_America']], positions = np.arange(1, 7), widths = 0.2, boxprops = colors1, medianprops=colors1, whiskerprops=colors1, capprops=colors1, flierprops = colors1)
bp2 = ax.boxplot([100 * (cobenefits_BAU_but_EV.air_pollution - cobenefits_BAU.air_pollution) / cobenefits_BAU.air_pollution, 100 * (cobenefits_FE.air_pollution - cobenefits_BAU.air_pollution) / cobenefits_BAU.air_pollution], positions = np.arange(1, 3), boxprops = colors2, medianprops=colors2, whiskerprops=colors2, capprops=colors2, flierprops = colors2)
plt.xticks(ticks = np.arange(1, 3), labels = ['EV with modal shares and locations as in BAU', 'EV'])
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['BAU', 'TAX'], loc='upper right')
plt.title("Air pollution")

cobenefits_BAU.active_modes[cobenefits_BAU.active_modes ==0] = 1

fig, ax = plt.subplots(figsize = (15, 10))
#bp1 = ax.boxplot([df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Africa'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Asia'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Europe'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'North_America'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'Oceania'], df_carbon_tax.avg_dist_city_center_BAU[df_carbon_tax.Continent == 'South_America']], positions = np.arange(1, 7), widths = 0.2, boxprops = colors1, medianprops=colors1, whiskerprops=colors1, capprops=colors1, flierprops = colors1)
bp2 = ax.boxplot([100 * (cobenefits_BAU_but_EV.active_modes - cobenefits_BAU.active_modes) / cobenefits_BAU.active_modes, 100 * (cobenefits_FE.active_modes - cobenefits_BAU.active_modes) / cobenefits_BAU.active_modes], positions = np.arange(1, 3), boxprops = colors2, medianprops=colors2, whiskerprops=colors2, capprops=colors2, flierprops = colors2)
plt.xticks(ticks = np.arange(1, 3), labels = ['EV with modal shares and locations as in BAU', 'EV'])
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['BAU', 'TAX'], loc='upper right')
plt.title("Active modes")


                         ##### BRT increases housing prices?

#par personne
path_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/capital_costs_25_0_01_50_5/'
df_BRT = pd.DataFrame(columns = ['City', 'median_rent_BRT'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df_BRT.City = df_BRT.index

for city in df_BRT.index:
    density2035 = np.load(path_BRT + city + "_density.npy")[20]
    rents2035 = np.load(path_BRT + city + "_rent.npy")[20]
    df_BRT['median_rent_BRT'][df_BRT.City == city] = weighted_percentile(np.array(rents2035)[~np.isnan(density2035) & ~np.isnan(rents2035)], 50, weights=density2035[~np.isnan(density2035) & ~np.isnan(rents2035)])
    
df_BAU = pd.DataFrame(columns = ['City', 'median_rent_BAU'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df_BAU.City = df_BAU.index

for city in df_BAU.index:
    density2035 = np.load(path_BAU + city + "_density.npy")[20]
    rents2035 = np.load(path_BAU + city + "_rent.npy")[20]
    df_BAU['median_rent_BAU'][df_BAU.City == city] = weighted_percentile(np.array(rents2035)[~np.isnan(density2035) & ~np.isnan(rents2035)], 50, weights=density2035[~np.isnan(density2035) & ~np.isnan(rents2035)])

df = df_BAU.merge(df_BRT, on = 'City')
df["var"] = 100 * (df.median_rent_BRT - df.median_rent_BAU) / df.median_rent_BAU
df["var"].astype(float).describe()

#par location
df = pd.DataFrame(columns = ['City', 'var'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df.City = df.index

for city in df.index:
    density2035BRT = np.load(path_BRT + city + "_density.npy")[20]
    rents2035BRT = np.load(path_BRT + city + "_rent.npy")[20]
    density2035BAU = np.load(path_BAU + city + "_density.npy")[20]
    rents2035BAU = np.load(path_BAU + city + "_rent.npy")[20]
    var_rent = rents2035BRT / rents2035BAU
    df['var'][df.City == city] = weighted_percentile(np.array(var_rent)[~np.isnan(density2035BAU) & ~np.isnan(var_rent)], 50, weights=density2035BAU[~np.isnan(density2035BAU) & ~np.isnan(var_rent)])

df["var"].astype(float).describe()

#En cumulé, en considérant la taille des logements

df_rent = pd.DataFrame(columns = ['City', 'loyers_BAU', 'loyecitrs_BRT', 'loyers_BAU_sqm', 'loyers_BRT_sqm'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))
df_rent.City = df_rent.index

for city in df_rent.index:
    density2035BRT = np.load(path_BRT + city + "_density.npy")[20]
    rents2035BRT = np.load(path_BRT + city + "_rent.npy")[20]
    density2035BAU = np.load(path_BAU + city + "_density.npy")[20]
    rents2035BAU = np.load(path_BAU + city + "_rent.npy")[20]
    size2035BRT = np.load(path_BRT + city + "_dwelling_size.npy")[20]
    size2035BAU = np.load(path_BAU + city + "_dwelling_size.npy")[20]
    df_rent["loyers_BAU"][df_rent.City == city] = sum(density2035BAU * size2035BAU * rents2035BAU)
    df_rent["loyers_BRT"][df_rent.City == city] = sum(density2035BRT * rents2035BRT * size2035BRT)
    df_rent["loyers_BAU_sqm"][df_rent.City == city] = sum(density2035BAU * size2035BAU * rents2035BAU) / sum(density2035BAU * size2035BAU)
    df_rent["loyers_BRT_sqm"][df_rent.City == city] = sum(density2035BRT * rents2035BRT * size2035BRT) / sum(density2035BRT * size2035BRT)
    
df_rent["var"] = 100 * (df_rent["loyers_BRT_sqm"] - df_rent["loyers_BAU_sqm"]) / df_rent["loyers_BAU_sqm"]
df_rent["var"].astype(float).describe()

df_rent["var"] = 100 * (df_rent["loyers_BRT"] - df_rent["loyers_BAU"]) / df_rent["loyers_BAU"]
df_rent["var"].astype(float).describe()