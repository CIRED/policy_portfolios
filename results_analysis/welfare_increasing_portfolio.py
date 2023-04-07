# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:26:37 2022

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.chdir("..")
from inputs.parameters import *
from inputs.transport import *
from inputs.data import *
from inputs.land_use import *
from calibration.calibration import *
from calibration.validation import *
from model.model import *
from outputs.outputs import *

# Define path
path_data = "C:/Users/charl/OneDrive/Bureau/City_dataStudy/"
path_folder = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Data/"
path_calibration = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/final_results/calibration_20211124/"

path_BAU = "C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BAU_20221221/" #BAU_20211124

#path_CT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/CT_20221221/'
#path_FE = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/FE_20221221/'
#path_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/TOD_20221221/'
#path_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BRT_20221221/'

#path_CT_FE = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/CT_FE_20221221/'
#path_CT_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/CT_BRT_20221221/'
#path_CT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/CT_UGB_20221221/'
#path_FE_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/FE_BRT_20221221/'
#path_FE_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/FE_UGB_20221221/'
#path_BRT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/BRT_UGB_20221221/'

#path_CT_FE_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/CT_FE_BRT_20221221/'
#path_CT_FE_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/CT_FE_UGB_20221221/'
#path_CT_BRT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/CT_BRT_UGB_20221221/'
#path_FE_BRT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/FE_BRT_UGB_20221221/'

#path_CT_FE_BRT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/all_TOD_20221221/'

path_CT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_CT_20230106/'
path_FE = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_FE_20230106/'
path_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_UGB_20230106/'
path_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_BRT_20230106/'

path_CT_FE = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_CT_FE_20230106/'
path_CT_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_CT_BRT_20230106/'
path_CT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_CT_UGB_20230106/'
path_FE_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_FE_BRT_20230106/'
path_FE_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_FE_UGB_20230106/'
path_BRT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_BRT_UGB_20230106/'

path_CT_FE_BRT = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_CT_FE_BRT_20230106/'
path_CT_FE_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_CT_FE_UGB_20230106/'
path_CT_BRT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_CT_BRT_UGB_20230106/'
path_FE_BRT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_FE_BRT_UGB_20230106/'

path_CT_FE_BRT_UGB = 'C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_all_20230106/'

# Import list of cities
list_city = list_of_cities_and_databases(path_data,'cityDatabase')

### STEP 1: SAMPLE SELECTION (SUPP SECTION C)

sample_of_cities = pd.DataFrame(columns = ['City', 'criterion1', 'criterion2', 'criterion3', 'final_sample'], index = np.unique(list_city.City))
sample_of_cities.City = sample_of_cities.index

# Exclusion criterion 1: real estate data availability
selected_cells = np.load(path_calibration + "d_selected_cells.npy", allow_pickle = True)
selected_cells = np.array(selected_cells, ndmin = 1)[0]

for city in list(np.delete(sample_of_cities.index, 153)):
    if (selected_cells[city] > 1):
        sample_of_cities.loc[city, "criterion1"] = 1
    elif (selected_cells[city] == 1):
        sample_of_cities.loc[city, "criterion1"] = 0
        
print("Number of cities excluded because of criterion 1:", sum(sample_of_cities.criterion1 == 0))

# Exclusion criterion 2: real estate data consistent with income data
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

for city in list(sample_of_cities.index):
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        
    density = density.loc[:,density.columns.str.startswith("density")],
    density = np.array(density).squeeze()
    rent = (rents_and_size.avgRent / conversion_rate) * 12
    size = rents_and_size.medSize
    if (city == 'Mashhad') |(city == 'Isfahan') | (city == 'Tehran'):
        rent = 100 * rent
    size.mask((size > 1000), inplace = True)
    
    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()   
    share_housing = rent * size / income
    if weighted_percentile(np.array(share_housing)[~np.isnan(density) & ~np.isnan(share_housing)], 80, weights=density[~np.isnan(density) & ~np.isnan(share_housing)]) > 1:
        sample_of_cities.loc[city, ['criterion2']] = 0
    else:
        sample_of_cities.loc[city, ['criterion2']] = 1

print("Number of cities excluded because of criterion 2:", sum(sample_of_cities.criterion2 == 0))

# Exclusion criterion 3: transportation data quality
sample_of_cities["criterion4"] = 1

for city in sample_of_cities.index:
    if city in ['Basel', 'Bern', 'Bilbao', 'Chicago', 'Cordoba', 'Cracow', 'Curitiba', 'Glasgow', 'Izmir', 'Johannesburg', 'Leeds', 'Liverpool', 'Malmo', 'Monterrey', 'Munich', 'New_York', 'Nottingham', 'Nuremberg', 'Salvador', 'San_Fransisco', 'Seattle', 'Singapore', 'Sofia', 'Stockholm', 'Valencia', 'Zurich']:
        sample_of_cities.loc[sample_of_cities.City == city, "criterion4"] = 0

print("Number of cities excluded because of criterion 3:", sum(sample_of_cities.criterion4 == 0))


# Exclusion criterion 4: reasonable fit of the model
d_corr_density_scells2 = np.load(path_calibration + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_calibration + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]

for city in list(sample_of_cities.index[sample_of_cities.criterion1 == 1 ]):
    if (d_corr_density_scells2[city][0] < 0) | (d_corr_rent_scells2[city][0] < 0):
        sample_of_cities.loc[city, "criterion3"] = 0
    else:
        sample_of_cities.loc[city, "criterion3"] = 1
        
print("Number of cities excluded because of criterion 4:", sum((sample_of_cities.criterion3 == 0) & (sample_of_cities.criterion2 == 1)& (sample_of_cities.criterion4 == 1)))

# Outcome of the sample selection
sample_of_cities.loc[(sample_of_cities.criterion1 == 0) | (sample_of_cities.criterion2 == 0) | (sample_of_cities.criterion3 == 0) | (sample_of_cities.criterion4 == 0), "final_sample"] = 0
sample_of_cities.loc[(sample_of_cities.criterion1 == 1) & (sample_of_cities.criterion2 == 1) & (sample_of_cities.criterion3 == 1) & (sample_of_cities.criterion4 == 1), "final_sample"] = 1

print("Number of cities kept in the end:", sum(sample_of_cities.final_sample == 1))

#### RESULTS WELFARE INCREASING

df = pd.DataFrame(columns = ['City', 'population_2035','emissions_2035_BAU', 'welfare_with_cobenefits_2035_BAU',
                             'emissions_2035_CT', 'welfare_with_cobenefits_2035_CT',
                             'emissions_2035_FE', 'welfare_with_cobenefits_2035_FE',
                             'emissions_2035_UGB', 'welfare_with_cobenefits_2035_UGB',
                             'emissions_2035_BRT', 'welfare_with_cobenefits_2035_BRT',
                             'emissions_2035_CT_FE', 'welfare_with_cobenefits_2035_CT_FE',
                             'emissions_2035_CT_BRT', 'welfare_with_cobenefits_2035_CT_BRT',
                             'emissions_2035_CT_UGB', 'welfare_with_cobenefits_2035_CT_UGB',
                             'emissions_2035_FE_BRT', 'welfare_with_cobenefits_2035_FE_BRT',
                             'emissions_2035_FE_UGB', 'welfare_with_cobenefits_2035_FE_UGB',
                             'emissions_2035_BRT_UGB', 'welfare_with_cobenefits_2035_BRT_UGB',
                             'emissions_2035_CT_FE_BRT', 'welfare_with_cobenefits_2035_CT_FE_BRT',
                             'emissions_2035_CT_FE_UGB', 'welfare_with_cobenefits_2035_CT_FE_UGB',
                             'emissions_2035_CT_BRT_UGB', 'welfare_with_cobenefits_2035_CT_BRT_UGB',
                             'emissions_2035_FE_BRT_UGB', 'welfare_with_cobenefits_2035_FE_BRT_UGB',
                             'emissions_2035_CT_FE_BRT_UGB', 'welfare_with_cobenefits_2035_CT_FE_BRT_UGB'], index = list(sample_of_cities.City[sample_of_cities.final_sample == 1]))

df.City = df.index
df["city"] = df.index

# Import results for all cities
for city in df.index:
    df.population_2035[df.City == city] = np.load(path_BAU + city + "_population.npy")[20]
    
    df.emissions_2035_BAU[df.City == city] = np.load(path_BAU + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_BAU[df.City == city] = np.load(path_BAU + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_CT[df.City == city] = np.load(path_CT + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT[df.City == city] = np.load(path_CT + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_FE[df.City == city] = np.load(path_FE + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_FE[df.City == city] = np.load(path_FE + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_UGB[df.City == city] = np.load(path_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_UGB[df.City == city] = np.load(path_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_BRT[df.City == city] = np.load(path_BRT + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_BRT[df.City == city] = np.load(path_BRT + city + "_total_welfare_with_cobenefits.npy")[20]
    ####
    df.emissions_2035_CT_FE[df.City == city] = np.load(path_CT_FE + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT_FE[df.City == city] = np.load(path_CT_FE + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_CT_BRT[df.City == city] = np.load(path_CT_BRT + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT_BRT[df.City == city] = np.load(path_CT_BRT + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_CT_UGB[df.City == city] = np.load(path_CT_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT_UGB[df.City == city] = np.load(path_CT_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_FE_BRT[df.City == city] = np.load(path_FE_BRT + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_FE_BRT[df.City == city] = np.load(path_FE_BRT + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_FE_UGB[df.City == city] = np.load(path_FE_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_FE_UGB[df.City == city] = np.load(path_FE_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_BRT_UGB[df.City == city] = np.load(path_BRT_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_BRT_UGB[df.City == city] = np.load(path_BRT_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    ####
    df.emissions_2035_CT_FE_BRT[df.City == city] = np.load(path_CT_FE_BRT + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT_FE_BRT[df.City == city] = np.load(path_CT_FE_BRT + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_CT_FE_UGB[df.City == city] = np.load(path_CT_FE_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT_FE_UGB[df.City == city] = np.load(path_CT_FE_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_CT_BRT_UGB[df.City == city] = np.load(path_CT_BRT_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT_BRT_UGB[df.City == city] = np.load(path_CT_BRT_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    
    df.emissions_2035_FE_BRT_UGB[df.City == city] = np.load(path_FE_BRT_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_FE_BRT_UGB[df.City == city] = np.load(path_FE_BRT_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    
    ###
    df.emissions_2035_CT_FE_BRT_UGB[df.City == city] = np.load(path_CT_FE_BRT_UGB + city + "_emissions.npy")[20]
    df.welfare_with_cobenefits_2035_CT_FE_BRT_UGB[df.City == city] = np.load(path_CT_FE_BRT_UGB + city + "_total_welfare_with_cobenefits.npy")[20]
    
    
df.iloc[:,1:34] = df.iloc[:,1:34].astype(float)
   
# Compute variations in outcomes
df["emissions_2035_BRT_var"] = 100 * (df.emissions_2035_BRT - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_UGB_var"] = 100 * (df.emissions_2035_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_FE_var"] = 100 * (df.emissions_2035_FE - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_CT_var"] = 100 * (df.emissions_2035_CT - df.emissions_2035_BAU) / df.emissions_2035_BAU

df["emissions_2035_CT_FE_var"] = 100 * (df.emissions_2035_CT_FE - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_CT_BRT_var"] = 100 * (df.emissions_2035_CT_BRT - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_CT_UGB_var"] = 100 * (df.emissions_2035_CT_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_FE_BRT_var"] = 100 * (df.emissions_2035_FE_BRT - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_FE_UGB_var"] = 100 * (df.emissions_2035_FE_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_BRT_UGB_var"] = 100 * (df.emissions_2035_BRT_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU

df["emissions_2035_CT_FE_BRT_var"] = 100 * (df.emissions_2035_CT_FE_BRT - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_CT_FE_UGB_var"] = 100 * (df.emissions_2035_CT_FE_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_CT_BRT_UGB_var"] = 100 * (df.emissions_2035_CT_BRT_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU
df["emissions_2035_FE_BRT_UGB_var"] = 100 * (df.emissions_2035_FE_BRT_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU

df["emissions_2035_CT_FE_BRT_UGB_var"] = 100 * (df.emissions_2035_CT_FE_BRT_UGB - df.emissions_2035_BAU) / df.emissions_2035_BAU

df["welfare_2035_BRT_var"] = 100 * (df.welfare_with_cobenefits_2035_BRT - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_FE_var"] = 100 * (df.welfare_with_cobenefits_2035_FE - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_CT_var"] = 100 * (df.welfare_with_cobenefits_2035_CT - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU

df["welfare_2035_CT_FE_var"] = 100 * (df.welfare_with_cobenefits_2035_CT_FE - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_CT_BRT_var"] = 100 * (df.welfare_with_cobenefits_2035_CT_BRT - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_CT_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_CT_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_FE_BRT_var"] = 100 * (df.welfare_with_cobenefits_2035_FE_BRT - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_FE_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_FE_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_BRT_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_BRT_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU

df["welfare_2035_CT_FE_BRT_var"] = 100 * (df.welfare_with_cobenefits_2035_CT_FE_BRT - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_CT_FE_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_CT_FE_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_CT_BRT_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_CT_BRT_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU
df["welfare_2035_FE_BRT_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_FE_BRT_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU

df["welfare_2035_CT_FE_BRT_UGB_var"] = 100 * (df.welfare_with_cobenefits_2035_CT_FE_BRT_UGB - df.welfare_with_cobenefits_2035_BAU) / df.welfare_with_cobenefits_2035_BAU

df = df.iloc[:,35:65]

df["emissions_2035_BRT_var"][df["welfare_2035_BRT_var"]<0] = np.nan
df["emissions_2035_UGB_var"][df["welfare_2035_UGB_var"]<0] = np.nan
df["emissions_2035_FE_var"][df["welfare_2035_FE_var"]<0] = np.nan
df["emissions_2035_CT_var"][df["welfare_2035_CT_var"]<0] = np.nan

df["emissions_2035_CT_FE_var"][df["welfare_2035_CT_FE_var"]<0] = np.nan
df["emissions_2035_CT_BRT_var"][df["welfare_2035_CT_BRT_var"]<0] = np.nan
df["emissions_2035_CT_UGB_var"][df["welfare_2035_CT_UGB_var"]<0] = np.nan
df["emissions_2035_FE_BRT_var"][df["welfare_2035_FE_BRT_var"]<0] = np.nan
df["emissions_2035_FE_UGB_var"][df["welfare_2035_FE_UGB_var"]<0] = np.nan
df["emissions_2035_BRT_UGB_var"][df["welfare_2035_BRT_UGB_var"]<0] = np.nan

df["emissions_2035_CT_FE_BRT_var"][df["welfare_2035_CT_FE_BRT_var"]<0] = np.nan
df["emissions_2035_CT_FE_UGB_var"][df["welfare_2035_CT_FE_UGB_var"]<0] = np.nan
df["emissions_2035_CT_BRT_UGB_var"][df["welfare_2035_CT_BRT_UGB_var"]<0] = np.nan
df["emissions_2035_FE_BRT_UGB_var"][df["welfare_2035_FE_BRT_UGB_var"]<0] = np.nan

df["emissions_2035_CT_FE_BRT_UGB_var"][df["welfare_2035_CT_FE_BRT_UGB_var"]<0] = np.nan

df = df.iloc[:,0:15]

# TABLES S13 AND S15

array_max = df.idxmin(axis=1)
array_max = array_max.str.replace('_var', '')
array_max = array_max.str.replace('emissions_2035_', '')
array_max.value_counts()

array_max.to_excel("C:/Users/charl/OneDrive/Bureau/mitigation_policies_city_characteristics/Sorties/robustness_welfare_increasing_policies.xlsx")