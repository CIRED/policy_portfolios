# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:50:54 2021

@author: charl
"""

criteria = pd.DataFrame(columns = ['City'], index = np.delete(np.delete(np.delete(np.unique(list_city.City), 168), 153), 78))

#a comparer avec income
criteria["share_housing"] = 1
for city in list(criteria.index):
    (country, proj, density, rents_and_size, land_use, land_cover_ESACCI, driving, transit, grille, 
     centre, distance_cbd, conversion_rate) = import_data(list_city, path_data, city, path_folder)        

    #Density, rents and dwelling sizes
    density = density.loc[:,density.columns.str.startswith("density")],
    density = np.array(density).squeeze()
    rent = (rents_and_size.avgRent / conversion_rate) * 12
    if city == "Addis_Ababa":
        rent[rent > 50000] = rent / 100
        rent[rent > 500] = rent / 100
    if city == 'Yerevan':
        rent[rent > 1000] = np.nan    
    size = rents_and_size.medSize
    size[size > 1000] = np.nan

    income = pd.read_excel(path_folder + "income.xlsx").income[pd.read_excel(path_folder + "income.xlsx").city == city].squeeze()
    #income = 1.5 * import_gdp_per_capita(path_folder, country, '2015')
    
    share_housing = rent * size / income
    if weighted_percentile(np.array(share_housing)[~np.isnan(density) & ~np.isnan(share_housing)], 80, weights=density[~np.isnan(density) & ~np.isnan(share_housing)]) > 1:
        criteria.loc[city, ['share_housing']] = 0
    
#R2, coeff_corr et rae

r2density_scells2 = np.load(path_outputs + "r2density_scells2.npy", allow_pickle = True)
r2density_scells2 = np.array(r2density_scells2, ndmin = 1)[0]
r2rent_scells2 = np.load(path_outputs + "r2rent_scells2.npy", allow_pickle = True)
r2rent_scells2 = np.array(r2rent_scells2, ndmin = 1)[0]

d_corr_density_scells2 = np.load(path_outputs + "d_corr_density_scells2.npy", allow_pickle = True)
d_corr_density_scells2 = np.array(d_corr_density_scells2, ndmin = 1)[0]
d_corr_rent_scells2 = np.load(path_outputs + "d_corr_rent_scells2.npy", allow_pickle = True)
d_corr_rent_scells2 = np.array(d_corr_rent_scells2, ndmin = 1)[0]

rae_density_scells2 = np.load(path_outputs + "rae_density_scells2.npy", allow_pickle = True)
rae_density_scells2 = np.array(rae_density_scells2, ndmin = 1)[0]
rae_rent_scells2 = np.load(path_outputs + "rae_rent_scells2.npy", allow_pickle = True)
rae_rent_scells2 = np.array(rae_rent_scells2, ndmin = 1)[0]


criteria["validation_density"] = 1
criteria["validation_rent"] = 1

#for city in list(criteria.index):
#    if (r2density_scells2[city] < 0) | (d_corr_density_scells2[city][0] < 0) | (rae_density_scells2[city] > 1):
#        criteria.loc[city, "validation_density"] = 0
        
#for city in list(criteria.index):
#    if (r2rent_scells2[city] < 0) | (d_corr_rent_scells2[city][0] < 0) | (rae_rent_scells2[city] > 1):
#        criteria.loc[city, "validation_rent"] = 0

for city in list(criteria.index):
    if (d_corr_density_scells2[city][0] < 0):
        criteria.loc[city, "validation_density"] = 0
        
for city in list(criteria.index):
    if (d_corr_rent_scells2[city][0] < 0):
        criteria.loc[city, "validation_rent"] = 0

print(sum((criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1)))

#Cohérent avec les welfare et émissions?

cobenetifs = pd.read_excel('C:/Users/charl/OneDrive/Bureau/cobenefits.xlsx')

criteria["cobenefits_baseline"] = 1
for city in list(criteria.index):
    if (cobenetifs.loc[cobenetifs["Unnamed: 0"] == city, "diff_welfare"].squeeze() < -25):
        criteria.loc[city, "cobenefits_baseline"] = 0

sum(criteria["cobenefits_baseline"] == 1)
print(sum((criteria["cobenefits_baseline"] == 1) & (criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1)))
#Garder juste share_housing + éventuellement creuser Mar del Plata et Recife

### Baseline
df_cobenefits = pd.read_excel('C:/Users/charl/OneDrive/Bureau/cobenefits_policy.xlsx')
baseline_2035 = df_cobenefits.loc[df_cobenefits.Pol == "None", ['City', 'emissions', 'welfare_with_cobenefits']]
baseline_2015 = cobenetifs.loc[:, ['Unnamed: 0', 'emissions', 'welfare_with_cobenefits']]
baseline_2015.columns = ['City', "emissions_2015", "welfare_2015"]
baseline_2035 = baseline_2035.merge(baseline_2015, on = 'City')
baseline_2035["evolution_emissions"] = baseline_2035.emissions / baseline_2035.emissions_2015
baseline_2035["evolution_utility"] = baseline_2035.welfare_with_cobenefits / baseline_2035.welfare_2015


criteria["evolution_utility_baseline"] = 1
for city in list(criteria.index):
    if (baseline_2035.loc[baseline_2035["City"] == city, "evolution_utility"].squeeze() < 0.5):
        criteria.loc[city, "evolution_utility_baseline"] = 0
    if (baseline_2035.loc[baseline_2035["City"] == city, "evolution_utility"].squeeze() > 10):
        criteria.loc[city, "evolution_utility_baseline"] = 0

print(sum((criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1)))
print(sum((criteria["evolution_utility_baseline"] == 1)))
print(sum((criteria["evolution_utility_baseline"] == 1) & (criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1)))

    
### WELFARE With policies


df_welfare = pd.DataFrame(columns = ['City'])
df_welfare['City'] = df_cobenefits['City'].iloc[0:192]
df_baseline = df_cobenefits.loc[df_cobenefits.Pol == 'None', ['City', 'welfare_with_cobenefits', 'emissions']]
df_baseline.columns = ['City', 'welfare_with_cobenefits_baseline', 'emissions_baseline']

for policy in ['UGB', 'carbon_tax', 'fuel_efficiency', 'BRT']:
    df_policy = df_cobenefits.loc[df_cobenefits.Pol == policy, ['City', 'welfare_with_cobenefits']]
    df_policy = df_policy.merge(df_baseline, on = 'City')
    df_policy[policy] = df_policy.welfare_with_cobenefits / df_policy.welfare_with_cobenefits_baseline
    df_policy = df_policy.loc[:, ['City', policy]]
    df_welfare = df_welfare.merge(df_policy, on = 'City')
 

criteria["welfare_policies"] = 1
for city in list(criteria.index):
    if ((df_welfare.loc[baseline_2035["City"] == city, "UGB"].squeeze() < 0.1) | (df_welfare.loc[baseline_2035["City"] == city, "UGB"].squeeze() > 3)):
        criteria.loc[city, "welfare_policies"] = 0
    if ((df_welfare.loc[baseline_2035["City"] == city, "carbon_tax"].squeeze() < 0.1) | (df_welfare.loc[baseline_2035["City"] == city, "carbon_tax"].squeeze() > 3)):
        criteria.loc[city, "welfare_policies"] = 0
    if ((df_welfare.loc[baseline_2035["City"] == city, "fuel_efficiency"].squeeze() < 0.1) | (df_welfare.loc[baseline_2035["City"] == city, "fuel_efficiency"].squeeze() > 3)):
        criteria.loc[city, "welfare_policies"] = 0
    if ((df_welfare.loc[baseline_2035["City"] == city, "BRT"].squeeze() < 0.1) | (df_welfare.loc[baseline_2035["City"] == city, "BRT"].squeeze() > 3)):
        criteria.loc[city, "welfare_policies"] = 0

print(sum((criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1)))
print(sum(criteria["welfare_policies"] == 1))    
print(sum((criteria["welfare_policies"] == 1) & (criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1)))
   
### EMISSIOS With policies


df_emissions = pd.DataFrame(columns = ['City'])
df_emissions['City'] = df_cobenefits['City'].iloc[0:192]
df_baseline = df_cobenefits.loc[df_cobenefits.Pol == 'None', ['City', 'welfare_with_cobenefits', 'emissions']]
df_baseline.columns = ['City', 'welfare_with_cobenefits_baseline', 'emissions_baseline']

for policy in ['UGB', 'carbon_tax', 'fuel_efficiency', 'BRT']:
    df_policy = df_cobenefits.loc[df_cobenefits.Pol == policy, ['City', 'emissions']]
    df_policy = df_policy.merge(df_baseline, on = 'City')
    df_policy[policy] = df_policy.emissions / df_policy.emissions_baseline
    df_policy = df_policy.loc[:, ['City', policy]]
    df_emissions = df_emissions.merge(df_policy, on = 'City')
 

criteria["emissions_policies"] = 1
for city in list(criteria.index):
    if ((df_welfare.loc[baseline_2035["City"] == city, "UGB"].squeeze() < 0.1) | (df_welfare.loc[baseline_2035["City"] == city, "UGB"].squeeze() > 3)):
        criteria.loc[city, "emissions_policies"] = 0
    if ((df_welfare.loc[baseline_2035["City"] == city, "carbon_tax"].squeeze() < 0.1) | (df_welfare.loc[baseline_2035["City"] == city, "carbon_tax"].squeeze() > 3)):
        criteria.loc[city, "emissions_policies"] = 0
    if ((df_welfare.loc[baseline_2035["City"] == city, "fuel_efficiency"].squeeze() < 0.1) | (df_welfare.loc[baseline_2035["City"] == city, "fuel_efficiency"].squeeze() > 3)):
        criteria.loc[city, "emissions_policies"] = 0
    if ((df_welfare.loc[baseline_2035["City"] == city, "BRT"].squeeze() < 0.1) | (df_welfare.loc[baseline_2035["City"] == city, "BRT"].squeeze() > 3)):
        criteria.loc[city, "emissions_policies"] = 0

print(sum((criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1)))
print(sum(criteria["emissions_policies"] == 1))    
print(sum((criteria["emissions_policies"] == 1) & (criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1)))

print(criteria.City[((criteria["emissions_policies"] == 0) & (criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1))])
print(criteria.City[((criteria["welfare_policies"] == 0) & (criteria["validation_density"] == 1) & (criteria["validation_rent"] == 1) & (criteria["share_housing"] == 1))])

print(criteria.City[((criteria["validation_density"] == 0) & (criteria["share_housing"] == 1))])
 print(criteria.City[((criteria["validation_rent"] == 0) & (criteria["share_housing"] == 1))])
     
print(sum(criteria["share_housing"] == 1))   
    
    
    
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
    