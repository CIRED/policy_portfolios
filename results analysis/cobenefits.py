# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:06:30 2021

@author: charl
"""

df_cobenefits = pd.read_excel('C:/Users/charl/OneDrive/Bureau/cobenefits_policy.xlsx')

### Cities where the welfare is decreasing due to the policies?
for policy in ['UGB', 'carbon_tax', 'fuel_efficiency', 'BRT']:
    for welfare_type in ['welfare_with_cobenefits', 'welfare_without_cobenefits']:
        welfare_database = df_cobenefits.loc[:, ['City', 'Pol', welfare_type]]
        array = 100 * (np.array(welfare_database.loc[welfare_database.Pol == policy, welfare_type]) - np.array(welfare_database.loc[welfare_database.Pol == 'None', welfare_type])) / np.array(welfare_database.loc[welfare_database.Pol == 'None', welfare_type])
        print(policy, "policy:", sum(array > 0), 'where the welfare is increasing and', sum(array < 0), 'cities where the welfare is decreasing (welfare type:', welfare_type, ').')

### Comparing welfare with and without cobenefits
for policy in ['None', 'UGB', 'carbon_tax', 'fuel_efficiency', 'BRT']:
    welfare_database = df_cobenefits.loc[df_cobenefits.Pol == policy, ['City', 'Pol', 'welfare_with_cobenefits', 'welfare_without_cobenefits']]
    print(policy, "policy:")
    print("Accounting for cobenefits increases the welfare in", sum(welfare_database.welfare_with_cobenefits > welfare_database.welfare_without_cobenefits), 'cities')
    print("Accounting for cobenefits decreases the welfare in", sum(welfare_database.welfare_with_cobenefits < welfare_database.welfare_without_cobenefits), 'cities')

### In which city is it more or less easy to reduce emissions?

df_cost_effectiveness = pd.DataFrame(columns = ['City'])
df_cost_effectiveness['City'] = df_cobenefits['City'].iloc[0:192]
df_baseline = df_cobenefits.loc[df_cobenefits.Pol == 'None', ['City', 'welfare_with_cobenefits', 'emissions']]
df_baseline.columns = ['City', 'welfare_with_cobenefits_baseline', 'emissions_baseline']

for policy in ['UGB', 'carbon_tax', 'fuel_efficiency', 'BRT']:
    df_policy = df_cobenefits.loc[df_cobenefits.Pol == policy, ['City', 'welfare_with_cobenefits', 'emissions']]
    df_policy = df_policy.merge(df_baseline, on = 'City')
    df_policy["variation_welfare"] = df_policy.welfare_with_cobenefits / df_policy.welfare_with_cobenefits_baseline
    df_policy["variation_emissions"] = df_policy.emissions / df_policy.emissions_baseline
    df_policy[policy] = df_policy["variation_emissions"] / df_policy["variation_welfare"]
    df_policy = df_policy.loc[:, ['City', policy]]
    df_cost_effectiveness = df_cost_effectiveness.merge(df_policy, on = 'City')

### 














### Detailed cobenefits analysis
for cobenefits in ['cost_air_pollution', 'cost_active_modes', 'cost_noise', 'cost_accidents']:
    
    print("COBENEFIT:", cobenefits)
    
    df_baseline = df_cobenefits.loc[df_cobenefits.Pol == 'None', ['City', cobenefits]]
    df_UGB = df_cobenefits.loc[df_cobenefits.Pol == 'UGB', ['City', cobenefits]]
    df_BRT = df_cobenefits.loc[df_cobenefits.Pol == 'BRT', ['City', cobenefits]]
    df_carbon_tax = df_cobenefits.loc[df_cobenefits.Pol == 'carbon_tax', ['City', cobenefits]]
    df_fuel_efficiency = df_cobenefits.loc[df_cobenefits.Pol == 'fuel_efficiency', ['City', cobenefits]]

    df_baseline.columns = ['City', 'baseline']
    df_UGB.columns = ['City', 'UGB']
    df_BRT.columns = ['City', 'BRT']
    df_carbon_tax.columns = ['City', 'carbon_tax']
    df_fuel_efficiency.columns = ['City', 'fuel_efficiency']

    df = df_baseline.merge(df_UGB, on = 'City')
    df = df.merge(df_BRT, on = 'City')
    df = df.merge(df_carbon_tax, on = 'City')
    df = df.merge(df_fuel_efficiency, on = 'City')

    print('UGB impact', np.nanmax(100 * (df.UGB - df.baseline) / df.baseline), '%')
    print('BRT impact', np.nanmax(100 * (df.BRT - df.baseline) / df.baseline), '%')
    print('Carbon tax impact', np.nanmax(100 * (df.carbon_tax - df.baseline) / df.baseline), '%')
    print('Fuel efficiency impact', np.nanmax(100 * (df.fuel_efficiency - df.baseline) / df.baseline), '%')