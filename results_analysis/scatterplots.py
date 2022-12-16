# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:58:01 2022

@author: charl
"""

df.to_excel('C:/Users/charl/OneDrive/Bureau/df_temp.xlsx')

plt.scatter(df.emissions_2035_CT_var, df.welfare_2035_CT_var)
plt.scatter(df.emissions_2035_FE_var, df.welfare_2035_FE_var)
plt.scatter(df.emissions_2035_UGB_var, df.welfare_2035_UGB_var)
plt.scatter(df.emissions_2035_BRT_var, df.welfare_2035_BRT_var)

plt.scatter(df.emissions_2035_CT_var, df.welfare_2035_CT_var_without)
plt.scatter(df.emissions_2035_FE_var, df.welfare_2035_FE_var_without)
plt.scatter(df.emissions_2035_UGB_var, df.welfare_2035_UGB_var_without)
plt.scatter(df.emissions_2035_BRT_var, df.welfare_2035_BRT_var_without)




df.loc[:,['City', 'population_2035', 'emissions_2035_CT_var', 'emissions_2035_FE_var', 'emissions_2035_UGB_var', 'emissions_2035_BRT_var',
          'welfare_2035_CT_var', 'welfare_2035_FE_var', 'welfare_2035_UGB_var', 'welfare_2035_BRT_var',
          'welfare_2035_CT_var_without', 'welfare_2035_FE_var_without', 'welfare_2035_UGB_var_without', 'welfare_2035_BRT_var_without',
          'cost_effectiveness_without_cobenefits_CT', 'cost_effectiveness_with_cobenefits_CT',
          'cost_effectiveness_without_cobenefits_FE', 'cost_effectiveness_with_cobenefits_FE',
          'cost_effectiveness_without_cobenefits_UGB', 'cost_effectiveness_with_cobenefits_UGB',
          'cost_effectiveness_without_cobenefits_BRT', 'cost_effectiveness_with_cobenefits_BRT']].to_excel('C:/Users/charl/OneDrive/Bureau/results_CT_FE_UGB_BRT.xlsx')

plt.figure(figsize = (15, 10))
plt.rcParams.update({'font.size': 20})
colors = list(df['Continent'].unique())
for i in range(0 , len(colors)):
    data = df.loc[df['Continent'] == colors[i]]
    plt.scatter(data.welfare_2035_UGB_var, data.emissions_2035_UGB_var, color=data.Continent.map(color_tab), label=colors[i], s = 100)
plt.xlabel("Welfare variation (with cobenefits)", size = 20)
plt.ylabel("Emissions variation", size = 20)
plt.legend()
plt.title("UGB - with health cobenefits")