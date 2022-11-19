# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:04:48 2022

@author: charl
"""

df["quartiles_CT"] = "Q0"
df["quartiles_CT"][df["cost_effectiveness_with_cobenefits_CT"] <  np.quantile(df["cost_effectiveness_with_cobenefits_CT"], 0.25)] = "Q4"
df["quartiles_CT"][(df["cost_effectiveness_with_cobenefits_CT"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_CT"], 0.25)) &  (df["cost_effectiveness_with_cobenefits_CT"] <  np.quantile(df["cost_effectiveness_with_cobenefits_CT"], 0.5))] = "Q3"
df["quartiles_CT"][(df["cost_effectiveness_with_cobenefits_CT"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_CT"], 0.5)) &  (df["cost_effectiveness_with_cobenefits_CT"] <  np.quantile(df["cost_effectiveness_with_cobenefits_CT"], 0.75))] = "Q2"
df["quartiles_CT"][df["cost_effectiveness_with_cobenefits_CT"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_CT"], 0.75)] = "Q1"


tidy_CT = pd.DataFrame(columns = ["Quartile", "Variable", "Value"])
tidy_CT["Quartile"] = np.array(["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"])
tidy_CT["Variable"] = np.array(["Average Emissions", "Average Emissions", "Average Emissions", "Average Emissions", "Average Welfare", "Average Welfare", "Average Welfare", "Average Welfare"])
tidy_CT["Value"] = np.array([np.nansum(df.emissions_2035_CT_var[df["quartiles_CT"] == "Q1"] * df.population_2035[df["quartiles_CT"] == "Q1"]) / np.nansum(df.population_2035[df["quartiles_CT"] == "Q1"]),
                                np.nansum(df.emissions_2035_CT_var[df["quartiles_CT"] == "Q2"] * df.population_2035[df["quartiles_CT"] == "Q2"]) / np.nansum(df.population_2035[df["quartiles_CT"] == "Q2"]),
                                np.nansum(df.emissions_2035_CT_var[df["quartiles_CT"] == "Q3"] * df.population_2035[df["quartiles_CT"] == "Q3"]) / np.nansum(df.population_2035[df["quartiles_CT"] == "Q3"]),
                                np.nansum(df.emissions_2035_CT_var[df["quartiles_CT"] == "Q4"] * df.population_2035[df["quartiles_CT"] == "Q4"]) / np.nansum(df.population_2035[df["quartiles_CT"] == "Q4"]),
                                np.nansum(df.welfare_2035_CT_var[df["quartiles_CT"] == "Q1"] * df.population_2035[df["quartiles_CT"] == "Q1"]) / np.nansum(df.population_2035[df["quartiles_CT"] == "Q1"]),
                                np.nansum(df.welfare_2035_CT_var[df["quartiles_CT"] == "Q2"] * df.population_2035[df["quartiles_CT"] == "Q2"]) / np.nansum(df.population_2035[df["quartiles_CT"] == "Q2"]),
                                np.nansum(df.welfare_2035_CT_var[df["quartiles_CT"] == "Q3"] * df.population_2035[df["quartiles_CT"] == "Q3"]) / np.nansum(df.population_2035[df["quartiles_CT"] == "Q3"]),
                                np.nansum(df.welfare_2035_CT_var[df["quartiles_CT"] == "Q4"] * df.population_2035[df["quartiles_CT"] == "Q4"]) / np.nansum(df.population_2035[df["quartiles_CT"] == "Q4"])])

plt.rcParams['figure.dpi'] = 360
plt.rc('font', weight='normal')
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,2))
sns.barplot(data=tidy_CT, x="Quartile", y="Value", hue = "Variable", palette=['#fc8d62', '#66c2a5'])#("Set2"))
plt.xlabel('')
plt.ylabel('')
plt.yticks([], [])
plt.ylim(-16, 4)
handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.45, 0.35), fontsize = 10, loc=2, borderaxespad=0.)#,  prop = {'weight':'bold'})
plt.legend().remove()
ax.tick_params(axis = 'both', labelsize=10, color='#4f4e4e')
sns.despine(left=True, top = True)
plt.text(x=0.2, y=1.5, s='+' +str(round(tidy_CT.Value[(tidy_CT.Quartile == "Q1") & (tidy_CT.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.2, y=0.7, s='+' +str(round(tidy_CT.Value[(tidy_CT.Quartile == "Q2") & (tidy_CT.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.2, y=0.7, s='+' +str(round(tidy_CT.Value[(tidy_CT.Quartile == "Q3") & (tidy_CT.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=3.2, y=0.7, s='+' +str(round(tidy_CT.Value[(tidy_CT.Quartile == "Q4") & (tidy_CT.Variable == "Average Welfare")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=-0.2, y=-2, s=str(round(tidy_CT.Value[(tidy_CT.Quartile == "Q1") & (tidy_CT.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=0.8, y=-2, s=str(round(tidy_CT.Value[(tidy_CT.Quartile == "Q2") & (tidy_CT.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.8, y=-2, s=str(round(tidy_CT.Value[(tidy_CT.Quartile == "Q3") & (tidy_CT.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.8, y=-2.5, s=str(round(tidy_CT.Value[(tidy_CT.Quartile == "Q4") & (tidy_CT.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')


df["quartiles_FE"] = "Q0"
df["quartiles_FE"][df["cost_effectiveness_with_cobenefits_FE"] <  np.quantile(df["cost_effectiveness_with_cobenefits_FE"], 0.25)] = "Q4"
df["quartiles_FE"][(df["cost_effectiveness_with_cobenefits_FE"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_FE"], 0.25)) &  (df["cost_effectiveness_with_cobenefits_FE"] <  np.quantile(df["cost_effectiveness_with_cobenefits_FE"], 0.5))] = "Q3"
df["quartiles_FE"][(df["cost_effectiveness_with_cobenefits_FE"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_FE"], 0.5)) &  (df["cost_effectiveness_with_cobenefits_FE"] <  np.quantile(df["cost_effectiveness_with_cobenefits_FE"], 0.75))] = "Q2"
df["quartiles_FE"][df["cost_effectiveness_with_cobenefits_FE"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_FE"], 0.75)] = "Q1"


tidy_FE = pd.DataFrame(columns = ["Quartile", "Variable", "Value"])
tidy_FE["Quartile"] = np.array(["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"])
tidy_FE["Variable"] = np.array(["Average Emissions", "Average Emissions", "Average Emissions", "Average Emissions", "Average Welfare", "Average Welfare", "Average Welfare", "Average Welfare"])
tidy_FE["Value"] = np.array([np.nansum(df.emissions_2035_FE_var[df["quartiles_FE"] == "Q1"] * df.population_2035[df["quartiles_FE"] == "Q1"]) / np.nansum(df.population_2035[df["quartiles_FE"] == "Q1"]),
                                np.nansum(df.emissions_2035_FE_var[df["quartiles_FE"] == "Q2"] * df.population_2035[df["quartiles_FE"] == "Q2"]) / np.nansum(df.population_2035[df["quartiles_FE"] == "Q2"]),
                                np.nansum(df.emissions_2035_FE_var[df["quartiles_FE"] == "Q3"] * df.population_2035[df["quartiles_FE"] == "Q3"]) / np.nansum(df.population_2035[df["quartiles_FE"] == "Q3"]),
                                np.nansum(df.emissions_2035_FE_var[df["quartiles_FE"] == "Q4"] * df.population_2035[df["quartiles_FE"] == "Q4"]) / np.nansum(df.population_2035[df["quartiles_FE"] == "Q4"]),
                                np.nansum(df.welfare_2035_FE_var[df["quartiles_FE"] == "Q1"] * df.population_2035[df["quartiles_FE"] == "Q1"]) / np.nansum(df.population_2035[df["quartiles_FE"] == "Q1"]),
                                np.nansum(df.welfare_2035_FE_var[df["quartiles_FE"] == "Q2"] * df.population_2035[df["quartiles_FE"] == "Q2"]) / np.nansum(df.population_2035[df["quartiles_FE"] == "Q2"]),
                                np.nansum(df.welfare_2035_FE_var[df["quartiles_FE"] == "Q3"] * df.population_2035[df["quartiles_FE"] == "Q3"]) / np.nansum(df.population_2035[df["quartiles_FE"] == "Q3"]),
                                np.nansum(df.welfare_2035_FE_var[df["quartiles_FE"] == "Q4"] * df.population_2035[df["quartiles_FE"] == "Q4"]) / np.nansum(df.population_2035[df["quartiles_FE"] == "Q4"])])

plt.rcParams['figure.dpi'] = 360
plt.rc('font', weight='normal')
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,2))
sns.barplot(data=tidy_FE, x="Quartile", y="Value", hue = "Variable", palette=['#fc8d62', '#66c2a5'])#("Set2"))
plt.xlabel('')
plt.ylabel('')
plt.yticks([], [])
plt.ylim(-16, 4)
handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.45, 0.35), fontsize = 10, loc=2, borderaxespad=0.)#,  prop = {'weight':'bold'})
plt.legend().remove()
ax.tick_params(axis = 'both', labelsize=10, color='#4f4e4e')
sns.despine(left=True, top = True)
plt.text(x=0.2, y=0.7, s='+' +str(round(tidy_FE.Value[(tidy_FE.Quartile == "Q1") & (tidy_FE.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.2, y=0.7, s='+' +str(round(tidy_FE.Value[(tidy_FE.Quartile == "Q2") & (tidy_FE.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.2, y=0.7, s='+' +str(round(tidy_FE.Value[(tidy_FE.Quartile == "Q3") & (tidy_FE.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=3.2, y=0.7, s='+' +str(round(tidy_FE.Value[(tidy_FE.Quartile == "Q4") & (tidy_FE.Variable == "Average Welfare")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=-0.2, y=-2, s=str(round(tidy_FE.Value[(tidy_FE.Quartile == "Q1") & (tidy_FE.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=0.8, y=-2, s=str(round(tidy_FE.Value[(tidy_FE.Quartile == "Q2") & (tidy_FE.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.8, y=-2, s=str(round(tidy_FE.Value[(tidy_FE.Quartile == "Q3") & (tidy_FE.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.8, y=-2, s=str(round(tidy_FE.Value[(tidy_FE.Quartile == "Q4") & (tidy_FE.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')


df["quartiles_BRT"] = "Q0"
df["quartiles_BRT"][df["cost_effectiveness_with_cobenefits_BRT"] <  np.quantile(df["cost_effectiveness_with_cobenefits_BRT"], 0.25)] = "Q4"
df["quartiles_BRT"][(df["cost_effectiveness_with_cobenefits_BRT"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_BRT"], 0.25)) &  (df["cost_effectiveness_with_cobenefits_BRT"] <  np.quantile(df["cost_effectiveness_with_cobenefits_BRT"], 0.5))] = "Q3"
df["quartiles_BRT"][(df["cost_effectiveness_with_cobenefits_BRT"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_BRT"], 0.5)) &  (df["cost_effectiveness_with_cobenefits_BRT"] <  np.quantile(df["cost_effectiveness_with_cobenefits_BRT"], 0.75))] = "Q2"
df["quartiles_BRT"][df["cost_effectiveness_with_cobenefits_BRT"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_BRT"], 0.75)] = "Q1"


tidy_BRT = pd.DataFrame(columns = ["Quartile", "Variable", "Value"])
tidy_BRT["Quartile"] = np.array(["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"])
tidy_BRT["Variable"] = np.array(["Average Emissions", "Average Emissions", "Average Emissions", "Average Emissions", "Average Welfare", "Average Welfare", "Average Welfare", "Average Welfare"])
tidy_BRT["Value"] = np.array([np.nansum(df.emissions_2035_BRT_var[df["quartiles_BRT"] == "Q1"] * df.population_2035[df["quartiles_BRT"] == "Q1"]) / np.nansum(df.population_2035[df["quartiles_BRT"] == "Q1"]),
                                np.nansum(df.emissions_2035_BRT_var[df["quartiles_BRT"] == "Q2"] * df.population_2035[df["quartiles_BRT"] == "Q2"]) / np.nansum(df.population_2035[df["quartiles_BRT"] == "Q2"]),
                                np.nansum(df.emissions_2035_BRT_var[df["quartiles_BRT"] == "Q3"] * df.population_2035[df["quartiles_BRT"] == "Q3"]) / np.nansum(df.population_2035[df["quartiles_BRT"] == "Q3"]),
                                np.nansum(df.emissions_2035_BRT_var[df["quartiles_BRT"] == "Q4"] * df.population_2035[df["quartiles_BRT"] == "Q4"]) / np.nansum(df.population_2035[df["quartiles_BRT"] == "Q4"]),
                                np.nansum(df.welfare_2035_BRT_var[df["quartiles_BRT"] == "Q1"] * df.population_2035[df["quartiles_BRT"] == "Q1"]) / np.nansum(df.population_2035[df["quartiles_BRT"] == "Q1"]),
                                np.nansum(df.welfare_2035_BRT_var[df["quartiles_BRT"] == "Q2"] * df.population_2035[df["quartiles_BRT"] == "Q2"]) / np.nansum(df.population_2035[df["quartiles_BRT"] == "Q2"]),
                                np.nansum(df.welfare_2035_BRT_var[df["quartiles_BRT"] == "Q3"] * df.population_2035[df["quartiles_BRT"] == "Q3"]) / np.nansum(df.population_2035[df["quartiles_BRT"] == "Q3"]),
                                np.nansum(df.welfare_2035_BRT_var[df["quartiles_BRT"] == "Q4"] * df.population_2035[df["quartiles_BRT"] == "Q4"]) / np.nansum(df.population_2035[df["quartiles_BRT"] == "Q4"])])

plt.rcParams['figure.dpi'] = 360
plt.rc('font', weight='normal')
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,2))
sns.barplot(data=tidy_BRT, x="Quartile", y="Value", hue = "Variable", palette=['#fc8d62', '#66c2a5'])#("Set2"))
plt.xlabel('')
plt.ylabel('')
plt.yticks([], [])
plt.ylim(-16, 4)
handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.45, 0.35), fontsize = 10, loc=2, borderaxespad=0.)#,  prop = {'weight':'bold'})
plt.legend().remove()
ax.tick_params(axis = 'both', labelsize=10, color='#4f4e4e')
sns.despine(left=True, top = True)
plt.text(x=0.2, y=1, s='+' +str(round(tidy_BRT.Value[(tidy_BRT.Quartile == "Q1") & (tidy_BRT.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.2, y=0.7, s='+' +str(round(tidy_BRT.Value[(tidy_BRT.Quartile == "Q2") & (tidy_BRT.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.2, y=0.7, s=str(round(tidy_BRT.Value[(tidy_BRT.Quartile == "Q3") & (tidy_BRT.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=3.2, y=0.7, s=str(round(tidy_BRT.Value[(tidy_BRT.Quartile == "Q4") & (tidy_BRT.Variable == "Average Welfare")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=-0.2, y=-2, s=str(round(tidy_BRT.Value[(tidy_BRT.Quartile == "Q1") & (tidy_BRT.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=0.8, y=-2, s=str(round(tidy_BRT.Value[(tidy_BRT.Quartile == "Q2") & (tidy_BRT.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.8, y=-2, s=str(round(tidy_BRT.Value[(tidy_BRT.Quartile == "Q3") & (tidy_BRT.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.8, y=0.7, s=str(round(tidy_BRT.Value[(tidy_BRT.Quartile == "Q4") & (tidy_BRT.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')


df["quartiles_UGB"] = "Q0"
df["quartiles_UGB"][df["cost_effectiveness_with_cobenefits_UGB"] <  np.quantile(df["cost_effectiveness_with_cobenefits_UGB"], 0.25)] = "Q4"
df["quartiles_UGB"][(df["cost_effectiveness_with_cobenefits_UGB"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_UGB"], 0.25)) &  (df["cost_effectiveness_with_cobenefits_UGB"] <  np.quantile(df["cost_effectiveness_with_cobenefits_UGB"], 0.5))] = "Q3"
df["quartiles_UGB"][(df["cost_effectiveness_with_cobenefits_UGB"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_UGB"], 0.5)) &  (df["cost_effectiveness_with_cobenefits_UGB"] <  np.quantile(df["cost_effectiveness_with_cobenefits_UGB"], 0.75))] = "Q2"
df["quartiles_UGB"][df["cost_effectiveness_with_cobenefits_UGB"] >=  np.quantile(df["cost_effectiveness_with_cobenefits_UGB"], 0.75)] = "Q1"


tidy_UGB = pd.DataFrame(columns = ["Quartile", "Variable", "Value"])
tidy_UGB["Quartile"] = np.array(["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"])
tidy_UGB["Variable"] = np.array(["Average Emissions", "Average Emissions", "Average Emissions", "Average Emissions", "Average Welfare", "Average Welfare", "Average Welfare", "Average Welfare"])
tidy_UGB["Value"] = np.array([np.nansum(df.emissions_2035_UGB_var[df["quartiles_UGB"] == "Q1"] * df.population_2035[df["quartiles_UGB"] == "Q1"]) / np.nansum(df.population_2035[df["quartiles_UGB"] == "Q1"]),
                                np.nansum(df.emissions_2035_UGB_var[df["quartiles_UGB"] == "Q2"] * df.population_2035[df["quartiles_UGB"] == "Q2"]) / np.nansum(df.population_2035[df["quartiles_UGB"] == "Q2"]),
                                np.nansum(df.emissions_2035_UGB_var[df["quartiles_UGB"] == "Q3"] * df.population_2035[df["quartiles_UGB"] == "Q3"]) / np.nansum(df.population_2035[df["quartiles_UGB"] == "Q3"]),
                                np.nansum(df.emissions_2035_UGB_var[df["quartiles_UGB"] == "Q4"] * df.population_2035[df["quartiles_UGB"] == "Q4"]) / np.nansum(df.population_2035[df["quartiles_UGB"] == "Q4"]),
                                np.nansum(df.welfare_2035_UGB_var[df["quartiles_UGB"] == "Q1"] * df.population_2035[df["quartiles_UGB"] == "Q1"]) / np.nansum(df.population_2035[df["quartiles_UGB"] == "Q1"]),
                                np.nansum(df.welfare_2035_UGB_var[df["quartiles_UGB"] == "Q2"] * df.population_2035[df["quartiles_UGB"] == "Q2"]) / np.nansum(df.population_2035[df["quartiles_UGB"] == "Q2"]),
                                np.nansum(df.welfare_2035_UGB_var[df["quartiles_UGB"] == "Q3"] * df.population_2035[df["quartiles_UGB"] == "Q3"]) / np.nansum(df.population_2035[df["quartiles_UGB"] == "Q3"]),
                                np.nansum(df.welfare_2035_UGB_var[df["quartiles_UGB"] == "Q4"] * df.population_2035[df["quartiles_UGB"] == "Q4"]) / np.nansum(df.population_2035[df["quartiles_UGB"] == "Q4"])])

plt.rcParams['figure.dpi'] = 360
plt.rc('font', weight='normal')
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6,2))
sns.barplot(data=tidy_UGB, x="Quartile", y="Value", hue = "Variable", palette=['#fc8d62', '#66c2a5'])#("Set2"))
plt.xlabel('')
plt.ylabel('')
plt.yticks([], [])
plt.ylim(-16, 4)
handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles[0:], labels=labels[0:], title = '', bbox_to_anchor=(0.45, 0.35), fontsize = 10, loc=2, borderaxespad=0.)#,  prop = {'weight':'bold'})
plt.legend().remove()
ax.tick_params(axis = 'both', labelsize=10, color='#4f4e4e')
sns.despine(left=True, top = True)
plt.text(x=0.2, y=0.7, s='+' +str(round(tidy_UGB.Value[(tidy_UGB.Quartile == "Q1") & (tidy_UGB.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.2, y=-2, s=str(round(tidy_UGB.Value[(tidy_UGB.Quartile == "Q2") & (tidy_UGB.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.2, y=-2, s=str(round(tidy_UGB.Value[(tidy_UGB.Quartile == "Q3") & (tidy_UGB.Variable == "Average Welfare")].squeeze(), 1))+ "%", 
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=3.2, y=-2, s=str(round(tidy_UGB.Value[(tidy_UGB.Quartile == "Q4") & (tidy_UGB.Variable == "Average Welfare")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=-0.2, y=-2, s=str(round(tidy_UGB.Value[(tidy_UGB.Quartile == "Q1") & (tidy_UGB.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=0.8, y=-2, s=str(round(tidy_UGB.Value[(tidy_UGB.Quartile == "Q2") & (tidy_UGB.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=1.8, y=-2, s=str(round(tidy_UGB.Value[(tidy_UGB.Quartile == "Q3") & (tidy_UGB.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')
plt.text(x=2.8, y=-2, s=str(round(tidy_UGB.Value[(tidy_UGB.Quartile == "Q4") & (tidy_UGB.Variable == "Average Emissions")].squeeze(), 1))+ "%",
                 color='black', fontsize=10, horizontalalignment='center')

