# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:36:31 2021

@author: charl
"""

#### HEAT ###

RR = 0.89

local_volume = 45*7
reference_volume = 168

pop = 25000
modal_share = 0.5
avg_distance = 2
avg_speed = 5.3

life_exp = 413.2051 / 100000


death_avoided_per_year = (local_volume / reference_volume) * (1 - RR) * life_exp * pop

### Regarder comment ça marche pour les modal shares + distance
### life29 expectancy data ?

def death_avoided_per_year(density, mode_choice, distance_cbd, WALKING_SPEED, mortality_rate, pop):
    pop = sum(density[mode_choice == 2])
    avg_distance = np.nansum(distance_cbd[mode_choice == 2] * density[mode_choice == 2]) / pop
    reference_volume = 168
    RR = 0.89
    return (60 * (avg_distance / WALKING_SPEED) / reference_volume) * (1 - RR) * mortality_rate * pop
