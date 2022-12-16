# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:36:55 2022

@author: charl
"""

density/simul_density
rent/simul_rent

plt.scatter(grille.XCOORD, grille.YCOORD, c = density, cmap='RdBu_r')
plt.colorbar()
#plt.clim(0, 8000) 
plt.axis('off')

plt.scatter(grille.XCOORD, grille.YCOORD, c = simul_density, cmap='RdBu_r')
plt.colorbar()
#plt.clim(0, 8000) 
plt.axis('off')

plt.scatter(grille.XCOORD, grille.YCOORD, c = rent, cmap='RdYlBu_r')
plt.colorbar()
plt.clim(0, 350) 
plt.axis('off')

plt.scatter(grille.XCOORD, grille.YCOORD, c = simul_rent, cmap='RdYlBu_r')
plt.colorbar()
plt.clim(0, 350) 
plt.axis('off')
