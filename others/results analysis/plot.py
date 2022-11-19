# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:05:36 2021

@author: charl
"""

def map2D(grid, x):
    plt.scatter(grid.XCOORD, grid.YCOORD, s = None, c = x, marker='.', cmap=plt.cm.RdYlGn)
    cbar = plt.colorbar()  
    plt.show()

def map3D(grid, x):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    ax.scatter(grid.XCOORD, grid.YCOORD, x, c = x, alpha = 0.2, marker='.')    
    ax.set_xlabel('coord_X')
    ax.set_ylabel('coord_Y')
    ax.set_zlabel('Value')    
    plt.show()