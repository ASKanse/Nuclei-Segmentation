# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:16:47 2020

pre-processing of CoNSeP data

refferences : https://stackoverflow.com/questions/58377015/counterclockwise-sorting-of-x-y-data

@author: Abhiraj
"""

import scipy.io as io
import numpy as np
import os
import math
import json

def sort_xy(x, y):

    x0 = np.mean(x)
    y0 = np.mean(y)

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))

    mask = np.argsort(angles)

    x_sorted = x[mask]
    y_sorted = y[mask]

    return [x_sorted, y_sorted]
            
d1 = './new_train/label'
d2 = './new_test/label'

dir = {}

filenames = os.listdir(d2)
for file in filenames:
    dir[file] = {}
    dir[file]["filename"] = file[:-3]+'png'
    dir[file]["regions"] = {}
    mat = io.loadmat(os.path.join(d2,file))
    conts = np.zeros_like(mat['inst_map'])
    for j in range(mat['inst_map'].shape[0]):
        for k in range(mat['inst_map'].shape[1]):
            if mat['inst_map'][j,k] != 0:
                tl = [max(j-1,0),max(k-1,0)]
                tm = [max(j-1,0),k]
                tr = [max(j-1,0),min(k+1,249)] #change this with size
                ml = [j,max(k-1,0)]
                mr = [j,min(k+1,249)] #change this with size
                bl = [min(j+1,249),max(k-1,0)] #change this with size
                bm = [min(j+1,249),k] #change this with size
                br = [min(j+1,249),min(k+1,249)] #change this with size
                if (mat['inst_map'][tm[0],tm[1]] != mat['inst_map'][j,k] or
                    mat['inst_map'][ml[0],ml[1]] != mat['inst_map'][j,k] or 
                    mat['inst_map'][mr[0],mr[1]] != mat['inst_map'][j,k] or
                    mat['inst_map'][bm[0],bm[1]] != mat['inst_map'][j,k]):
                    conts[j,k] = mat['inst_map'][j,k]
    val = np.zeros_like(mat['inst_map'])
    for j in range(mat['inst_map'].shape[0]):
        for k in range(mat['inst_map'].shape[1]):
            if conts[j,k] != 0 and val[j,k] == 0:
                dir[file]['regions'][str(int(conts[j,k]))] = {}
                dir[file]['regions'][str(int(conts[j,k]))]["shape_attributes"] = {}
                dir[file]["regions"][str(int(conts[j,k]))]["region_attributes"] = {}
                contours = np.argwhere(conts == (conts[j,k])) #returns all_x and all_y
                for idx in contours:
                    val[idx[0],idx[1]] = 1
                for i in range(len(contours)): 
                    contours[i] = np.flipud(contours[i])
                sort_contours = np.array(contours, dtype='int16')
                all_x = sort_contours[:,0]
                all_y = sort_contours[:,1]
                [all_x, all_y] = sort_xy(all_x, all_y)
                all_x = all_x.tolist()
                all_y = all_y.tolist()
                all_x.append(all_x[0])
                all_y.append(all_y[0])
                dir[file]["regions"][str(int(conts[j,k]))]["shape_attributes"]["all_points_x"] = all_x
                dir[file]["regions"][str(int(conts[j,k]))]["shape_attributes"]["all_points_y"] = all_y
                #dir[file]["regions"][str(int(conts[j,k]))]["shape_attributes"]["category_id"] = int(mat['type_map'][j,k])-1
                dir[file]["regions"][str(int(conts[j,k]))]["shape_attributes"]["category_id"] = 0
    print('completeted ',file)
    
                
with open('new_test_regions.json', 'w') as fp:
    json.dump(dir, fp) 
