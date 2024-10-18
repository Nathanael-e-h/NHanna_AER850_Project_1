"""
Created on Thu Oct  3 12:31:53 2024

@author: Engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###############################
# STEP 1: Data Processing
###############################
df = pd.read_csv("Project_1_Data.csv")
print(df.info())


###############################
# STEP 2: Data Visualization
###############################
# do a 3d plot here

# color code points based on step 1-13

X_min, X_max = df['X'].min(), df['X'].max()
Y_min, Y_max = df['Y'].min(), df['Y'].max()
Z_min, Z_max = df['Z'].min(), df['Z'].max()

DataVivPicture = plt.figure()
ax = plt.axes(projection ='3d')

# Plot the points
ax.scatter(df['X'], df['Y'], df['Y'])

# Set axis limits
ax.set_xlim(X_min, X_max)
ax.set_ylim(Y_min, Y_max)
ax.set_zlim(Z_min, Z_max)


###############################
# STEP 3: 
###############################
# is the correlation graph


###############################
# STEP 4: 
###############################
# 3 with grid search, one with randomized
"""Data Splitting Hell Yeah"""
# Stratified Sampling (strongly recommended by the professor)


# REPORT: intro, step step step, plots, explanation, conclusion w/ github URL