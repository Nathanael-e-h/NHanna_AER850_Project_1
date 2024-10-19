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
DataVivPicture = plt.figure()
ax = plt.axes(projection ='3d')

# Plot the points, w/ color coding
plot = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'])
color_bar = plt.colorbar(plot, label='Steps')

# Set axis limits (I'm leaving this out unless I discover it's necessary, which it doesn't seem to be)
# X_min, X_max = df['X'].min(), df['X'].max()
# Y_min, Y_max = df['Y'].min(), df['Y'].max()
# Z_min, Z_max = df['Z'].min(), df['Z'].max()
# ax.set_xlim(X_min, X_max)
# ax.set_ylim(Y_min, Y_max)
# ax.set_zlim(Z_min, Z_max)

# Statistical Analysis
print(df.describe())

# Create histogram
DataHistPicture = plt.figure()
plt.hist(df.iloc[:, 3], bins=13, edgecolor='black', color='purple', align='mid', rwidth=0.8)
plt.xticks(range(1, 14))
plt.title('Frequency of Points per Step')
plt.xlabel('Step')
plt.ylabel('Frequency')



###############################
# STEP 3: Correlation Analysis 
###############################
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Correlation Matrix - Steps & Axes')


###############################
# STEP 4: 
###############################
# 3 with grid search, one with randomized
"""Data Splitting Hell Yeah"""
# Stratified Sampling (strongly recommended by the professor)


# REPORT: intro, step step step, plots, explanation, conclusion w/ github URL