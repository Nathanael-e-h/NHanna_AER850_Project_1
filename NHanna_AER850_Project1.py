"""
Created on Thu Oct  3 12:31:53 2024

@author: Engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Project_1_Data.csv")
print(df.info())

"""Data Splitting Hell Yeah"""
#Stratified Sampling (strongly recommended by the professor)
df["income_categories"] = pd.cut(df["median_income"],
                          bins=[0, 2, 4, 6, np.inf],
                          labels=[1, 2, 3, 4])
my_splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)
for train_index, test_index in my_splitter.split(df, df["income_categories"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
strat_df_train = strat_df_train.drop(columns=["income_categories"], axis = 1)
strat_df_test = strat_df_test.drop(columns=["income_categories"], axis = 1)
