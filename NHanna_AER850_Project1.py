import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


##########################################################
# STEP 1: Data Processing
##########################################################
df = pd.read_csv("Project_1_Data.csv")
print(df.info())


##########################################################
# STEP 2: Data Visualization
##########################################################
DataVivPicture = plt.figure()
ax = plt.axes(projection ='3d')

# Plot the points, w/ color coding
plot = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'])
color_bar = plt.colorbar(plot, label='Step')

# Statistical Analysis
print(df.describe())

# Create histogram
DataHistPicture = plt.figure()
plt.hist(df.iloc[:, 3], bins=13, edgecolor='black', color='purple', align='mid', rwidth=0.8)
plt.xticks(range(1, 14))
plt.title('Frequency of Points per Step')
plt.xlabel('Step')
plt.ylabel('Frequency')



##########################################################
# STEP 3: Correlation Analysis 
##########################################################
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Correlation Matrix - Steps & Axes')


##########################################################
# STEP 4: Classification Model Development/Engineering 
##########################################################

# Stratified Sampling (strongly recommended by the Dr Reza)
# Using DataIn and DataOut since I can't really use X and Y lmao
DataIn = df[['X', 'Y', 'Z']]
DataOut = df['Step']
DataIn_train, DataIn_test, DataOut_train, DataOut_test = train_test_split(DataIn, DataOut, test_size=0.4, random_state=212291, stratify=DataOut)

##### Model 1: Decision Tree

# Define the model
dt = DecisionTreeClassifier(random_state=212291)

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}

# Grid Search
grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search_dt.fit(DataIn_train, DataOut_train)

# Debugging
print("\nDecision Tree:")
print(f'Best parameters: {grid_search_dt.best_params_}')
print(f'Best score: {grid_search_dt.best_score_}')
print()

##### Model 2: Random Forest

# Define the model
rf = RandomForestClassifier(random_state=212291)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5]
}

# Grid Search
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search_rf.fit(DataIn_train, DataOut_train)

# Best parameters and score
print("Random Forest:")
print(f'Best parameters: {grid_search_rf.best_params_}')
print(f'Best score: {grid_search_rf.best_score_}')
print()


##### Model 3: Gradient Boosting Machine

# Define the model
gbm1 = GradientBoostingClassifier(random_state=212291)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.9, 1.0]
}

# Grid Search
grid_search_gbm1 = GridSearchCV(estimator=gbm1, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search_gbm1.fit(DataIn_train, DataOut_train)

# Best parameters and score
print("Gradient Boosting Machine:")
print(f'Best parameters: {grid_search_gbm1.best_params_}')
print(f'Best score: {grid_search_gbm1.best_score_}')
print()


##### Model 4: RandomizedSearchCV (model is Decision Tree again)

# Define the parameter distributions
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Define the model
dt2 = DecisionTreeClassifier(random_state=212291)

# Randomized Search
random_search_dt2 = RandomizedSearchCV(dt2, param_distributions=param_dist, n_iter=20, cv=3, random_state=212291, n_jobs=-1, scoring='accuracy')

# Fit the RandomizedSearchCV
random_search_dt2.fit(DataIn_train, DataOut_train)

# Best parameters and score
print("Randomized Search Decision Tree:")
print(f'Best parameters: {random_search_dt2.best_params_}')
print(f'Best score: {random_search_dt2.best_score_}')
print()

##########################################################
# STEP 5: Model Performance Analysis  
##########################################################

##### Model 1: Decision Tree
# Predict on the test set
DataOut_pred_dt = grid_search_dt.predict(DataIn_test)

# Calculate accuracy, precision, and F1 score
accuracy_dt = accuracy_score(DataOut_test, DataOut_pred_dt)
precision_dt = precision_score(DataOut_test, DataOut_pred_dt, average='weighted')
f1_dt = f1_score(DataOut_test, DataOut_pred_dt, average='weighted')

# Print the results
print("Decision Tree Model Performance:")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision (Weighted): {precision_dt:.4f}")
print(f"F1 Score (Weighted): {f1_dt:.4f}")


##### Model 2: Random Forest
# Predict on the test set
DataOut_pred_rf = grid_search_rf.predict(DataIn_test)

# Calculate accuracy, precision, and F1 score
accuracy_rf = accuracy_score(DataOut_test, DataOut_pred_rf)
precision_rf = precision_score(DataOut_test, DataOut_pred_rf, average='weighted')
f1_rf = f1_score(DataOut_test, DataOut_pred_rf, average='weighted')

# Print the results
print("RandomForest Model Performance:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision (Weighted): {precision_rf:.4f}")
print(f"F1 Score (Weighted): {f1_rf:.4f}")


##### Model 3: GBM
# Predict on the test set
DataOut_pred_gbm1 = grid_search_gbm1.predict(DataIn_test)

# Calculate accuracy, precision, and F1 score
accuracy_gbm1 = accuracy_score(DataOut_test, DataOut_pred_gbm1)
precision_gbm1 = precision_score(DataOut_test, DataOut_pred_gbm1, average='weighted')
f1_gbm1 = f1_score(DataOut_test, DataOut_pred_gbm1, average='weighted')

# Print the results
print("GBM Model Performance:")
print(f"Accuracy: {accuracy_gbm1:.4f}")
print(f"Precision (Weighted): {precision_gbm1:.4f}")
print(f"F1 Score (Weighted): {f1_gbm1:.4f}")


##### Model 4: RandomSearch
# Predict on the test set
DataOut_pred_dt2 = random_search_dt2.predict(DataIn_test)

# Calculate accuracy, precision, and F1 score
accuracy_dt2 = accuracy_score(DataOut_test, DataOut_pred_dt2)
precision_dt2 = precision_score(DataOut_test, DataOut_pred_dt2, average='weighted')
f1_dt2 = f1_score(DataOut_test, DataOut_pred_dt2, average='weighted')

# Print the results
print("Decision Tree Model (Randomsearch) Performance:")
print(f"Accuracy: {accuracy_dt2:.4f}")
print(f"Precision (Weighted): {precision_dt2:.4f}")
print(f"F1 Score (Weighted): {f1_dt2:.4f}")



### THE WINNER WAS THE GBM MODEL (it was the only model I could get to work reliably)
# Confusion Matrix
cm_gbm1 = confusion_matrix(DataOut_test, DataOut_pred_gbm1)
cm_gbm1_df = pd.DataFrame(cm_gbm1, index=np.unique(DataOut_test),
                          columns=np.unique(DataOut_pred_gbm1))
print("\nConfusion Matrix for GBM Model:")
print(cm_gbm1_df)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_gbm1, display_labels=np.unique(DataOut_test))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Gradient Boosting Machine Model')
plt.show()

##########################################################
# STEP 6: Stacked Model Performance Analysis 
##########################################################
# Create base models
base_models = [
    ('gbm', grid_search_gbm1.best_estimator_),  # Gradient Boosting Machine (dropping the 1 since I didn't end up doing two gbm models)
    ('dt', grid_search_dt.best_estimator_)      # Decision Tree ('dt', with no suffix; dt2 was unreliable)
]

# Create a StackingClassifier
stacked_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier())

# Fit the stacked model on the training data
stacked_model.fit(DataIn_train, DataOut_train)

# Make predictions on the test set
DataOut_pred_stacked = stacked_model.predict(DataIn_test)

# Calculate metrics
accuracy_stacked = accuracy_score(DataOut_test, DataOut_pred_stacked)
precision_stacked = precision_score(DataOut_test, DataOut_pred_stacked, average='weighted')
f1_stacked = f1_score(DataOut_test, DataOut_pred_stacked, average='weighted')

# Print the results
print("Stacked Model Performance:")
print(f"Accuracy: {accuracy_stacked:.4f}")
print(f"Precision (Weighted): {precision_stacked:.4f}")
print(f"F1 Score (Weighted): {f1_stacked:.4f}")

# Confusion Matrix
stackedMatrix = confusion_matrix(DataOut_test, DataOut_pred_stacked)
disp_stacked = ConfusionMatrixDisplay(confusion_matrix=stackedMatrix, display_labels=np.unique(DataOut_test))
disp_stacked.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Stacked Model')
plt.show()


##########################################################
# STEP 7:  Model Evaluation 
##########################################################
# Package the stacked model instead of the GBM model (it seemed to work ok)
joblib.dump(stacked_model, 'stacked_model.joblib')
loaded_model = joblib.load('stacked_model.joblib')

EvalCoordinates = [
                [9.375, 3.0625, 1.51],
                [6.995, 5.125, 0.3875],
                [0, 3.0625, 1.93],
                [9.4, 3, 1.8],
                [9.4, 3, 1.3]
]

coordinates_df = pd.DataFrame(EvalCoordinates, columns=['X', 'Y', 'Z'])
predictions = loaded_model.predict(coordinates_df)
print("Predictions for the given coordinates:")
print(predictions)

# REPORT: intro, step step step, plots, explanation, conclusion w/ github URL