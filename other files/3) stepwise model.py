import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd 

#first set correct working directory
from pathlib import Path

# Get the directory of the current script
wd = Path(__file__).parent

# Change the current directory to that folder
import os
datasets_dir = wd / "Datasets"
os.chdir(datasets_dir)

print("Current working directory:", Path.cwd())


test_data_reduced = pd.read_csv('test_data_reduced.csv')
train_data_reduced = pd.read_csv('train_data_reduced.csv')

print(train_data_reduced.columns)

predictors = ['max_power', 'trm_len_6', 'engine_type_dissel', 'low_education_ind_0.0',
       'marital_status_M', 'time_of_week_driven_weekday',
       'time_driven_6am - 12pm', 'gender_F', 'veh_age_1', 'area_B',
       'veh_age_4', 'veh_body_STNWG', 'agecat_3', 'area_A', 'veh_color_black',
       'area_D', 'agecat_1', 'engine_type_hybrid', 'veh_color_yellow',
       'agecat_2', 'veh_color_red', 'time_driven_6pm - 12am', 'veh_body_COUPE',
       'veh_body_RDSTR', 'veh_body_UTE', 'veh_body_CONVT',
       'driving_history_score', 'veh_body_HDTOP', 'veh_color_brown']

response = ['claim_amt']

xtrain = train_data_reduced[predictors]
ytrain = train_data_reduced[response]
#print(xtrain.head())

xtest = test_data_reduced[predictors]
ytest = test_data_reduced[response]
#print(ytest.head())


# Select predictors and response
X = xtrain.values
y = ytrain.values.ravel()  # Flatten y for sklearn

# Discretize all predictors
kbd = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')
X_binned = kbd.fit_transform(X)

# Fit model
model = LinearRegression().fit(X_binned, y)

# Predict on test data
Xtest_binned = kbd.transform(xtest.values)
y_pred_test = model.predict(Xtest_binned)

# Print performance
from sklearn.metrics import r2_score
print("R² on training:", model.score(X_binned, y))
print("R² on test:", r2_score(ytest, y_pred_test))