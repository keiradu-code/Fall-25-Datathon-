import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

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

xtrain = train_data_reduced[predictors].copy()
ytrain = train_data_reduced[response].copy()
#print(xtrain.head())

xtest = test_data_reduced[predictors].copy()
ytest = test_data_reduced[response].copy()
#print(ytest.head())

# --- Convert categorical columns ---
for col in predictors:
    if xtrain[col].dtype == 'object':
        xtrain[col] = xtrain[col].astype('category')
    if xtest[col].dtype == 'object':
        xtest[col] = xtest[col].astype('category')

# --- Prepare DMatrix for XGBoost ---
dtrain = xgb.DMatrix(xtrain, label=ytrain, enable_categorical=True)
dtest = xgb.DMatrix(xtest, label=ytest, enable_categorical=True)

# --- XGBoost Tweedie Regression Parameters ---
params = {
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.5,
    'learning_rate': 0.01,
    'max_depth': 5,
    'min_child_weight': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0.0,
    'nthread': -1,
    'seed': 42
}

# --- Train model directly (no CV) ---
num_boost_round = 1000  # you can adjust this if needed
print(f"Training XGBoost model with {num_boost_round} boosting rounds...")
model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

# --- Predictions ---
train_pred = model.predict(dtrain)
test_pred = model.predict(dtest)

# --- Compute R² and RMSE ---
r2_train = r2_score(ytrain, train_pred)
r2_test = r2_score(ytest, test_pred)
rmse_train = np.sqrt(mean_squared_error(ytrain, train_pred))
rmse_test = np.sqrt(mean_squared_error(ytest, test_pred))

print("\nModel Performance Metrics (no cross-validation):")
print(f"R2 (Train): {r2_train:.2f}")
print(f"R2 (Test):  {r2_test:.2f}")
print(f"RMSE (Train): {rmse_train:.2f}")
print(f"RMSE (Test):  {rmse_test:.2f}")

print("\n Model training, prediction, and evaluation completed successfully.")