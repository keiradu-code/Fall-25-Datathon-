import os
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import itertools



#######################################################
#                  Prepare Data                       #    
# #####################################################
     
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









###################################################
#           XGBoost MODEL FUNCTION                #
###################################################

def train_xgb_model(train_df, test_df, predictors, response, params, num_boost_round=1000):
    """
    Train an XGBoost model on training data and evaluate on test data

    Parameters:
    train_df : training data
    test_df : testing data
    predictors : list of str; names of predictors/columns
    response : str; name of the response column
    params : dict of XGBoost parameters
    num_boost_round : int, default=1000; number of bosting rounds

    returns a dictionary with trained model, predictions, and performance metrics (R2 and RMSE)
    """
    
    #split predictors and response
    xtrain = train_df[predictors].copy()
    ytrain = train_df[response].copy()
    
    xtest = test_df[predictors].copy()
    ytest = test_df[response].copy()
    
    #convert categorical columns 
    for col in predictors:
        if xtrain[col].dtype == 'object':
            xtrain[col] = xtrain[col].astype('category')
        if xtest[col].dtype == 'object':
            xtest[col] = xtest[col].astype('category')
    
    #prepare data for xgboost to read
    dtrain = xgb.DMatrix(xtrain, label=ytrain, enable_categorical=True)
    dtest = xgb.DMatrix(xtest, label=ytest, enable_categorical=True)
    
    #train model
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    
    #make predictions
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)
    
    #compute metrics
    r2_train = r2_score(ytrain, train_pred)
    r2_test = r2_score(ytest, test_pred)
    rmse_train = np.sqrt(mean_squared_error(ytrain, train_pred))
    rmse_test = np.sqrt(mean_squared_error(ytest, test_pred))
    
    metrics = {
        "R2_Train": r2_train,
        "R2_Test": r2_test,
        "RMSE_Train": rmse_train,
        "RMSE_Test": rmse_test,
        "Train_Pred": train_pred,
        "Test_Pred": test_pred,
        "Model": model
    }
    
    return metrics







###########################################################
#       RUNNING ON COPIED PARAMETERS (BASELINE MODEL?)    #
###########################################################
params1 = {
    #regression function; tweedie is good because it is made for datasets with many 0's and only positive values
    'objective': 'reg:tweedie',
    'tweedie_variance_power': 1.5,
    #smaller learning rate = learns slower
    'learning_rate': 0.01,
    #depth of the tree
    'max_depth': 5,
    #?
    'min_child_weight': 100,
    #?
    'subsample': 0.8,
    #?
    'colsample_bytree': 0.8,
    #?
    'lambda': 1.0,
    #?
    'alpha': 0.0,
    #?
    'nthread': -1,
    #for reproducibility
    'seed': 42
}

metrics1 = train_xgb_model(train_data_reduced, test_data_reduced, predictors, response, params1, num_boost_round=1000)

print("For Model 1: ")
print("R2 Train:", metrics1["R2_Train"])
print("R2 Test:", metrics1["R2_Test"])
print("RMSE Train:", metrics1["RMSE_Train"])
print("RMSE Test:", metrics1["RMSE_Test"])

###############################################
#       LOOPING OVER PARAMETER VALUES         #
###############################################
#loops over 192 combinations and saves the best outputs
param_grid = {
    #how much each tree contributes to the final prediction
    'learning_rate': [0.01, 0.05],
    
    #maximum depth/complexity of each tree
    #smaller trees are less likely to overfit
    'max_depth': [3, 5, 7],
    
    #minimum sum of instance weights needed in a child node
    #prevents creating nodes with too few samples
    #low value allows splitting nodes with few samples --> more complex trees --> risk of overfitting
    'min_child_weight': [1, 10],

    #fraction of traninng samples used to grow each tree
    #smaller introduces more randomess and reduces overfitting
    'subsample': [0.6, 0.8],

    #fraction of features/columns used per tree
    #<1 introduces randomness and reduces overfitting
    'colsample_bytree': [0.6, 0.8],

    #L2 regularization term
    #0 = no regularization
    'lambda': [0, 1.0],

    #L1 regularization term
    #Encourages sparsity as it increases
    'alpha': [0, 0.5]
}

#unchanging parameters
fixed_params = {
    'objective': 'reg:tweedie',
    'seed': 42,
    'nthread' : -1, 
    #Controls variance-mean relationship in tweedie regression
    #p = 1 is poisson-like
    # 1<p<2 is compount poisson-gamma (best for our case)
    #p = 2 is gamme-like
    'tweedie_variance_power': 1.5
}

#generate all possible combinations of parameter values
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#store the best model for each metric
best_r2_train = {"score": -np.inf, "params": None, "metrics": None}
best_r2_test  = {"score": -np.inf, "params": None, "metrics": None}
best_rmse_train = {"score": np.inf, "params": None, "metrics": None}
best_rmse_test  = {"score": np.inf, "params": None, "metrics": None}


#loop over all of these combinations
for i, combo in enumerate(combinations, start=1):
    params = fixed_params.copy()
    params.update(combo)
    
    print(f"\n===== Model {i} =====")
    print("Parameters:", params)
    
    metrics = train_xgb_model(train_data_reduced, test_data_reduced, predictors, response, params, num_boost_round=1000)
    
    r2_train = metrics['R2_Train']
    r2_test = metrics['R2_Test']
    rmse_train = metrics['RMSE_Train']
    rmse_test = metrics['RMSE_Test']
    
    print(f"R2 Train: {r2_train:.4f}, R2 Test: {r2_test:.4f}")
    print(f"RMSE Train: {rmse_train:.4f}, RMSE Test: {rmse_test:.4f}")
    
    #check and save best train r^2
    if r2_train > best_r2_train['score']:
        best_r2_train['score'] = r2_train
        best_r2_train['params'] = params
        best_r2_train['metrics'] = metrics
    
    #check and save best test r^2
    if r2_test > best_r2_test['score']:
        best_r2_test['score'] = r2_test
        best_r2_test['params'] = params
        best_r2_test['metrics'] = metrics
    
    #check adn save best train rmse
    if rmse_train < best_rmse_train['score']:
        best_rmse_train['score'] = rmse_train
        best_rmse_train['params'] = params
        best_rmse_train['metrics'] = metrics
    
    #check and save best test rmse
    if rmse_test < best_rmse_test['score']:
        best_rmse_test['score'] = rmse_test
        best_rmse_test['params'] = params
        best_rmse_test['metrics'] = metrics


#view top models after all combinations have been evaluated
print("\n=== Best Models ===")
print("Best R2 Train Model:", best_r2_train['score'], "Params:", best_r2_train['params'])
print("Best R2 Test Model: ", best_r2_test['score'], "Params:", best_r2_test['params'])
print("Best RMSE Train Model:", best_rmse_train['score'], "Params:", best_rmse_train['params'])
print("Best RMSE Test Model: ", best_rmse_test['score'], "Params:", best_rmse_test['params'])