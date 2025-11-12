import pandas as pd
from sklearn.discriminant_analysis import \
    (LinearDiscriminantAnalysis as LDA,
     QuadraticDiscriminantAnalysis as QDA)
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


##load data and split into predictor and response dataset

test_data_reduced = pd.read_csv('/Users/aleyahsidhu/vscode projects/Fall-25-Datathon-/Datasets/test_data_reduced.csv')
train_data_reduced = pd.read_csv('/Users/aleyahsidhu/vscode projects/Fall-25-Datathon-/Datasets/train_data_reduced.csv')

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


##BUILD MODEL

# XGBoost Tweedie Regression with Cross-Validation Early Stopping and Native Categorical Support
import xgboost as xgb

# Define features and target
#y = train_data['claim_amt_capped_per_exposure']
#X = train_data[pred_lst].copy()
# weights = train_data['exposure']  # Assuming exposure is in years, convert to months

# XGBoost DMatrix with offset
dtrain = xgb.DMatrix(xtrain, label=ytrain, enable_categorical=True)

# # Compute offset (log of exposure, or any other offset variable)
# offset = np.log(train_data['exposure'])
# dtrain.set_base_margin(offset)

# Define XGBoost parameters for Tweedie regression
params = {
    'objective': 'reg:tweedie',         # Tweedie regression objective
    'tweedie_variance_power': 1.5,      # Tweedie power (1=Poisson, 2=Gamma, 1<p<2 for insurance)
    'learning_rate': 0.01,              # Step size shrinkage (smaller = more robust, slower learning)
    'max_depth': 5,                     # Maximum tree depth (controls model complexity)
    'min_child_weight': 100,            # Minimum sum of instance weight (hessian) needed in a child (min samples per leaf, set for 45,300 samples)
    'subsample': 0.8,                   # Fraction of samples used per tree (prevents overfitting)
    'colsample_bytree': 0.8,            # Fraction of features used per tree (prevents overfitting)
    'lambda': 1.0,                      # L2 regularization term (prevents overfitting)
    'alpha': 0.0,                       # L1 regularization term (prevents overfitting)
    'nthread': -1,                      # Use all CPU cores
    'seed': 42                          # Random seed for reproducibility
}

# Cross-validation with early stopping
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=5000,
    nfold=5,  # 5-fold cross-validation
    metrics='rmse',  # or another appropriate metric
    early_stopping_rounds=50,
    seed=42,
    verbose_eval=50
 )

# The best number of boosting rounds is:
best_num_boost_round = len(cv_results)
print(f"Best num_boost_round from CV: {best_num_boost_round}")

# Train final model on all data using best_num_boost_round
model = xgb.train(
    params,
    dtrain,
    # num_boost_round=500
    num_boost_round=best_num_boost_round
 )
