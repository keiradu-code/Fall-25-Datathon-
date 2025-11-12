import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


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


scaler = StandardScaler()
scaler.fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)
xtest_scaled = scaler.transform(xtest)

tweedie_model = TweedieRegressor(power=1.5, alpha=0.5)
tweedie_model.fit(xtrain_scaled, ytrain)

ypred = tweedie_model.predict(xtest_scaled)
tweedie_test_mse = mean_squared_error(ytest, ypred)
tweedie_test_r2 = r2_score(ytest, ypred)

ypred_train = tweedie_model.predict(xtrain_scaled)
tweedie_train_mse = mean_squared_error(ytrain,ypred_train)
tweedie_train_r2 = r2_score(ytrain, ypred_train)
print('Tweedie Train MSE: ', tweedie_train_mse)
print('Tweedie Train R^2: ', tweedie_train_r2)

print('Tweedie Test MSE: ', tweedie_test_mse)
print('Tweedie Test R^2: ', tweedie_test_r2)


