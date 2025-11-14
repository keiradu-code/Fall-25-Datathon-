import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

test_data_reduced = pd.read_csv('/Users/aleyahsidhu/vscode projects/Fall-25-Datathon-/Datasets/test_data_reduced.csv')
train_data_reduced = pd.read_csv('/Users/aleyahsidhu/vscode projects/Fall-25-Datathon-/Datasets/train_data_reduced.csv')


test_data_encoded = pd.read_csv('/Users/aleyahsidhu/vscode projects/Fall-25-Datathon-/Datasets/test_data_encoded.csv')
train_data_encoded = pd.read_csv('/Users/aleyahsidhu/vscode projects/Fall-25-Datathon-/Datasets/train_data_encoded.csv')


#print(train_data_encoded.columns)

predictors = ['max_power', 'trm_len_6', 'engine_type_dissel', 'low_education_ind_0.0',
       'marital_status_M', 'time_of_week_driven_weekday',
       'time_driven_6am - 12pm', 'gender_F', 'veh_age_1', 'area_B',
       'veh_age_4', 'veh_body_STNWG', 'agecat_3', 'area_A', 'veh_color_black',
       'area_D', 'agecat_1', 'engine_type_hybrid', 'veh_color_yellow',
       'agecat_2', 'veh_color_red', 'time_driven_6pm - 12am', 'veh_body_COUPE',
       'veh_body_RDSTR', 'veh_body_UTE', 'veh_body_CONVT',
       'driving_history_score', 'veh_body_HDTOP', 'veh_color_brown']

xtrain = train_data_encoded[predictors]

#generate a histogram and heatmap showing the correlation between predictors to guide variable selection
corr_val_df = round(xtrain.corr(method='kendall'), 2)
corr_val_vec = pd.Series( [item for sublist in corr_val_df.values.tolist() for item in sublist if item < 1.0])
rcParams['figure.figsize'] = 15,4
corr_val_vec.plot.hist(bins = 50, figsize = (15,5), xlim=(-1,1), grid=True)

plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_val_df, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()



