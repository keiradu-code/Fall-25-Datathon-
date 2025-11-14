import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from varclushi import VarClusHi

#first set correct working directory
from pathlib import Path

# Get the directory of the current script
wd = Path(__file__).parent

# Change the current directory to that folder
import os
datasets_dir = wd / "Datasets"
os.chdir(datasets_dir)

print("Current working directory:", Path.cwd())

model_data = pd.read_csv("model_data.csv")
inference_data = pd.read_csv("inference_data.csv")




####################################################
#           ENCODING CATEGORICAL VARS              #
####################################################



###MODEL ON ONLY ALL MODEL DATA, then split into train/test
#define claim severity from his bootcamp code: 
# Rename columns for clarity and consistency
model_data = model_data.rename(columns={'numclaims': 'claim_cnt'})
model_data = model_data.rename(columns={'claimcst0': 'claim_amt'})


# Create a new column 'claim_sev' (claim severity) as claim_amt divided by claim_cnt
# If claim_cnt is zero, set claim_sev to NaN to avoid division by zero
model_data['claim_sev'] = model_data.apply(
    lambda row: row['claim_amt'] / row['claim_cnt'] if row['claim_cnt'] != 0 else np.nan,
    axis=1
)

#define claim frequency: numclaims (per policy)/exposure 
model_data["claim_freq"] = model_data.apply(
    lambda row: row["claim_cnt"] / row["exposure"]
    if row["exposure"] != 0 else np.nan, #prevent division by 0
    axis = 1 #apply row-wise
) 

#So claim_freq == 1 means one claim per full policy year (for example if claim_cnt = 1 and exposure = 1); 
#claim_fre1 == 4 means four claims per full policy year (for example, if claim_cnt = 2 and exposure = 0.5; 2 claims were made in 6 months, meaning the pattern extends to 4 being made in a year)
#Why does this pattern apply? We're normalizing the number of claims to a standard exposure unit (one year == exposure = 1) 



#Break features into related groups for variable reduction purposes
#Features describing the vehicle: 
veh_pred_lst = ['veh_value', 'veh_body', 'veh_age', 'engine_type', 'max_power', 'veh_color', ]
#Features describing the Policy
policy_pred_lst = ['e_bill']
#Features describing the driving behavior
driving_behavior_pred_lst = ['area', 'time_of_week_driven', 'time_driven']
#Features describing the driver
demo_pred_lst = ['marital_status', 'low_education_ind', 'credit_score', 'driving_history_score', 'gender', 'agecat']

# Identify categorical perdictors
categorical_cols = ["veh_body", "veh_age", "gender", "area", "agecat", "engine_type", "veh_color", "marital_status", "time_of_week_driven", "time_driven", "trm_len", "low_education_ind"]

# One-hot encode categorical variables 
model_data_encoded = pd.get_dummies(model_data, columns=categorical_cols, drop_first=False)

inference_data_encoded = pd.get_dummies(inference_data, columns=categorical_cols, drop_first=False)

#this saves them as TRUE/FALSE. Let's change that to 1/0
model_data_encoded[model_data_encoded.select_dtypes(bool).columns] = model_data_encoded.select_dtypes(bool).astype(int)

inference_data_encoded[inference_data_encoded.select_dtypes(bool).columns] = inference_data_encoded.select_dtypes(bool).astype(int)

inference_data_encoded = inference_data_encoded.rename(columns={'low_education_ind_0': 'low_education_ind_0.0'})


#how many variables do we have now? 
print(f"Num of variables (including id, sample, fold, etc.): {len(model_data_encoded.columns)}")

#how many predictors do we have now?
predictors = []
keywords = [
    "veh_value", "max_power", "driving_history", "credit_score", "numclaims",
    "veh_body", "veh_age", "gender", "area", "agecat", "engine_type",
    "veh_color", "marital_status", "time_of_week_driven", "time_driven",
    "trm_len", "low_education_ind", "e_bill"
]
for var in model_data_encoded.columns:
    if any(k in var for k in keywords):
        predictors.append(var)
print(f"Num of predictors: {len(predictors)}")








####################################################
#      VISUALIZING VARIABLE RELATIONSHIPS          #
####################################################


#Scatterplots of claim_amt to all predictors
def pairwise_scatterplots_for_feature_reduction(dataset, predictors):
    """
    creates scatterplots for each predictor vs claim_amt to help with feature reduction.
    """
    for column in predictors:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=dataset, y=column, x="claim_amt", alpha=1)
        plt.title(f"Scatterplot: {column} vs claim_amt")
        plt.xlabel("claim_amt")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()
#plot pairwise scatterplots with claim_amt on full encoded model data
pairwise_scatterplots = pairwise_scatterplots_for_feature_reduction(model_data_encoded, predictors)


#break up training and testing data
train_data_encoded = model_data_encoded[model_data_encoded["sample"] == "1|bld"]
test_data_encoded = model_data_encoded[model_data_encoded["sample"] == "2|val"]

#save data
#train_data_encoded.to_csv("train_data_encoded.csv", index=False)
#test_data_encoded.to_csv("test_data_encoded.csv", index=False)











####################################################
#           VARIABLE REDUCTION                     #
####################################################

#apply varclushi variable reduction 
#https://github.com/jingtt/varclushi/blob/master/README.md

model_data_encoded_vc = VarClusHi(model_data_encoded[predictors],maxeigval2=1,maxclus=None)
model_data_encoded_vc.varclus()

#get the number of clusters, variables in each cluster (N_Vars), and variance explained by each cluster
print(model_data_encoded_vc.info)
#get the (1-rsquare) ratio of each variable. meaning...?
print(model_data_encoded_vc.rsquare)

#now we can look at the clusters and choose one or a few variables per cluster to include in the model (usually the one with highest RS_Own)
#This helps reduce multicollinearity and redundancy. 
#if varprop == 1, then the variables are perfectly correlated

df_varlevel = model_data_encoded_vc.rsquare.reset_index()
df_varlevel = df_varlevel.drop(columns='index')
df_varlevel.columns = ['Cluster', 'Variable', 'RS_Own', 'RS_NC', 'RS_Ratio']
#find the row index of the variable with the highest RS_Own in a cluster, and selects the name of the variable corresponding to that row
#so representative_vars is a series where each cluster has the variable with the highest RS_Own
representative_vars = df_varlevel.groupby('Cluster').apply(lambda g: g.loc[g['RS_Own'].idxmax(), 'Variable'])
print(representative_vars.tolist())
# Convert representative_vars from Series to list
rep_vars_list = representative_vars.tolist()

# Reduced dataset with only representative variables + target
reduced_data = model_data_encoded[rep_vars_list + ['claim_amt', 'claim_cnt', 'claim_sev', 'claim_freq', 'sample']]


# Split into train/test
train_data_reduced = reduced_data[reduced_data['sample'] == '1|bld']
test_data_reduced = reduced_data[reduced_data['sample'] == '2|val']

# Save reduced datasets
#train_data_reduced.to_csv("train_data_reduced.csv", index=False)
#test_data_reduced.to_csv("test_data_reduced.csv", index=False)


predictors = ['max_power', 'trm_len_6', 'engine_type_dissel', 'low_education_ind_0.0',
       'marital_status_M', 'time_of_week_driven_weekday',
       'time_driven_6am - 12pm', 'gender_F', 'veh_age_1', 'area_B',
       'veh_age_4', 'veh_body_STNWG', 'agecat_3', 'area_A', 'veh_color_black',
       'area_D', 'agecat_1', 'engine_type_hybrid', 'veh_color_yellow',
       'agecat_2', 'veh_color_red', 'time_driven_6pm - 12am', 'veh_body_COUPE',
       'veh_body_RDSTR', 'veh_body_UTE', 'veh_body_CONVT',
       'driving_history_score', 'veh_body_HDTOP', 'veh_color_brown']

inference_data_encoded = inference_data_encoded[predictors]

inference_data_encoded.to_csv("inference_data_reduced.csv", index=False)

#####################################################
#                   SANITY CHECKS!                  #   
#####################################################

#making sure we didn't lose any datapoints -- we're all good! :)
print(len(train_data_encoded) == len(train_data_reduced))
print(len(test_data_encoded) == len(test_data_reduced))