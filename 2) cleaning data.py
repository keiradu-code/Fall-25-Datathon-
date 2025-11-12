import pandas as pd
import matplotlib.pyplot as plt
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

model_data = pd.read_csv("model_data.csv")


#joe steps
#don't seperate model data
#don't consider inference data (yet)
#define claim severity and frequency
#apply dummy encoding


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


#break up training and testing data
train_data_encoded = model_data_encoded[model_data_encoded["sample"] == "1|bld"]
test_data_encoded = model_data_encoded[model_data_encoded["sample"] == "2|val"]

#save data
train_data_encoded.to_csv("train_data_encoded.csv", index=False)
test_data_encoded.to_csv("test_data_encoded.csv", index=False)





#next steps: 
#visualize data as claimcst = [all variables] scatterplots
#tree 
#variable reduction on models





#https://github.com/jingtt/varclushi/blob/master/README.md

