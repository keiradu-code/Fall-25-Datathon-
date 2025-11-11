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
inference_data = pd.read_csv("inference_data.csv")

model_data_vars = model_data.columns.tolist()
inference_data_vars = inference_data.columns.tolist()

#visiualize distributions of variables; numerical are blue and categorical are orange
def feature_distribution(dataframe, n_cols=3):
    """
    plots the distribution of all features in given dataframe
    """
    columns = dataframe.columns.tolist()
    n_rows = (len(columns) + n_cols - 1) // n_cols  # number of subplot rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        series = dataframe[col]

        if pd.api.types.is_numeric_dtype(series):
            series.plot(kind='hist', bins=20, ax=ax, color='blue', edgecolor='black')
            ax.set_title(f"distribution of {col}")
        else:
            series.value_counts().plot(kind='bar', ax=ax, color='orange', edgecolor='black')
            ax.set_title(f"Value counts of {col}")
        
        ax.set_xlabel(col)
        ax.set_ylabel("frequency")

    # Hide unused subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

feature_distribution(model_data)
feature_distribution(inference_data)

#for those instances where clm > 0, what does the distribution look like? 
model_data_subset = model_data[model_data['clm'] > 0].copy()
feature_distribution(model_data_subset)

#how many have clm > 0? 
print(len(model_data) - len(model_data_subset))
#13978
#That's over 93% of the data

#Any missing data? 
model_data.isna().any().any()
#nope! yay :)

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




#split into training/testing 
training = model_data[model_data["sample"] == "1|bld"].copy() 
#11,204 training points
testing = model_data[model_data["sample"] == "2|val"].copy()
#3796 testing points

#sanity check -- true!
#print((len(training) + len(testing)) == len(model_data))



#Break features into related groups for variable reduction purposes
#Features describing the vehicle: 
veh_pred_lst = ['veh_value', 'veh_body', 'veh_age', 'engine_type', 'max_power', 'veh_color', ]
#Features describing the Policy
policy_pred_lst = ['e_bill']
#Features describing the driving behavior
driving_behavior_pred_lst = ['area', 'time_of_week_driven', 'time_driven']
#Features describing the driver
demo_pred_lst = ['marital_status', 'low_education_ind', 'credit_score', 'driving_history_score', 'gender', 'agecat']

#what does this do?
pred_lst = veh_pred_lst + policy_pred_lst + driving_behavior_pred_lst + demo_pred_lst
cols = ['exposure'] + pred_lst

# Concatenate model_data[cols] and inference_data[cols] vertically (ensures the same dummy variable structure across both)
combined_expo_pred_data = pd.concat(
    [model_data[cols], 
     inference_data[cols]], 
     axis=0, 
     ignore_index=True
     )
print('Combined data shape:', combined_expo_pred_data.shape)
combined_expo_pred_data.head()

# Identify categorical perdictors
categorical_cols = [
    col for col in pred_lst 
    if training[col].dtype == 'object' 
    or str(training[col].dtype).startswith('category') 
    or col == "agecat"
    or col == "veh_age"
]
# One-hot encode categorical variables in pred_lst
train_data_encoded = pd.get_dummies(training, columns=categorical_cols, drop_first=False)

# Update pred_lst to include new dummy variable columns
new_pred_lst = []
for col in pred_lst:
    if col in categorical_cols:
        new_pred_lst.extend([c for c in train_data_encoded.columns if c.startswith(col + '_')])
    else:
        new_pred_lst.append(col)

print('Categorical columns one-hot encoded:', categorical_cols)
print('New predictor list:', new_pred_lst)



#Make the testing data match!!
# apply the same dummy encoding
test_data_encoded = pd.get_dummies(testing, columns=categorical_cols, drop_first=False)

# Ensure test data has the same columns as training
# (fill in any missing dummy columns with zeros)
for col in train_data_encoded.columns:
    if col not in test_data_encoded.columns:
        test_data_encoded[col] = 0

# Align column order to match training data
test_data_encoded = test_data_encoded[train_data_encoded.columns]




#variable reduction



