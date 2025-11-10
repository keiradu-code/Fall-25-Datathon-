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

model_data_columns = model_data.columns.tolist()
inference_data_columns = inference_data.columns.tolist()

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

#define claim frequency 
