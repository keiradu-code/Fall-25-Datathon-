
#lets find the eval

#pandas
import pandas as pd
import os
# Ensure the parent directory is in the path for module import
import sys
sys.path.append(os.path.abspath(".."))
from Datasets.data_exploration import PredictivenessCheck
from Datasets.model_selection import ModelEvaluation

# Select training samples for predictiveness check
data = pd.read_csv('Datasets/train_data_reduced.csv')

# Define variables
exp_var = 'exposure'
pred_var = 'train_pred1'
var_1 = 'claim_amt'
var_2 = 'train_pred1'
nbins = 10


# Create and run PredictivenessCheck
pc = PredictivenessCheck(
    df=data,
    pred_var=pred_var,
    exp_var=exp_var,
    var_1=var_1,
    var_2=var_2
)

pc.binning(nbins=nbins)
pc.aggregate()

# Compute top lift
pc.top_lift()

# Compute R^2, RMSE, and MAE
me = ModelEvaluation(data[var_1], data[var_2])
gini = me.gini()
r2 = me.r2()
rmse = me.rmse()
mae = me.mae()

# Print results
print(f"Top Lift: {pc.top_lift:.4f}")
print(f"Gini: {gini:.4f}")
print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
pc.plot(figsize=(10, 4))