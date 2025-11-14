import pandas as pd


'''
import warnings
warnings.filterwarnings("ignore")

import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import shap

from scipy import stats
from xgboost import XGBClassifier

# Import local package
from utils import waterfall
# Force package to be reloaded
importlib.reload(waterfall);
'''

predictors = ['max_power', 'trm_len_6', 'engine_type_dissel', 'low_education_ind_0.0',
       'marital_status_M', 'time_of_week_driven_weekday',
       'time_driven_6am - 12pm', 'gender_F', 'veh_age_1', 'area_B',
       'veh_age_4', 'veh_body_STNWG', 'agecat_3', 'area_A', 'veh_color_black',
       'area_D', 'agecat_1', 'engine_type_hybrid', 'veh_color_yellow',
       'agecat_2', 'veh_color_red', 'time_driven_6pm - 12am', 'veh_body_COUPE',
       'veh_body_RDSTR', 'veh_body_UTE', 'veh_body_CONVT',
       'driving_history_score', 'veh_body_HDTOP', 'veh_color_brown']

inference_data_reduced = pd.read_csv('/Users/aleyahsidhu/vscode projects/Fall-25-Datathon-/Datasets/inference_data_reduced.csv')


