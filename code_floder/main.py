import logging

logger = logging.getLogger(__name__)

import xgboost as xgb
import pandas as pd 
import numpy as np
import shap

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

import configparser

# bulid ConfigParser
config = configparser.ConfigParser()

# setting conf doc
config.read('job.conf')

# read conf parameter
param1 = config.get('Parameters', 'param1')
param2 = config.get('Parameters', 'param2')
param3 = config.get('Parameters', 'param3')

