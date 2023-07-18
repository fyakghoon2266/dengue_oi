# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import configparser

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

# bulid ConfigParser
config = configparser.ConfigParser()

# setting conf doc
config.read('./conf/job.conf')

logger.info('======== filter the IR more than 0 and drop the columns ========')

class function_process():

    def __init__(self, config):
        # bulid ConfigParser
        config = configparser.ConfigParser()

        # setting conf doc
        config.read('./conf/job.conf')
        self.config=config.get()

    def load_data(self):
        data = pd.read_csv(self.config.get('path')+self.config.get('table_name'))
        return data

    def filter_df(self, df):
        filter = df['IR'] > 0
        df = df.loc[filter]

        data = data.drop(self.config.get('drop_list'), axis=1)

        return df

    def split_data(self, df):
        Train_data, Test_data = train_test_split(df, random_state=777, train_size=0.8, shuffle=True)
        
        Train_data.to_csv(self.config.get('path')+'Train_data.csv', index=False)
        Test_data.to_csv(self.config.get('path')+'Test_data.csv', index=False)

        X_Train = Train_data.drop('OI', axis=1)
        X_Test = Test_data.drop('OI', axis=1)
        Y_Train = Train_data.loc[:, ['OI']]
        Y_Test = Test_data.loc[:, ['OI']]
        X_Train, X_validation, Y_Train, Y_validation = train_test_split(X_Train, Y_Train, test_size=0.25)

        X_Train.to_csv(self.config.get('path')+'X_Train.csv', index=False)
        Y_Train.to_csv(self.config.get('path')+'Y_Train.csv', index=False)

        X_validation.to_csv(self.config.get('path')+'X_validation.csv', index=False)
        Y_validation.to_csv(self.config.get('path')+'Y_validation.csv', index=False)

        return X_Train, X_Test, X_validation, Y_Train, Y_Test, Y_validation

    def model_train(self, X_Train, Y_Train, X_validation, Y_validation):
        other_params ={'objective' : 'reg:squarederror'}

        model = xgb.XGBRegressor(**other_params)

        space={'max_depth': np.linspace(6, 10, 5, dtype=int),
                'gamma': np.linspace(0, 1, 1,dtype=int),
            'colsample_bytree': np.linspace(0.6, 1, 5),
                'min_child_weight': np.linspace(1, 2, 1, dtype=int),
            'subsample': np.linspace(0.6, 1, 5),
                'n_estimators': np.linspace(50, 250, 5, dtype=int),
            'eta': np.linspace(0.1, 0.3, 3)
            }
        
        gs = GridSearchCV(model, space, verbose=2, refit=True, cv=5,n_jobs=1)
        gs.fit(X_Train, Y_Train, eval_set=[(X_Train, Y_Train), (X_validation, Y_validation)], early_stopping_rounds=20)

        return gs


