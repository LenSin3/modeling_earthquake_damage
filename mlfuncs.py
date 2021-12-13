# import base libraries and dependencies
from re import UNICODE
from numpy.lib import utils
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import funcs
import eda_plots
import utils

# import machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# import modules for machine learning models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn import svm
import xgboost
import catboost

# set seed for reproducibility
SEED = 24
# process data before machine learning preprocessing
def pre_ml_preprocessing(df, initial_to_drop, num_cols, target_var = None, num_cols_threshold = 0.9, low_var_threshold = 0.9):
    # check for valid dataframe
    if isinstance(df, pd.DataFrame):
        # extract dataframe columns
        df_cols = df.columns.tolist()
        # check if all columns to drop are in df_cols
        membership = all(col in df_cols for col in initial_to_drop)
        # if membership
        if membership:
            for col in initial_to_drop:
                # drop col
                print("Dropping: {}".format(col))
                df.drop(col, axis=1, inplace=True)
        else:
            not_cols = []
            for col in initial_to_drop:
                if col not in df_cols:
                    not_cols.append(col)
            raise utils.InvalidColumn(not_cols)
        # drop high correlated features
        df = funcs.dim_redux(df, num_cols, threshold = num_cols_threshold)
        # drop low variance features
        df = funcs.drop_low_var_cols(df, target_var = target_var, unique_val_threshold = low_var_threshold)
    
    return df