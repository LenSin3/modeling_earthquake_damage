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
    """Process data for machine learning preprocessing.

    Low variance categorical features are high correlated numerical features are dropped from DataFrame. This process helps in dimensionality reduction.

    Parameters
    ----------
    df: DataFrame
        DataFrame to process.
    initial_to_drop: list
        List of initial columns to drop.
    target_var: str
        Target variable to exclude from analysis.
        Default(value = None)
    num_cols_threshold: float64
        Threshold correlation value for numerical features.
        Default(value = 0.9)
    low_var_threshold: str
        Threshold normalized unique value of max value counts.
        Default(value = 0.9)
    
    Returns
    -------
    DataFrame
    """
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
    else:
        raise utils.InvalidDataFrame(df)
    
    return df

def split_data(df, target_var, stratify = True, test_size = 0.25):
    """Split data into train and test set.

    Data is split into training and test set for model fit and evaluation.

    Parameters
    ----------
    df: DataFrame
        DataFrame to split.
    target_var: str
        Target variable.
    stratify: bool
        Boolean value to indicate if target_var should be stratified to mitigate imbalance classes.
    test_size: float64
        Percentage size of test set.
        Default(value = 0.25)
    
    Returns
    -------
    X_train: DataFrame
        Features in train data set.
    X_test: DataFrame
        Features in test data set.
    y_train: DataFrame
        Labels in train data set.
    y_test: DataFrame
        Labels in test data set.
    """
    # check for valid dataframe
    if isinstance(df, pd.DataFrame):
        # extract columns
        df_cols = df.columns.tolist()
        if target_var in df_cols:
            # extract X
            X = df.drop(target_var, axis=1)
            # extract y
            y = df[target_var]
            # if target_var is categorical
            if df[target_var].dtypes == 'object':
                # if stratify is true
                if stratify:
                    # split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = SEED, stratify = y)
                else:
                    # split data with stratify parameter
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = SEED)
            else:
                # if target_var is int (example, for regression tasks)
                if df[target_var].dtypes == 'int64' or df[target_var].dtypes == 'int32' or df[target_var].dtypes == 'float64':
                    # if stratify is true
                    if stratify:
                        # raise error (stratify should only be set for target variables of type object)
                        raise utils.StratifyError(target_var)
                    # else
                    elif not stratify:
                        # split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = SEED)
        else:
            raise utils.InvalidColumn(target_var)
    else:
        raise utils.InvalidDataFrame(df)

    return X_train, X_test, y_train, y_test

def extract_cat_num_text_feats(X_train, text_feats = None):
    return None