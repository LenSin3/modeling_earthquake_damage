from re import UNICODE
from numpy.lib import utils
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import funcs
import mlfuncs
import eda_plots
import utils
import dlfuncs
import new_mlfuncs

# import machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

SEED = 42

params = {}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
# params['max_depth']=10
params['num_class']=3

def lightgbm_model(df, target_var, feats_to_exclude, text_feats, stratify = True, test_size = 0.25):
    
    # extract train and test set
    # split data
    X_train, X_test, y_train, y_test = mlfuncs.split_data(df, target_var=target_var, stratify = stratify, test_size = test_size)
    # extract cat_feats, num_feats, text_feats
    cat_feats, num_feats, text_feats = dlfuncs.extract_cat_num_text_feats_for_keras(X_train, feats_to_exclude=feats_to_exclude, text_feats = text_feats)
    # extract processor pipeline
    preprocessor = dlfuncs.preprocess_col_transformer_for_keras(cat_feats, num_feats, text_feats = text_feats)
    # instantiate XGBoost Classifier
    clf_name = 'LightGBM Classifier'
    clf = LGBMClassifier(n_estimators=1500, random_state=SEED)
    print("Creating pipeline for {}.".format(clf_name))
    pipe = make_pipeline(preprocessor, clf)
    # fit training data to pipe
    print("Fitting training data to pipeline for {}.".format(clf_name))
    pipe.fit(X_train, y_train)
    # get predictions
    print("Predicting test values for {}.".format(clf_name))
    y_pred = pipe.predict(X_test)
    # get accuracy score
    print("Calculating accuracy score for {}.".format(clf_name))
    clf_f1_scr = accuracy_score(y_test, y_pred)
    print("Accuracy Score for {}: {}".format(clf_name, clf_f1_scr))
    return clf_f1_scr, pipe

def lightgbm_model_1(df, target_var, feats_to_exclude, text_feats, stratify = True, test_size = 0.25):
    
    # extract train and test set
    # split data
    X_train, X_test, y_train, y_test = mlfuncs.split_data(df, target_var=target_var, stratify = stratify, test_size = test_size)
    # extract cat_feats, num_feats, text_feats
    cat_feats, num_feats, text_feats = dlfuncs.extract_cat_num_text_feats_for_keras(X_train, feats_to_exclude=feats_to_exclude, text_feats = text_feats)
    # extract processor pipeline
    preprocessor = new_mlfuncs.preprocess_col_transformer(cat_feats, num_feats, text_feats = text_feats)
    # instantiate XGBoost Classifier
    clf_name = 'LightGBM Classifier'
    clf = LGBMClassifier(n_estimators=3000, learning_rate=0.1, random_state=SEED)
    print("Creating pipeline for {}.".format(clf_name))
    pipe = make_pipeline(preprocessor, clf)
    # fit training data to pipe
    print("Fitting training data to pipeline for {}.".format(clf_name))
    pipe.fit(X_train, y_train)
    # get predictions
    print("Predicting test values for {}.".format(clf_name))
    y_pred = pipe.predict(X_test)
    # get accuracy score
    print("Calculating accuracy score for {}.".format(clf_name))
    clf_f1_scr = accuracy_score(y_test, y_pred)
    print("Accuracy Score for {}: {}".format(clf_name, clf_f1_scr))
    return clf_f1_scr, pipe
