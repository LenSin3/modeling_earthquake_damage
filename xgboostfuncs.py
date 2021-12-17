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

# import machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

SEED = 42




def xgboost_model(df, target_var, stratify = True, test_size = 0.25):
    # # label encode labels
    # le = LabelEncoder()
    # encoded_labels = le.fit_transform(df['labels'])
    # df['encoded_labels'] = [x for x in encoded_labels]
    # split data into X and y
    # X = df.drop(target_var, axis = 1)
    # y = df[target_var]
    # convert data into Dmatrix
    # data_matrix = xgb.DMatrix(data = X, label = y)
    # extract train and test set
    X_train, X_test, y_train, y_test = mlfuncs.split_data(df, target_var=target_var, test_size = test_size, stratify = stratify)
    # extract column names
    cat_feats, num_feats, text_feats = mlfuncs.extract_cat_num_text_feats(X_train)
    # extract processor objects
    preprocessor = mlfuncs.preprocess_col_transformer(cat_feats, num_feats, text_feats)
    # instantiate XGBoost Classifier
    clf_name = 'XGBoost Classifier'
    clf = xgb.XGBClassifier(n_estimators = 10)
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
    clf_f1_scr = f1_score(y_test, y_pred, average='micro')
    print("Accuracy Score for {}: {}".format(clf_name, clf_f1_scr))
    return clf_f1_scr


