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
import models

# import machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
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
            # if stratify is true
            if stratify:
                # split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = SEED, stratify = y)
            else:
                # split data without stratify parameter
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = SEED)
        else:
            raise utils.InvalidColumn(target_var)
    else:
        raise utils.InvalidDataFrame(df)

    return X_train, X_test, y_train, y_test

def extract_cat_num_text_feats(X_train, text_feats = None):
    """Extract categorical and numerical features.

    Categorical and numerical features are extracted from X_train.

    Parameters
    ----------
    X_train: DataFrame
        Features in train data set.
    text_feats: str, optional
        Column with text values.
    
    Returns
    -------
    cat_feats: list
        List of categorical column names.
    num_feats: list
        List of numerical column names.
    text_feats: str
        Name of column with text values.
    """
    # check for valid dataframe
    if isinstance(X_train, pd.DataFrame):
        # list to hold categorical and numerical features
        cat_feats = []
        num_feats = []
        # extract dataframe columns
        Xtrain_cols = X_train.columns.tolist()
        if not text_feats:
            # loop over dataframe columns
            for col in Xtrain_cols:
                # get object data columns
                if X_train[col].dtypes == 'object':
                    # append to cat_feats list
                    cat_feats.append(col)
                # get numerical data type columns
                elif X_train[col].dtypes == 'int64' or X_train[col].dtypes == 'float64' or X_train[col].dtypes == 'int32':
                    # append to num_cols
                    num_feats.append(col)
        else:
            # if text_feats is specified
            if text_feats:
                # check if text_feats in Xtrain_cols
                if text_feats in Xtrain_cols:
                    # check if text_feats data is of type object
                     if X_train[text_feats].dtypes == 'object':
                        # subset dataframe with all columns except text_feats
                        df_no_text = X_train.drop(text_feats, axis=1)
                        # loop over dataframe columns
                        for col in Xtrain_cols:
                            # get object data columns
                            if df_no_text[col].dtypes == 'object':
                                # append to cat_feats list
                                cat_feats.append(col)
                            # get numerical data type columns
                            elif df_no_text[col].dtypes == 'int64' or df_no_text[col].dtypes == 'float64' or df_no_text[col].dtypes == 'int32':
                                # append to num_cols
                                num_feats.append(col)
                else:
                    raise utils.InvalidDataType(text_feats)
            else:
                raise utils.InvalidColumn(text_feats)
    else:
        raise utils.InvalidDataFrame(X_train)
    
    return cat_feats, num_feats, text_feats

def preprocess_col_transformer(cat_feats, num_feats, text_feats = None):
    """Create column_transformer pipeline object.

    Create pipeline to tranform categorical and numerical features in dataframe.

    Parameters
    ----------
    cat_feats: list
        List of categorical features.
    num_feats: list
        List of numerical features.
    text_feats: str, optional
        name of column with text data.
    
    Returns
    -------
    sklearn.compose.make_column_transformer
    """
    # check if cat_feats is a list
    if isinstance(cat_feats, list):
        # check if num_feats is a list
        if isinstance(num_feats, list):
            # if text_feats is not specified
            if not text_feats:
                # create instances for imputation and encoding of categorical variables
                cat_imp = SimpleImputer(strategy = 'constant', fill_value = 'missing')
                ohe = OneHotEncoder(handle_unknown = 'ignore')
                cat_pipeline = make_pipeline(cat_imp, ohe)

                # create instances for imputation and encoding of numerical variables
                num_imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
                std = StandardScaler()
                num_pipeline = make_pipeline(num_imp, std)

                # create a preprocessor object
                preprocessor = make_column_transformer(
                    (cat_pipeline, cat_feats),
                    (num_pipeline, num_feats),
                    remainder = 'passthrough'
                )
            # if text feats is specified
            elif text_feats:
                # create instances for imputation and encoding of categorical variables
                cat_imp = SimpleImputer(strategy = 'constant', fill_value = 'missing')
                ohe = OneHotEncoder(handle_unknown = 'ignore')
                cat_pipeline = make_pipeline(cat_imp, ohe)

                # create instances for imputation and encoding of numerical variables
                num_imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
                std = StandardScaler()
                num_pipeline = make_pipeline(num_imp, std)

                # create instance for imputation for text column
                text_vectorize = TfidfVectorizer()
                text_pipeline = make_pipeline(text_vectorize)

                # create a preprocessor object
                preprocessor = make_column_transformer(
                    (cat_pipeline, cat_feats),
                    (num_pipeline, num_feats),
                    (text_pipeline, 'processed_essay'),
                    remainder = 'passthrough'
                )
        else:
            raise utils.InvalidList(num_feats)
    else:
        raise utils.InvalidList(cat_feats)
               
    return preprocessor

def multi_models_classifiers(df, target_var, stratify = True, test_size = 0.25, classifier_type = 'multi_class', text_feats = None):
    # split data
    X_train, X_test, y_train, y_test = split_data(df, target_var=target_var, stratify = stratify, test_size = test_size)
    # extract cat_feats, num_feats, text_feats
    cat_feats, num_feats, text_feats = extract_cat_num_text_feats(X_train, text_feats = text_feats)
    # extract processor pipeline
    processor = preprocess_col_transformer(cat_feats, num_feats, text_feats = text_feats)
    # get classifiers list
    classifiers = models.classifiers_ensemble(type = classifier_type)
    # dictionaries to hold accuracy and f1 scores
    acc_scores = {}
    f1_scores = {}
    # iterate over the classifiers
    # iterate over the classifiers
    for clf_name, clf in classifiers:
        # instantiate pipeline
        print("Creating pipeline for {}.".format(clf_name))
        pipe = make_pipeline(processor, clf)
        # fit training data to pipe
        print("Fitting training data to pipeline for {}.".format(clf_name))
        pipe.fit(X_train, y_train)
        # print(pipe.fit(np.ravel(X_train), y_train))
        # get predictions
        print("Predicting test values for {}.".format(clf_name))
        y_pred = pipe.predict(X_test)
        # get accuracy score
        print("Calculating accuracy score for {}.".format(clf_name))
        clf_acc_scr = accuracy_score(y_test, y_pred)
        print("Accuracy Score for {}: {}".format(clf_name, clf_acc_scr))
        # create key, value pair in acc_scores
        acc_scores[clf_name] = clf_acc_scr

        # get fl score
        print("Calculating f1 score for {}.".format(clf_name))
        f1_scr = f1_score(y_test, y_pred, average = 'micro')
        print("F1 Score for {}: {}".format(clf_name, f1_scr))
        # create key, value pair in f1_scores
        f1_scores[clf_name] = f1_scr

    # create dataframe of acc_scores dict
    df_acc_scores = pd.DataFrame(acc_scores.items(), columns=['Classifier', 'AccScore'])
    max_acc_scor = df_acc_scores.loc[df_acc_scores['AccScore'] == df_acc_scores['AccScore'].max()]
    best_classifier_acc_score = max_acc_scor['Classifier'].values.tolist()[0]
    # create dataframe of acc_scores dict
    df_f1_scores = pd.DataFrame(f1_scores.items(), columns=['Classifier', 'F1Score'])
    max_f1_score =  df_f1_scores.loc[df_f1_scores['F1Score'] == df_f1_scores['F1Score'].max()]
    # best classifier
    best_classifier_f1_score = max_f1_score['Classifier'].values.toist()[0]
    # plot scores
    eda_plots.bar_plot(df_acc_scores, 'Classifier', 'AccScore')
    eda_plots.bar_plot(df_f1_scores, 'Classifier', 'F1Score')
    return best_classifier_acc_score, best_classifier_f1_score




      

                    