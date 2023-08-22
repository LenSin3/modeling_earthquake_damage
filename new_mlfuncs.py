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
import dlfuncs

# import machine learning modules
from sklearn.model_selection import train_test_split, KFold
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

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.externals
import joblib

# set seed for reproducibility
SEED = 24

# process data before machine learning preprocessing
def pre_ml_preprocessing(df, initial_to_drop, num_cols, target_var = None, missing_val_threshold = 40, num_cols_threshold = 0.9, low_var_threshold = 0.9):
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
        # drop column with high missing values
        df = funcs.missing_val_selector(df, missing_val_threshold = missing_val_threshold)
        # drop high correlated features
        df = funcs.dim_redux(df, num_cols, threshold = num_cols_threshold)
        # drop low variance features
        df = funcs.drop_low_var_cols(df, target_var = target_var, unique_val_threshold = low_var_threshold)
    else:
        raise utils.InvalidDataFrame(df)
    
    return df

## target encoding  
def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()
    
    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()
    
    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)
    
    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values

def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)
    
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
      
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature       
    return train_feature.values

def mean_target_encoding(train, test, target, categorical, alpha=5):
  
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)
  
    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    
    # Return new features to add to the model
    return train_feature, test_feature  

def target_encode_categorical_features(train, test, target, alpha=5):
    train = train.copy().dropna()
    test = test.copy().dropna()
    le = LabelEncoder()
    le = le.fit(train[target])
    train['target_encoded'] = le.transform(train[target])
    train.drop(target, axis=1, inplace=True)
    target = 'target_encoded'
    train_cols = train.columns.tolist()
    for col in train_cols:
        if col in target:
            pass
        else:
            if col not in target:
                if train[col].dtypes == 'object':
                    uniques = funcs.unique_vals_column(train, col, normalize = False)
                    if len(uniques) <= 2:
                        pass
                    elif len(uniques) > 2:
                        train_feature, test_feature = mean_target_encoding(train=train, test=test, target=target, categorical=col, alpha=alpha)
                        train[col] = train_feature
                        test[col] = test_feature

    return train, test, le


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
            # le = LabelEncoder()
            # le = le.fit(y)
            # y_le = le.transform(y)
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
                le_binary = LabelEncoder()
                cat_pipeline = make_pipeline(cat_imp, le_binary)

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
                le_binary = LabelEncoder()
                cat_pipeline = make_pipeline(cat_imp, le_binary)

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


def multi_models_classifiers(df, target_var,  feats_to_exclude, stratify = True, test_size = 0.25, classifier_type = 'multi_class', text_feats = None):
    # split data
    X_train, X_test, y_train, y_test = split_data(df, target_var=target_var, stratify = stratify, test_size = test_size)
    # extract cat_feats, num_feats, text_feats
    cat_feats, num_feats, text_feats = extract_cat_num_text_feats(X_train, text_feats = text_feats)
    # extract processor pipeline
    preprocessor = preprocess_col_transformer(cat_feats, num_feats, text_feats = text_feats)
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
        pipe = make_pipeline(preprocessor, clf)
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

        # # get fl score
        # print("Calculating f1 score for {}.".format(clf_name))
        # f1_scr = f1_score(y_test, y_pred, average = 'micro')
        # print("F1 Score for {}: {}".format(clf_name, f1_scr))
        # # create key, value pair in f1_scores
        # f1_scores[clf_name] = f1_scr

    # create dataframe of acc_scores dict
    df_acc_scores = pd.DataFrame(acc_scores.items(), columns=['Classifier', 'AccScore'])
    max_acc_score = df_acc_scores.loc[df_acc_scores['AccScore'] == df_acc_scores['AccScore'].max()]
    best_classifier_acc_score = max_acc_score['Classifier'].values.tolist()[0]
    # # create dataframe of acc_scores dict
    # df_f1_scores = pd.DataFrame(f1_scores.items(), columns=['Classifier', 'F1Score'])
    # max_f1_score =  df_f1_scores.loc[df_f1_scores['F1Score'] == df_f1_scores['F1Score'].max()]
    # best classifier
    # best_classifier_f1_score = max_f1_score['Classifier'].values.tolist()[0]

    
    print("####################################################################################")
    print("{} model yielded the highest Accuracy Score of: {:.2f}".format(max_acc_score['Classifier'].values.tolist()[0], max_acc_score['AccScore'].values.tolist()[0]))
    # plot scores
    eda_plots.bar_plot(df_acc_scores, 'Classifier', 'AccScore')
    print("####################################################################################")
    # print("{} model yielded the highest F1 Score of: {:.2f}".format(max_f1_score['Classifier'].values.tolist()[0], max_f1_score['F1Score'].values.tolist()[0]))
    # eda_plots.bar_plot(df_f1_scores, 'Classifier', 'F1Score')
    return best_classifier_acc_score, pipe, X_train.columns.tolist()

def best_classifier_hyperparameter(df, best_classifier, target_var, feats_to_exclude, stratify = True, test_size = 0.25, classifier_type = 'multi_class', text_feats = None):
    print("hyperparameter tuning for the Best Classifier: {}".format(best_classifier))
     # split data
    X_train, X_test, y_train, y_test = split_data(df, target_var=target_var, stratify = stratify, test_size = test_size)
    # extract cat_feats, num_feats, text_feats
    cat_feats, num_feats, text_feats = extract_cat_num_text_feats(X_train, feats_to_exclude=feats_to_exclude, text_feats = text_feats)
    # extract processor pipeline
    preprocessor = preprocess_col_transformer(cat_feats, num_feats, text_feats = text_feats)
    # extract best regressor model and grid params
    grid_params, grid_model = models.classifier_ensemble_hyperparameters(best_classifier)
    # instantiate pipeline
    print("Creating pipeline for {}.".format(best_classifier))
    pipe = make_pipeline(preprocessor, grid_model)
    # dictionary to hold gridsearch model output
    GridSearchCV_model_output = {}
    # create gridsearch object
    grid_search = GridSearchCV(pipe, param_grid=grid_params, cv = 3, scoring = 'f1_micro', refit = True, return_train_score = True)
    # grid_search = RandomizedSearchCV(pipe, param_distributions=grid_params, n_iter=100, cv = 5, scoring = 'f1_micro', random_state=SEED, refit = True, n_jobs = 4, return_train_score = True)
    # fit gridsearch to training data
    grid_search.fit(X_train, y_train)
    # create key, value pair for best regressor
    GridSearchCV_model_output['best_classifier_grid_object'] = grid_search.best_estimator_
    # get best params
    grid_best_params = grid_search.best_params_
    print("Best parameters after GridSearchCV:\n {}".format(grid_best_params))
    GridSearchCV_model_output['best_params'] = grid_best_params
    # best score
    grid_best_score = grid_search.best_score_
    print("Best score after GridSearchCV:\n {:.2f}".format(grid_best_score))
    # get feature names
    cat_feature_names = grid_search.best_estimator_.named_steps['columntransformer'].named_transformers_['pipeline-1'].\
        named_steps['labelencoder'].get_feature_names(input_features = cat_feats)
    all_feature_names = np.r_[cat_feature_names, num_feats]
    # list to hold coefs or feature imporatnces in the case of decision trees and random forest
    coefs_or_feats_imp = []
    if not 'RandomForestClassifier' or not 'DecisionTreeClassifier':
        # grab the coefficients
        best_classifier_coef = list(grid_search.best_estimator_.named_steps[best_classifier.lower()].coef_)
        best_classifier_coef_x = [x for x in best_classifier_coef]
        coefs_or_feats_imp.append(best_classifier_coef_x)
        # grab the intercept
        grid_search_intercept = grid_search.best_estimator_.named_steps[best_classifier.lower()].intercept_
        GridSearchCV_model_output['best_classifier_intercept'] = grid_search_intercept
    else:
        # grab the coefficients
        best_classifier_coef = list(grid_search.best_estimator_.named_steps[best_classifier.lower()].feature_importances_)
        best_classifier_coef_x = [x for x in best_classifier_coef]
        coefs_or_feats_imp.append(best_classifier_coef_x)

    # create a dataframe of feature names and coeeficients
    coef_dict = dict(zip(all_feature_names, coefs_or_feats_imp[0]))
    df_coef = pd.DataFrame(coef_dict.items(), columns = ['Feature', 'Coefficient'])
    df_coef = df_coef.sort_values(by = ['Coefficient'], ascending = False)
    df_coef = df_coef.reset_index(drop = True)
    GridSearchCV_model_output['best_classifier_feature_importances'] = df_coef
    # grab names of predictors
    GridSearchCV_model_output['final_predictors'] = X_train.columns.tolist()
    return GridSearchCV_model_output

def dump_estimator(GridSearchCV_model_output, dump_path):
    # convert tuple to list
    grid_list = list(GridSearchCV_model_output)
    # extract best estimator
    best_estimator = grid_list[0]['best_classifier_grid_object']
    # save as pickle
    joblib.dump(best_estimator, dump_path)
    return None

def make_predictios(test, submission_format_path,  my_submission_path, model_path, final_predictors, le):
    # load model
    # predictors = list(GridSearchCV_model_output)[0]['final_predictors']
    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
        if os.path.exists(submission_format_path):
            df_to_submit = pd.read_csv(submission_format_path, index_col = 'id')
            test_cols = test.columns.tolist()
            membership = all(col in test_cols for col in final_predictors)
            if membership:
                X_predictors = test[final_predictors]
                y_predicted = best_model.predict(X_predictors)
                labels = le.inverse_transform(y_predicted)
                submission = pd.DataFrame(data = labels, columns = df_to_submit.columns, index = df_to_submit.index)
                submission.to_csv(my_submission_path)
            else:
                not_cols = []
                for col in final_predictors:
                    if col not in test_cols:
                        not_cols.append(col)
                raise utils.InvalidColumn(not_cols)
        else:
            raise utils.InvalidFilePath(submission_format_path)
    else:
        raise utils.InvalidFilePath(model_path)
    return submission