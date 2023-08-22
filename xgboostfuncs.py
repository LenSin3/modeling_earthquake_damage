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

# import machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

SEED = 42


def xgboost_model(df, target_var, feats_to_exclude, text_feats, stratify = True, test_size = 0.25):
    
    # extract train and test set
    # split data
    X_train, X_test, y_train, y_test = mlfuncs.split_data(df, target_var=target_var, stratify = stratify, test_size = test_size)
    # extract cat_feats, num_feats, text_feats
    cat_feats, num_feats, text_feats = dlfuncs.extract_cat_num_text_feats_for_keras(X_train, feats_to_exclude=feats_to_exclude, text_feats = text_feats)
    # extract processor pipeline
    preprocessor = dlfuncs.preprocess_col_transformer_for_keras(cat_feats, num_feats, text_feats = text_feats)
    # instantiate XGBoost Classifier
    clf_name = 'XGBoost Classifier'
    clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=2000, n_jobs = 4, random_state = SEED)
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
    return clf_f1_scr, pipe

def best_classifier_hyperparameter(df, target_var, feats_to_exclude, stratify = True, test_size = 0.25, text_feats = None):
    best_classifier = 'XGBClassifier'
    print("hyperparameter tuning for the Best Classifier: {}".format(best_classifier))
     # split data
    X_train, X_test, y_train, y_test = mlfuncs.split_data(df, target_var=target_var, stratify = stratify, test_size = test_size)
    # extract cat_feats, num_feats, text_feats
    cat_feats, num_feats, text_feats = dlfuncs.extract_cat_num_text_feats_for_keras(X_train, feats_to_exclude=feats_to_exclude, text_feats = text_feats)
    # extract processor pipeline
    preprocessor = dlfuncs.preprocess_col_transformer_for_keras(cat_feats, num_feats, text_feats = text_feats)
    # extract best classifier model and grid params
    # grid_params, grid_model = models.classifier_ensemble_hyperparameters(best_classifier)
    xgb_clf = xgb.XGBClassifier(objective='multi:softmax', n_estimators=2000, n_jobs = 4, random_state = SEED)
    grid_params = {
            'xgbclassifier__learning_rate': [0.1, 0.01, 0.05],
            'xgbclassifier__n_estimators': [2000, 3000, 5000],
            # 'xgbclassifier__max_depth': range (2, 12, 2),
            # 'xgbclassifier__min_child_weight': [6, 12],
            # 'xgb_clf__gamma': [i/10.0 for i in range(3)]
        }
    # instantiate pipeline
    print("Creating pipeline for {}.".format(best_classifier))
    pipe = make_pipeline(preprocessor, xgb_clf)
    # dictionary to hold gridsearch model output
    GridSearchCV_model_output = {}
    # create gridsearch object
    # grid_search = GridSearchCV(pipe, param_grid=grid_params, cv = 5, scoring = 'f1_micro', refit = True, n_jobs = 4, return_train_score = True)
    grid_search = RandomizedSearchCV(pipe, param_distributions=grid_params, cv = 5, scoring = 'f1_micro', random_state=SEED, refit = True, return_train_score = True)
    print("Fitting {} Grid Model to training data.".format(best_classifier))
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
        named_steps['onehotencoder'].get_feature_names(input_features = cat_feats)
    all_feature_names = np.r_[cat_feature_names, num_feats]
    # list to hold coefs or feature imporatnces in the case of decision trees and random forest
    coefs_or_feats_imp = []
    
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


    return GridSearchCV_model_output, eda_plots.bar_plot(df_coef, 'Feature', 'Coefficient')



