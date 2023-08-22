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
import models

# import machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from tensorflow.keras import callbacks, layers
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier

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
        # df = funcs.dim_redux(df, num_cols, threshold = num_cols_threshold)
        # drop low variance features
        # df = funcs.drop_low_var_cols(df, target_var = target_var, unique_val_threshold = low_var_threshold)

    else:
        raise utils.InvalidDataFrame(df)
    
    return df

def extract_cat_num_text_feats_for_keras(X_train, feats_to_exclude, text_feats = None):
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
                if col in feats_to_exclude:
                    # num_feats.append(col)
                    pass
                else:
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

def preprocess_col_transformer_for_keras(cat_feats, num_feats, text_feats = None):
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









def split_data_for_keras(df, target_var, stratify = True, test_size = 0.3):
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
        df = df.copy()
        # extract columns
        df_cols = df.columns.tolist()
        if target_var in df_cols:
            # extract X
            X = df.drop(target_var, axis=1)
            # extract y
            df[target_var] = pd.Categorical(df[target_var])
            df[target_var] = df[target_var].cat.codes
            y = to_categorical(df[target_var])
            
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

def transformed_train_test(df, target_var, feats_to_exclude, stratify = True, test_size = 0.3, text_feats = None):
    X_train, X_test, y_train, y_test = split_data_for_keras(df, target_var, stratify = stratify, test_size = test_size)
    cat_feats, num_feats, text_feats = extract_cat_num_text_feats_for_keras(X_train, feats_to_exclude = feats_to_exclude, text_feats = text_feats)
    preprocessor = preprocess_col_transformer_for_keras(cat_feats, num_feats)
    pipe = make_pipeline(preprocessor)
    X_td = pipe.fit_transform(X_train)
    X_tdt = pipe.transform(X_test)
    n_cols = X_td.shape[1]
    return X_td, X_tdt, y_train, y_test, pipe, n_cols

def classification_model(n_cols, optimizer = 'adam', activation='relu', loss = 'categorical_crossentropy', metrics = 'accuracy'):
    model = Sequential()
    model.add(Dense(200, input_shape = (n_cols,), activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(150, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(100, activation=activation))
    model.add(Dense(100, activation=activation))
    model.add(Dense(10, activation = activation))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation = activation))
    model.add(BatchNormalization())
    model.add(Dense(3, activation = 'softmax'))
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics = [metrics])
    return model

def train_model(df, optimizer, loss, target_var, feats_to_exclude, metrics, stratify = True, test_size = 0.3, text_feats = None):
    
    X_train, X_test, y_train, y_test, pipe, n_cols = transformed_train_test(df, target_var=target_var, feats_to_exclude=feats_to_exclude, stratify = stratify, test_size = test_size, text_feats = text_feats)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
    model_save = ModelCheckpoint('models/best_model.hdf5',save_best_only = True)
    n_cols = X_train.shape[1]
    model = classification_model(n_cols = n_cols, optimizer = optimizer, loss = loss, metrics = metrics)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 50, batch_size=32, callbacks = [early_stopping, model_save], use_multiprocessing = True)
    result = model.evaluate(X_test, y_test)
    y_preds = model.predict(X_test)
    return history, result, y_test, y_preds, model, pipe

def plot_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()

def make_predictios(df_to_predict_path, submission_format_path,  my_submission_path, model_path, pipe, cols_to_drop):
    # load model
    preds = []
    if os.path.exists(model_path):
        best_model = load_model(model_path)
        if os.path.exists(df_to_predict_path):
            df_to_predict = pd.read_csv(df_to_predict_path)
            df_to_predict_cols = df_to_predict.columns.tolist()
            df_to_predict = df_to_predict.drop(cols_to_drop, axis=1)
            X_predictors = pipe.transform(df_to_predict)
            if os.path.exists(submission_format_path):
                df_to_submit = pd.read_csv(submission_format_path, index_col = 'building_id')
                y_predicted = best_model.predict(X_predictors)
                for val in y_predicted:
                    # print(predictions)
                    y_classes = val.argmax(axis=-1)
                    # print(y_classes)
                    label_index = y_classes
                    # print(label_index)
                    # preds.append(label_index)
                    if label_index == 0:
                        preds.append(1)
                    elif label_index == 1:
                        preds.append(2)
                    elif label_index == 2:
                        preds.append(3)                        
            else:
                raise utils.InvalidFilePath(submission_format_path)
        else:
            raise utils.InvalidFilePath(df_to_predict_path)
    else:
        raise utils.InvalidFilePath(model_path)

    submission = pd.DataFrame(data = preds, columns = df_to_submit.columns, index = df_to_submit.index)
    submission.to_csv(my_submission_path)

    return submission

def keras_hyperparameter_tuning(df, target_var, feats_to_exclude, params, cv, stratify, test_size, text_feats = None):
    RandomSearchCVObjects = {}
    X_train, X_test, y_train, y_test, pipe, n_cols = transformed_train_test(df, target_var=target_var, feats_to_exclude = feats_to_exclude, stratify = stratify, test_size = test_size, text_feats = text_feats)
    
    # n_cols = X_train.shape[1]

    def classification_model(n_cols = n_cols, optimizer = 'adam', activation='relu', loss = 'categorical_crossentropy', metrics = 'accuracy'):
        model = Sequential()
        model.add(Dense(128, input_shape = (n_cols,), activation=activation))
        model.add(Dense(100, activation=activation))
        model.add(Dense(100, activation = activation))
        model.add(Dense(32, activation=activation))
        model.add(BatchNormalization())
        model.add(Dense(3, activation = 'softmax'))
        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics = [metrics])
        return model

    models = KerasClassifier(build_fn=classification_model, epochs = 20, batch_size = 32)
    random_search = RandomizedSearchCV(models, param_distributions=params, cv=cv)
    random_search_results = random_search.fit(X_train, y_train)
    RandomSearchCVObjects['best_score'] = random_search_results.best_score_
    RandomSearchCVObjects['best_params'] = random_search_results.best_params_
    RandomSearchCVObjects['data_prep_pipe'] = pipe
    return RandomSearchCVObjects




