# import dependencies
from ctypes import windll
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import utils

## function to read data
## check for data quality issues

def read_and_qa(pth_str):
    # check if path exists
    if os.path.exists(pth_str):
        # read file
        df = pd.read_csv(pth_str)
        # take 10000 random rows
        # df = df.iloc[np.random.choice(np.arange(len(df)), size=10000)]
        # df.reset_index(drop = True, inplace = True)
        df = df.copy()
        ##############################
        # check for nulls
        data_types = []
        non_nulls = []
        nulls = []
        null_column_percent = []
        null_df_percent = []
        df_cols = df.columns
        print("There are {} columns and {} records in the dataframe.".format(len(df_cols), len(df)))
        # loop through columns and capture the variables above
        print("Extracting count and percentages of nulls and non nulls")
        for col in df_cols:
            # extract null count
            null_count = df[col].isna().sum()
            nulls.append(null_count)
                
            # extract non null count
            non_null_count = len(df) - null_count
            non_nulls.append(non_null_count)
                
            # extract % of null in column
            col_null_perc = 100 * null_count/len(df)
            null_column_percent.append(col_null_perc)
            
            if null_count == 0:
                null_df_percent.append(0)
            else:
                # extract % of nulls out of total nulls in dataframe
                df_null_perc = 100 * null_count/df.isna().sum().sum()
                null_df_percent.append(df_null_perc)
                
            # capture data types
            data_types.append(df[col].dtypes) 
    
    else:
        raise utils.InvalidFilePath(pth_str)
            
    # create zipped list with column names, data_types, nulls and non nulls
    lst_data = list(zip(df_cols, data_types, non_nulls, nulls, null_column_percent, null_df_percent))
    # create dataframe of zipped list
    df_zipped = pd.DataFrame(lst_data, columns = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsInColumn', 'PercentOfNullsInData'])
    return df, df_zipped


def unique_vals_counts(df: DataFrame) -> DataFrame:
    """Count unique values in dataframe.

    The count is perfromed on all columns in the dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe to check for unique values per column.
        

    Returns
    -------
    DataFrame
    """
    if isinstance(df, pd.DataFrame):
        df_cats = df.select_dtypes(include ='object')
        df_cats_cols = df_cats.columns.tolist()
        for col in df_cats_cols:
            if 'essay' in col or 'last_online' in col:
                df_cats = df_cats.drop(col, axis=1)
        vals = df_cats.nunique().reset_index()
    else:
        raise utils.InvalidDataFrame(df)
    return vals.rename(columns = {'index': 'column', 0: 'count'})

def unique_vals_column(df: DataFrame, col: str, normalize = False) -> DataFrame:
    """Count unique values in a single column in a dataframe.

    Value counts are calculated for a single column and tabulated.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check 
        for unique values.
    col : str
        Name of column to check for unique values.
    normalized : bool, optional
        If true this function normalizes the counts.
         (Default value = False)
         

    Returns
    -------
    DataFrame
    """
    if isinstance(df, pd.DataFrame):
        if col in df.columns:
            uniques = df[col].value_counts().reset_index().rename(columns = {'index': col, col : 'count'})
            if normalize:
                uniques = df[col].value_counts(normalize = True).reset_index().rename(columns = {'index': col, col : 'percentOfTotal'})
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)
    return uniques

def merge_val_labels(df1, df2, col_to_merge):
    """Merge two DataFrames

    Two DataFrames are merged given a shared merging column.

    Parameters
    ----------
    df1: DataFrame
        First DataFrame to merge.
    df2: DataFrame
        Second DataFrame to merge.
    col: str
        name of column to merge on.
    
    Returns
    -------
    DataFrame
    """
    # check if data are valid dataframes
    if isinstance(df1, pd.DataFrame):
        if isinstance(df2, pd.DataFrame):
            # extract column names
            df1_cols = df1.columns.tolist()
            df2_cols = df2.columns.tolist()
            # check if column to merge is in both dataframe columns
            if col_to_merge in df1_cols:
                if col_to_merge in df2_cols:
                    df_merged = df1.merge(df2, how = 'left', on = col_to_merge)
                else:
                    raise utils.InvalidColumn(col_to_merge)
            else:
                raise utils.InvalidColumn(col_to_merge)
        else:
            raise utils.InvalidDataFrame(df2)
    else:
        raise utils.InvalidDataFrame(df1)
    return df_merged


def transform_to_cat(df, cols_to_cat):
    """Convert column values to object data type.

    DataFrame columns are converted to oject data types.

    Parameters
    ----------
    df: DataFrame 
        Contains columns to convert to object data type.
    cols_to_cat: list
        List of columns names to convert to object data type.

    Returns
    -------
    DataFrame
    """
    # check for valid dataframe
    if isinstance(df, pd.DataFrame):
        # extract columns
        df_cols = df.columns.tolist()
        # check if all columns to convert are in dataframe
        membership = all(col in df_cols for col in cols_to_cat)
        if membership:
            # convert each col to object data type
            for col in cols_to_cat:
                df[col] = df[col].astype(object)
        else:
            not_cols = []
            for col in cols_to_cat:
                if col not in df_cols:
                    not_cols.append(col)
            raise utils.InvalidColumn(not_cols)
    else:
        raise utils.InvalidDataFrame(df)
    return df
