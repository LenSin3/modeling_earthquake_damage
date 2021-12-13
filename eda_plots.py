from seaborn import palettes
from seaborn.matrix import heatmap
from seaborn.utils import ci
import funcs
import utils
# import dependencies
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re



def plot_qa(df_null):
    """Plot Count of Nulls of columns with nulls.

    The plot is done for columns with nulls in Dataframe.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe to check and plot for nulls.
        

    Returns
    -------
    Seaborn.barplot
    """
        
    null_columns = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsInColumn', 'PercentOfNullsInData']
    df_null_cols = df_null.columns

    # check if columns match
    mem_cols = all(col in null_columns for col in df_null_cols)
    if mem_cols:
        if df_null['CountOfNulls'].sum() == 0:
            print("There are zero nulls in the DataFrame.")
            print("No plot to display!")
        else:
            null_df = df_null.loc[df_null['CountOfNulls'] > 0]
            null_df.reset_index(drop = True, inplace = True)

            fig, ax = plt.subplots()
            # the size of A4 paper lanscape
            fig.set_size_inches(15, 10)
            sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
            bar = sns.barplot(y = 'PercentOfNullsInColumn', x = 'Feature' , data = null_df, ci = False)
            for i in range(len(null_df)):
                bar.text(i, null_df['PercentOfNullsInColumn'][i] + 0.25, str(round(null_df['PercentOfNullsInColumn'][i], 2)),
                fontdict= dict(color = 'blue', fontsize = 10, horizontalalignment = 'center'))
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.title('Percentage of Nulls in Column')
            plt.show()
    else:
        raise utils.UnexpectedDataFrame(df_null)

def plot_qa_mtpltlib(df_null):
    """Plot Count of Nulls of columns with nulls.

    The plot is done for columns with nulls in Dataframe.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe to check and plot for nulls.
        

    Returns
    -------
    matplotlib.pyplot.barplot 
    """
        
    null_columns = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsInColumn', 'PercentOfNullsInData']
    df_null_cols = df_null.columns

    # check if columns match
    mem_cols = all(col in null_columns for col in df_null_cols)
    if mem_cols:
        if df_null['CountOfNulls'].sum() == 0:
            print("There are zero nulls in the DataFrame.")
            print("No plot to display!")
        else:
            null_df = df_null.loc[df_null['CountOfNulls'] > 0]
            null_df.reset_index(drop = True, inplace = True)

            plt.figure(figsize=(15, 8))
            plt.bar('Feature', 'CountOfNulls', data = df_null, color = 'orange', width = 0.9, align = 'center', edgecolor = 'blue')
            # i = 1.0
            # j = 2000
            # for i in range(len(null_df)):
            #     plt.annotate(null_df['PercentOfNullsInColumn'][i], (-0.1 + i, null_df['PercentOfNullsInColumn'][i] + j))
            plt.xticks(rotation = 90)
            plt.xlabel("Columns")
            plt.ylabel("Percentage")
            plt.title("Count of Nulls in Column")
            plt.show()
    else:
        raise utils.UnexpectedDataFrame(df_null)


def plot_unique_vals_count(df):
    """Plot count of unique values per column.

    The plot is done for unique values of all columns.

    Parameters
    ----------
    df : Dataframe
        Dataframe to check for unique values per column.        

    Returns
    -------
    seaborn.barplot
    """
    unique_vals = funcs.unique_vals_counts(df) 
    fig, ax = plt.subplots()
    # the size of A4 paper lanscape
    fig.set_size_inches(15, 12)
    sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
    bar = sns.barplot(y = 'count', x = 'column' , data = unique_vals)
    for i in range(len(unique_vals)):
                 bar.text(i, unique_vals['count'][i] + 0.25, str(unique_vals['count'][i]),
                 fontdict= dict(color = 'blue', fontsize = 10, horizontalalignment = 'center'))
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.title('Count of Unique Values per Column')
    plt.show()

    # plt.figure(figsize=(15, 8))
    # plt.bar('column', 'count', data = unique_vals, color = 'orange', width = 0.7, align = 'center', edgecolor = 'blue')
    # plt.xticks(rotation = 90)
    # plt.xlabel("Column")
    # plt.ylabel("Count")
    # plt.title("Count of Unique Values in Column")
    # plt.show()

def plot_unique_vals_column(df, col, normalize = False):
    """Plot value counts in a column.

    Value counts are calculated for a single column and plotted.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to check for unique values
    normalized : bool, optional
        If true this function normalizes the counts.
        (Default value = False)

    Returns
    -------
    seaborn.barplot
    """
    if normalize:
            unique_col_vals = funcs.unique_vals_column(df, col, normalize = True) 
            fig, ax = plt.subplots()
            # the size of A4 paper lanscape
            fig.set_size_inches(15, 12)
            sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
            sns.barplot(y = 'percentOfTotal', x = col , data = unique_col_vals)
            plt.setp(ax.get_xticklabels(), rotation=0)
            plt.title('Percentage of Unique Values in {}'.format(col))
            plt.show()
    else:
            unique_col_vals = funcs.unique_vals_column(df, col, normalize = False) 
            fig, ax = plt.subplots()
            # the size of A4 paper lanscape
            fig.set_size_inches(15, 12)
            sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
            sns.barplot(y = 'count', x = col , data = unique_col_vals)
            plt.setp(ax.get_xticklabels(), rotation=0)
            plt.title('Count of Unique Values in {}'.format(col))
            plt.show()

def count_plot(df, col, **hue):
    """Plot value counts in a column.

    Value counts are calculated for a single column and plotted.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to check for unique values.
    **hue: dict
        Keyword arguments.
    

    Returns
    -------
    seaborn.countplot
    """
    var = hue.get('var', None)
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols and not var:
            fig, ax = plt.subplots()
            # the size of A4 paper lanscape
            fig.set_size_inches(15, 8)
            sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
            sns.countplot(y = df[col], order=df[col].value_counts(ascending=False).index, ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.title('Count of {}'.format(col))
            plt.savefig('images/{}.png'.format(col))
            plt.show()
        elif col in df_cols and var:
            if var in df_cols:
                fig, ax = plt.subplots()
                # the size of A4 paper lanscape
                fig.set_size_inches(15, 8)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                sns.countplot(y = df[col], order=df[col].value_counts(ascending=False).index, hue = df[var],  ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=90)
                plt.title('{} vs {}'.format(col.title(), var.title()))
                plt.savefig('images/{}v{}.png'.format(col, var))
                plt.show()
            else:
                raise utils.InvalidColumn(var)
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)


def bar_plot(df, col_x, col_y):
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col_x in df_cols:
            if col_y in df_cols:
                fig, ax = plt.subplots()
                # the size of A4 paper lanscape
                fig.set_size_inches(15, 12)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                bar = sns.barplot(x = col_x, y = col_y, data = df, ci = None, ax=ax)
                # for i in range(len(df)):
                #     bar.text(i, df[col_y][i] + 0.25, str(round(df[col_y][i], 2)),
                #     fontdict= dict(color = 'blue', fontsize = 10, horizontalalignment = 'center'))
                plt.setp(ax.get_xticklabels(), rotation=90)
                plt.title('Rank of {} by {}'.format(col_x, col_y))
                plt.savefig('images/{}_{}.png'.format(col_x, col_y))
                plt.show()
            else:
                raise utils.InvalidColumn(col_y)
        else:
            raise utils.InvalidColumn(col_x)
    else:
        raise utils.InvalidDataFrame(df)


def pie_plot(df, col):
    """Plot pie plot of values in a column.

    Percentage of values counts are calculated for a single column and plotted.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to plot.


    Returns
    -------
    pandas.DataFrame.plot.pie
    """
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols:
            pie_data = df[col].value_counts().reset_index()
            pie_data.set_index('index', inplace=True)
            if len(pie_data) > 5:
                print("{} contains more than 5 values.".format(col))
                print("Visualization best practices recommends using a barplot for variables with more than 5 unique values.")
            elif len(pie_data) <= 5:
                pie_data.plot.pie(y=col, autopct='%0.1f%%', figsize=(10, 10))
                plt.savefig('images/{}.png'.format(col))
                plt.title('{} Composition'.format(col.title()))
                plt.show()
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)

def hist_distribution(df, col, bins = 30, kde = False):
    """Plot distribution of values in a column.

    Histogram with kde(kernel density estimate) of values in a numerical column are plotted.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to plot.
    bins : integer, optional
        Number of bins in distribution.
        (Default value = 30)
    kde : boolean, optional
        If true, a kde(kernel density estimate) plot is included in histogram.
        (Default value = False)

    Returns
    -------
    seaborn.histplot
    """
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols:
            if df[col].dtypes == 'int64' or df[col].dtypes == 'int32' or df[col].dtypes == 'float64':
                fig, ax = plt.subplots()
                # the size of A4 paper lanscape
                fig.set_size_inches(15, 8)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                sns.histplot(df[col], bins=bins, kde=kde, ax=ax, color="orange")
                plt.setp(ax.get_xticklabels(), rotation=90)
                plt.title('Distribution of {}'.format(col.title()))
                plt.savefig('images/{}_distribution.png'.format(col))
                plt.show()
            else:
                raise utils.InvalidDataType(col)
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)

def box_distribution(df, col, x =None, hue = None):
    """Plot distribution of values in a column.

    Box plot distribution of values in a column are plotted. If column to plot distribution is numerical, a single plot is generated.
    If column is categorical, x value should be provided together with hue to plot distribution of categorical values varied by the hue.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check for unique values.
    col : str
        Name of column to plot.
    x : str, optional
        Column in DataFrame.
        (Default value = None)
    hue : str, optional
        Column in DataFrame.
        (Default value = None)

    Returns
    -------
    seaborn.boxplot
    """
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols and not x and not hue:
            if df[col].dtypes == 'int64' or df[col].dtypes == 'int32' or df[col].dtypes == 'float64':
                fig, ax = plt.subplots()
                # the size of A4 paper lanscape
                fig.set_size_inches(15, 8)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                sns.boxplot(x = df[col])
                plt.setp(ax.get_xticklabels(), rotation=0)
                plt.title('Box Distribution of {}'.format(col.title()))
                plt.savefig('images/{}_distribution.png'.format(col))
                plt.show()
            else:
                raise utils.InvalidDataType(col)
        elif (col in df_cols and df[col].dtype == 'object') and (x and hue):
            if x in df_cols and hue in df_cols:
                if (df[x].dtypes == 'int64' or df[x].dtypes == 'int32' or df[x].dtypes == 'float64') and (df[hue].dtypes == 'object'):
                    fig, ax = plt.subplots()
                    # the size of A4 paper lanscape
                    fig.set_size_inches(8, 15)
                    sns.boxplot(y= col, x = x,
                                hue = hue, palette='vlag',
                                data=df)
                    sns.despine(offset=10, trim=True)
                    # plt.setp(ax.get_xticklabels(), rotation=45)
                    plt.title('Distribution of {} vs {}'.format(x.title(), col.title()))
                    plt.savefig('images/{}_{}_bxplt.png'.format(x, col))
                    plt.show()
                elif (df[x].dtypes != 'int64' and df[x].dtypes != 'int32' and df[x].dtypes != 'float64'):
                    raise utils.InvalidDataType(x)
                elif df[hue].dtypes != 'object':
                    raise utils.InvalidDataType(hue)
                # else:
                #     if df[x].dtypes != 'int64' or df[x].dtypes != 'int32' or df[x].dtypes != 'float64':
                #         raise utils.InvalidDataType(x)
                #     elif df[hue].dtypes != 'object':
                #         raise utils.InvalidDataType(hue)
            else:
                if x not in df_cols:
                    raise utils.InvalidColumn(x)
                elif hue not in df_cols:
                    raise utils.InvalidColumn(hue)
        else:
            if col not in df_cols:
                raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)


def box_plot(df, col, y = None):
    if isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        if col in df_cols:
            if not y:
                if df[col].dtypes == 'int64' or df[col].dtypes == 'float64':
                    fig, ax = plt.subplots()
                    # the size of A4 paper lanscape
                    fig.set_size_inches(15, 8)
                    sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                    sns.boxplot(x = df[col], color = 'green')
                    plt.setp(ax.get_xticklabels(), rotation=0)
                    plt.title('Box Distribution of {}'.format(col.title()))
                    plt.savefig('images/{}_distribution.png'.format(col))
                    plt.show()
                else:
                    raise utils.InvalidDataType(col)
            if y:
                if y in df_cols:
                    if df[y].dtypes == 'object':
                        fig, ax = plt.subplots()
                        # the size of A4 paper lanscape
                        fig.set_size_inches(15, 8)
                        sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                        sns.boxplot(y = df[col], x = df[y])
                        plt.setp(ax.get_xticklabels(), rotation=0)
                        plt.title('{} Distribution by {}'.format(col.title(), y.title()))
                        plt.savefig('images/{}_{}_distribution.png'.format(col,y))
                        plt.show()
                    else:
                        raise utils.InvalidDataType(y)
                else:
                    raise utils.InvalidColumn(y)
        else:
            raise utils.InvalidColumn(col)

def check_for_correlation(df, num_cols):
    """Plot correlation heatmap of numerical variables.

    Masked correlation heatmap is created of numerical values specified.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing numerical features to check for correlation.
    num_cols: List
        Numerical columns to evaluate for correlation.

    Returns
    -------
    None
    """
    # check for valid dataframe
    if isinstance(df, pd.DataFrame):
        # extract dataframe columns
        df_cols = df.columns.tolist()
        # check if numeric columns are in dataframe
        membership = all(col in df_cols for col in num_cols)
        # if in dataframe
        if membership:
            # extract data of numeric columns
            df_corr = df[num_cols]
            # build the heatmap
            sns.set(font_scale = 1)
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 8)
            mask = np.triu(np.ones_like(df_corr.corr(), dtype = bool))
            heatmap = sns.heatmap(df_corr.corr(), vmin = -1, vmax = 1, mask = mask, annot = True)
            heatmap.set_title('Correlation Heatmap', fontdict = {'fontsize': 12}, pad = 12)
            plt.setp(ax.get_xticklabels(), rotation = 90)
            plt.setp(ax.get_yticklabels(), rotation = 0)
            plt.show()
        else:
            not_cols = []
            for col in num_cols:
                if col not in df_cols:
                    not_cols.append(col)
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)
