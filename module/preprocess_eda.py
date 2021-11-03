import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import warnings
import scipy
from scipy.stats import boxcox, boxcox_normmax
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from typing import List

warnings.filterwarnings('ignore')


def preprocess_null_values(
    data: pd.DataFrame, dropped_columns: list, kept_columns: list, filled_values: str
) -> pd.DataFrame:
    '''
    Process unmeaningful NA and meaningful Na:
    1. Drop columns with unmeaningful null values.
    2. Fill 'No'(filled_values) into columns with meaningful null values.
    '''
    preprocess_data = data.loc[:, ~data.columns.isin(dropped_columns)]
    preprocess_data[kept_columns] = preprocess_data[kept_columns].fillna(filled_values)
    return preprocess_data


def select_column_names_with_types(data: pd.DataFrame, column_type: list) -> list:
    '''
    Select the names of columns with specific types, such as integer, float, object ... etc.
    Return: list of column names
    '''
    column_names = data.select_dtypes(include=column_type).columns.tolist()
    return column_names


def count_null_nonnull_variables_in_columns(
    data: pd.DataFrame, name_exit: str, name_null: str
) -> pd.DataFrame:
    '''
    Count Null and Non-null values in columns with null values.
    NOTE: Transpose dataframe is convinent to plot graphs
    '''
    null_columns = data.columns[data.isnull().any()]
    null_data = data.loc[:, null_columns]
    for column in null_data.columns:
        null_data.loc[~null_data[column].isnull(), column] = name_exit
        null_data.loc[null_data[column].isnull(), column] = name_null
    null_data = null_data.apply(lambda x: pd.Series.value_counts(x)).T
    null_data = null_data.sort_values(by=[name_exit])
    return null_data


def set_graph_null_nonnull_values_in_columns(
    data: pd.DataFrame, name: str, color_marker: str, color_line: str, line_width: int
) -> go.Bar:
    '''
    Use graph_obj to plot 'bars' for null and non-null values in columns with null values
    Args:
        name: the name of null and non-null in columns
    NOTE: customized
    '''
    graph_setting = go.Bar(
        x=data[name],
        y=data.index,
        name=name,
        orientation='h',
        marker=dict(
            color=color_marker,
            line=dict(
                color=color_line,
                width=line_width
            )
        )
    )
    return graph_setting


def display_bar_graph(graphs: list, barmode: str, title: str) -> py.iplot:
    '''
    Display graphs with graph settings.
    Args:
        graphs: settings of graphs,
        barmode: barmode corresponding to bar
    NOTE: layout can be added customized arguments for graphs of different types
    NOTE: customized
    '''
    layout = go.Layout(
        width=800,
        height=400,
        barmode=barmode,
        title=title
    )
    fig = go.Figure(data=graphs, layout=layout)
    py.iplot(fig)


def calculate_correlation_two_columns(
    data: pd.DataFrame, column_one: str, column_two: str
) -> float:
    '''
    Calculate correlation between two columns.
    Return: correlation
    '''
    correlation = data[[column_one, column_two]].corr().loc[column_one, column_two].round(2)
    return correlation


def calculate_max_correlation_one_column_to_multiple_columns(
    data: pd.DataFrame, column: str
) -> float:
    '''
    Calculate the corrlation between one variable to other variables
    and select max absoulte correlation
    Return: max correlation
    '''
    max_correlation = round(max(abs(data.corr().loc[column, :].drop(column))), 2)
    return max_correlation


def plot_jointplot(
    data: pd.DataFrame, column1: str, column2: str,
    kind: str, xlabel: str, ylabel: str, title: str
):
    '''
    Plot jointplot for two columns, which can display a relationship between 2 variables
    TODO: customized function for kind and figsize
    '''
    plt.figure(figsize=(6, 8))
    sns.jointplot(data[column1], data[column2], kind=kind)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_scatter_on_levels(
    data: pd.DataFrame, plot_settings, color_number: int,
    levels: range, xlabel: str, ylabel: str, title: str,
    column_level: str, column_x: str, column_y: str
):
    '''
    Plot scatter plots based on difference levels/conditions.
    TODO: customized for color, legend location, values of levels in for loop
    '''
    plot_settings.set_prop_cycle(cycler(color=sns.color_palette('Set2_r', 10)))
    for level in levels:
        index_level = data[column_level] == level
        plot_settings.scatter(
            data.loc[index_level, column_x], data.loc[index_level, column_y], label=level
            )
    plot_settings.legend(bbox_to_anchor=[1.1, 1])
    plot_settings.set_xlabel(xlabel)
    plot_settings.set_ylabel(ylabel)
    plot_settings.set_title(title)


def plot_box(data: pd.DataFrame, column_x: str, column_y: str, plot_settings, title: str):
    '''
    Plot box for two variables
    TODO: customized for fontsize
    '''
    sns.boxplot(x=column_x, y=column_y, data=data, ax=plot_settings)
    plot_settings.set_title(title, fontsize=12)


def predict_linear_model(data: pd.DataFrame, columns_x: list, column_y: str):
    '''
    Predict results for linear models
    '''
    train = data.copy()
    model = linear_model.LinearRegression()
    model.fit(train[columns_x], train[column_y])
    predict_result = model.predict(train[columns_x])
    return predict_result


def plot_multiple_plots(
    data: pd.DataFrame, column_x: str, column_y: str, plot_settings, multiple: bool,
    title: str, xlabel: str, ylabel: str):
    '''
    Plot multiple plots
    TODO: customized for multiple settings
    '''
    if multiple:
        plot_settings.scatter(data[column_x], column_y, color='r')
    plot_settings.plot(data[column_x], column_y, color='blue', linewidth=3)
    plot_settings.set_title(title, fontsize=12)
    plot_settings.set_xlabel(xlabel)
    plot_settings.set_ylabel(ylabel)
# TEST


def checkOutlier(df, m = 4):
    uniOutlier = dict().fromkeys(df.columns, None)
    outSample = abs(df - df.mean()) > 4 * df.std()
    outSum = (abs(df - df.mean()) > 4 * df.std()).sum()
    for key in uniOutlier.keys():
        uniOutlier[key] = set(outSample.index[outSample.loc[:, key]])
    outportion = outSum / df.shape[0]
    outportion = outportion[outportion != 0].sort_values()
    outlierLst = outportion.index.tolist()
    return uniOutlier, outlierLst


def outlierCounter(outlierDict, exceptionLst = ['SalePrice']):
    inter = Counter()
    name = defaultdict(list)
    coreKey = set(outlierDict.keys()).difference(exceptionLst)
    for key in coreKey:
        value = outlierDict[key]
        for val in value:
            inter[val] += 1
            name[val].append(key)
    res = pd.DataFrame([inter, name], index = ['count', 'variable']).T
    res = res.sort_values('count', ascending = False)
    return res


def bivarCI(dfNum, y = 'SalePrice', outer = 10, z_score = 0.00006, cols = 2):
    
    colNum = dfNum.shape[1]
    row, col = divmod(colNum-1, cols)
    if row == 1 and col == 0: row += 1
    if col != 0: row += 1
    
    
    z_under = z_score * 0.98
    z_upper = z_score * 1.02
    
    biOutlier = dict().fromkeys(dfNum.columns, None)
    #f, axes = plt.subplots(row, cols, figsize = (4*cols, 4*row))
    #f.suptitle('Bivaraite CI', fontsize = 12)
    for ix, var1 in enumerate(dfNum.columns):
        if var1 == y: break
        r,c = divmod(ix, cols)
        dfPart = dfNum.loc[:, [var1,y]]
        dfPart = dfPart[~dfPart.isnull()].copy()
        dfPart = dfPart.loc[dfPart.loc[:, var1] != 0,:]
        dfPart = (dfPart - dfPart.mean()) / dfPart.std()
        F, X, Y, posProb = bivarConverter(dfPart, outer, z_under, z_upper, N = 700)
        #axes[r,c].contourf(X, Y, posProb)
        #axes[r,c].scatter(dfPart.loc[:, var1], dfPart.loc[:, y], alpha = 1)
        #axes[r,c].set_title('Bivaraite CI ' + var1)
        dfPartProb = F.pdf(dfPart.values)
        outIndex = dfPart.index[dfPartProb < z_score]
        biOutlier[var1] = set(outIndex.tolist())
    #f.tight_layout(rect = [0, 0.03, 1, 0.95])
    #plt.show()
    
    return biOutlier

def bivarConverter(df, outer, z_under, z_upper, N = 500):
    x_init, y_init = df.min() - outer
    x_end, y_end = df.max() + outer
    X = np.linspace(x_init, x_end, N)
    Y = np.linspace(y_init, y_end, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:,:,0] = X
    pos[:,:,1] = Y
    F = multivariate_normal(mean=df.mean().values, cov=df.corr().values)
    posProb = F.pdf(pos)
    posProb[(z_under < posProb) & (posProb < z_upper)] = 1
    posProb[(z_under > posProb) | (posProb < z_upper)] = 0
       
    return F , X, Y, posProb