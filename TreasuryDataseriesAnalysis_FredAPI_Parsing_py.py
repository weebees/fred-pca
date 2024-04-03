"""
    author: vbs
"""

# coding: utf-8

# ## Install all required packages
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "fred_requirements.txt"])

## Import Utility module
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime

## Import FRED API to query and retrieve data using JSON and HTTP.
import requests, json, bs4

## Import sklearn module for Dimensionality reduction functionality
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD

"""
# Step 1: Query and retrieve the following data series from online macroeconomic database FRED using their API (https://fred.stlouisfed.org/):

_ | Data series | FRED Ticker

1 | US 1Y Treasury | GS1

2 | US 2Y Treasury | GS2

3 | US 3Y Treasury | GS3

4 | US 5Y Treasury | GS5

5 | US 7Y Treasury | GS7

6 | US 10Y Treasury | GS10
"""


# ## Utility Functions

# Sources/Endpoint:
# 1. Documentation and API source: https://fred.stlouisfed.org/docs/api/fred/
# 2. FRED API call for series: https://fred.stlouisfed.org/docs/api/fred/release_series.html

## Utility function for querying and retrieving the data from FRED API.
def get_series_using_fred(api_key: str, series: str) -> pd.DataFrame:
    """
        Utility function that helps to convert the JSON response from FRED API into a dataframe for analysis.
        Input:
            api_key, the string API Key for FRED API.
            series, a string representing the series to be queried and retrieved from FRED API.
        Output:
            df, a pandas DataFrame version of the FRED API response.
            returns df
    """
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={api_key}"
    response = requests.get(url)
    bs = bs4.BeautifulSoup(response.content, ["lxml", "xml"])
    obs = bs.find_all("observation")  ## All observation tags from the JSON xml tags
    cols = (obs[0].attrs.keys())  ## Column names from the observation xml tag

    df = pd.DataFrame()
    for node in obs:
        node_dict = {}
        for col in cols:
            if col == "value":
                node_dict[series] = node[col]
            else:
                node_dict[col] = node[col]
        df = pd.concat([df, pd.DataFrame(node_dict, index=[0])])
    return df


## Utility function for plotting the Treasury data series.
def plot_treasury(df: pd.DataFrame, name: str) -> None:
    """
        Plot the graph with the first two columns of dataframe as the axes.
        Saves an SVG image of the graph that is plotted.

        Input:
            df, a DataFrame that is to be plotted -
                First column is treated as the x-axis,
                Second column is treated as the y-axis.
            name, a string which is passed to save the svg image
                  of the graph.
        Output:
            Saves an SVG image of the graph that is plotted.
            returns None [Void function]
    """
    if len(df.columns) == 2:
        columns = list(df.columns)
        X, y = df[columns[0]], df[columns[1]]
        plt.plot(df[columns[0]], df[columns[1]].astype(float), label=f"GS{name}")
        plt.xlabel("Time (year)")
        plt.ylabel("Yield (%)")
        plt.suptitle("Yield of the 6 data series")
        plt.legend()


## Get the data series in a dictionary datastructure for easy access.
def get_all_series(API_KEY: str) -> dict:
    """
        Define a dictionary data structure for easy access to the six different series.
        Plot and save the data series on a single graph to visualize the difference in yields.

        Input:
            API_KEY, string that represents the FRED API Key.

        Output:
            _dict, a dictionary with key->Treasury series name and value->dataframe with monthly values
            returns _dict
    """
    _dict = {}
    figure(figsize=(10, 4), dpi=100)
    for i in [1, 2, 3, 5, 7, 10]:
        rq = ["date", f"GS{i}"]
        req = get_series_using_fred(API_KEY, f'GS{i}')

        ## Since we retrieve information as strings we convert respective columns to float and datetime formats by casting.
        req[f"GS{i}"] = req[f"GS{i}"].astype(float)
        req["date"] = pd.to_datetime(req["date"], format="%Y-%m-%d")
        req = req[rq]
        _dict[i] = req
        plot_treasury(req, i)
        req.to_csv(f"./Results/GS{i}.csv", index=False)

    plt.savefig(f"./Results/Dataset_value_plot.svg", dpi=500)
    return _dict


## Merge all the treasury bond datapoints as features then we compute the monthly change.
def merge_data_series(series: dict) -> pd.DataFrame:
    """
        Merges the six series into one dataframe.

        Input:
            series, a dictionary with key->Treasury series name and value->dataframe with monthly values

        Output:
            dx, a dataframe with each column representing a series
    """
    dx = None
    for k, v in treasury_bills.items():
        v[f"GS{k}_CHG"] = v[f"GS{k}"].pct_change()
        if dx is None:
            dx = v
        else:
            dx = pd.merge(dx, v, how='outer', on=["date"])

    dx.to_csv(os.path.join(directory, "Merged Treasury File.csv"), index=False)
    return dx


## Standardize the dataset using sklearn pre processing libraries and transform data using an imputer.
def standardize_columns(data: pd.DataFrame, strategy: str) -> np.ndarray:
    """
        Plot the graph with the first two columns of dataframe as the axes.
        Saves an SVG image of the graph that is plotted.

        Updates the given dataframe by replacing the nan values with 0.
        Fits the data in the dataframe using StandardScaler from sklearn.

        Input: data, input DataFrame to be transformed using StandardScaler.
               strategy, the string input for strategy argument in SimpleImputer method.

        Output: standardized_data, a numpy array which consists of transformed
                                   values for all rows and columns.
    """
    ## Reason for imputing the nan values ->
    ## we need the dataset to be free from nan values since the dimensionality reduction techniques are impacted by each value.
    if strategy == "median":
        data = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(data)
    elif strategy == "constant":
        ## Reason for filling nan with zeroes -> We use zero since these bonds have not yielded any returns.
        data = SimpleImputer(missing_values=np.nan, fill_value=0).fit_transform(data)
    else:
        data = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data)
    standardized_data = StandardScaler().fit_transform(data)
    return standardized_data


## We prepare the data such that it is Standardized for Principal Component Analysis:
def dimensionality_reduction_function(features: list, data: pd.DataFrame, reduction_type: str):
    """
        Given a dataset with features - we perform Principal Component Analysis(PCA)/Truncated Singular Value
        Decomposition (tSVD) to get reduced dimensions.

        Input: features, a list of the feature colums.
               data, a pandas dataframe of the dataset.
               reduction_type, a string representing PCA or tSVD for dimensionality reduction.
        Output: Perform PCA on imputed and standardized data to get reduced dimensions.
                Also, print relavant information while simulating the PCA dimensionality reduction.
                Returns model_df, which is has the data with reduced dimensions.
    """
    n_components = range(1, len(features) + 1)
    main_model = None
    main_model_df = None
    print(f"{reduction_type} Simulation:")
    for n in n_components:
        if reduction_type.upper() == "PCA":
            model = PCA(n_components=n, random_state=10)
            model_data = model.fit_transform(data)
            model_features = [f"Principal Component {x + 1}" for x in range(len(model_data[0]))]
            model_df = pd.DataFrame(model_data, columns=model_features)
            cols = ["date"] + list(model_df.columns)
            model_df["date"] = dx["date"]
            model_importance = model.explained_variance_ratio_
        elif reduction_type.upper() == "TSVD":
            model = TruncatedSVD(n_components=n, random_state=20)
            model_data = model.fit_transform(data)
            model_features = [f"Feature {x + 1}" for x in range(len(model_data[0]))]
            model_df = pd.DataFrame(model_data, columns=model_features)
            model_df["date"] = dx["date"]
            cols = ["date"] + list(model_df.columns)
            model_importance = model.explained_variance_ratio_

        ## Print the simulation:
        print("--" * 10)
        if n == len(features):
            print(f"If we do not reduce the dimensions using {reduction_type}:")
            print(f"Number of feature dimensions = {data.shape[1]}")
            print(f"Number of dimensions reduced = {6 - n}")
            print(f"Number of feature dimensions after dimensionality reduction = {model_data.shape[1]}")
            print(f"Variance explained by the feature = {round(sum(model_importance) * 100, 2)}%")
        else:
            print(f"If we reduce the dimensions from 6 to {n} using {reduction_type}:")
            print(f"Number of feature dimensions = {data.shape[1]}")
            print(f"Number of dimensions reduced = {6 - n}")
            print(f"Number of feature dimensions after dimensionality reduction = {model_data.shape[1]}")
            print(f"Variance explained by the feature = {round(sum(model_importance) * 100, 2)}%")

        for i in range(len(model_importance)):
            print(f"{round(model_importance[i] * 100, 2)}% of variance is explained by Principal Component {i + 1}")
        model_df = model_df[cols]
        if n == 2:
            main_model = model
            main_model_df = model_df
        model_df.to_csv(os.path.join(directory, f"{reduction_type}_DimensionsReduced_to_{n}.csv"), index=False)
    return model_df, model, main_model


## Given the sklearn model used to fit the dataset used for dimensionality reduction we extract the explained variance.
def visualize_reduction(model, features: list) -> (pd.DataFrame, pd.DataFrame):
    """
        Extract the variance explained metrics to visualize the model for both PCA and tSVD
        Input: model, the sklearn model either PCA or tSVD.
               features, the list of feature columns.
        Output: dffx, a dataframe:
            (1) the first principal component rows -> 2 since the variance explained is above 95%
                threshold which is enough for reducing the dimensions down.
            (2) The remaining four principal component rows.
    """
    n_pcs = model.components_.shape[0]

    dffx = pd.DataFrame(model.components_, columns=features)
    dffx["PC"] = dffx.index + 1

    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
    variances = [model.explained_variance_ratio_[i] * 100 for i in range(n_pcs)]
    dffx["Explained Variance Ratio"] = variances

    most_important_names = [features[most_important[i]] for i in range(n_pcs)]

    dffx["Cummulative Explained Variance"] = dffx["Explained Variance Ratio"].cumsum()
    dffx["Most important feature in Principal Component"] = most_important_names

    return dffx


## Utility function that helps in generating scree plots
def scree_plot(df: pd.DataFrame, name: str) -> None:
    """
        Generates a scree plot for visualizing the results of dimensionality reductions.
        Input:
            df, a dataframe which contains the principal components and cummulative variance explained by them.
            name, a string representing the model - either PCA or tSVD.

        Ouptut:
            Plots and saves the graph for visualizing the variance explained by principal components
            returns None (Void function)
    """
    plt.scatter(df["PC"], df["Cummulative Explained Variance"], c='r')
    plt.plot(df["PC"], df["Cummulative Explained Variance"])
    x_ticks = df["PC"].tolist()
    plt.xlabel("Principal Components")
    plt.ylabel("Percentage of Variance explained \nby the Principal Components")
    plt.xticks(x_ticks)
    plt.suptitle(f"{name}")
    plt.savefig(f"./Results/{name}.svg", dpi=500)
    plt.show()


if "__main__" in __name__:

    ## Configure settings, directory management and API initialization operations.

    ## Setting the right formatting for pandas numeric data, for better readability.
    pd.options.display.float_format = '{:,.5f}'.format

    ## Fetching the FRED API key from a text file (The text file only contains the API Key)
    FILE_API_KEY = "./fred_api_key.txt"

    ## Initialize the results directory where all the outputs are stored.
    directory = "./Results/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    API_KEY = None
    with open(FILE_API_KEY) as f:
        API_KEY = f.read()

    treasury_bills = get_all_series(API_KEY)
    dx = merge_data_series(treasury_bills)
    features = [col for col in dx.columns if ("CHG" in col)]
    dataset = dx[features]

    # #### Using PCA to reduce dimensions - we assume the data as sparse and run the model.
    df_pca = standardize_columns(dataset.copy(), "median")
    df_tsvd = standardize_columns(dataset.copy(), "constant")

    pd.DataFrame(df_pca, columns=features)
    pca_result, pca_model, pca_all = dimensionality_reduction_function(features, df_pca, "PCA")

    pd.DataFrame(df_tsvd, columns=features)
    tsvd_result, tsvd_model, tsvd_all = dimensionality_reduction_function(features, df_tsvd, "tSVD")

    print(dx)
    print(features)
    print(dataset)

    figure()
    v_pca = visualize_reduction(pca_model, features)

    print(v_pca[:2])
    print(v_pca[2:])
    scree_plot(v_pca, "PCA")

    v_tsvd = visualize_reduction(tsvd_model, features)
    print(v_tsvd[:2])
    print(v_tsvd[2:])
    scree_plot(v_tsvd, "tSVD")
