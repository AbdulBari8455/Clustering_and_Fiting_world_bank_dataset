from sklearn.metrics import silhouette_score
import sklearn.cluster as cluster
import cluster_tools as ct
import errors as err
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'
pd.options.mode.chained_assignment = None


def read_file(loc):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    ------------
    loc (str): The filename of the CSV file to be read.

    Returns:
    ---------
    df (pandas.DatFrame): The DataFrame containing the data
    read from the CSV file.
    """
    address = loc
    print(address)
    df = pd.read_csv(address, skiprows=4)
    df = df.drop(
        columns=[
            'Country Code',
            'Indicator Name',
            'Indicator Code',
            'Unnamed: 67'])
    return df


def indicators_data(first_ind_name, Second_ind_name, df1, df2, Year):
    """


    Parameters
    ----------
    first_ind_name : String
        Name of First Indicator.
    Second_ind_name : String
        Name of Second Indicator.
    df1 : Pandas Data frame
        Data Frame to extract data.
    df2 : Pandas Data Frame
        Data Frame fo Extract Data.
    Year : String
        Year who's Data required.

    Returns
    -------
    df_cluster : Pandas DataFrame
        Data For Clustering.

    """
    df1 = df1[['Country Name', Year]]
    df2 = df2[['Country Name', Year]]
    df = pd.merge(df1, df2,
                  on="Country Name", how="outer")
    df = df.dropna()
    df = df.rename(
        columns={
            Year +
            "_x": first_ind_name,
            Year +
            "_y": Second_ind_name})
    df_cluster = df[[first_ind_name, Second_ind_name]].copy()
    return df_cluster


def merging_datasets(first_ind_name, Second_ind_name, df1, df2, Year):
    """


    Parameters
    ----------
    first_ind_name : String
        Name of first indicator.
    Second_ind_name : String
        Name of second indicator.
    df1 : Pandas DataFrame
        Data Frame 01.
    df2 : Pandas DataFrame
        DESCRIPTION.
    Year : String 
        DESCRIPTION.

    Returns
    -------
    df_cluster : Pandas Data Frame
        It will return a Pandas DataFrame for Clustering.

    """
    df1 = df1[['Country Name', Year]]
    df2 = df2[['Country Name', Year]]
    df = pd.merge(df1, df2,
                  on="Country Name", how="outer")
    df = df.dropna()
    df = df.rename(
        columns={
            Year +
            "_x": first_ind_name,
            Year +
            "_y": Second_ind_name})
    df_cluster = df[['Country Name', first_ind_name, Second_ind_name]].copy()
    return df_cluster


def logistics(t, a, k, t0):
    """


    Parameters
    ----------
    t : int
        initial Year.
    a : initial Value
        From Where to Start.
    k : int
        Constant.
    t0 : int
        Last Year.

    Returns
    -------
    f : List of Values
        Its the List of Values after computations.

    """
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f


def fit_and_predict(df, Country_name, Ind, title, tit_forecast, initial):
    """


    Parameters
    ----------
    df : Pandas Data Frame
        For Fitting.
    Country_name : String
        Name of Countries.
    Ind : String
        Indicator to Forecast.
    title : String
        Title of graph.
    tit_forecast : TYPE
        Title of Forecast Graph.
    initial : list
        List of initial values for fitting.

    Returns
    -------
    None.

    """
    popt, pcorr = opt.curve_fit(logistics, df.index, df[Country_name],
                                p0=initial)
    df["pop_log"] = logistics(df.index, *popt)
    plt.figure()
    plt.plot(df.index, df[Country_name], label="data")
    plt.plot(df.index, df["pop_log"], label="fit")
    plt.legend()
    plt.xlabel('Years')
    plt.ylabel(Ind)
    plt.title(title)
    plt.savefig(Country_name + 'b.png', dpi=300)
    years = np.linspace(1995, 2030)
    print(*popt)
    popt, pcorr = opt.curve_fit(logistics, df.index, df[Country_name],
                                p0=initial)

    pop_log = logistics(years, *popt)
    sigma = err.error_prop(years, logistics, popt, pcorr)
    low = pop_log - sigma
    up = pop_log + sigma
    plt.figure()
    plt.title(tit_forecast)
    plt.plot(df.index, df[Country_name], label="data")
    plt.plot(years, pop_log, label="Forecast")
    # plot error ranges with transparency
    plt.fill_between(years, low, up, alpha=0.5, color="y")
    plt.legend(loc="upper left")
    plt.xlabel('Years')
    plt.ylabel(Ind)
    plt.savefig(Country_name + 'b_forecast.png', dpi=300)
    plt.show()


def specific_country_data(df, country_name, start_year, end_year):
    """
    To Get Specific Country's Data
    Parameters
    ----------
    df : Pandas dataframe
        Dataset to Extract information from.
    country_name : String
        Country name who's info is required.
    start_year : int
         Start Year.
    end_year : int
        Ending year.

    Returns
    -------
    df : Pandas DataFrame
        Data Set For Specific Country.

    """
    # Taking the Transpose of DataSet
    df = df.T
    df.columns = df.iloc[0]
    # droping the row of dataset.
    df = df.drop(['Country Name'])
    df = df[[country_name]]
    # changing the Type of index
    df.index = df.index.astype(int)
    # Filtering Data for our Range
    df = df[(df.index > start_year) & (df.index <= end_year)]
    df[country_name] = df[country_name].astype(float)
    return df


def plot_clusters(
        df,
        ind1,
        ind2,
        xlabel,
        ylabel,
        tit,
        n_clu_cen,
        df_fit,
        df_min,
        df_max):
    """


    Parameters
    ----------
    df : Pandas Dataframe
        Original Data Frame for ploting clusters.
    ind1 : String
        X-axis Indicator.
    ind2 : String
        Y-axis Indicator.
    xlabel : String
        xlabel of graph.
    ylabel : String
        ylabel of graph.
    tit : String
        Title of Clustring Graph.
    n_clu_cen : int
        Nummber of Clusters.
    df_fit : Pandas Data Frame
        Normalize DataFrame.
    df_min : int
        Minimum Value of Data.
    df_max : int
        Maximum Value of Data.

    Returns
    -------
    labels : Array
        Cluster Number of Each Datapoint.

    """
    nc = n_clu_cen  # number of cluster centres
    kmeans = cluster.KMeans(n_clusters=nc, n_init=10, random_state=0)
    kmeans.fit(df_fit)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    # Setting the Figure Size
    plt.figure(figsize=(8, 8))
    # scatter plot with colours selected using the cluster numbers
    # now using the original dataframe
    scatter = plt.scatter(df[ind1], df[ind2], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(tit)
    plt.savefig('Clustering_plot.png', dpi=300)
    plt.show()
    return labels


def plot_silhouette_score(data, max_clusters=10):
    """
    Evaluate and plot silhouette scores for different numbers of clusters.

    Parameters:
    - data: The input data for clustering.
    - max_clusters: The maximum number of clusters to evaluate.

    Returns:
    """

    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        # Perform clustering using KMeans
        kmeans = cluster.KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


CO2_emissions_metric_tons_per_capita = read_file(
    'CO2_emissions_metric_tons_per_capita.csv')
GDP_per_capita_current_US = read_file('GDP_per_capita_current_US$.csv')
df_cluster = indicators_data(
    'GDP_per_capita_current_US$',
    'CO2_emissions_metric_tons_per_capita',
    GDP_per_capita_current_US,
    CO2_emissions_metric_tons_per_capita,
    '2020')

df_fit, df_min, df_max = ct.scaler(df_cluster)
plot_silhouette_score(df_fit, 12)

labels = plot_clusters(
    df_cluster,
    'GDP_per_capita_current_US$',
    'CO2_emissions_metric_tons_per_capita',
    'GDP_per_capita_current_US$',
    'CO2_emissions_metric_tons_per_capita',
    'CO2_emissions_metric_tons vs GDP current_US$ (per capita) in 2020',
    3,
    df_fit,
    df_min,
    df_max)
df = merging_datasets(
    'GDP_per_capita_current_US$',
    'CO2_emissions_metric_tons_per_capita',
    GDP_per_capita_current_US,
    CO2_emissions_metric_tons_per_capita,
    '2020')
df['label'] = labels
df[df['Country Name'].isin(['United Kingdom', 'Pakistan'])]

df = specific_country_data(GDP_per_capita_current_US, 'Pakistan', 1990, 2020)
df = df.fillna(0)
fit_and_predict(
    df,
    'Pakistan',
    'GDP_per_capita_current_US',
    "GDP per Capita Current US$ In Pakistan 1990-2020",
    "GDP per Capita Current US$ In Pakistan Forecast Untill 2030",
    (10e4,
     0.04,
     1990.0))
df = specific_country_data(
    GDP_per_capita_current_US,
    'United Kingdom',
    1990,
    2020)
df = df.fillna(0)
fit_and_predict(
    df,
    'United Kingdom',
    'GDP_per_capita_current_US',
    "GDP per Capita Current US$ In United Kingdom 1990-2020",
    "GDP per Capita Current US$ In United Kingdom Forecast Untill 2030",
    (10e4,
     0.04,
     1990.0))

df = specific_country_data(
    CO2_emissions_metric_tons_per_capita,
    'Pakistan',
    1990,
    2020)
df = df.fillna(0)
fit_and_predict(
    df,
    'Pakistan',
    'CO2_emissions_metric_tons_per_capita',
    "CO2 Emissions Metric Tons Per Capita In Pakistan 1990-2020",
    "CO2 Emissions Metric Tons Per Capita In Pakistan Forecast Untill 2030",
    (10,
     0.04,
     1990.0))
df = specific_country_data(
    CO2_emissions_metric_tons_per_capita,
    'United Kingdom',
    1990,
    2020)
df = df.fillna(0)
fit_and_predict(
    df,
    'United Kingdom',
    'CO2_emissions_metric_tons_per_capita',
    "CO2 Emissions Metric Tons Per Capita In UK 1990-2020",
    "CO2 Emissions Metric Tons Per Capita In UK Forecast Untill 2030",
    (10,
     0.04,
     1990.0))
