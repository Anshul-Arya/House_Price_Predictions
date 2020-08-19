# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 04:16:58 2020

@author: Anshul Arya
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from scipy import stats
import matplotlib.gridspec as gridspec
import matplotlib.style as style

# Function to add figure number
def add_fignum(caption):
    figtext_args = (0.5, -0.2, caption) 
  
    figtext_kwargs = dict(horizontalalignment ="center",  
                          fontsize = 14, color ="black",
                          wrap = True)
    return figtext_args, figtext_kwargs


def plotting_3_charts(df, feature, cap, filename):
    style.use('fivethirtyeight')
    figtext_args, figtext_kwargs = add_fignum(cap)
    ## Creating a custom chart and giving in figsize and everything
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## Creating a gridspec of 3 rows and 3 columns
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ## Customizing the histogram grid
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title
    ax1.set_title('Histogram')
    ## Plot the histogram
    sns.distplot(df.loc[:,feature], norm_hist=True, ax=ax1)
    
    ## Customizing the QQplot.
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title
    ax2.set_title("QQ Plot")
    ## Plotting the QQ Plot
    stats.probplot(df.loc[:,feature], plot=ax2)
    
    ## Customizing the box plot
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set the title
    ax3.set_title('Box Plot')
    ## Plotting the Box Plot
    sns.boxplot(df.loc[:, feature], orient = 'v', ax=ax3)
    
    plt.figtext(*figtext_args, **figtext_kwargs)
    plt.savefig(filename)
    

# Define a function to get the missing values
def missing(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(
                                                        ascending=False) * 100
    missing_data = pd.concat([total, percent], axis=1, 
                             keys=['Total', 'Percent']).reset_index()
    missing_data = missing_data[missing_data['Total'] != 0]
    missing_data = missing_data.rename(
            columns={'index':'Variable', 
                     'Total':'Total', 
                     'Percent':'Percent'})
    return missing_data

# Define a function to plot the missing data percentage
def plot_missing_data(i, df, figname):
    i = i
    missing_data = missing(df=df)
    plt.figure(figsize=(15,7))
    chart = sns.barplot(
        x='Variable', 
        y = 'Percent',
        palette='Set1',
        data=missing_data)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.title("Percentage of Missing Values by Variable")
    fmt = '%.0f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    chart.yaxis.set_major_formatter(xticks)
    plt.figtext(0.3,-0.3, 
                "Fig %s. Display Missing data percentage by variable" % i)
    for index, row in missing_data.iterrows():
        chart.text(row.name,row.Percent, round(row.Percent,2), 
                   color='black', ha="center")
    
    plt.savefig(figname)

