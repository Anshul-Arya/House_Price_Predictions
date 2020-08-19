# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 04:33:43 2020

@author: Anshul Arya
"""

"""
### Exploratory Data Analysis
In statistics, exploratory data analysis (EDA) is an approach to analyzing data 
sets to summarize their main characteristics, often with visual methods. A 
statistical model can be used or not, but primarily EDA is for seeing what the 
data can tell us beyond the formal modeling or hypothesis testing task. Exploratory 
data analysis was promoted by John Tukey to encourage statisticians to explore the
data, and possibly formulate hypotheses that could lead to new data collection and 
experiments. EDA is different from initial data analysis (IDA),[1] which focuses 
more narrowly on checking assumptions required for model fitting and hypothesis 
testing, and handling missing values and making transformations of variables as 
needed. EDA encompasses IDA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
from functions import add_fignum

# Read all the cleaned data files.
cleaned_train = pd.read_csv("Cleaned_train.csv")
cleaned_test = pd.read_csv("Cleaned_test.csv")

# Keep numerical features
num_features = cleaned_train.select_dtypes(include = np.number)
correl = num_features.corr()

# SalePrice correlation matrix
k = 11
plt.figure(figsize = (10,10))
sns.set_style(style = 'white')
figtext_args, figtext_kwargs = add_fignum(
        "Fig 8. Correlation Matrix Heatmap of Sale Price")
cols = correl.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(cleaned_train[cols].values.T)
sns.set(font_scale = 1.25)
plt.title(
        "Correlation Heatmap of Sale Price with 10 most related variable\n", 
        weight = 'bold')
mask = np.triu(np.ones_like(cm, dtype=np.bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
hm = sns.heatmap(cm, mask=mask, cmap=cmap, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, 
                 xticklabels=cols.values)
plt.figtext(*figtext_args, **figtext_kwargs)

"""
#Impact of Overall Quality on SalePrice
In normal marketting terms, better quality products generally costs more, which 
is exactly the thing we are expecting in case of house 
"""
plt.figure(figsize=(12,7))
figtext_args, figtext_kwargs = add_fignum(
        "Fig 9. Impact of Overall Quality on Sale Price") 
style.use('fivethirtyeight')
sns.boxplot(x='OverallQual', y='SalePrice', data=cleaned_train)
plt.title("Impact of Overall Quality on Sale Price")
plt.figtext(*figtext_args, **figtext_kwargs)

"""
As Expected, as the overall quality improves, the sale price of the houses in Ames, 
Iowa increases which is expected
"""

# Sale Price vs Ground Living Area
plt.figure(figsize=(12,7))
figtext_args, figtext_kwargs = add_fignum(
        "Fig 10. Sale Price by Above ground Living Area") 
sns.regplot(x='GrLivArea', 
            y = 'SalePrice', 
            color = 'brown',
            data = cleaned_train)
plt.title("House Prices in Ames, Iowa", 
          loc='left', fontdict=dict(fontsize = 18))
plt.xlabel("Above Ground Living Area", fontsize = 12)
plt.figtext(*figtext_args, **figtext_kwargs)

"""
There seems to be a linear relationship between Sale Price and Above ground 
Living area. i.e. as the Above Ground living area increases the price of the 
house increases, but there are outliers in this relationship, the two points at 
the bottom right of plot are clearly outliers as it seems highly impossible for 
the house of size with greater than 4000 sq ft. have such low price, so it is 
safe to remove these two entries from the dataset
"""
# Delete outliers
cleaned_train = cleaned_train.drop(
        cleaned_train[(cleaned_train['GrLivArea'] > 4000) & 
                    (cleaned_train['SalePrice'] < 300000)].index)

plt.figure(figsize=(12,7))
figtext_args, figtext_kwargs = add_fignum(
        "Fig 11. Sale Price by Above ground Living Area (without outlier)") 
sns.regplot(x='GrLivArea', 
            y = 'SalePrice', 
            color = 'brown',
            data = cleaned_train)
plt.title("House Prices in Ames, Iowa", 
          loc='left', fontdict=dict(fontsize = 18))
plt.xlabel("Above Ground Living Area", fontsize = 12)
plt.figtext(*figtext_args, **figtext_kwargs)

# Plot the Sale Price vs overall condition
plt.figure(figsize = (10,7))
style.use('fivethirtyeight')
figtext_args, figtext_kwargs = add_fignum(
        "Fig 12. Sale Price vs Overall Condition")
sns.boxplot(x="OverallCond",
            y="SalePrice",
            palette = "colorblind",
            data = cleaned_train)
plt.title("House Price in Ames, Iowa",
          loc= 'left', fontdict = dict(fontsize=18))
plt.xlabel("Overall Material and Finish of the house", fontsize = 12)
plt.figtext(*figtext_args, **figtext_kwargs)

"""
The price of the house increases as the overall condition of the house 
increases, although the trend is not as clear as with overall house quality but
nevertheless the mean price of house in Ames, increases with better condition 
of the house.
"""

# Sale Price by Neighborhood
plt.figure(figsize = (12,7))
style.use("bmh")
figtext_args, figtext_kwargs = add_fignum(
        "Fig 13. Sale Price by Neighborhood")
sns.boxplot(x="Neighborhood",
            y="SalePrice",
            palette = "colorblind",
            data = cleaned_train)
plt.title("House Prices in Ames, Iowa",
          loc = "left", fontdict=dict(fontsize=18))
plt.xticks(rotation = 45)
plt.figtext(*figtext_args, **figtext_kwargs)

# Sale Price by bedroom size
plt.figure(figsize=(7,7))
sns.set_style(style='whitegrid')
figtext_args, figtext_kwargs = add_fignum(
        "Fig 14. Sale Price by Number of Bedrooms") 
sns.boxplot(x='BedroomAbvGr', 
            y = 'SalePrice', 
            palette='colorblind',
            data = cleaned_train)
plt.title("House Prices in Ames, Iowa", 
          loc='left', fontdict=dict(fontsize = 18))
plt.xticks(rotation = 45)
plt.figtext(*figtext_args, **figtext_kwargs)

# Sale Price by Sale Zoning classification
plt.figure(figsize=(7,7))
sns.set_style(style="dark")
figtext_args, figtext_kwargs = add_fignum("Fig 15. Sale Price by Sale Zoning classification") 
sns.boxplot(x='MSZoning', 
            y = 'SalePrice', 
            palette='colorblind',
            data = cleaned_train)
plt.title("House Price in Ames, Iowa by Sale Zoning Identification\n", 
          loc='left', fontdict=dict(fontsize = 18))
plt.xticks(rotation = 45)
plt.figtext(*figtext_args, **figtext_kwargs)

# Sale Price vs Garage Area
plt.figure(figsize=(12,5))
sns.set_context('paper')
sns.set_style(style="ticks")
style.use('fivethirtyeight')
figtext_args, figtext_kwargs = add_fignum(
        "Fig 16. Sale Price by Garage Area") 
sns.scatterplot(x=cleaned_train.drop(
    cleaned_train[(cleaned_train['LotArea'] > 55000) & 
                (cleaned_train['GarageArea'] < 500000)].index).LotArea, 
                y='SalePrice',data=cleaned_train,
                color = 'orange')
plt.title("House Price in Ames, Iowa\n", 
          loc='center', fontdict=dict(fontsize = 18))
plt.xlabel('Garage Area', fontsize = 15, weight = 'bold')
plt.ylabel('Sale Price', fontsize = 15, weight = 'bold')
plt.figtext(*figtext_args, **figtext_kwargs)

# Sale Price by Sale Zoning classification and Lot Area
plt.figure(figsize=(12,5))
sns.set_context('paper')
sns.set_style(style="ticks")
figtext_args, figtext_kwargs = add_fignum(
        "Fig 17. Sale Price by Sale Zoning classification and Lot Area") 
sns.scatterplot(x=cleaned_train.drop(
    cleaned_train[(cleaned_train['LotArea'] > 55000) & 
                (cleaned_train['SalePrice'] < 500000)].index).LotArea, 
                y='SalePrice', hue = 'MSZoning',data=cleaned_train)
plt.title("House Price in Ames, Iowa\n", 
          loc='center', fontdict=dict(fontsize = 18))
plt.xlabel('Lot Area', fontsize = 15, weight = 'bold')
plt.ylabel('Sale Price', fontsize = 15, weight = 'bold')
plt.figtext(*figtext_args, **figtext_kwargs)

# Sale Price by Total square feet of basement area
plt.figure(figsize=(12,7))
figtext_args, figtext_kwargs = add_fignum(
        "Fig 18. Sale Price by Total square feet of basement area") 
style.use('fivethirtyeight')
sns.regplot(x='TotalBsmtSF', 
            y = 'SalePrice', 
            color='crimson',
            data = cleaned_train)
plt.title("House Prices in Ames, Iowa", 
          loc='left', fontdict=dict(fontsize = 18))
plt.xlabel("Total square feet of basement area", 
           fontsize = 15, weight = 'bold')
plt.ylabel('Sale Price', fontsize = 15, weight = 'bold')
plt.figtext(*figtext_args, **figtext_kwargs)

# Sale Price by First Floor square feet
plt.figure(figsize=(12,7))
figtext_args, figtext_kwargs = add_fignum(
        "Fig 19. Sale Price by First Floor square feet") 
style.use('fivethirtyeight')
sns.scatterplot(x='1stFlrSF', 
            y = 'SalePrice', 
            color='olive',
            data = cleaned_train)
plt.title("House Prices in Ames, Iowa", 
          loc='left', fontdict=dict(fontsize = 18))
plt.xlabel("First Floor square feet", fontsize = 15, weight = 'bold')
plt.ylabel('Sale Price', fontsize = 15, weight = 'bold')
plt.figtext(*figtext_args, **figtext_kwargs)

# Sale Price by Full bathrooms above grade
plt.figure(figsize=(12,7))
figtext_args, figtext_kwargs = add_fignum(
        "Fig 20. Sale Price by Full bathrooms above grade") 
sns.boxplot(x='FullBath', 
            y = 'SalePrice', 
            color='khaki',
            data = cleaned_train)
plt.title("House Prices in Ames, Iowa", 
          loc='left', fontdict=dict(fontsize = 18))
plt.xlabel("Full bathrooms above grade", fontsize = 15, weight = 'bold')
plt.ylabel('Sale Price', fontsize = 15, weight = 'bold')
plt.figtext(*figtext_args, **figtext_kwargs)

# Sale Price by Total rooms above grade (does not include bathrooms)
plt.figure(figsize=(12,7))
figtext_args, figtext_kwargs = add_fignum(
        "Fig 21. Sale Price by Total rooms above grade (does not include bathrooms)") 
sns.boxplot(x='TotRmsAbvGrd', 
            y = 'SalePrice', 
            color='indianred',
            data = cleaned_train)
plt.title("House Prices in Ames, Iowa", 
          loc='left', fontdict=dict(fontsize = 18))
plt.xlabel("Total rooms above grade (does not include bathrooms)", 
           fontsize = 15, weight = 'bold')
plt.ylabel('Sale Price', fontsize = 15, weight = 'bold')
plt.figtext(*figtext_args, **figtext_kwargs)

cleaned_train.to_csv("cleaned_train.csv", index = False, header = True)
