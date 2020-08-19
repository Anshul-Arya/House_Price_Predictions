# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 03:25:05 2020

@author: Anshul Arya
"""
"""
# House Price Prediction

Ask a home buyer to describe their dream house, and they probably won't begin 
with the height of the basement ceiling or the proximity to an east-west 
railroad. But this playground competition's dataset proves that much more 
influences price negotiations than the number of bedrooms or a white-picket 
fence.
With 79 explanatory variables describing (almost) every aspect of residential 
homes in Ames, Iowa, this competition challenges you to predict the final price
of each home.

### Acknowledgement
The Ames Housing dataset was compiled by Dean De Cock for use in data science 
education. It's an incredible alternative for data scientists looking for a 
modernized and expanded version of the often cited Boston Housing dataset.
"""
#### Load all required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import (
        add_fignum, 
        plotting_3_charts, 
        missing,
        plot_missing_data
)

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 100)
sns.set_color_codes("dark")
sns.set_style(style='whitegrid')

# Specify the folder where the files are stored
file_folder = "C:\\Users\Anshul Arya\Desktop\DataScience\HousePrice\Data" 
# Read Train.csv file
house_train = pd.read_csv(file_folder + '\\' + 'train.csv')
# Read Test.csv file
house_test = pd.read_csv(file_folder + '\\' + 'test.csv')

print("The Training dataset has %s rows and %s columns" 
      %(house_train.shape[0],house_train.shape[1]))
print("The Test dataset has %s rows and %s columns" 
      %(house_test.shape[0],house_test.shape[1]))

combine_data = pd.concat((house_train, house_test), sort=False).reset_index(
        drop=True)
# Drop the Target Variable from the combined dataset
combine_data.drop(['SalePrice'], axis = 1, inplace = True)
print("The Combined dataset has %s rows and %s columns"
      %(combine_data.shape[0],combine_data.shape[1]))

df_sp = pd.DataFrame(house_train.SalePrice.describe()).T
df_sp['Skew'] = round(house_train.SalePrice.skew(),2)
df_sp['Kurtosis'] = round(house_train.SalePrice.kurt(),2)
round(df_sp,2)


"""
#### Checking Normality
In statistics, normality tests are used to determine if a data set is 
well-modeled by a normal distribution and to compute how likely it is for a 
random variable underlying the data set to be normally distributed.
We will be checking:<br>
- **Histogram**: Kurtosis and Skewness
- **Normal Probability Plot**:  Data distribution should closely follow the 
diagonal that represents the normal distribution.
"""

plotting_3_charts(house_train, 'SalePrice', 
                  cap="Fig 1. Histogram and normal probability plot",
                  filename = "SalePrice_Normality.png")

"""
Ok, 'SalePrice' is not normal. It shows 'peakedness', positive skewness and 
does not follow the diagonal line.
But everything's not lost. A simple data transformation can solve the problem. 
This is one of the awesome things you can learn in statistical books: 
in case of positive skewness, log transformations usually works well. 
When I discovered this, I felt like an Hogwarts' student discovering a new cool
spell.
"""

# Check Skewness and Kurtosis
print("Skewness: %.2f" % house_train['SalePrice'].skew())
print("Kurtosis: %.2f" % house_train['SalePrice'].kurt())

# Applying Log Transformation
house_train['SalePrice'] = np.log(house_train['SalePrice'])
plotting_3_charts(house_train, 'SalePrice', 
                  cap="Fig 2. Transformed Histogram and normal probability plot",
                  filename = "Transformed_SalePrice_Normality.png")

# Check Skewness and Kurtosis
print("Skewness: %.2f" % house_train['SalePrice'].skew())
print("Kurtosis: %.2f" % house_train['SalePrice'].kurt())

df_num = house_train._get_numeric_data()
df_mis = pd.DataFrame(df_num.isnull().sum(), columns=['Count']).reset_index()
df_mis.rename(columns = {'index': 'Variable'})
df_mis[df_mis.Count != 0]

"""
Relationship between Target variable and Other variable
   Sale Price vs Lot Frontage, MasVnrArea and GarageYrBlt
"""
figtext_args, figtext_kwargs = add_fignum("Fig 3. Sale Price vs Garage, MasVnr and Garage year built")
fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace = 1, wspace = 1)
ax = fig.add_subplot(1,3,1)
sns.scatterplot(x="SalePrice", y="LotFrontage", data=house_train, ax=ax, color = 'red')
plt.title("Sale Price vs LotFrontage")
ax = fig.add_subplot(1,3,2)
sns.scatterplot(x="SalePrice", y="MasVnrArea", data=house_train, ax=ax, color = 'green')
plt.title("Sale Price vs MasVnrArea")
ax = fig.add_subplot(1,3,3)
sns.scatterplot(x="SalePrice", y="GarageYrBlt", data=house_train, ax=ax, color = 'blue')
plt.title("Sale Price vs Garage Built Year")
plt.figtext(*figtext_args,**figtext_kwargs)
plt.savefig("Fig_2.png")

"""
From the three numerical variable that has missing values, we checked the 
relation of these three variable with our target variable i.e. Sale Price and 
none of the three variables has significant relationship with Sale Price so we 
can safely remove these columns from our dataset instead of handling missing 
values.
"""

plot_missing_data(i=4, df=combine_data, figname = "missing_percent_1.png")

"""
Let's analyse this to understand how to handle the missing data
We'll consider that when more than 15% of the data is missing, we should delete
the corresponding variable and pretend it never existed. This means that we will
not try any trick to fill the missing data in these cases. According to this, 
there is a set of variables (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.) that 
we should delete. The point is: will we miss this data? I don't think so. None 
of these variables seem to be very important, since most of them are not aspects
in which we think about when buying a house (maybe that's the reason why data is
missing?). Moreover, looking closer at the variables, we could say that 
variables like 'PoolQC', 'MiscFeature' and 'FireplaceQu' are strong candidates 
for outliers, so we'll be happy to delete them.
In what concerns the remaining cases, we can see that 'GarageX' variables have 
the same number of missing data. I bet missing data refers to the same set of 
observations (although I will not check it; it's just 5% and we should not spend
20 in  problems). Since the most important information regarding garages is 
expressed by 'GarageCars' and considering that we are just talking about 5% of 
missing data, I'll delete the mentioned 'GarageX' variables. The same logic 
applies to 'BsmtX' variables.
Regarding 'MasVnrArea' and 'MasVnrType', we can consider that these variables 
are not essential. Furthermore, they have a strong correlation with 'YearBuilt' 
and 'OverallQual' which are already considered. Thus, we will not lose 
information if we delete 'MasVnrArea' and 'MasVnrType'.
Finally, we have one missing observation in 'Electrical'. Since it is just one 
observation, we'll delete this observation and keep the variable.
In summary, to handle missing data, we'll delete all the variables with missing
data, except the variable 'Electrical'. In 'Electrical' we'll just delete the 
observation with missing data.
"""

def drop_col(cols, df):
    df.drop(columns=cols, inplace=True)

cols = ['PoolQC', 'MiscFeature', 'Alley', 
        'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt']
# Drop columns from combined data
drop_col(cols, df=combine_data)
# Drop columns from train data
drop_col(cols, df=house_train)
# Drop columns from test data
drop_col(cols, df=house_test)

plot_missing_data(i = 5, df=combine_data, figname = "missing_percent_2.png")

"""
#### Let's Explore the relationship of Variables with missing values with 
     Target Variables
##### GarageX Variables
"""

figtext_args, figtext_kwargs = add_fignum("Fig 6. Sale Price vs GarageX variable")
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace = 1, wspace = 1)
ax = fig.add_subplot(2,2,1)
sns.boxplot(x="GarageQual", y='SalePrice', data=house_train, ax=ax)
plt.title("Sale Price Vs Garage Quality")
ax = fig.add_subplot(2,2,2)
sns.boxplot(x="GarageType", y='SalePrice', data=house_train, ax=ax)
plt.title("Sale Price Vs Garage Type")
ax = fig.add_subplot(2,2,3)
sns.boxplot(x="GarageFinish", y='SalePrice', data=house_train, ax=ax)
plt.title("Sale Price Vs Garage Finish")
ax = fig.add_subplot(2,2,4)
sns.boxplot(x="GarageCond", y='SalePrice', data=house_train, ax=ax)
plt.title("Sale Price Vs Garage Condition")
plt.figtext(*figtext_args, **figtext_kwargs)

"""
After exploring the four garage variables, it seems that the garage X's 
variable does not have any impact on the Sale Price, so it is advisable not to 
waste time on handling missing values for these variables, as we will any how 
drop these variable when doing the modelling for Price prediction.
"""

cols = ['GarageQual', 'GarageType', 'GarageFinish', 'GarageCond']
# Drop columns from combined data
drop_col(cols, df=combine_data)
# Drop columns from train data
drop_col(cols, df=house_train)
# Drop columns from test data
drop_col(cols, df=house_test)

# BasementX Variables
figtext_args, figtext_kwargs = add_fignum("Fig 7. Sale Price vs BasementX Variables")
fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace = 1, wspace = 1)
ax = fig.add_subplot(2,3,1)
sns.boxplot(x='BsmtFinType2', y='SalePrice', data=house_train, ax = ax)
plt.title("Sale Price Vs Basement Finish Type 2")
ax = fig.add_subplot(2,3,2)
sns.boxplot(x = 'BsmtExposure', y = 'SalePrice', data=house_train, ax = ax)
plt.title("Sale Price vs Basement Exposure")
ax = fig.add_subplot(2,3,3)
sns.boxplot(x = 'BsmtFinType1', y = 'SalePrice', data=house_train, ax=ax)
plt.title("Sale Price vs Basement Finish Type 1")
ax = fig.add_subplot(2,3,4)
sns.boxplot(x = 'BsmtCond', y = 'SalePrice', data=house_train, ax=ax)
plt.title("Sale Price vs Basement Condition")
ax = fig.add_subplot(2,3,5)
sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data=house_train, ax=ax)
plt.title("Sale Price vs Basement Quality")
plt.figtext(*figtext_args, **figtext_kwargs)

"""
After exploring the five Basement variables, it seems that the Basement X's 
variable does not have any impact on the Sale Price, so it is advisable not to 
waste time on handling missing values for these variables, as we will any how 
drop these variable when doing the modelling for Price prediction.
"""

cols = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 
        'BsmtFinType2', 'BsmtExposure']
# Drop columns from combined data
drop_col(cols, df=combine_data)
# Drop columns from train data
drop_col(cols, df=house_train)
# Drop columns from test data
drop_col(cols, df=house_test)

plot_missing_data(i=7, df=combine_data, figname = "missing_percent_3.png")

cols = ['MasVnrType','MSZoning', 'BsmtFullBath', 'Functional', 'Utilities', 
        'BsmtHalfBath', 'Exterior1st', 'KitchenQual','GarageCars','GarageArea',
        'TotalBsmtSF', 'SaleType', 'BsmtUnfSF','BsmtFinSF2','BsmtFinSF1',
        'Exterior2nd', 'MasVnrArea', 'Electrical']

# Define a function to replace the missing value with mode in all the column with missing values
def handle_missing(col):
    combine_data[col] = combine_data[col].fillna(combine_data[col].mode()[0])
    house_train[col]  = house_train[col].fillna(house_train[col].mode()[0])
    house_test[col]   = house_test[col].fillna(house_test[col].mode()[0])

for value in cols:
    handle_missing(value)

if missing(df=combine_data).empty:
    print('No More Columns with missing values')

house_train.to_csv("Cleaned_train.csv", index=False, header=True)
house_test.to_csv("Cleaned_test.csv", index=False, header=True)