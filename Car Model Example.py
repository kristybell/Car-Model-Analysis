#!/usr/bin/env python
# coding: utf-8

# ## Car Model

# ### Importing the Relevant Libraries

# In[2]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# ### Load the Raw Data

# In[3]:


raw_data = pd.read_csv('1.04. Real-life example.csv')
# show only the first five lines of the data
raw_data.head()


# ### Preprocessing

# #### Exploring the Descriptive Statistics of the Variables

# In[5]:


# include not only numerical data but also descriptive data
raw_data.describe(include='all')


# #### Determining the Variables of the Interest

# In[7]:


# 'DataFrame.drop(columns,axis)' returns new object with the
# indicated columns dropped
# rows: axis=0
# columns: axis=1
data = raw_data.drop(['Model'],axis=1)
data.describe(include='all')


# #### Dealing with Missing Values

# In[11]:


# sums all the missing values and give number of null values
# rule of thumb: if you are removing <5% of the observations
# you are free to just remove all that observations with missing values
data.isnull().sum()


# In[12]:


# drop observations
data_no_mv = data.dropna(axis=0)


# In[13]:


data_no_mv.describe(include='all')


# #### Exploring the PDFs

# In[14]:


sns.distplot(data_no_mv['Price'])


# #### Dealing with Outliers

# In[15]:


# one way to deal with outliers is to remove top 1% of observations
# 'DataFrame.quantile(the quantile)' returns the value at the 
# given quantile (=np.percentile)


# In[16]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')


# In[17]:


sns.distplot(data_1['Price'])


# In[18]:


sns.distplot(data_no_mv['Mileage'])


# In[21]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[23]:


sns.distplot(data_2['Mileage'])


# In[24]:


sns.distplot(data_no_mv['EngineV'])


# In[25]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[27]:


sns.distplot(data_3['EngineV'])


# In[28]:


sns.distplot(data_no_mv['Year'])


# In[29]:


q = data_3['Year'].quantile(0.01)


# In[30]:


data_4 = data_3[data_3['Year']>q]


# In[31]:


sns.distplot(data_4['Year'])


# In[32]:


data_cleaned = data_4.reset_index(drop=True)


# In[33]:


data_cleaned.describe(include='all')


# ### Checking the OLS Assumptions

# In[34]:


# check for linearity using scatter plots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize = (15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['EngineV'])
ax2.set_title('Price and Engine')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Mileage'])
ax3.set_title('Price and Mileage')

plt.show()


# In[35]:


sns.distplot(data_cleaned['Price'])


# #### Relaxing the assumptions

# In[39]:


# 'np.log(x)' returns the natural logarithm 
# of a number or array of numbers
# taking the natural logarithm will relax the data ensuring linearity
log_price = np.log(data_cleaned['Price'])
# add this new processed data to a new column in data frame
data_cleaned['log_price'] = log_price
data_cleaned


# In[40]:


# check for linearity using scatter plots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize = (15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('Log Price and Engine')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')

plt.show()


# In[ ]:


data_cleaned = data_cleaned.drop(['Price'], axis=1)


# #### Multicollinearity

# In[41]:


data_cleaned.columns.values


# In[45]:


# unfortunately, sklearn does not have a dedicated method 
# to check the assumption of multicollinearity
# therefore, must turn to Multi StatsModel using VIF
# VIF: variance inflation method; produces R^2

from statsmodels.stats.outliers_influence import variance_inflation_factor
# define the features we want to check for multicollinearity
variables = data_cleaned[['Mileage','Year', 'EngineV']]
# syntax
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
vif


# In[47]:


# when vif=1: no multicollinearity
# 1< vif < 5: okay
# 10< vif: unacceptable

data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


# ## Create Dummy Variables

# In[49]:


# 'pd.get_dummies(df[, drop_first])' spots all categorical 
# variables and creates dummies automatically
# if we have N categories for a feature, we have to create N-1 dummies
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
# show only the first 5 lines
data_with_dummies.head()


# ### Rearrange a bit

# In[50]:


data_with_dummies.columns.values


# In[51]:


cols = ['log_price', 'Mileage', 'EngineV',
       'Brand_BMW', 'Brand_Mercedes-Benz', 'Brand_Mitsubishi',
       'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch',
       'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
       'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
       'Registration_yes']


# In[52]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# ### Linear Regression Model

# #### Declare the inputs and the targets

# In[53]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)


# #### Scale the data

# In[54]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)


# In[55]:


inputs_scaled = scaler.transform(inputs)


# In[56]:


# it is not recommended to standardize dummy variables
# scaling has no effect on the predictive power of dummies
# once scaled, they lose all their dummy meaning


# ### Train Test Split

# In[57]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)


# ### Create the Regression

# In[58]:


reg = LinearRegression()
reg.fit(x_train, y_train)


# In[59]:


# a simple way to check the final results is to plot the 
# predicted values against the observed values
y_hat = reg.predict(x_train)


# In[66]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[67]:


# a second check is for residual
# residual: differences between the targets and the predictions
sns.distplot(y_train - y_hat)
plt.title("Residuals PDF", size=18)


# In[69]:


# almost normaly distributed, but some outlier overestimating targeton left
reg.score(x_train, y_train)


# #### Finding the Weights and Bias

# In[70]:


reg.intercept_


# In[71]:


reg.coef_


# In[73]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=["Features"])
reg_summary['Weighs'] = reg.coef_
reg_summary


# In[74]:


# Weights Interpretation (Continuous Variables)
# Positive Weight: shows that as a feature increases in value,
# so do the log_price and 'Price' respectively

# Negative Weight: shows that as a feature increase in value,
# log_price and 'Price' decrease


# In[75]:


data_cleaned['Brand'].unique()


# In[76]:


# Audi is the benchmark

# Weights Interpretation (Dummy Variables)
# Positive Weight: shows that the respective category (Brand) is
# more expensive than the benchmark (Audi)

# Negative Weight: shows that the respective category (Brand) is
# less expensive than the benchmark (Audi)


# ### Testing

# In[79]:


# find the predictions
y_hat_test = reg.predict(x_test)


# In[81]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[82]:


# note that for the lower prices, the model does not fit as
# well against linearity as the higher prices
# 'plt.scatter(x,y, [,alpha])' creates a scatter plot
# alpha: specifies the opacity


# In[83]:


# create a new Data Frame


# In[85]:


# 'np.exp(x)' returns the exponential of x (the Euler number 'e'
# to the power of 'x')
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[86]:


df_pf['Target'] = np.exp(y_test)
df_pf.head()


# In[89]:


y_test = y_test.reset_index(drop=True)


# In[91]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[92]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


# In[93]:


# lower error -> better explanatory power


# In[94]:


df_pf['Difference%'] = np.absolute(df_pf['Residual'] / df_pf['Target'] * 100)


# In[95]:


df_pf.describe()


# In[96]:


# for most of our predicitions, we got pretty close looking at the
#'Difference%' for the percentiles 25 through 75


# In[100]:


# let's see if we can get closer
pd.options.display.max_rows = 999  #to show more rows
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])


# In[ ]:




