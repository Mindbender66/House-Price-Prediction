#!/usr/bin/env python
# coding: utf-8

# # Housing Price Prediction

# In[3]:


# Import the numpy and pandas package

import numpy as n
import pandas as p

# Data Visualisation

import matplotlib.pyplot as plt 
import seaborn as sns


# ## Reading The Data

# In[4]:


data = p.read_csv("Housing.csv")
data


# In[5]:


data.head()


# In[6]:


data.tail()


# * Data Inspection

# In[7]:


data.shape


# In[8]:


data.info()


# In[9]:


data.describe()


# * Data Cleaning

# In[10]:


data.isnull().sum()


# In[11]:


# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(data['price'], ax = axs[0,0])
plt2 = sns.boxplot(data['area'], ax = axs[0,1])
plt3 = sns.boxplot(data['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(data['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(data['stories'], ax = axs[1,1])
plt3 = sns.boxplot(data['parking'], ax = axs[1,2])

plt.tight_layout()
plt.show()


# In[12]:


data.columns


# ## EDA

# In[13]:


sns.pairplot(data)
plt.show()


# In[14]:


sns.distplot(data['price'])
plt.show()


# In[15]:


data.corr()


# In[16]:


sns.heatmap(data.corr())
plt.show()


# * By doing the below operation we can convert the categorical to numerical data.
# * There are two methods to convert.

# In[17]:


data


# In[18]:


dum0 = p.get_dummies(data['mainroad'])
dum0


# * yes = 1
# * no =0

# In[19]:


data = p.concat ((data,dum0),axis=1)
data = data.drop(['no'],axis=1)
data = data.drop(['mainroad'],axis=1)
data = data.rename(columns={"yes":"mainroad"})


# In[20]:


data


# In[21]:


dum1 = p.get_dummies(data['guestroom'])
dum1


# In[22]:


data = p.concat ((data,dum1),axis=1)
data = data.drop(['no'],axis=1)
data = data.drop(['guestroom'],axis=1)
data = data.rename(columns={"yes":"guestroom"})


# In[23]:


data


# In[24]:


dum2 = p.get_dummies(data['basement'])
dum2


# In[25]:


data = p.concat ((data,dum2),axis=1)
data = data.drop(['no'],axis=1)
data = data.drop(['basement'],axis=1)
data = data.rename(columns={"yes":"basement"})


# In[26]:


data


# In[27]:


dum3 = p.get_dummies(data['hotwaterheating'])
dum3


# In[28]:


data = p.concat ((data,dum3),axis=1)
data = data.drop(['no'],axis=1)
data = data.drop(['hotwaterheating'],axis=1)
data = data.rename(columns={"yes":"hotwaterheating"})


# In[29]:


data


# In[30]:


dum4 = p.get_dummies(data['airconditioning'])
dum4


# In[31]:


data = p.concat ((data,dum4),axis=1)
data = data.drop(['no'],axis=1)
data = data.drop(['airconditioning'],axis=1)
data = data.rename(columns={"yes":"airconditioning"})


# In[32]:


data


# In[33]:


dum5 = p.get_dummies(data['prefarea'])
dum5


# In[34]:


data = p.concat ((data,dum5),axis=1)
data = data.drop(['no'],axis=1)
data = data.drop(['prefarea'],axis=1)
data = data.rename(columns={"yes":"prefarea"})


# In[35]:


data


# In[36]:


dum6 = {
    "furnished":0,
    "semi-furnished":1,
    "unfurnished":2,
}


# In[37]:


data["furnishingstatus"] = data["furnishingstatus"].map(dum6)
data


# ## Train Test Split

# In[38]:


X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnishingstatus']]
y = data['price']


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Creating and Training the Model

# In[40]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[41]:


# print the intercept
print(lm.intercept_)


# In[42]:


coeff_df = p.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# ## Predictions from our Model
# 

# In[43]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()


# * Residual Histogram

# In[44]:


sns.distplot((y_test-predictions),bins=50);
plt.show()


# In[45]:


from sklearn import metrics


# In[46]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', n.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## In Linear Regression the classification metrices is Mean Absolute Error, Mean Square Error and Root Mean Square Error these values have boundaries.
