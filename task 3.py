#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')


# In[103]:


df = pd.read_csv('car data.csv')
df.head()


# In[104]:


df.info()


# In[105]:


df.describe()


# In[106]:


df.dtypes


# In[107]:


df.isnull()


# In[108]:


df.isnull().sum()


# In[109]:


df.corr()


# In[110]:


df.boxplot()


# In[111]:


df.drop('Car_Name', axis=1, inplace=True)

# Calculate the age of the car
df['Car_Age'] = 2024 - df['Year']  # Use the correct current year
df.drop('Year', axis=1, inplace=True)


# In[ ]:





# In[112]:


# One-hot 
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
df.head()


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = linear_regressor.predict(X_test)
y_pred


# In[114]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, MSE: {mse}, R²: {r2}')


# In[ ]:





# In[115]:


# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True)
plt.show()


# In[116]:


from sklearn.ensemble import RandomForestRegressor

# Define features and target variable
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Initialize and fit Random Forest model
random_forest = RandomForestRegressor()
random_forest.fit(X, y)


# In[117]:


feature_importances = pd.Series(random_forest.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importances 


# In[118]:


plt.figure(figsize=(10,7))
feature_importances.plot(kind='bar')
plt.title('Feature Importance')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




