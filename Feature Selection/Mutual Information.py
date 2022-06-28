#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('Mining Process mean Final.csv')


# In[4]:


from sklearn.model_selection import train_test_split
X = df.drop(['EX3', 'date','Silica Concentrate', 'Iron Concentrate'],axis=1)
y = df['EX3']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.feature_selection import mutual_info_regression
mutual_info = mutual_info_regression(train_x, train_y)
mutual_info


# In[5]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = train_x.columns
mutual_info.sort_values(ascending=False)


# In[6]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(15,5))


# Percentile

# In[7]:


from sklearn.feature_selection import SelectPercentile
selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
selected_top_columns.fit(train_x.fillna(0), train_y)
selected_top_columns.get_support()


# In[8]:


train_x.columns[selected_top_columns.get_support()]


# In[ ]:




