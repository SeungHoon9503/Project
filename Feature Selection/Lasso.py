#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('Mining Process mean Final.csv')


# In[5]:


get_ipython().system('conda install -c conda-forge -y sklearn-genetic')


# In[12]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso


# In[13]:


X = df.drop(['EX3', 'date','Silica Concentrate', 'Iron Concentrate'],axis=1)
y = df['EX3']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[14]:


pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])


# In[15]:


search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )


# In[17]:


search.fit(train_x,train_y)


# In[18]:


search.best_params_


# In[19]:


coefficients = search.best_estimator_.named_steps['model'].coef_


# In[20]:


importance = np.abs(coefficients)


# In[28]:


feature_columns = list(df.columns.difference(['date','EX3', 'Silica Concentrate', 'Iron Concentrate']))


# In[29]:


np.array(feature_columns)[importance > 0]


# In[31]:


np.array(feature_columns)[importance == 0]


# In[ ]:




