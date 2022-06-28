#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('Mining Process mean Final.csv')


# In[2]:


from genetic_selection import GeneticSelectionCV


# In[3]:


from sklearn.tree import DecisionTreeRegressor


# In[5]:


import numpy as np
from sklearn.model_selection import train_test_split
X = df.drop(['EX3', 'date','Silica Concentrate', 'Iron Concentrate'],axis=1)
y = df['EX3']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[6]:


estimator = DecisionTreeRegressor()
model = GeneticSelectionCV(
    estimator, cv=5, verbose=0,
    scoring="accuracy", max_features=5,
    n_population=100, crossover_proba=0.5,
    mutation_proba=0.2, n_generations=50,
    crossover_independent_proba=0.5,
    mutation_independent_proba=0.04,
    tournament_size=3, n_gen_no_change=10,
    caching=True, n_jobs=-1)
model = model.fit(X, y)
print('Features:', X.columns[model.support_])


# In[ ]:




