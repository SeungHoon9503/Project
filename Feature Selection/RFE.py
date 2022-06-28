#!/usr/bin/env python
# coding: utf-8

# RFE(Recursive Feature Elimination)

# In[9]:


import pandas as pd
df = pd.read_csv('Mining Process mean Final.csv')


# In[10]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[11]:


X = df.drop(['EX3', 'date','Silica Concentrate', 'Iron Concentrate'],axis=1)
y = df['EX3']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[13]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#모델 입력, n_features_to_select: 골라낼 변수의 수, step: 한 번에 몇 개씩 제거할지 선택
rfe = RFE(lr, n_features_to_select=8, step=1)
model = rfe.fit(X,y)


# In[14]:


model


# In[9]:


#선택될 변수
model.support_


# In[10]:


#변수 중요도(숫자가 높을수록 불필요함)
model.ranking_


# Correlation Method

# In[2]:


#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# In[6]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[7]:


#Correlation with output variable
cor_target = abs(cor["EX3"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.05]
relevant_features


# Backward Elimination

# In[19]:


#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues


# In[20]:


#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# In[ ]:




