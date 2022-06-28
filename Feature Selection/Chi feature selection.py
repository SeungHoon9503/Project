#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('Mining Process mean Final.csv')


# In[2]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[3]:


X = df.drop(['EX3', 'date','Silica Concentrate', 'Iron Concentrate'],axis=1)
y = df['EX3']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[4]:


from sklearn.feature_selection import chi2


# In[6]:


import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)
        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)


# In[7]:


cT = ChiSquare(df)


# In[9]:


cT


# In[13]:


feature_columns


# In[14]:


feature_columns = list(df.columns.difference(['date','EX3', 'Silica Concentrate', 'Iron Concentrate']))
testColumns = feature_columns
for var in testColumns:
    cT.TestIndependence(colX=var,colY= "EX3" ) 


# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[2]:


import sys
print(sys.version)


# In[ ]:




