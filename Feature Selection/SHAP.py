#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('Mining Process mean Final.csv')


# In[2]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[3]:


df


# In[4]:


X = df.drop(['EX3', 'date','Silica Concentrate', 'Iron Concentrate'],axis=1)
y = df['EX3']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 42) 
# train/test 비율을 7:3
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 확인


# In[5]:


df.dtypes


# In[6]:


# lightgbm을 구현하여 shap value를 예측할 것
# ligthgbm 구현

# library
import lightgbm as lgb  # 없을 경우 cmd/anaconda prompt에서 install
from math import sqrt
from sklearn.metrics import mean_squared_error

# lightgbm model
lgb_dtrain = lgb.Dataset(data = train_x, label = train_y) # LightGBM 모델에 맞게 변환
lgb_param = {'max_depth': 10,
            'learning_rate': 0.01, # Step Size
            'n_estimators': 1000, # Number of trees
            'objective': 'regression'} # 목적 함수 (L2 Loss)
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain) # 학습 진행
lgb_model_predict = lgb_model.predict(test_x) # test data 예측
print("RMSE: {}".format(sqrt(mean_squared_error(lgb_model_predict, test_y)))) # RMSE


# In[7]:


# shap value를 이용하여 각 변수의 영향도 파악

# !pip install shap (에러 발생시, skimage version 확인 (0.14.2 이상 권장))
# import skimage -> skimage.__version__ (skimage version 확인)
# skimage version upgrade -> !pip install --upgrade scikit-image

# shap value 
import shap
explainer = shap.TreeExplainer(lgb_model) # Tree model Shap Value 확인 객체 지정
shap_values = explainer.shap_values(test_x) # Shap Values 계산


# In[8]:


# version 확인
import skimage
skimage.__version__


# In[9]:


shap.initjs() # javascript 초기화 (graph 초기화)
shap.force_plot(explainer.expected_value, shap_values[1,:], test_x.iloc[1,:])


# In[10]:


# 전체 검증 데이터 셋에 대해서 적용
shap.force_plot(explainer.expected_value, shap_values, test_x) 


# In[11]:


import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


rc('font', family='AppleGothic')

plt.rcParams['axes.unicode_minus'] = False


# In[13]:


# summary
shap.summary_plot(shap_values, test_x)


# In[14]:


# 각 변수에 대한 |Shap Values|을 통해 변수 importance 파악
shap.summary_plot(shap_values, test_x, plot_type = "bar")


# In[15]:


shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([test_x.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df


# In[12]:


import sys
print("--sys.version==")
print(sys.version)


# In[14]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:




