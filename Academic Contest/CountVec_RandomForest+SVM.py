#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd

import matplotlib.pylab as plt
import datetime
import seaborn as sns
from matplotlib import rc


plt.rcParams["figure.figsize"] = (10,4)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True 
rc('font', family='AppleGothic')


# In[2]:


claim = pd.read_csv('1005_data.csv')


# In[3]:


claim.head()


# In[4]:


claim = claim.dropna(subset = ['특이사항'])
claim  = claim.astype({'특이사항' : 'str'})
claim.head()


# In[5]:


claim['현상코드_y'] = claim['원인부품숫자'].astype('category')
claim['현상코드_y'] = claim['현상코드_y'].cat.codes


# In[6]:


from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split (claim['특이사항'], claim['현상코드_y'], test_size = 0.20, random_state=42)


# In[7]:


X_train


# In[8]:


from KaggleWord2VecUtility import KaggleWord2VecUtility
import nltk


# In[9]:


nltk.download('punkt')


# In[10]:


sentences_all = []
for review in claim['특이사항'].values:
    sentences_all += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)


# In[11]:


sentences_all[0][:10]


# In[12]:


import gensim


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# 튜토리얼과 다르게 파라메터 값을 수정
# 파라메터 값만 수정해도 캐글 스코어 차이가 많이 남
model = CountVectorizer(analyzer = 'word', 
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 1, # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 3), 
                             max_features = 100
                            )
vectorizer


# In[17]:


def review_to_words( raw_review ):
    # 1. HTML 제거
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. 소문자 변환
    words = letters_only.lower().split()
    # 4. 파이썬에서는 리스트보다 세트로 찾는게 훨씬 빠르다.
    # stopwords 를 세트로 변환한다.
#     stops = set(stopwords.words('english'))
    # 5. Stopwords 불용어 제거
#     meaningful_words = [w for w in words if not w in stops]
    # 6. 어간추출
    stemming_words = [stemmer.stem(w) for w in words]
    # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
    return( ' '.join(stemming_words) )

from multiprocessing import Pool
import numpy as np

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    # 키워드 항목 중 workers 파라메터를 꺼냄
    workers = kwargs.pop('workers')
    # 위에서 가져온 workers 수로 프로세스 풀을 정의
    pool = Pool(processes=workers)
    # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    # 작업 결과를 합쳐서 반환
    return pd.concat(list(result))


# In[24]:


fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)
ax.set_title("단어 CountVec 시각화", fontsize=50)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()


# In[25]:


get_ipython().system('pip install adjustText')


# In[26]:


sentences_train = []
for review in X_train:
    sentences_train += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)

sentences_test = []
for review in X_test:
    sentences_test += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# 튜토리얼과 다르게 파라메터 값을 수정
# 파라메터 값만 수정해도 캐글 스코어 차이가 많이 남
def getC2V(sentence_param, name_param):
    model = CountVectorizer(analyzer = 'word', 
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 1, # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 3), 
                             max_features = 100
                            )

    vocab = vectorizer.get_feature_names()
    X = model[vocab]
    return X


# In[32]:


claim_df = claim[['사용일수', '주행거리(km로 환산)', '현상코드_y']]


# In[33]:


#Normalizing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_df = pd.DataFrame()

features = claim_df[['사용일수', '주행거리(km로 환산)']]
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns =features.columns )
features_scaled = features_scaled.reset_index(drop=True)

label_col = claim_df[['현상코드_y']].reset_index(drop=True)

model1_df = pd.concat([features_scaled, label_col], axis=1)
model1_df.tail()


# In[34]:


X = model1_df[features.columns]
y = model1_df['현상코드_y'].values
X_train, X_test, Y_train, Y_test = train_test_split (X, y, test_size = 0.20, random_state=42)


# In[37]:


from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트 분류기에 100개의 의사결정 트리를 사용한다.
forest = RandomForestClassifier(n_estimators=100)

# 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작한다.
forest.fit(X_train, Y_train)
y_pred = forest.predict(X_test)


# In[38]:


from sklearn import metrics
print('정확도 : ', metrics.accuracy_score(Y_test, y_pred))

print('f1-score macro: ', metrics.f1_score(Y_test, y_pred, average='macro'))

print('f1-score micro: ', metrics.f1_score(Y_test, y_pred, average='micro'))

print('f1-score weighted: ', metrics.f1_score(Y_test, y_pred, average='weighted'))


# In[42]:


# 멀티스레드로 4개의 워커를 사용해 처리한다.
def getCleanReviews(data):
    clean_reviews = []
    clean_reviews = KaggleWord2VecUtility.apply_by_multiprocessing(        data["특이사항"], KaggleWord2VecUtility.review_to_wordlist,        workers=4)
    return clean_reviews


# In[43]:


claim_df = claim[['사용일수', '주행거리(km로 환산)', '특이사항', '현상코드_y']]
claim_df.head()


# In[44]:


X = claim_df[['사용일수', '주행거리(km로 환산)', '특이사항']]
y = claim_df['현상코드_y']

X_train, X_test, Y_train, Y_test = train_test_split (X, y, test_size = 0.20, random_state=42)


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = 'word', max_features = 5000)

train_data_features = vectorizer.fit_transform(claim['특이사항'])


# In[46]:


train_data_features


# In[47]:


X_train, X_test, Y_train, Y_test = train_test_split (train_data_features, y, test_size = 0.20, random_state=42)


# In[48]:


from sklearn.ensemble import RandomForestClassifier

# 랜덤 포레스트 분류기에 100개의 의사결정 트리를 사용한다.
forest = RandomForestClassifier(n_estimators=100)

# 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작한다.
forest.fit(X_train, Y_train)
y_pred = forest.predict(X_test)


# In[49]:


from sklearn import metrics
print('정확도 : ', metrics.accuracy_score(Y_test, y_pred))

print('f1-score macro: ', metrics.f1_score(Y_test, y_pred, average='macro'))

print('f1-score micro: ', metrics.f1_score(Y_test, y_pred, average='micro'))

print('f1-score weighted: ', metrics.f1_score(Y_test, y_pred, average='weighted'))


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[51]:


ml = SVC(kernel='linear', C=1.0, random_state=0)
ml.fit(X_train, Y_train)
y_pred = ml.predict(X_test)


# In[52]:


from sklearn import metrics
print('정확도 : ', metrics.accuracy_score(Y_test, y_pred))

print('f1-score macro: ', metrics.f1_score(Y_test, y_pred, average='macro'))

print('f1-score micro: ', metrics.f1_score(Y_test, y_pred, average='micro'))

print('f1-score weighted: ', metrics.f1_score(Y_test, y_pred, average='weighted'))


# In[ ]:




