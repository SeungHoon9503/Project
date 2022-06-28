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


# In[13]:


# 파라메터값 지정
num_features = 300 # 문자 벡터 차원 수
min_word_count = 40 # 최소 문자 수
num_workers = 4 # 병렬 처리 스레드 수
context = 10 # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도 수 Downsample

# 초기화 및 모델 학습
from gensim.models import word2vec
model = word2vec.Word2Vec(sentences_all, 
                           workers=num_workers, 
                           size=num_features, 
                           min_count=min_word_count,
                           window=context,
                           sample=downsampling)

# # 모델 학습
# model = word2vec.Word2Vec(sentences_all, 
#                           workers=num_workers, 
#                           size=num_features, 
#                           min_count=min_word_count,
#                           window=context,
#                           sample=downsampling)
# model


# In[14]:


# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace=True)

model_name = '300features_40minwords_10text'
# model_name = '300features_50minwords_20text'
model.save(model_name)


# In[15]:


# 참고 https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

model_name = '300features_40minwords_10text'
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X_all = model[vocab]

print(len(X_all))
print(X_all[0][:10])
tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X_all[:100,:])
# X_tsne = tsne.fit_transform(X)


# In[16]:


df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
print(df.shape)

df.head()


# In[17]:


fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)
ax.set_title("단어 W2V 시각화", fontsize=50)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()


# In[18]:


get_ipython().system('pip install adjustText')


# In[19]:


sentences_train = []
for review in X_train:
    sentences_train += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)

sentences_test = []
for review in X_test:
    sentences_test += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)


# In[20]:


# 모델 학습

def getW2V(sentence_param, name_param):
    model = word2vec.Word2Vec(sentence_param, 
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)
    

    # 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
    model.init_sims(replace=True)

#     model = '300features_40minwords_10text_EEEEE'
    model_name = name_param
    model.save(model_name)


    model = g.Doc2Vec.load(model_name)

    vocab = list(model.wv.vocab)
    X = model[vocab]
    
    return X


# In[21]:


get_ipython().run_line_magic('time', 'X_train_W2V = getW2V(sentences_train, "train_300features_w2v")')
get_ipython().run_line_magic('time', 'X_test_W2V = getW2V(sentences_test, "test_300features_w2v")')


# In[22]:


X_train_W2V


# In[23]:


claim.head(3)


# In[24]:


claim_df = claim[['사용일수', '주행거리(km로 환산)', '현상코드_y']]


# In[25]:


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


# In[40]:


X = model1_df[features.columns]
y = model1_df['현상코드_y'].values
X_train, X_test, Y_train, Y_test = train_test_split (X, y, test_size = 0.20, random_state=42)


# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[47]:


ml = SVC(kernel='linear', C=1.0, random_state=0)
ml.fit(X_train, Y_train)
y_pred = ml.predict(X_test)


# In[49]:


from sklearn import metrics
print('정확도 : ', metrics.accuracy_score(Y_test, y_pred))

print('f1-score macro: ', metrics.f1_score(Y_test, y_pred, average='macro'))

print('f1-score micro: ', metrics.f1_score(Y_test, y_pred, average='micro'))

print('f1-score weighted: ', metrics.f1_score(Y_test, y_pred, average='weighted'))


# In[50]:


import numpy as np

def makeFeatureVec(words, model, num_features):
    """
    주어진 문장에서 단어 벡터의 평균을 구하는 함수
    """
    # 속도를 위해 0으로 채운 배열로 초기화 한다.
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.
    # Index2word는 모델의 사전에 있는 단어명을 담은 리스트이다.
    # 속도를 위해 set 형태로 초기화 한다.
    index2word_set = set(model.wv.index2word)
    # 루프를 돌며 모델 사전에 포함이 되는 단어라면 피처에 추가한다.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 결과를 단어수로 나누어 평균을 구한다.
    featureVec = np.divide(featureVec,nwords)
    return featureVec


# In[51]:


def getAvgFeatureVecs(reviews, model, num_features):
    # 리뷰 단어 목록의 각각에 대한 평균 feature 벡터를 계산하고 
    # 2D numpy 배열을 반환한다.
    
    # 카운터를 초기화 한다.
    counter = 0.
    # 속도를 위해 2D 넘파이 배열을 미리 할당한다.
    reviewFeatureVecs = np.zeros(
        (len(reviews),num_features),dtype="float32")
    
    for review in reviews:
       # 매 1000개 리뷰마다 상태를 출력
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
       # 평균 피처 벡터를 만들기 위해 위에서 정의한 함수를 호출한다.
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model,            num_features)
       # 카운터를 증가시킨다.
        counter = counter + 1.
    return reviewFeatureVecs


# In[52]:


# 멀티스레드로 4개의 워커를 사용해 처리한다.
def getCleanReviews(data):
    clean_reviews = []
    clean_reviews = KaggleWord2VecUtility.apply_by_multiprocessing(        data["특이사항"], KaggleWord2VecUtility.review_to_wordlist,        workers=4)
    return clean_reviews


# In[53]:


claim_df = claim[['사용일수', '주행거리(km로 환산)', '특이사항', '현상코드_y']]
claim_df.head()


# In[54]:


X = claim_df[['사용일수', '주행거리(km로 환산)', '특이사항']]
y = claim_df['현상코드_y']

X_train, X_test, Y_train, Y_test = train_test_split (X, y, test_size = 0.20, random_state=42)


# In[55]:


get_ipython().run_line_magic('time', 'trainDataVecs = getAvgFeatureVecs(    getCleanReviews(X_train), model, num_features )')


# In[56]:


get_ipython().run_line_magic('time', 'testDataVecs = getAvgFeatureVecs(        getCleanReviews(X_test), model, num_features )')


# In[57]:


trainDataVecs


# In[58]:


testDataVecs


# In[59]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[60]:


ml = SVC(kernel='linear', C=1.0, random_state=0)
ml.fit(trainDataVecs, Y_train)
y_pred = ml.predict(testDataVecs)


# In[61]:


from sklearn import metrics
print('정확도 : ', metrics.accuracy_score(Y_test, y_pred))

print('f1-score macro: ', metrics.f1_score(Y_test, y_pred, average='macro'))

print('f1-score micro: ', metrics.f1_score(Y_test, y_pred, average='micro'))

print('f1-score weighted: ', metrics.f1_score(Y_test, y_pred, average='weighted'))


# In[ ]:




