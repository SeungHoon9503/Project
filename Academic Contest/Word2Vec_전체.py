#!/usr/bin/env python
# coding: utf-8

# In[26]:


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


# In[27]:


claim = pd.read_csv('1005_data.csv')


# In[28]:


claim.head()


# In[29]:


claim = claim.dropna(subset = ['특이사항'])
claim  = claim.astype({'특이사항' : 'str'})
claim.head()


# In[30]:


claim['현상코드_y'] = claim['원인부품숫자'].astype('category')
claim['현상코드_y'] = claim['현상코드_y'].cat.codes


# In[31]:


from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split (claim['특이사항'], claim['현상코드_y'], test_size = 0.20, random_state=42)


# In[32]:


X_train


# In[33]:


from KaggleWord2VecUtility import KaggleWord2VecUtility
import nltk


# In[34]:


nltk.download('punkt')


# In[35]:


sentences_all = []
for review in claim['특이사항'].values:
    sentences_all += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)


# In[36]:


sentences_all[0][:10]


# In[37]:


import gensim


# In[38]:


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


# In[39]:


# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace=True)

model_name = '300features_40minwords_10text'
# model_name = '300features_50minwords_20text'
model.save(model_name)


# In[40]:


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


# In[41]:


df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
print(df.shape)

df.head()


# In[42]:


fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)
ax.set_title("W2V", fontsize=50)
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


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn import metrics

X = model1_df[features.columns]
y = model1_df['현상코드_y'].values

X_train, X_test, Y_train, Y_test = train_test_split (X, y, test_size = 0.20, random_state=42)

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=100)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)


print('정확도 : ', metrics.accuracy_score(Y_test, y_pred))

print('f1-score macro: ', metrics.f1_score(Y_test, y_pred, average='macro'))

print('f1-score micro: ', metrics.f1_score(Y_test, y_pred, average='micro'))

print('f1-score weighted: ', metrics.f1_score(Y_test, y_pred, average='weighted'))


# In[29]:


plt.figure(figsize=(12,8))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')  


# In[30]:


model1_df.head()


# In[31]:


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


# In[32]:


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


# In[33]:


# 멀티스레드로 4개의 워커를 사용해 처리한다.
def getCleanReviews(data):
    clean_reviews = []
    clean_reviews = KaggleWord2VecUtility.apply_by_multiprocessing(        data["특이사항"], KaggleWord2VecUtility.review_to_wordlist,        workers=4)
    return clean_reviews


# In[34]:


claim_df = claim[['사용일수', '주행거리(km로 환산)', '특이사항', '현상코드_y']]
claim_df.head()


# In[35]:


X = claim_df[['사용일수', '주행거리(km로 환산)', '특이사항']]
y = claim_df['현상코드_y']

X_train, X_test, Y_train, Y_test = train_test_split (X, y, test_size = 0.20, random_state=42)


# In[36]:


get_ipython().run_line_magic('time', 'trainDataVecs = getAvgFeatureVecs(    getCleanReviews(X_train), model, num_features )')


# In[37]:


get_ipython().run_line_magic('time', 'testDataVecs = getAvgFeatureVecs(        getCleanReviews(X_test), model, num_features )')


# In[38]:


trainDataVecs


# In[39]:


testDataVecs


# In[40]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1, random_state=2018)


# In[41]:


get_ipython().run_line_magic('time', 'forest = forest.fit( trainDataVecs, Y_train)')


# In[42]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn import metrics

y_pred = forest.predict(testDataVecs)


print('정확도 : ', metrics.accuracy_score(Y_test, y_pred))

print('f1-score macro: ', metrics.f1_score(Y_test, y_pred, average='macro'))

print('f1-score micro: ', metrics.f1_score(Y_test, y_pred, average='micro'))

print('f1-score weighted: ', metrics.f1_score(Y_test, y_pred, average='weighted'))


# In[43]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn import metrics


# In[47]:


claim[['현상명', '현상코드', '현상코드_y']]


# In[48]:


from sklearn.metrics import classification_report

print(classification_report(Y_test, y_pred))


# In[49]:


from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('time', "score = np.mean(cross_val_score(    forest, trainDataVecs,     Y_train, cv=10, scoring='accuracy'))")


# In[50]:


score


# In[51]:


output = pd.DataFrame(data={"predict":result})
output


# In[ ]:




