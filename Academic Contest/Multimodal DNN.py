#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
claim = pd.read_csv('1005_data.csv')
claim.head()


# In[55]:


sample = claim['특이사항'].values[:-1]
sample


# In[56]:


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
find_between(str(sample[0]), '[', ']')


# In[57]:


unique_window = []

for i in claim['특이사항'].values:
    claims = str(i).split('\n')
    print(claims)
    for c in claims:
        w = find_between(c,'[', ']')
        unique_window.append(w)


# In[58]:


set(unique_window)


# In[59]:


claims_text = claim['특이사항'].values

p = [c for c in claims_text if '[@]' in str(c)]
T = [c for c in claims_text if '[T]' in str(c)]
A = [c for c in claims_text if '[A]' in str(c)]
S = [c for c in claims_text if '[S]' in str(c)]
L = [c for c in claims_text if '[L]' in str(c)]
R = [c for c in claims_text if '[R]' in str(c)]
D = [c for c in claims_text if '[D]' in str(c)]
O = [c for c in claims_text if '[O]' in str(c)]
M = [c for c in claims_text if '[M]' in str(c)]
C = [c for c in claims_text if '[C]' in str(c)]


# In[60]:


print(len(claims_text))
print(len(C), len(T), len(A), len(S),len(R))


# In[61]:


claim = claim.dropna()


# In[62]:


claim.tail()


# In[63]:


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
    print(result)
    pool.close()
    # 작업 결과를 합쳐서 반환
    return pd.append(result)


# In[64]:


fcode_with_window = pd.DataFrame(columns = ['사용일수','주행거리(km로 환산)','원인부품숫자', '특이사항', 'C', 'T', 'S' 'A'
                                            ,'R']) 


# In[65]:


fcode_with_window.tail()


# In[66]:


claim_top = claim


# In[67]:


def mapWindow(row):
    c_text = ""
    t_text = ""
    s_text = ""
    a_text = ""
    r_text = ""
    
    fcode = row['원인부품숫자']
    claim_text = str(row['특이사항'])
    day = int(row['사용일수'])
    km = int(row['주행거리(km로 환산)'])
    
    print(fcode, claim_text)
    claims = claim_text.split('\n')
    for c in claims:
        w = find_between(c,'[', ']')
        if 'C' in w:
            c_text = c
            
        if 'T' in w:
            t_text = c
        if 'S' in w:
            s_text = c
        if 'A' in w:
            a_text = c
        if 'R' in w:
            r_text = c
       
    row_dict = {'사용일수':day,'주행거리':km,'원인부품숫자': fcode, '특이사항':claim_text,'C':c_text,'T':t_text, 'S':s_text, 'A':a_text,
                'R':r_text}
    return row_dict


result = claim_top.apply(lambda row : mapWindow(row), axis=1)


# In[68]:


result


# In[69]:


fcode_with_window = pd.DataFrame()

for index, items in result.items():
#     print(items)
#     print(type(items))
    fcode_with_window = fcode_with_window.append(items, ignore_index=True)


# In[70]:


fcode_with_window


# In[72]:


fcode_with_window.to_csv('11count.csv')


# count

# In[73]:


fcode_window_numeric = pd.read_csv('11count.csv')
fcode_window_numeric.head()

fcode_window_numeric = fcode_window_numeric.astype({'C' : 'str'})
fcode_window_numeric = fcode_window_numeric.astype({'A' : 'str'})
fcode_window_numeric = fcode_window_numeric.astype({'R' : 'str'})
fcode_window_numeric = fcode_window_numeric.astype({'S' : 'str'})
fcode_window_numeric = fcode_window_numeric.astype({'T' : 'str'})
fcode_window_numeric = fcode_window_numeric.astype({'특이사항' : 'str'})


# In[74]:


import numpy as np
from keras.utils import np_utils


# In[75]:


from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split (fcode_window_numeric[['C', 'A',
                                                                           'R', 'S','T', '사용일수', '주행거리']], fcode_window_numeric['원인부품숫자'], test_size = 0.20, random_state=42)


Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)


# In[76]:


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


# In[77]:


import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()


# In[78]:


get_ipython().run_line_magic('time', "c_clean = apply_by_multiprocessing(    X_train['C'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "T_clean = apply_by_multiprocessing(    X_train['T'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "A_clean = apply_by_multiprocessing(    X_train['A'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "S_clean = apply_by_multiprocessing(    X_train['S'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "R_clean = apply_by_multiprocessing(    X_train['R'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "c_clean_test = apply_by_multiprocessing(    X_test['C'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "T_clean_test = apply_by_multiprocessing(    X_test['T'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "A_clean_test = apply_by_multiprocessing(    X_test['A'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "S_clean_test = apply_by_multiprocessing(    X_test['S'], review_to_words, workers=4)")

get_ipython().run_line_magic('time', "R_clean_test = apply_by_multiprocessing(    X_test['R'], review_to_words, workers=4)")


# In[79]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# 튜토리얼과 다르게 파라메터 값을 수정
# 파라메터 값만 수정해도 캐글 스코어 차이가 많이 남
vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 1, # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 3), 
                             max_features = 100
                            )
vectorizer


# In[80]:


pipeline = Pipeline([
    ('vect', vectorizer),
])  


# In[81]:


get_ipython().run_line_magic('time', 'c_train = pipeline.fit_transform(c_clean)')

get_ipython().run_line_magic('time', 'a_train = pipeline.fit_transform(A_clean)')

get_ipython().run_line_magic('time', 't_train = pipeline.fit_transform(T_clean)')

get_ipython().run_line_magic('time', 's_train = pipeline.fit_transform(S_clean)')

get_ipython().run_line_magic('time', 'r_train = pipeline.fit_transform(R_clean)')


# In[82]:


get_ipython().run_line_magic('time', 'c_test = pipeline.fit_transform(c_clean_test)')

get_ipython().run_line_magic('time', 'a_test = pipeline.fit_transform(A_clean_test)')

get_ipython().run_line_magic('time', 't_test = pipeline.fit_transform(T_clean_test)')

get_ipython().run_line_magic('time', 's_test = pipeline.fit_transform(S_clean_test)')

get_ipython().run_line_magic('time', 'r_test = pipeline.fit_transform(R_clean_test)')


# In[83]:


print(c_train.shape, c_test.shape)
print(a_train.shape, a_test.shape)
print(t_train.shape, t_test.shape)
print(s_train.shape, s_test.shape)
print(r_train.shape, r_test.shape)


# In[84]:


vocab = vectorizer.get_feature_names()
print(len(vocab))
print(vocab[:10])


# In[85]:


vocab


# In[86]:


# 벡터화 된 피처를 확인해 봄
import numpy as np
dist = np.sum(s_train, axis=0)
    
for tag, count in zip(vocab, dist):
    print(count, tag)
    
pd.DataFrame(dist, columns=vocab)


# In[87]:


# 벡터화 된 피처를 확인해 봄
import numpy as np
dist = np.sum(c_train, axis=0)
    
for tag, count in zip(vocab, dist):
    print(count, tag)
    
pd.DataFrame(dist, columns=vocab)


# In[88]:


# 벡터화 된 피처를 확인해 봄
import numpy as np
dist = np.sum(a_train, axis=0)
    
for tag, count in zip(vocab, dist):
    print(count, tag)
    
pd.DataFrame(dist, columns=vocab)


# In[89]:


# 벡터화 된 피처를 확인해 봄
import numpy as np
dist = np.sum(r_train, axis=0)
    
for tag, count in zip(vocab, dist):
    print(count, tag)
    
pd.DataFrame(dist, columns=vocab)


# In[90]:


# 벡터화 된 피처를 확인해 봄
import numpy as np
dist = np.sum(t_train, axis=0)
    
for tag, count in zip(vocab, dist):
    print(count, tag)
    
pd.DataFrame(dist, columns=vocab)


# In[91]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# %matplotlib inline 설정을 해주어야지만 노트북 안에 그래프가 디스플레이 된다.
get_ipython().run_line_magic('matplotlib', 'inline')

def displayWordCloud(data = None, backgroundcolor = 'white', width=800, height=600 ):
    wordcloud = WordCloud(stopwords = STOPWORDS, 
                          background_color = backgroundcolor, 
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show() 


# In[92]:


# 학습 데이터의 모든 단어에 대한 워드 클라우드를 그려본다.
get_ipython().run_line_magic('time', "displayWordCloud(' '.join(c_clean))")


# In[93]:


# 학습 데이터의 모든 단어에 대한 워드 클라우드를 그려본다.
get_ipython().run_line_magic('time', "displayWordCloud(' '.join(T_clean))")


# In[94]:


# 학습 데이터의 모든 단어에 대한 워드 클라우드를 그려본다.
get_ipython().run_line_magic('time', "displayWordCloud(' '.join(A_clean))")


# In[95]:


# 학습 데이터의 모든 단어에 대한 워드 클라우드를 그려본다.
get_ipython().run_line_magic('time', "displayWordCloud(' '.join(S_clean))")


# In[96]:


# 학습 데이터의 모든 단어에 대한 워드 클라우드를 그려본다.
get_ipython().run_line_magic('time', "displayWordCloud(' '.join(R_clean))")


# In[97]:


from keras.utils import plot_model

from keras.utils import to_categorical
from keras import regularizers
from keras import layers

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input

import tensorflow as tf

import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# In[98]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.f1s.append(sklm.f1_score(targ, predict,average='micro'))
        self.confusion.append(sklm.confusion_matrix(targ.argmax(axis=1),predict.argmax(axis=1)))

        return


# In[154]:


from keras import backend as K
import tensorflow as tf

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


# In[155]:


def create_model_for_merge_5(optimizer_, X0, X1, X2, X3, X4, X5, y1, rate1):
#with tf.device('/cpu:0'):

    print(X0.shape[1], X1.shape[1], X2.shape[1], X3.shape[1], X4.shape[1])
    
    input_tensor0 = Input(shape=(X0.shape[1],))
    input_tensor1 = Input(shape=(X1.shape[1],))
    input_tensor2 = Input(shape=(X2.shape[1],))
    input_tensor3 = Input(shape=(X3.shape[1],))
    input_tensor4 = Input(shape=(X4.shape[1],))
    input_tensor5 = Input(shape=(X5.shape[1],))

    # model generation
    model0 = Dense(128, activation='relu')(input_tensor0)
    model0 = Dropout(rate1)(model0)
    model0 = Dense(64, activation='relu')(model0)
    model0 = Dropout(rate1)(model0)
    model0 = Dense(64, activation='relu')(model0)
    model0 = Dropout(rate1)(model0)
    
    # model generation
    model1 = Dense(128, activation='relu')(input_tensor1)
    model1 = Dropout(rate1)(model1)
    model1 = Dense(64, activation='relu')(model1)
    model1 = Dropout(rate1)(model1)
    model1 = Dense(64, activation='relu')(model1)
    model1 = Dropout(rate1)(model1)
    
    # model generation
    model2 = Dense(128, activation='relu')(input_tensor2)
    model2 = Dropout(rate1)(model2)
    model2 = Dense(64, activation='relu')(model2)
    model2 = Dropout(rate1)(model2)
    model2 = Dense(64, activation='relu')(model2)
    model2 = Dropout(rate1)(model2)
    
    # model generation
    model3 = Dense(128, activation='relu')(input_tensor3)
    model3 = Dropout(rate1)(model3)
    model3 = Dense(64, activation='relu')(model3)
    model3 = Dropout(rate1)(model3)
    model3 = Dense(64, activation='relu')(model3)
    model3 = Dropout(rate1)(model3)
    
   # model generation
    model4 = Dense(128, activation='relu')(input_tensor4)
    model4 = Dropout(rate1)(model4)
    model4 = Dense(64, activation='relu')(model4)
    model4 = Dropout(rate1)(model4)
    model4 = Dense(64, activation='relu')(model4)
    model4 = Dropout(rate1)(model4)
    
     # model generation
    model5 = Dense(128, activation='relu')(input_tensor5)
    model5 = Dropout(rate1)(model5)
    model5 = Dense(64, activation='relu')(model5)
    model5 = Dropout(rate1)(model5)
    model5 = Dense(64, activation='relu')(model5)
    model5 = Dropout(rate1)(model5)
    
    
    # model merge
    merge = layers.concatenate([model0, model1, model2, model3, model4, model5], name='softmax')
    merge = Dense(128, activation='relu')(merge)
    merge = Dropout(rate1)(merge)
    output = Dense(y1.shape[1], activation='softmax')(merge)

    model_merged = Model([input_tensor0, input_tensor1, input_tensor2, input_tensor3, input_tensor4, input_tensor5], output)    
    print(model_merged.summary())

    model_merged.compile(loss='categorical_crossentropy',
                         optimizer=optimizer_,
                         metrics=['accuracy', f1score, f1_macro])


    return model_merged


# # model with numeric

# In[156]:


c_train_1 = pd.DataFrame(c_train.todense())
t_train_1 = pd.DataFrame(t_train.todense())
a_train_1 = pd.DataFrame(a_train.todense())
s_train_1 = pd.DataFrame(s_train.todense())
r_train_1 = pd.DataFrame(r_train.todense())


# In[148]:


Y_train


# In[146]:


c_train_1


# In[ ]:





# In[157]:


#for rate1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
for rate1 in [0.3]:
    NN_merged = create_model_for_merge_5('adam', X_train[['사용일수', '주행거리']], c_train_1, t_train_1, a_train_1, s_train_1, r_train_1,                                         Y_train, rate1)

    metrics = Metrics()
    hist_numeric = NN_merged.fit([X_train[['사용일수', '주행거리']], c_train_1, t_train_1, a_train_1, s_train_1, r_train_1], Y_train, 
                                 validation_split=0.2, 
                                 epochs=50, 
                                 batch_size=64, 
                                 verbose=1, 
                                 shuffle=True)


# In[158]:


import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist_numeric.history['loss'], 'y', label='train loss')
loss_ax.plot(hist_numeric.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist_numeric.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist_numeric.history['val_accuracy'], 'g', label='val acc')
acc_ax.plot(hist_numeric.history['val_f1_macro'], 'g', label='val macro')

acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

#plt.show()


# In[ ]:





# In[ ]:




