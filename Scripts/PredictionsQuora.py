import sklearn
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import nltk
import pandas as pd
from nltk.stem import *
import logging
from sklearn.grid_search import GridSearchCV
import sys
import gensim
from .Features import n_grams,freq_hash, preprocess, word2vec_features
import copy
from collections import Counter
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_name = None

np.random.seed(42)

#Loading Data
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
Training_Set = pd.read_csv('train.csv').dropna()
clean_data = Training_Set[['question1','question2','is_duplicate']].values
sentences_train = clean_data[:,:2]
labels = clean_data[:,2]


#Size of Training Set used
if len(sys.argv)==1:
    size_train = sentences_train.shape[0]
else:
    size_train = int(sys.argv[1])



#tfidf stuff
print('diverse tfidf and basic features')
df_train = Training_Set.iloc[:size_train]
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)

counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
stops = stpwds

def word_shares(row):
    q1 = set(str(row['question1']).lower().split())
    q1words = q1.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0'

    q2 = set(str(row['question2']).lower().split())
    q2words = q2.difference(stops)
    if len(q2words) == 0:
        return '0:0:0:0:0'
    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    if len(shared_words) > 0:
        R1 = np.sum(shared_weights) / np.sum(total_weights)  # tfidf share
    else:
        R1 = 0
    R2 = len(shared_words) / (len(q1words) + len(q2words))  # count share
    R31 = len(q1stops) / len(q1words)  # stops in q1
    R32 = len(q2stops) / len(q2words)  # stops in q2
    return '{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32)

df = copy.deepcopy(df_train)
df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

x = pd.DataFrame()

x['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
x['shared_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

x['stops1_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
x['stops2_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']

x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
x['duplicated'] = df.duplicated(['question1','question2']).astype(int)

word_shares_features = x.values

#word2vec
# print('Training Word2Vec')
sentences = preprocess(sentences_train[:size_train,:],stpwds)
model = gensim.models.Word2Vec(sentences, min_count=1,size=248,workers=5)
model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
for i in range(10):
    print('epoch',i)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
print('calculating word2vec features')
word2vec_features_ = word2vec_features(model,sentences)



# #Doc2Vec Features
# print('calculating doc2vec features')
# doc_2_vec_features_train = doc2vecs_features(sentences_train[:size_train,:],stpwds,model_name=model_name)

#frequency and hash Features
print('calucalting hash and freq features')
train_orig =  pd.read_csv('train.csv', header=0)
test_orig =  pd.read_csv('test.csv', header=0)

train_comb,test_comb = freq_hash(train_orig,test_orig)

train_comb_features_train = train_comb[['q1_hash','q2_hash','q1_freq','q2_freq']].iloc[:size_train].values



# N-grams features
print('Calculating n grams features')
n_grams_features = n_grams(sentences_train[:size_train,:],stpwds)


#Training
X = np.column_stack([train_comb_features_train,n_grams_features,word2vec_features_,word_shares_features])
X = preprocessing.scale(X)
y = labels[:size_train].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

param_grid = {"max_depth": [3, 5, 10, 12],
              "max_features": [3, 5, 10, 12],
              "n_estimators": [100],
              "class_weight": [None,{1: 0.46, 0: 1.32}]
              }

# run grid search
clf = RandomForestClassifier()
grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='log_loss')
grid_search.fit(X_train, y_train)
clf = grid_search.best_estimator_
clf.fit(X_train, y_train)
print('Training Score:', sklearn.metrics.log_loss(y_train, clf.predict_proba(X_train)[:, 1]))
print('Testing Score:', sklearn.metrics.log_loss(y_test, clf.predict_proba(X_test)[:, 1]))



print('Predicting Submission Set')
Testing_Set= pd.read_csv('test.csv')


print('calucalting hash and freq features')
test_comb_features = test_comb[['q1_hash','q2_hash','q1_freq','q2_freq']].values


print('\n Computing Share Words and Word Match')

df_test = Testing_Set
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
words = (" ".join(test_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
stops = stpwds

df = copy.deepcopy(Testing_Set)
df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

x = pd.DataFrame()

x['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
x['shared_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

x['stops1_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
x['stops2_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']

x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
x['duplicated'] = df.duplicated(['question1','question2']).astype(int)

word_shares_features_test = x.values


print('\n Computing Word2Vec Features')

sentences_test = Testing_Set[['question1','question2']].values
sentences_submission = preprocess(sentences_test,stpwds)

model = gensim.models.Word2Vec(sentences_submission, min_count=1,size=248,workers=5)
model.train(sentences_submission,total_examples=model.corpus_count,epochs=model.iter)


print('\n Computing n_grams Features on Submission Set')
n_grams_test = n_grams(sentences_test,stpwds)

for i in range(10):
    print('epoch',i)
    model.train(sentences_submission, total_examples=model.corpus_count, epochs=model.iter)

word2vec_features_test = word2vec_features(model,sentences_submission)
X_submission = np.column_stack([test_comb_features,n_grams_test,word2vec_features_test,word_shares_features_test])
X_submission = preprocessing.scale(X_submission)
y_submission = clf.predict_proba(X_submission)[:,1]
Testing_Set['is_duplicate'] = y_submission
submission = Testing_Set[['test_id','is_duplicate']]
submission = submission.to_csv('/Users/pascalsitbon/work/Kaggle/pred.csv',index=False)