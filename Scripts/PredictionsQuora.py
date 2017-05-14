import sklearn
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import gensim
import numpy as np
import timeit
import nltk
import pandas as pd
from nltk.stem import *
import logging
import sys
from Features.py import doc2vecs_features,n_grams_features,freq_hash
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_name = None

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

#Doc2Vec Features
print('calculating doc2vec features')
doc_2_vec_features_train = doc2vecs_features(sentences_train[:size_train,:],stpwds,model_name=model_name)

#frequency and hash Features
print('calucalting hash and freq features')
train_orig =  pd.read_csv('train.csv', header=0)
test_orig =  pd.read_csv('test.csv', header=0)
train_comb,test_comb = freq_hash(train_orig,test_orig)
train_comb_features_train = train_comb[['q1_hash','q2_hash','q1_freq','q2_freq']].iloc[:size_train].values

# N-grams features
print('Calculating n grams features')
n_grams_features = n_grams_features(sentences_train[:size_train,:],stpwds)



pdb.set_trace()
X = np.column_stack([train_comb_features_train,n_grams_features,doc_2_vec_features_train])
X = preprocessing.scale(X)
y = labels[:size_train].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

clf = RandomForestClassifier(n_estimators=200,max_depth = 2,max_features=10)
clf.fit(X_train,y_train)
print('Training Score:',sklearn.metrics.log_loss(y_train,clf.predict_proba(X_train)[:,1]))
print('Testing Score:',sklearn.metrics.log_loss(y_test,clf.predict_proba(X_test)[:,1]))

pdb.set_trace()
