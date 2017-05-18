import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import nltk
from nltk import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import scipy
from sklearn.grid_search import GridSearchCV
import sys
import pdb


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
    size_train = min(sentences_train.shape[0],int(sys.argv[1]))




def tf_idf_kronecker(data_set,max_features):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  max_features=max_features,
                                   stop_words='english',analyzer = 'word')
    tfidf_sentences= tfidf_vectorizer.fit_transform(data_set.flatten())
    res = scipy.sparse.lil_matrix((data_set.shape[0], tfidf_sentences.shape[1]))
    for i in range(data_set.shape[0]):
        res[i,:] = (tfidf_sentences[2*i,:].multiply(tfidf_sentences[2*i+1,:]))
    return scipy.sparse.csr_matrix(res)




print('tfidf features')
X = tf_idf_kronecker(sentences_train[:size_train,:],max_features= 10000)
LR = LogisticRegression()
print(X.shape)
#X = preprocessing.scale(X).astype(float)
y = labels[:size_train].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
print('Grid Search for the parameters of the logistic regression')
param_grid = {"C": [ 0.1, 1, 10,100],
              "penalty": ['l1', 'l2'],
              "class_weight": [{1: 0.46, 0: 1.32}]
              }
print('learning')
# run grid search
grid_search = GridSearchCV(LR, param_grid=param_grid, scoring='log_loss')
grid_search.fit(X_train, y_train)
clf = grid_search.best_estimator_
clf.fit(X_train, y_train)

print('Training Score:', sklearn.metrics.log_loss(y_train, clf.predict_proba(X_train)[:, 1]))
print('Testing Score:', sklearn.metrics.log_loss(y_test, clf.predict_proba(X_test)[:, 1]))
pdb.set_trace()
clf.fit(X, y)

print('\n ==============')
print('Predicting Submission Set')
print('\n ==============')
Testing_Set = pd.read_csv('test.csv')
print('\n ==============')
print('calucalting hash and freq features')
sentences_test = Testing_Set[['question1','question2']].values
X_submission = tf_idf_kronecker(sentences_test)
y_submission = clf.predict_proba(X_submission)[:, 1]
Testing_Set['is_duplicate'] = y_submission
submission = Testing_Set[['test_id', 'is_duplicate']]
submission = submission.to_csv('/Users/pascalsitbon/work/Kaggle/pred.csv', index=False)