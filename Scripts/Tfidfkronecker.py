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


np.random.seed(42)

#Loading Data
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
print('\n ==========')
print('Loeading Data')
Training_Set = pd.read_csv('train.csv').dropna()
Testing_Set = pd.read_csv('test.csv')
sentences_train = (Training_Set[['question1','question2']].values).astype(str)
sentences_test = (Testing_Set[['question1','question2']].values).astype(str)
labels = Training_Set['is_duplicate'].values.astype(float)


def tf_idf_kronecker(data_set,max_features):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  max_features=max_features,
                                   stop_words='english',analyzer = 'word')
    tfidf_sentences= tfidf_vectorizer.fit_transform(data_set.flatten())
    res = scipy.sparse.lil_matrix((data_set.shape[0], max_features))
    for i in range(data_set.shape[0]):
        res[i,:] = (tfidf_sentences[2*i,:].multiply(tfidf_sentences[2*i+1,:]))
    return scipy.sparse.csr_matrix(res)



print('\n ==========')
print('tfidf features')
sentences = np.concatenate([sentences_train,sentences_test])
TFIDF = tf_idf_kronecker(sentences,max_features= 10000)
X = TFIDF[:sentences_train.shape[0],:]
print('\n ==========')
print('Training on', X.shape[0],'examples')
X_submission = TFIDF[sentences_train.shape[0]:,:]
LR = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, labels,
                                                    test_size=0.2,
                                                    random_state=42)
print('\n ==========')
print('Grid Search for the parameters of the logistic regression')
param_grid = {"C": [ 0.1, 1, 10,100],
              "penalty": ['l1', 'l2'],
              "class_weight": [{1: 0.46, 0: 1.32}]
              }

print('\n ==========')
print('learning')
# run grid search
grid_search = GridSearchCV(LR, param_grid=param_grid, scoring='log_loss')
grid_search.fit(X_train, y_train)
clf = grid_search.best_estimator_
clf.fit(X_train, y_train)

print('Training Score:', sklearn.metrics.log_loss(y_train, clf.predict_proba(X_train)[:, 1]))
print('Testing Score:', sklearn.metrics.log_loss(y_test, clf.predict_proba(X_test)[:, 1]))
clf.fit(X, labels)

print('\n ==============')
print('Predicting Submission Set')

print('\n ==============')
y_submission = clf.predict_proba(X_submission)[:, 1]
Testing_Set['is_duplicate'] = y_submission
submission = Testing_Set[['test_id', 'is_duplicate']]
submission = submission.to_csv('/Users/pascalsitbon/work/Kaggle/pred.csv', index=False)