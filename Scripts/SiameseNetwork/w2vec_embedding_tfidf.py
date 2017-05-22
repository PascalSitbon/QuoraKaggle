import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

print('=======================','\n','Loading Data')
df = pd.read_csv("train.csv")
df_test= pd.read_csv('test.csv')
nlp = spacy.load('en')

np.random.seed(42)

questions = list(df['question1'].values.astype(str)) + list(df['question2'].values.astype(str))
questions_ = list(df_test['question1'].values.astype(str)) + list(df_test['question2'].values.astype(str))

tfidf = TfidfVectorizer(lowercase=False, stop_words='english', )
tfidf.fit_transform(questions+questions_)
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

vecs1 = []
for qu in tqdm(list(df['question1'])):
    doc = nlp(qu)
    mean_vec = np.zeros([len(doc), 300])
    cpt=0
    for word in doc:
        vec = word.vector
        try:
            idf = word2tfidf[str(word)]
            cpt+=1
        except:
            idf = 0
        mean_vec[cpt,:] = vec * idf
    mean_vec = mean_vec.sum(axis=0)
    vecs1.append(mean_vec)
df['q1_feats'] = list(vecs1)

vecs2 = []
for qu in tqdm(list(df['question2'])):
    doc = nlp(qu)
    mean_vec = np.zeros([len(doc), 300])
    cpt=0
    for word in doc:
        # word2vec
        vec = word.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word)]
        except:
            #print word
            idf = 0
        # compute final vec
        mean_vec[cpt,:] = vec * idf
    mean_vec = mean_vec.sum(axis=0)
    vecs2.append(mean_vec)
df['q2_feats'] = list(vecs2)

df.to_pickle('/Users/PascTrainingW2vectfidf.csv')