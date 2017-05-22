import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm

print('=======================','\n','Loading Data')
df = pd.read_csv("train.csv")
nlp = spacy.load('en')
vecs1 = []

for qu in tqdm(list(df['question1'])):
    doc = nlp(qu)
    mean_vec = np.zeros([len(doc), 300])
    cpt=0
    for word in doc:
        vec = word.vector
        mean_vec[cpt,:] = vec
        cpt+=1
    mean_vec = mean_vec.mean(axis=0)
    vecs1.append(mean_vec)
df['q1_feats'] = list(vecs1)

vecs2 = []
for qu in tqdm(list(df['question2'])):
    doc = nlp(str(qu))
    mean_vec = np.zeros([len(doc), 300])
    cpt=0
    for word in doc:
        vec = word.vector
        mean_vec[cpt,:] = vec
        cpt+=1
    mean_vec = mean_vec.mean(axis=0)
    vecs2.append(mean_vec)
df['q2_feats'] = list(vecs2)

df.to_pickle('/Users/PascTrainingW2vec.csv')