import spacy
import pandas as pd
from siamesefeatures import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from keras.optimizers import RMSprop, SGD, Adam
df = pd.read_csv("train.csv")
nlp = spacy.load('en')

# merge texts
questions = list(df['question1'].values.astype(str)) + list(df['question2'].values.astype(str))

tfidf = TfidfVectorizer(lowercase=False, )
tfidf.fit_transform(questions)


# dict key:word and value:tf-idf score
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

vecs1 = []
for qu in tqdm(list(df['question1'])):
    doc = nlp(qu)
    mean_vec = np.zeros([len(doc), 300])
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
        mean_vec += vec * idf
    mean_vec = mean_vec.mean(axis=0)
    vecs1.append(mean_vec)
df['q1_feats'] = list(vecs1)

vecs2 = []
import pdb
for qu in tqdm(list(df['question2'])):
    doc = nlp(str(qu))
    mean_vec = np.zeros([len(doc), 300])
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
        mean_vec += vec * idf
    mean_vec = mean_vec.mean(axis=0)
    vecs2.append(mean_vec)
df['q2_feats'] = list(vecs2)
# shuffle df
df = df.reindex(np.random.permutation(df.index))

# set number of train and test instances
num_train = int(df.shape[0] * 0.98)
num_test = df.shape[0] - num_train
print("Number of training pairs: %i" % (num_train))
print("Number of testing pairs: %i" % (num_test))

# init data data arrays
X_train = np.zeros([num_train, 2, 300])
X_test = np.zeros([num_test, 2, 300])
Y_train = np.zeros([num_train])
Y_test = np.zeros([num_test])

# format data
b = [a[None, :] for a in list(df['q1_feats'].values)]
q1_feats = np.concatenate(b, axis=0)



b = [a[None, :] for a in list(df['q2_feats'].values)]
q2_feats = np.concatenate(b, axis=0)

# fill data arrays with features
X_train[:, 0, :] = q1_feats[:num_train]
X_train[:, 1, :] = q2_feats[:num_train]
Y_train = df[:num_train]['is_duplicate'].values

X_test[:, 0, :] = q1_feats[num_train:]
X_test[:, 1, :] = q2_feats[num_train:]
Y_test = df[num_train:]['is_duplicate'].values

# remove useless variables
del b
del q1_feats
del q2_feats

net = create_network(300)

# train
# optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
optimizer = Adam(lr=0.17,decay= 0.005)
net.compile(loss='binary_crossentropy', optimizer=optimizer)
net.fit([X_train[:, 0, :], X_train[:, 1, :]], Y_train,
        validation_data=([X_test[:, 0, :], X_test[:, 1, :]], Y_test),
        batch_size=128, epochs=20, shuffle=True,class_weight={1: 0.46, 0: 1.32})

print('Preparing Submission Data')

df_test= pd.read_csv('test.csv')
questions = list(df_test['question1'].values.astype(str)) + list(df_test['question2'].values.astype(str))
tfidf.transform(questions)
# tfidf = TfidfVectorizer(lowercase=False, )
# tfidf.fit_transform(questions)
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))



vecs1 = []
for qu in tqdm(list(df_test['question1'])):
    doc = nlp(str(qu))
    mean_vec = np.zeros([len(doc), 300])
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
        mean_vec += vec * idf
    mean_vec = mean_vec.mean(axis=0)
    vecs1.append(mean_vec)
df_test['q1_feats'] = list(vecs1)

vecs2 = []
for qu in tqdm(list(df_test['question2'])):
    doc = nlp(str(qu))
    mean_vec = np.zeros([len(doc), 300])
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
        mean_vec += vec * idf
    mean_vec = mean_vec.mean(axis=0)
    vecs2.append(mean_vec)
df_test['q2_feats'] = list(vecs2)


## other alternative
seperators= [750000,1500000]

X_sub1 = np.zeros([750000, 2, 300])
b = [a[None, :] for a in list(df_test['q1_feats'].iloc[:seperators[0]].values)]
q1_feats = np.concatenate(b, axis=0)
b = [a[None, :] for a in list(df_test['q2_feats'].iloc[:seperators[0]].values)]
q2_feats = np.concatenate(b, axis=0)
X_sub1[:, 0, :] = q1_feats
X_sub1[:, 1, :] = q2_feats
del b
del q1_feats
del q2_feats


X_sub2 = np.zeros([750000, 2, 300])
b = [a[None, :] for a in list(df_test['q1_feats'].iloc[seperators[0]:seperators[1]].values)]
q1_feats = np.concatenate(b, axis=0)
b = [a[None, :] for a in list(df_test['q2_feats'].iloc[seperators[0]:seperators[1]].values)]
q2_feats = np.concatenate(b, axis=0)
X_sub2[:, 0, :] = q1_feats
X_sub2[:, 1, :] = q2_feats
del b
del q1_feats
del q2_feats


X_sub3 = np.zeros([845796, 2, 300])
b = [a[None, :] for a in list(df_test['q1_feats'].iloc[seperators[1]:].values)]
q1_feats = np.concatenate(b, axis=0)
b = [a[None, :] for a in list(df_test['q2_feats'].iloc[seperators[1]:].values)]
q2_feats = np.concatenate(b, axis=0)
X_sub3[:, 0, :] = q1_feats
X_sub3[:, 1, :] = q2_feats
del b
del q1_feats
del q2_feats

testPredictions1 = net.predict([X_sub1[:, 0, :],X_sub1[:, 1, :]])
testPredictions2 = net.predict([X_sub2[:, 0, :],X_sub2[:, 1, :]])
testPredictions3 = net.predict([X_sub3[:, 0, :],X_sub3[:, 1, :]])
testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))

submissionName = 'Siamese'

submission = pd.DataFrame()
submission['test_id'] = df_test['test_id']
submission['is_duplicate'] = testPredictions
submission.to_csv(submissionName + '.csv', index=False)