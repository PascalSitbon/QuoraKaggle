import spacy
import pandas as pd
from siamesefeatures import *
import numpy as np
from tqdm import tqdm
np.random.seed(42)
from keras.optimizers import Adam

print('=======================','\n','Loading Data')
nlp = spacy.load('en')
w2vec = '/Users/PascTrainingW2vec.csv'
w2vec_tfidf = '/Users/PascTrainingW2vectfidf.csv'
file_name = w2vec



#Loading or computing Word2Vec Embedding
try :
    df = pd.read_pickle(file_name)
except:
    df = pd.read_csv("train.csv")
    print('=======================','\n','File not found - Recalculating the embedding via Word2Vec')
    vecs1 = []
    for qu in tqdm(list(df['question1'])):
        doc = nlp(str(qu))
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


#df = pd.read_pickle(file_name)
df = df.reindex(np.random.permutation(df.index))

# set number of train and test instances
num_train = int(df.shape[0] * 0.90)
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
optimizer = Adam(lr=0.01)
net.compile(loss='binary_crossentropy', optimizer=optimizer)
net.fit([X_train[:, 0, :], X_train[:, 1, :]], Y_train,
        validation_data=([X_test[:, 0, :], X_test[:, 1, :]], Y_test),
        batch_size=128, epochs=30, shuffle=True, class_weight= {0: 1.309028344, 1: 0.472001959})

print('Preparing Submission Data')

df_test= pd.read_csv('test.csv')

vecs1 = []
for qu in tqdm(list(df_test['question1'])):
    doc = nlp(str(qu))
    mean_vec = np.zeros([len(doc), 300])
    cpt=0
    for word in doc:
        # word2vec
        vec = word.vector
        mean_vec[cpt,:] = vec
        cpt+=1
    mean_vec = mean_vec.mean(axis=0)
    vecs1.append(mean_vec)
df_test['q1_feats'] = list(vecs1)

vecs2 = []
for qu in tqdm(list(df_test['question2'])):
    doc = nlp(str(qu))
    mean_vec = np.zeros([len(doc), 300])
    cpt=0
    for word in doc:
        # word2vec
        vec = word.vector
        mean_vec[cpt,:] = vec
        cpt+=1
    mean_vec = mean_vec.mean(axis=0)
    vecs2.append(mean_vec)
df_test['q2_feats'] = list(vecs2)

seperators= [750000, 1500000]

X_sub1 = np.zeros([750000, 2, 300])
b = [a[None, :] for a in list(
    df_test['q1_feats'].iloc[:seperators[0]].values)]
q1_feats = np.concatenate(b, axis=0)
b = [a[None, :] for a in list(
    df_test['q2_feats'].iloc[:seperators[0]].values)]
q2_feats = np.concatenate(b, axis=0)
X_sub1[:, 0, :] = q1_feats
X_sub1[:, 1, :] = q2_feats
del b
del q1_feats
del q2_feats


X_sub2 = np.zeros([750000, 2, 300])
b = [a[None, :] for a in list(
    df_test['q1_feats'].iloc[seperators[0]:seperators[1]].values)]
q1_feats = np.concatenate(b, axis=0)
b = [a[None, :] for a in list(
    df_test['q2_feats'].iloc[seperators[0]:seperators[1]].values)]
q2_feats = np.concatenate(b, axis=0)
X_sub2[:, 0, :] = q1_feats
X_sub2[:, 1, :] = q2_feats
del b
del q1_feats
del q2_feats


X_sub3 = np.zeros([845796, 2, 300])
b = [a[None, :] for a in list(
    df_test['q1_feats'].iloc[seperators[1]:].values)]
q1_feats = np.concatenate(b, axis=0)
b = [a[None, :] for a in list(
    df_test['q2_feats'].iloc[seperators[1]:].values)]
q2_feats = np.concatenate(b, axis=0)
X_sub3[:, 0, :] = q1_feats
X_sub3[:, 1, :] = q2_feats
del b
del q1_feats
del q2_feats

testPredictions1 = net.predict([X_sub1[:, 0, :],X_sub1[:, 1, :]])
testPredictions2 = net.predict([X_sub2[:, 0, :],X_sub2[:, 1, :]])
testPredictions3 = net.predict([X_sub3[:, 0, :],X_sub3[:, 1, :]])
testPredictions = np.concatenate([testPredictions1,testPredictions2,testPredictions3])

submissionName = 'Siamese'

submission = pd.DataFrame()
submission['test_id'] = df_test['test_id']
submission['is_duplicate'] = testPredictions
submission.to_csv(submissionName + '.csv', index=False)
