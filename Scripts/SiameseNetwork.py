import spacy
import pandas as pd
from sklearn.metrics import log_loss
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
num_train = int(df.shape[0] * 0.88)
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
optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)

for epoch in range(10):
    net.fit([X_train[:, 0, :], X_train[:, 1, :]], Y_train,
            validation_data=([X_test[:, 0, :], X_test[:, 1, :]], Y_test),
            batch_size=128, nb_epoch=1, shuffle=True, )

    pred_test = net.predict([X_test[:, 0, :], X_test[:, 1, :]], batch_size=128)
    pred_train = net.predict([X_train[:, 0, :], X_train[:, 1, :]], batch_size=128)

    te_acc = compute_accuracy(pred_test, Y_test)
    tr_acc = compute_accuracy(pred_train,Y_train)


    print('Log Loss on train set:', log_loss(Y_train, pred_train>0.5))
    print('Log Loss on test set:',log_loss(Y_test,pred_test>0.5))

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))