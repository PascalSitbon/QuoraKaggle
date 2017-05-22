#!/usr/bin/python
# -*- coding: utf-8 -*-

import sklearn
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as skm

np.random.seed(42)
#On "fixe" le random afin de pouvoir comparer les résultats successivement.

#Loading Data
print('---------------------------------------------')
print('loading training data')
trainDF = pd.read_csv('train.csv')
trainDF = trainDF.dropna(how="any").reset_index(drop=True)

print('---------------------------------------------')
print('Training Tf Idf')
count_vectorizer = CountVectorizer(max_df=0.999, min_df=50,  max_features=300000,
                                   stop_words='english', analyzer='word',ngram_range=(1,10))

#min_df ==> mots plus de 50 fois pas comptés
#analyzer "word", "char"
#max_features = nb max de colonnes dans ta matrice. Une colonne = 1 document. Une ligne = 1 terme par rapport au corpus.
#ngram_range = (1,10) ===> laisse le à (1,10), checker sur wikipedia

count_vectorizer.fit(pd.concat((trainDF.ix[:,'question1'],trainDF.ix[:,'question2'])).unique())

#fitting du vectorizer

trainQuestion1_BOW_rep = count_vectorizer.transform(trainDF.ix[:,'question1'])
#Matrice dont les lignes sont toutes les questions 1 de la base

trainQuestion2_BOW_rep = count_vectorizer.transform(trainDF.ix[:,'question2'])
#Matrice dont les lignes sont toutes les questions 1 de la base
labels = np.array(trainDF.ix[:,'is_duplicate'])
#labels = boolean "is duplicate"

crossValidationStartTime = time.time()

numCVSplits = 8
#On sépare la base initiale, je prends 75 pcent pour apprendre et je check sur les 25% restants
#Training score VS validation score
#CV = cross validation ==> 8: je train sur 7 sets et j'en predis un, puis sur 7 autres et un autre, etc,

numSplitsToBreakAfter = 6

X = (trainQuestion1_BOW_rep.multiply(trainQuestion2_BOW_rep).astype(float))
#Multiply = mutliplication terme à terme de matrice sparse dans SciPy
#Retourne une matrice multipliant les deux matrices terme à terme ==> produit de Kronecker ou tensoriel
#Multiplie la 1ere ligne de train question 1 par la 1ere ligne de train question 2
#Multiplie chaque intensité de terme. Va augmenter les intensités si les deux ne sont pas nuls.
#Checker countvectorizer au lieu de tfidf_vectorizer.

y = labels

Cs = [1,10,100]
penaltys = ['l1','l2']
best_param = {'C': 0.1, 'penalty': 'l1'}
best_score = np.inf
for C in Cs:
    for penalty in penaltys:
        print('Parameters: C:',C,'penalty:',penalty)
        logisticRegressor = LogisticRegression(C=C,penalty=penalty)

        logRegAccuracy = []
        logRegLogLoss = []
        logRegAUC = []

        print('---------------------------------------------')
        stratifiedCV = skm.StratifiedKFold(n_splits=numCVSplits, random_state=2)
        #Création des sous bases pour la cross validation
        for k, (trainInds, validInds) in enumerate(stratifiedCV.split(X, y)):
            #Pour toutes les sous-bases qu'on vient de créer
            foldTrainingStartTime = time.time()

            X_train_cv = X[trainInds, :]
            X_valid_cv = X[validInds, :]

            y_train_cv = y[trainInds]
            y_valid_cv = y[validInds]

            logisticRegressor.fit(X_train_cv, y_train_cv)

            y_train_hat = logisticRegressor.predict_proba(X_train_cv)[:, 1]
            y_valid_hat = logisticRegressor.predict_proba(X_valid_cv)[:, 1]

            logRegAccuracy.append(sklearn.metrics.accuracy_score(y_valid_cv, y_valid_hat > 0.5))
            logRegLogLoss.append(sklearn.metrics.log_loss(y_valid_cv, y_valid_hat))
            logRegAUC.append(sklearn.metrics.roc_auc_score(y_valid_cv, y_valid_hat))

            foldTrainingDurationInMinutes = (time.time() - foldTrainingStartTime) / 60.0
            print('fold %d took %.2f minutes: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (k + 1,
                                                                                               foldTrainingDurationInMinutes,
                                                                                               logRegAccuracy[-1],
                                                                                               logRegLogLoss[-1],
                                                                                               logRegAUC[-1]))

            if (k + 1) >= numSplitsToBreakAfter:
                break

        crossValidationDurationInMinutes = (time.time() - crossValidationStartTime) / 60.0

        print('---------------------------------------------')
        print('cross validation took %.2f minutes' % (crossValidationDurationInMinutes))
        print('mean CV: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (np.array(logRegAccuracy).mean(),
                                                                         np.array(logRegLogLoss).mean(),
                                                                         np.array(logRegAUC).mean()))
        print('---------------------------------------------')

        if np.array(logRegLogLoss).mean() <= best_score:
            best_param['C'] = C
            best_param['penalty'] = penalty
        print('---------------------------------------------')
import pdb
pdb.set_trace()
print('Training on whole data')
trainingStartTime = time.time()

logisticRegressor = sklearn.linear_model.LogisticRegression(C=best_param['C'],penalty=best_param['penalty'],
                                                    class_weight={1: 0.46, 0: 1.32})
logisticRegressor.fit(X, y)

trainingDurationInMinutes = (time.time()-trainingStartTime)/60.0
print('full training took %.2f minutes' % (trainingDurationInMinutes))


testPredictionStartTime = time.time()

testDF = pd.read_csv('test.csv')
testDF.ix[testDF['question1'].isnull(),['question1','question2']] = 'random empty question'
testDF.ix[testDF['question2'].isnull(),['question1','question2']] = 'random empty question'

testQuestion1_BOW_rep = count_vectorizer.transform(testDF.ix[:,'question1'])
testQuestion2_BOW_rep = count_vectorizer.transform(testDF.ix[:,'question2'])

X_test = (testQuestion1_BOW_rep.multiply(testQuestion2_BOW_rep)).astype(float)


# quick fix to avoid memory errors
seperators = [750000,1500000]
testPredictions1 = logisticRegressor.predict_proba(X_test[:seperators[0],:])[:,1]
testPredictions2 = logisticRegressor.predict_proba(X_test[seperators[0]:seperators[1],:])[:,1]
testPredictions3 = logisticRegressor.predict_proba(X_test[seperators[1]:,:])[:,1]
testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))


submissionName = 'CountKronecker'

submission = pd.DataFrame()
submission['test_id'] = testDF['test_id']
submission['is_duplicate'] = testPredictions

submission.to_csv(submissionName + '.csv', index=False)