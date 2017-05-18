import sklearn
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as skm

np.random.seed(42)

#Loading Data
print('---------------------------------------------')
print('loading training data')
trainDF = pd.read_csv('train.csv')
trainDF = trainDF.dropna(how="any").reset_index(drop=True)

print('---------------------------------------------')
print('Training Tf Idf')
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=50,  max_features=300000,
                                   stop_words='english', analyzer='word',ngram_range=(1,7))
tfidf_vectorizer.fit(pd.concat((trainDF.ix[:,'question1'],trainDF.ix[:,'question2'])).unique())
trainQuestion1_BOW_rep = tfidf_vectorizer.transform(trainDF.ix[:,'question1'])
trainQuestion2_BOW_rep = tfidf_vectorizer.transform(trainDF.ix[:,'question2'])
labels = np.array(trainDF.ix[:,'is_duplicate'])

crossValidationStartTime = time.time()

numCVSplits = 8
numSplitsToBreakAfter = 2

X = (trainQuestion1_BOW_rep.multiply(trainQuestion2_BOW_rep).astype(float))
y = labels

logisticRegressor = LogisticRegression(C=0.1,penalty='l1')

logRegAccuracy = []
logRegLogLoss = []
logRegAUC = []

print('---------------------------------------------')
stratifiedCV = skm.StratifiedKFold(n_splits=numCVSplits, random_state=2)
for k, (trainInds, validInds) in enumerate(stratifiedCV.split(X, y)):
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


print('---------------------------------------------')
print('Training on whole data')
trainingStartTime = time.time()

logisticRegressor = sklearn.linear_model.LogisticRegression(C=0.1, solver='sag',
                                                    class_weight={1: 0.46, 0: 1.32})
logisticRegressor.fit(X, y)

trainingDurationInMinutes = (time.time()-trainingStartTime)/60.0
print('full training took %.2f minutes' % (trainingDurationInMinutes))


testPredictionStartTime = time.time()

testDF = pd.read_csv('test.csv')
testDF.ix[testDF['question1'].isnull(),['question1','question2']] = 'random empty question'
testDF.ix[testDF['question2'].isnull(),['question1','question2']] = 'random empty question'

testQuestion1_BOW_rep = tfidf_vectorizer.transform(testDF.ix[:,'question1'])
testQuestion2_BOW_rep = tfidf_vectorizer.transform(testDF.ix[:,'question2'])

X_test = (testQuestion1_BOW_rep.multiply(testQuestion2_BOW_rep)).astype(float)


# quick fix to avoid memory errors
seperators= [750000,1500000]
testPredictions1 = logisticRegressor.predict_proba(X_test[:seperators[0],:])[:,1]
testPredictions2 = logisticRegressor.predict_proba(X_test[seperators[0]:seperators[1],:])[:,1]
testPredictions3 = logisticRegressor.predict_proba(X_test[seperators[1]:,:])[:,1]
testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))


submissionName = 'TfidfKronecker'

submission = pd.DataFrame()
submission['test_id'] = testDF['test_id']
submission['is_duplicate'] = testPredictions
submission.to_csv(submissionName + '.csv', index=False)