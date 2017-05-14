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
from nltk import word_tokenize, ngrams
import logging
import sys
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
def doc2vecs_features(sentences_train, nb_epochs=100, alpha=0.025, min_alpha=0.025):
    sentences = []
    stemmer = PorterStemmer()
    for i in range(sentences_train.shape[0]):
        source_sentence = sentences_train[i, 0].lower().split(" ")
        source_sentence = [token for token in source_sentence if token not in stpwds]
        unigrams_que1 = [stemmer.stem(token) for token in source_sentence]
        sentences.append(unigrams_que1)

        target_sentence = sentences_train[i, 1].lower().split(" ")
        target_sentence = [token for token in target_sentence if token not in stpwds]
        unigrams_que2 = [stemmer.stem(token) for token in target_sentence]
        sentences.append(unigrams_que2)

    texts = sentences.copy()
    documents = []
    ct = 0
    for doc in texts:
        doc = gensim.models.doc2vec.LabeledSentence(words=doc, tags=['SENT_' + str(ct)])
        ct += 1
        documents.append(doc)
    model = gensim.models.Doc2Vec(alpha=alpha, min_alpha=min_alpha, min_count=1, workers=4)
    model.build_vocab(documents)

    for epoch in range(nb_epochs):
        model.train(documents, total_examples=model.corpus_count, epochs=model.iter)

    most_similar_is_duo_1_2 = []
    most_similar_is_duo_2_1 = []

    most_similar_score_if_duo_1_2 = []
    most_similar_score_if_duo_2_1 = []

    n_similarities = []

    for i in range(sentences_train.shape[0]):
        most_sim_1 = model.docvecs.most_similar(["SENT_" + str(2 * i)])[0]
        most_sim_2 = model.docvecs.most_similar(["SENT_" + str(2 * i + 1)])[0]

        most_similar_is_duo_1_2.append(int(most_sim_1[0] == "SENT_" + str(2 * i + 1)))
        most_similar_is_duo_2_1.append(int(most_sim_2[0] == "SENT_" + str(2 * i)))

        most_similar_score_if_duo_1_2.append(int(most_sim_1[0] == "SENT_" + str(2 * i + 1)) * most_sim_1[1])
        most_similar_score_if_duo_2_1.append(int(most_sim_2[0] == "SENT_" + str(2 * i)) * most_sim_2[1])

        n_similarities.append(model.n_similarity(sentences[2 * i], sentences[2 * i + 1]))

        doc_2_vec_features = np.array([n_similarities,
                                       most_similar_score_if_duo_1_2,
                                       most_similar_score_if_duo_2_1,
                                       most_similar_is_duo_1_2,
                                       most_similar_is_duo_2_1]).T

    return doc_2_vec_features
print('calculating doc2vec features')
doc_2_vec_features_train = doc2vecs_features(sentences_train[:size_train,:])


#frequency and hash Features
print('calucalting hash and freq features')
train_orig =  pd.read_csv('train.csv', header=0)
test_orig =  pd.read_csv('test.csv', header=0)
def freq_hash(train_orig,test_orig):
    tic0=timeit.default_timer()
    df1 = train_orig[['question1']].copy()
    df2 = train_orig[['question2']].copy()
    df1_test = test_orig[['question1']].copy()
    df2_test = test_orig[['question2']].copy()

    df2.rename(columns = {'question2':'question1'},inplace=True)
    df2_test.rename(columns = {'question2':'question1'},inplace=True)

    train_questions = df1.append(df2)
    train_questions = train_questions.append(df1_test)
    train_questions = train_questions.append(df2_test)
    #train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
    train_questions.drop_duplicates(subset = ['question1'],inplace=True)

    train_questions.reset_index(inplace=True,drop=True)
    questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
    train_cp = train_orig.copy()
    test_cp = test_orig.copy()
    train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

    test_cp['is_duplicate'] = -1
    test_cp.rename(columns={'test_id':'id'},inplace=True)
    comb = pd.concat([train_cp,test_cp])

    comb['q1_hash'] = comb['question1'].map(questions_dict)
    comb['q2_hash'] = comb['question2'].map(questions_dict)

    q1_vc = comb.q1_hash.value_counts().to_dict()
    q2_vc = comb.q2_hash.value_counts().to_dict()

    def try_apply_dict(x,dict_to_apply):
        try:
            return dict_to_apply[x]
        except KeyError:
            return 0
    #map to frequency space
    comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
    comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

    train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]
    test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]

    return train_comb,test_comb
train_comb,test_comb = freq_hash(train_orig,test_orig)
train_comb_features_train = train_comb[['q1_hash','q2_hash','q1_freq','q2_freq']].iloc[:size_train].values

#TfIdf Cosine Similarity
def tf_idf_cosin(data_set):
    tf_idf_cosin_sim = []
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  # max_features=n_features,
                                   stop_words='english')
    tfidf_sentences= tfidf_vectorizer.fit_transform(data_set.flatten())

    for i in range(data_set.shape[0]):
        tf_idf_cosin_sim.append(cosine_similarity(tfidf_sentences[2*i,:], tfidf_sentences[2*i+1,:])[0,0])

    return np.array([tf_idf_cosin_sim]).T

#Lsi Indexing

# N-grams features
def n_grams_features(data_set):
    stemmer = PorterStemmer()
    dif_len = []
    common_unigrams_lens = []
    common_unigrams_ratios = []
    common_bigrams_lens = []
    common_bigrams_ratios = []
    common_trigrams_lens = []
    common_trigrams_ratios = []
    sentences = []
    for i in range(data_set.shape[0]):

        source_sentence = data_set[i, 0].lower().split(" ")
        source_sentence = [token for token in source_sentence if token not in stpwds]
        unigrams_que1 = [stemmer.stem(token) for token in source_sentence]
        sentences.append(unigrams_que1)

        target_sentence = data_set[i, 1].lower().split(" ")
        target_sentence = [token for token in target_sentence if token not in stpwds]
        unigrams_que2 = [stemmer.stem(token) for token in target_sentence]
        sentences.append(unigrams_que2)

        # get unigram features #
        common_unigrams_len = len(set(unigrams_que1).intersection(set(unigrams_que2)))
        common_unigrams_lens.append(common_unigrams_len)
        common_unigrams_ratios.append(
            float(common_unigrams_len) / max(len(set(unigrams_que1).union(set(unigrams_que2))), 1))

        # get bigram features #
        bigrams_que1 = [i for i in ngrams(unigrams_que1, 2)]
        bigrams_que2 = [i for i in ngrams(unigrams_que2, 2)]
        common_bigrams_len = len(set(bigrams_que1).intersection(set(bigrams_que2)))
        common_bigrams_lens.append(common_bigrams_len)
        common_bigrams_ratios.append(
            float(common_bigrams_len) / max(len(set(bigrams_que1).union(set(bigrams_que2))), 1))

        # get trigram features #
        trigrams_que1 = [i for i in ngrams(unigrams_que1, 3)]
        trigrams_que2 = [i for i in ngrams(unigrams_que2, 3)]
        common_trigrams_len = len(set(trigrams_que1).intersection(set(trigrams_que2)))
        common_trigrams_lens.append(common_trigrams_len)
        common_trigrams_ratios.append(
            float(common_trigrams_len) / max(len(set(trigrams_que1).union(set(trigrams_que2))), 1))

        dif_len.append(abs(len(source_sentence) - len(target_sentence)))

    # examples as rows, features as columns
    features = np.array([common_unigrams_lens,
                         common_unigrams_ratios,
                         common_bigrams_lens,
                         common_bigrams_ratios,
                         common_trigrams_lens,
                         common_trigrams_ratios,
                         dif_len]).T


    return features
print('Calculating n grams features')
n_grams_features = n_grams_features(sentences_train[:size_train,:])



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
