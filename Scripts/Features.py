import gensim
import numpy as np
import timeit
import pandas as pd
from nltk.stem import *
from nltk import word_tokenize, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csr import csr_matrix

#Ideas on features based on word2vec:

#remove duplicates in questions and calculate the max/min (max can be took without removing duplicate words
#distance between word2vec representation of words. sounds like a good plan

#L2 norm/similarity between centroids of sentences.

#

def preprocess(sentences_train,stpwds):
    sentences = []
    stemmer = PorterStemmer()
    for i in range(sentences_train.shape[0]):

        source_sentence = sentences_train[i,0].lower().split(" ")
        source_sentence = [token for token in source_sentence if token not in stpwds]
        unigrams_que1 = [stemmer.stem(token) for token in source_sentence]
        sentences.append(unigrams_que1)

        target_sentence = sentences_train[i,1].lower().split(" ")
        target_sentence= [token for token in target_sentence if token not in stpwds]
        unigrams_que2 = [stemmer.stem(token) for token in target_sentence]
        sentences.append(unigrams_que2)
    return sentences


def word2vec_features(model, sentences):
    'sentences :  [[token1,...,tokend],..,[token1,...,tokend]]'
    'model : Trained Word 2 Vec model'

    max_distance_tokens = []
    min_distance_tokens_duplic_removed = []
    min_distances = []
    centroid_distances = []

    for i in range(int(len(sentences) / 2)):
        if i%10000 == 0:
            print(i)
        d_min_classic = 0
        d_min = 0
        d_max = 0
        distance_centroid = 0

        set1 = set(sentences[2 * i])
        set2 = set(sentences[2 * i + 1])

        sym_dif = set1.symmetric_difference(set2)

        # in the else condition we assume set1 and set2 non empty
        if len(set1) > 0 and len(set2) > 0:
            for token1 in set1:
                for token2 in set2:
                    distance_ = cosine_similarity(model[token1].reshape(1, -1), model[token2].reshape(1, -1))[0, 0]
                    if distance_ <= d_min_classic:
                        d_min_classic = distance_

                        # following condition is same with adding len(set2)>0
            # means set1==set2 and non empty sets
            if len(sym_dif) == 0:
                d_min = 1
                d_max = 1
            else:
                if set1.issubset(set2) or set2.issubset(set1):
                    d_min = 0
                    d_max = 0
                else:
                    for token1 in set1 & sym_dif:
                        for token2 in set2 & sym_dif:
                            distance_tokens = \
                            cosine_similarity(model[token1].reshape(1, -1), model[token2].reshape(1, -1))[0, 0]
                            if distance_tokens <= d_min:
                                d_min = distance_tokens
                            if distance_tokens >= d_max:
                                d_max = distance_tokens

        #this can be improved by taking tf_idf weights over each tokens.
        if min(len(set1), len(set2)) > 0:
            centroid1 = np.sum([model[token1] for token1 in set1], axis=0) / len(set1)
            centroid2 = np.sum([model[token2] for token2 in set2], axis=0) / len(set2)
            distance_centroid = cosine_similarity(centroid1.reshape(1, -1), centroid2.reshape(1, -1))[0, 0]

        max_distance_tokens.append(d_max)
        min_distances.append(d_min_classic)
        min_distance_tokens_duplic_removed.append(d_min)
        centroid_distances.append(distance_centroid)

    word2vec_features = np.array(
        [centroid_distances, min_distances, min_distance_tokens_duplic_removed, max_distance_tokens]).T

    return word2vec_features


def doc2vecs_features(sentences_train,stpwds,model_name = None ,nb_epochs=100, alpha=0.025, min_alpha=0.025):
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
    if model_name is None:
        for doc in texts:
            doc = gensim.models.doc2vec.LabeledSentence(words=doc, tags=['SENT_' + str(ct)])
            ct += 1
            documents.append(doc)
        model = gensim.models.Doc2Vec(alpha=alpha, min_alpha=min_alpha, min_count=1, workers=4)
        model.build_vocab(documents)

        for epoch in range(nb_epochs):
            model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
        else:
            model = gensim.models.Doc2Vec.load(model_name)

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
        if len(sentences[2 * i]) !=0 and len(sentences[2*i+ 1] )!=0:
            n_similarities.append(model.n_similarity(sentences[2 * i], sentences[2 * i + 1]))
        else:
            n_similarities.append(0)

        doc_2_vec_features = np.array([n_similarities,
                                       most_similar_score_if_duo_1_2,
                                       most_similar_score_if_duo_2_1,
                                       most_similar_is_duo_1_2,
                                       most_similar_is_duo_2_1]).T

    return doc_2_vec_features

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

def tf_idf_cosin(data_set):
    tf_idf_cosin_sim = []
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  max_features=30,
                                   stop_words='english')
    tfidf_sentences= tfidf_vectorizer.fit_transform(data_set.flatten())

    for i in range(data_set.shape[0]):
        tf_idf_cosin_sim.append(cosine_similarity(tfidf_sentences[2*i,:], tfidf_sentences[2*i+1,:])[0,0])

    return np.array([tf_idf_cosin_sim]).T


def tf_idf_f1(data_set,max_features = 300):
    res = csr_matrix((int(data_set.shape[0]), max_features))
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,  max_features=max_features,
                                   stop_words='english')
    tfidf_sentences= tfidf_vectorizer.fit_transform(data_set)

    for i in range(data_set.shape[0]):
        res[i,:] = tfidf_sentences[2*i,:].multiply(tfidf_sentences[2*i+1,:])
    return res


def n_grams(data_set,stpwds):
    # Select basic features
    stemmer = PorterStemmer()
    lens1 = []
    lens2 = []
    common_unigrams_lens = []
    common_unigrams_ratios = []
    common_bigrams_lens = []
    common_bigrams_ratios = []
    common_trigrams_lens = []
    dif_len = []
    common_trigrams_ratios = []
    for i in range(int(data_set.shape[0])):
        if i%10000 == 0:
            print(i)
        source_sentence = data_set[i, 0].lower().split(" ")
        source_sentence = [token for token in source_sentence if token not in stpwds]
        unigrams_que1 = [stemmer.stem(token) for token in source_sentence]

        target_sentence = data_set[i, 1].lower().split(" ")
        target_sentence = [token for token in target_sentence if token not in stpwds]
        unigrams_que2 = [stemmer.stem(token) for token in target_sentence]

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

        lens1.append(len(source_sentence))
        lens2.append(len(target_sentence))
        dif_len.append(abs(len(source_sentence) - len(source_sentence)))

    features = np.array([common_unigrams_lens,
                         common_unigrams_ratios,
                         common_bigrams_lens,
                         common_bigrams_ratios,
                         common_trigrams_lens,
                         common_trigrams_ratios,
                         lens1,
                         lens2,
                         dif_len]).T
    return features