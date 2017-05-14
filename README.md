# QuoraKaggle
Scripts And Notebooks for Quora Challenge on Kaggle

# Scripts

  ## Features.py
      Contains functions relative to feature egineering for building a training set from the sentences

    ### doc2vecs_features 
      (sentences_train: np array shape (N,2) containing sentences, stopwords ,model_name : str if model has already been              train, since it s useless to train it more than once, model can ben then loaded and we can compute similarities)
      
      Compute features based on the similiraty by Doc2Vec (doc to vectors - adapted from word2vec).
      
    ### freq_hash
        Give hash to id questions and also compute frequency of questions
        
    ### tf_idf_cosin 
        Compute cosine similarities on tf_idf representation of documents
    
    ### lsi_indexing
        Compute lsi indexing similarity
    
    ### n_grams_features
        Compute n grams (uni, duo and tri)-grams number and ratio
  
  ## Predictions.py
      Scripts to run from scratch to train predict and submission file
      sys.argv[1] = Number of desired training exemples
        
