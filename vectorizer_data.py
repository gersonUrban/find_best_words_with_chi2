### Creating Vectorizer

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_vectorizer_tfidf(text_series,ngram=(1,1)):
    '''
    This Function receive pandas series with texts and a ngram tuple. Return a trained tfidf model
    text_series: pandas series with strigs
    ngram: tuple containing range of ngram to be used
    return: Vectorizer tfidf model
    '''
    # Create tfidf vectorizer object
    vectorizer = TfidfVectorizer(ngram_range=ngram)
    # Fitting tfidf weights to text
    vectorizer = vectorizer.fit(text_series)
    return vectorizer

def get_df_features(df, TEXT_COL, CLASS_COL,CLASS_VALUE,ngram=(1,1)):
    '''
    This Function receive pandas DataFrame  with texts and its classification, the both columns and a ngram tuple. Returning a pandas DataFrame with all words ant it tfidf values
    df: pandas DataFrame
    TEXT_COL: string with the name of column with texts
    CLASS_COL: string with the name of column with classifications
    CLASS_VALUE: the class value to trained the model
    ngram: tuple containing lenth of ngram
    return: numpy array with all tfidf values and numpy array with all ngrams (words)
    '''
    # Getting text only from wanted class
    df_aux = df[df[CLASS_COL] == CLASS_VALUE]
    # Getting tfidf model
    vec = get_vectorizer_tfidf(df_aux[TEXT_COL],ngram=ngram)
    # Transforming all data using tfdif model
    vectors = vec.transform(df[TEXT_COL])
    # Getting feature names (words)
    feature_names = vec.get_feature_names()
    # Getting idf wright from each ngram
    idf = vec.idf_
    weights = zip(feature_names, idf)
    
    return vectors, feature_names, weights
   

def get_vectorizer_df(text_col_series):
    '''
    This function receivs a pandas series with texts, transform that in a count vector of words,
    Update ir in a boolean vector of words and pass as dataFrame in function return
    text_col_series: pandas series with strigs
    return: DataFrame with all words from text_col_series as a collumn, informing with there is this word in the prahse id
    '''
    # Init Vectorizer
    vectorizer = CountVectorizer()
    # Transforming Text
    X = vectorizer.fit_transform(text_col_series)
    # Getting all words
    X = X.toarray()
    # Setting words to 0 or 1, instead a count of words
    X = np.where(X > 0.5, 1, 0)
    # Putting in DataFrame
    X = pd.DataFrame(X)
    # Setting Collumn names
    X.columns = vectorizer.get_feature_names()
    
    return X
