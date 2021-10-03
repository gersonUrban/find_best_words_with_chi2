### Creating Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def get_df_features(df, text_col, class_col, class_value, ngram=(1,1)):
    '''
    This Function receive pandas DataFrame with texts and its classification, the both columns and a ngram tuple. Returning a pandas DataFrame with all words ant it tfidf values
    df: pandas DataFrame
    text_col: string with the name of column with texts
    CLASS_COL: string with the name of column with classifications
    CLASS_VALUE: the class value to trained the model
    ngram: tuple containing tuple with lenth of ngram
    return: numpy array with all tfidf values, array with ngrams (words)
    '''
    # Getting text only from wanted class
    df_aux = df[df[class_col] == class_value]
    # Create tfidf vectorizer object
    vectorizer = TfidfVectorizer(ngram_range=ngram)
    # Fitting tfidf weights to text
    vectorizer = vectorizer.fit(df_aux[text_col])
    # Transforming all data using tfdif model
    vectors = vectorizer.transform(df[text_col])
    # Getting feature names (words)
    feature_names = vectorizer.get_feature_names()
    # Getting idf wright from each ngram
    # weights = vectorizer.idf_
    
    return vectors, feature_names
   

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
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
