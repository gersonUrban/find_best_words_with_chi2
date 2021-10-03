
import numpy as np
from vectorizer_data import get_df_features
from chi2_utils import get_top_chi2, get_doc_freq, get_chi2_x_doc_freq

def get_main_words_pipeline(df, 
                            text_col, 
                            class_col, 
                            sentiment, 
                            ngram=(1,1), 
                            max_words=40
                           ):
    '''
    Pipeline that execute functions to get most representative words, using chi2 and tfidf
    df: Pandas DataFrame with text column and text class column
    text_col: string with text column name
    class_col: string with class column name
    sentiment: class name to be analysed
    ngram: range of ngrams to be used in vectorize text
    max_words: most important features to be returned
    return: tuple with most important features to class and its importance value
    '''
    target_col='target',
    # Training tfidf model with ngrams and selexted df, and getting transformed features with tfidf
    X, feat_names = get_df_features(df, text_col, class_col,sentiment,ngram=ngram)

    # Creating a col with 0 in target
    df[target_col] = 0
    
    # Setting as 1 texts with sentiment that will be analysed
    y = df[target_col].copy()
    y[df[class_col]== sentiment] = 1

    # Executing and get top chi2 result
    topchi2 = get_top_chi2(X,y,feat_names,1000)
    
    # Getting feature index (col index)
    cols = np.array(topchi2)[:,2]

    ## Getting Sparse matrix to numpy matriz to select rows
    dense = X.todense()
    
    ## Selecting only desired rows(target rows)
    X2 = dense[y[y==1].index,:]
    # Removing dense to empty memory
    del dense

    # Getting doc frequencys
    doc_freq = get_doc_freq(X2, cols.astype(int))
    
    # Getting top chi2 features
    result_chi2_freq = get_chi2_x_doc_freq(topchi2, list(np.array(doc_freq)[0]),n=max_words)
    
    return topchi2, result_chi2_freq

