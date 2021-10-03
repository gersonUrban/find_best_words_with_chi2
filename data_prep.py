from nltk.corpus import stopwords

def basic_preprocess_text(text_series, language='english'):
    '''
    Function to make a basic data prep in text, according to sentiment analysis dataset
    text_series: pandas series with texts to be treated
    language: string indicating stopwords language to be used
    return: Pandas Series with treated text
    '''
    # Passing text to lowercase
    text_series = text_series.str.lower()
    # Defining stopwords to be removed
    pat = r'\b(?:{})\b'.format('|'.join(stopwords.words(language)))
    # removing stopwords
    text_series = text_series.str.replace(pat,'')
    # Removing ponctuation from text
    text_series = text_series.str.replace(r'\s+',' ')
    # Normalizing to NFDK, if have words with special characteres
    text_series = text_series.str.normalize('NFKD')
    text_series = text_series.str.replace('[^\w\s]','')
    # Removing numeric substrings from text
    text_series = text_series.str.replace(' \d+','')
    
    return text_series
