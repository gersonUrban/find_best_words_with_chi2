# Find most relevant words for each class using Chi² and TFIDF

#### Suposing we want analyse which words is most importante accoding to each class. Which analysis we could do and how visualize this words?

#### We can do a word cloud from our text, but when we do that, the bigger words in our cloud(the most important words) not necessarily are the most important words to us and our classes. In general are the most repeated words.

#### Therefore, if we want analyze witch words have a bigger relevance to each categoty, in order to get these keywords to check if there is words that must be removed or to understand how our model is working, we should understanding our data and find the most important features to each class. 

#### So we will discuss some points related to this problematic and later a posible solution to this case, in this way we will talk about:
1. DataBase used in this case
2. DataBase Preprocessing
3. Text Preprocessing
4. Basic Word Cloud
5. TF-IDF
6. Chi²
7. Chi² x TF-IDF
8. Final Results


## 1 - DataBase

#### To do our analysis we will use the **"Sentiment Analysis on Movie Reviews"**, available in https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/.

#### This DataBase contains 3 main columns.
 1 - SentenceId: Indicating what review does that Phrase belong. (A review can have more than one Phrase);
 2 - Phrase: containing each frase of each review;
 3 - Sentiment: Indicating the sentiment contained in Phrase. Where we have 5 levels:
| Value | Sentment |
| ------ | ------ |
| 0 | Negative |
| 1 | Somewhat Negative |
| 2 | Neutral |
| 3 | Somewhat Positive |
| 4 | Positive |

## 2 - DataBase Preprocessing
#### We separate our preprocessing in two steps, in very first we need change our Data structure to our application and analysis.
#### After read data, we decide remove all intermediate Sentiments in order to improve the separability between **Negative** and **Positive** Sentiments, and in facilitate viewing of the most signficant words.
#### Finally we divide class by 4, in order to obtain only 0 and 1 values.

```python
import pandas as pd
import numpy as np

########## Initing Constants and Variables ##########
TEXT_COL = 'Phrase'
AUX_CLASS_COL = 'new_sent'
class_col = 'Sentiment'


########## Reading Data ##########
df = pd.read_csv('sentiment-analysis-on-movie-reviews/train.tsv',sep='\t')

# Removing 'neutral' Analysis to improve separability in Negative and Positive Classes
df = df[df[class_col] != 2]
df = df[df[class_col] != 1]
df = df[df[class_col] != 3]

# Doing Sentment Became only 0 and 1 values
df['Sentiment'] = df['Sentiment']//4
```

#### We can make do another data transformations, group all Phrases from the same review, use a sentment mean to analyse, but for now, that's enough.

## 3 - Text Preprocessing

#### In order to focus on data analisys, our text preprocessing is very simple. First we merely transform text to lower case, remove stopwords, ponctuations, numbers and change encoding type to NFDK in order to remove possible special ponctuation characteres(such as 'ç') in our text.

```python
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
```

#### Finally we make text process, remove texts with less than 2 characteres and reset our data index.

```python
from data_prep import basic_preprocess_text

# Basic Text Processing
df[TEXT_COL] = basic_preprocess_text(df[TEXT_COL], language='english')

# Removing Duplicates
df = df.drop_duplicates(subset=[TEXT_COL])
# Removing texts with less than 2 characteres
df = df[df['Phrase'].str.len()>2].reset_index(drop=True)
# Reseting Index
df.reset_index(drop=True,inplace=True)
```

#### In this step we can add other text treatments, such as lemmatization, add other stop words related to our domain, like **"films"** and **"movies"**, we could also treat better negative sentences as "not funny" but as before we cherish the basics.

## 4 - Basic Word Cloud
#### Done the basic text preproccess we can begin our text analysis. In order to compare our results and demonstrate a simpler data analysis, let's plot a word cloud from each class of our current text data.


```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_word_cloud(df, text_col, class_col):
    '''
    plot a basic word cloud to each class of DataFrame
    df: Pandas DataFrame with text column and text class column
    text_col: string with text column name
    class_col: string with class column name
    '''
    
    # Initializing Figure to plot
    plt.figure(1,figsize=(20, 20))
    # Getting a sorted vector from class values
    n_classes = sorted(df[class_col].unique())
    # create dict to get all values
    for i, c in enumerate(n_classes):
        print("Doing {} graph".format(c))
        # Getting all words from class c
        words = ' '.join(df[df[class_col]==c][text_col].values)
        # Generating WordCloud
        wordcloud = WordCloud(background_color='black',width=3000,height=2500).generate(words)
        # Plotting WordCloud
        plt.subplot(320+(i+1))
        plt.imshow(wordcloud)
        plt.title('Sentiment {}'.format(c))
        plt.axis('off')
    plt.show()
```
```python
from plot_word_cloud import plot_word_cloud
plot_word_cloud(df, TEXT_COL, class_col)
```


![alt text](https://github.com/gersonUrban/find_best_words_with_chi2/blob/master/images/basicWordCloud.png)

#### We can view that the words do not represent very well the differences between Positive and Negative Sentiments. It occurs because we get only frequency of words to each Sentiment and plot it, giving importance only to the frequency of words in each class.
#### To improve it we can find the most relevante features according to each class, analyzing correlation between our features(words) and classes.

## 5 - TF-IDF
#### To make possible the Chi² analysis we need transform our text data in a vector of features where each feature is a ngram. When we have a 1gram(n=1), than each feature is a different word.
#### There are some ways to vectorize our text, we can do a simple bag of words, were we only count how many times each word/feature appear in a document. In this way, we can found statistics of each word in each document and consequently in each class. A better approach is use TFIDF [REF]. 
#### TFIDF calculate **Term Frequency** to check how many times word appears, and **Inverse Document Frequency** in order to penalize words that appear in too many documents. In this way is possible to find the importance of each term.

#### To summarize we will only apply this technique, if you want to delve in this topic you can read the references.

#### Because TFIDF calculates the most important features to a corpus, we will calculate it 2 times, one for each Sentiment variation, in order to get the statistical difference words in those Classes.

#### Than we create a DataFrame only with desired Sentiment Reviews, create a TFIDF weights model and transform all data with these weights. Finally getting the vectors, containg weights and the feature_names containing all ngrams(words) of our vector.

#### The script to get this results can be seen bellow.
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def get_df_features(df, text_col, class_col, class_value, ngram=(1,1)):
    '''
    This Function receive pandas DataFrame with texts and its classification, 
    the both columns and a ngram tuple. 
    Returning a pandas DataFrame with all words ant it tfidf values
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
    
    return vectors, feature_names
   
```
#### Now we have each word weight to each sentence in our corpus, and than we can find the Chi², using this values and comparing with our targets.

## 6 - Chi²
#### Resuming Chi² analyze the dependency between two features, the higher value, the higher dependency[REF]. In this way we can use each word of our corpus as a distinct feature and verify how diferent it are from our target. thus obtaining the correlation of each word in relation to our Sentiment values.

#### As the Chi² calculation is performed only in a binary way, analyzing the hypotesis of the feature being or not related to the target variable, we need to do calculos with binary classes. So we will do 2 classification variations, one for each different class. Being Negative and not negative, and Being Positive and not Positive.

#### Than we will use the code above to obtain Chi² values.

```python
from sklearn.feature_selection import chi2

def get_top_chi2(X,y,feature_names, n):
    '''
    Function to receive X with columns and y as target 
    and return the n best features according with chi2
    X: pandas Dataframe with
    y: pandas series with 1 in index that be analysed and 0 for another ones
    return: list of n tuples with features name and chi2 value, ordered by chi2 value
    '''
    # Getting chi2 score
    chi2score = chi2(X, y)[0]
    # Join feature names and chi2 results
    wscores = zip(feature_names, chi2score, range(0,len(feature_names)))
    # Sorting results
    wchi2 = sorted(wscores, key=lambda x:x[1])
    # Select only the n best values
    topchi2 = wchi2[::-1][:n]
    
    return topchi2
```
    
#### Now we can do this to each Class and plot our main Chi2 features for each one. In this way we get these two graphs below.

![alt text](https://github.com/gersonUrban/find_best_words_with_chi2/blob/master/images/top_negative_x_top_positive_tokens_chi2.png)

![alt text](https://github.com/gersonUrban/find_best_words_with_chi2/blob/master/images/top_negative_tokens_from_chi2.png)

![alt text](https://github.com/gersonUrban/find_best_words_with_chi2/blob/master/images/top_positive_tokens_from_chi2.png)


Como o calculo de chi2 é realizado apenas de forma binária, analisando a hipótese da feature ser ou não relacionada àquela vairavel target, precisamos realizar os calculos com classes binárias, portanto faremos 5 variações de classificação uma para cada classe diferente.


Uma vez encontradas as melhores features com chi2, podemos utiliza-las para gerar nosso gráficos e obter nossa informação.
Entretanto como pode ser visualizado nos resultados obtidos, as palavras mais relevantes não são tão relacionadas a nossa classe desejada, isto porque palavras que tem baixa ocorrencia mas aparecem apenas quando a classe é desejada, se sobressaem em relação a palavras com alta ocorrencia mas que tem uma variação maior entre as classes analisadas.
Portanto, para contornar este problema podemos analisar a frequencia das palavras de forma a penalizar palavras com baixa ocorrencia em nossos documentos, desta forma obtemos um ajuste melhor dos resultados encontrados com o chi2. Podemos utilzar outros tipos de analise de frequencia como TFIDF, porém a fim de simplificar, utilizaremos apenas a frenquencia da feature em relação a todos os documentos.

## 7 Chi² x TF-IDF

## 8 - Final Results
