# Find most relevant words for each class using Chi² and TFIDF

#### Suposing we want analyse which words is most importante accoding to each class. Which analysis we could do and how visualize this words?

#### We can do a word cloud from our text, but when we do that, the bigger words in our cloud(the most important words) not necessarily are the most important words to us and our classes. In general are the most repeated words.

#### Therefore, if we want analyze witch words have a bigger relevance to each categoty, in order to get these keywords to check if there is words that must be removed or to understand how our model is working, we should understanding our data and find the most important features to each class. 

#### So we will discuss some points related to this problematic and later a posible solution to this case, in this way we will talk about:
 1 - DataBase used in this case
 2 - DataBase Preprocessing
 3 - Text Preprocessing
 4 - Basic Word Cloud
 5 - TFIDF
 6 - Chi²
 7 - Final Results


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
```

#### Finally we divide class by for, in order to obtain only 0 and 1 values
```python
# Doing Sentment Became only 0 and 1 values
df['Sentiment'] = df['Sentiment']//4
```
#### We can make do another data transformations, group all Phrases from the same review, use a sentment mean to analyse, but for now, that's enough.

## 3 - Text Preprocessing

#### In order to focus on data analisys, our text preprocessing is very simple. First we merely remove stopwords, ponctuations, numbers and change encoding type to NFDK in order to remove possible special ponctuation characteres(such as 'ç').

```python
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
```

## 4 - Basic Word Cloud
#### Doind the basic text preproccess we can begin our text analysis. In order to compare our results and demonstrate a simpler data analysis, let's plot a word cloud our current text data.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Show one word cloud to each Sentiment
d = {}
plt.figure(1,figsize=(20, 20))
n_classes = range(0, len(df[CLASS_COL].value_counts().index))
for i in n_classes:
    print("Doing {} graph".format(i))
    d[i] = df[df[CLASS_COL]==i]
    words = ' '.join(d[i][TEXT_COL])
    split_word = " ".join([word for word in words.split()])
    wordcloud = WordCloud(background_color='black',width=3000,height=2500).generate(split_word)
    plt.subplot(320+(i+1))
    plt.imshow(wordcloud)
    plt.title('Sentiment {}'.format(i))
    plt.axis('off')
plt.show()
```

![alt text](https://github.com/gersonUrban/find_best_words_with_chi2/blob/master/images/basicWordCloud.png)

#### We can view that the words do not represent very well the differences between Positive and Negative Sentiments. It occurs because we get only frequency of words to each Sentiment and plot it, giving importance only to the frequency of words in each class.
#### To improve it we can find the most relevante features according to each class, analyzing correlation between our features(words) and classes.

## 5 - TFIDF
#### To make possible the Chi² analysis we need transform our text data in a vector of features where each feature is a ngram. When we have a 1gram(n=1), than each feature is a different word.
#### There are some ways to vectorize our text, we can do a simple bag of words, were we only count how many times each word/feature appear in a document. In this way, we can found statistics of each word in each document and consequently in each class. A better approach is use TFIDF [REF]. 
#### TFIDF calculate **Term Frequency** to check 'how many times' word appears, and **Inverse Document Frequency** in order to penalize words that appear in too many documents. In this way is possible to find the importance of each term.

#### To summarize we will only apply this technique, if you want to delve in this topic you can read the references.

#### Because TFIDF calculates the most important features to a corpus, we will calculate it 2 times, one for each Sentiment variation, in order to get the statistical difference words in those Classes.

#### Than we create a DataFrame only with desired Sentiment Reviews, create a TFIDF weights model and transform all data with these weights. Finally getting the vectors, containg weights and the feature_names containing all ngrams(words) of our vector.
```python
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Getting text only from wanted class
    df_aux = df[df[CLASS_COL] == CLASS_VALUE]
    # Getting tfidf model
    # Create tfidf vectorizer object
    vectorizer = TfidfVectorizer(ngram_range=ngram)
    # Fitting tfidf weights to text
    vec = vectorizer.fit(df_aux[TEXT_COL])
    # Transforming all data using tfdif model
    vectors = vec.transform(df[TEXT_COL])
    # Getting feature names (words)
    feature_names = vec.get_feature_names()
```
#### Now we have each word weight to each sentence in our corpus, and than we can find the Chi², using this values and comparing with our targets.

## 6 - Chi²
#### Resuming Chi² analyze the dependency between two features, the higher value, the higher dependency[REF]. In this way we can use each word of our corpus as a distinct feature and verify how diferent it are from our target. thus obtaining the correlation of each word in relation to our Sentiment values.


#### As the Chi² calculation is performed only in a binary way, analyzing the hypotesis of the feature being or not related to the target variable, we need to do calculos with binary classes. So we will do 2 classification variations, one for each diferent class. Being Negative and not negative, and Being Positive and not Positive. 
Como o calculo de chi2 é realizado apenas de forma binária, analisando a hipótese da feature ser ou não relacionada àquela vairavel target, precisamos realizar os calculos com classes binárias, portanto faremos 5 variações de classificação uma para cada classe diferente.


Uma vez encontradas as melhores features com chi2, podemos utiliza-las para gerar nosso gráficos e obter nossa informação.
Entretanto como pode ser visualizado nos resultados obtidos, as palavras mais relevantes não são tão relacionadas a nossa classe desejada, isto porque palavras que tem baixa ocorrencia mas aparecem apenas quando a classe é desejada, se sobressaem em relação a palavras com alta ocorrencia mas que tem uma variação maior entre as classes analisadas.
Portanto, para contornar este problema podemos analisar a frequencia das palavras de forma a penalizar palavras com baixa ocorrencia em nossos documentos, desta forma obtemos um ajuste melhor dos resultados encontrados com o chi2. Podemos utilzar outros tipos de analise de frequencia como TFIDF, porém a fim de simplificar, utilizaremos apenas a frenquencia da feature em relação a todos os documentos.

## 7 - Final Results
