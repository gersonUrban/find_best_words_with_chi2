# Find most relevant words for each class using Chi² and TFIDF

### Suposing we want analyse which words is most importante accoding to each class. Which analysis we could do and how visualize this words?

Supondo que quero analisar quais as palavras mais importantes de acordo com cada classe dos meus dados, quais analises devo fazer e como posso visualizar essas palavras?

### We can do a word cloud from our text, but when we do that, the bigger words in our cloud(the most important words) not necessarily are the most important words to us and our classes. In general are the most repeated words.
Quando fazemos uma nuvem de palavras clássica as palavras com maior relevancia não são necessariamente as mais representativas para aquela categoria específica. Em geral são as palavras que mais aparecem que acabam tendo um tamanho maior em nossa nuvem.

### Therefore, if we want analyze witch words have a bigger relevance to each categoty, in order to get these keywords to check if there is words that must be removed or to understand how our model is working, we should understanding our data and find the most important features to each class. 
Portanto, se quisermos fazer uma análise de quais palavras tem maior importancia para cada categoria, a fim de obter essas palavras chaves, ou entender como um possivel modelo de similaridade de textos ou classificação pode funcionar, devemos entender a fundo quais os termos mais relevantes para nossa modelagem, possibilitando também melhorar o préprocessamento de texto realizado inicialmente.

### So we will discuss some points related to this problematic and later a posible solution to this case, in this way we will talk about:
 1 - DataBase used in this case
 2 - DataBase Preprocessing
 3 - Text Preprocessing
 4 - Basic Word Cloud
 5 - Chi²
 6 - TFIDF
 7 - Final Results
 
Contudo discutiremos alguns pontos para introduzir e levantar a problemática e posteriormente uma possivel solução para o caso, desta forma falaremos sobre:
1 - Base de dados
2 - pre processamento (basico)
3 - nuvem de palavras (basico)
4 - Chi2
5 - TFIDF
6 - analise dos dados
7 - nuvem de palavras final



## 1 - DataBase

#### To do our analysis we will use the **"Sentiment Analysis on Movie Reviews"**, available in https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/.
Para realizar esta análise vamos utilizar a base "Sentiment Analysis on Movie Reviews" disponivel em https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/.
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
 
Esta base contém 3 colunas principais, sendo a SentenceId, que indica a qual review aquela frase pertence, Phrase contendo cada frase de cada review( cada review pode ter mais de uma Phrase) e Sentiment que indica o sentimento contido na frase sendo:

0 - negative; 1 - somewhat negative; 2 - neutral; 3 - somewhat positive; 4 - positive




## 2 - DataBase Preprocessing
#### We separate our preprocessing in two steps, in very first we need change our Data structure to our applicationand analysis.
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
# Removing 'neutral' Analysis to improve separability in Negative and Positive Class
df = df[df[class_col] != 2]
df = df[df[class_col] != 1]
df = df[df[class_col] != 3]
```

#### After that we group all Phrases by SentenceId, joining all Phrases from a reviewer. We do that because there are a lot of short Phrases, with less than 100 characters.
```python
# Grouping by SentenceId and transforming data as a list
df2 = df.groupby('SentenceId').agg(lambda x: x.tolist())
# Getting review sentiment mean 
df2['sent_mean'] = df2[class_col].apply(lambda x: np.array(x).mean())
```

#### Posterioly we get the mean of each review and create a new classification column, with 1 if sentmente mean was > 2 and 0 otherwise.
```python
# Transform sentiment values to only positive and negative values
# Setting all instaces as negative (0)
df2[AUX_CLASS_COL] = 0 
# Setting only most than 2 class as 1 (positive)
df2.loc[df2['sent_mean'] > 2,AUX_CLASS_COL] = 1
```
#### Finaly we join all Phrases, reset index and change our class name variable.
```python
# Join text from the text list created in groupby agg
df2[TEXT_COL] = df2[TEXT_COL].apply(lambda x: ' '.join(x))
# Reset index
df2.reset_index(drop=True, inplace=True)
# Redefining df DataFrame
df = df2.copy()
# Changing class_col name
class_col = AUX_CLASS_COL
```

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

# Colocar imagem aqui

#### We can view that the words do not represent very well the differences between Positive and Negative Sentiments. It occurs because we get only frequency of words to each Sentiment and plot it, giving importance only to the frequency of words in each class.
#### To improve it we can find the most relevante features according to each class, analyzing correlation between our features(words) and classes.

## 5 - Chi²
## 6 - TFIDF
## 7 - Final Results
