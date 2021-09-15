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

Como pode ser visualizado, a nuvem de palavras não corresponde a uma boa representação de cada tipo de review de filme(cada classe diferente).
Portanto podemos mudar nossa abordagem e em vez de utilizar apenas as palavras de acordo com sua frequencia, podemos utilizar as palavras de acordo com sua relevancia para cada classe específica.

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
De forma bem resumida o chi2 analisa a dependencia entre duas features, quanto maior o valor de chi2 maior a dependecia entre as features, desta forma podemos utilizar cada palavra de nosso texto como uma feature distinta e analisar o quão independente elas são de nossa variavel target, obtendo assim a correlação de cada palavra em relação a nossos valores de sentimentos.
#### As the Chi² calculation is performed only in a binary way, analyzing the hypotesis of the feature being or not related to the target variable, we need to do calculos with binary classes. So we will do 2 classification variations, one for each diferent class. Being Negative and not negative, and Being Positive and not Positive. 
Como o calculo de chi2 é realizado apenas de forma binária, analisando a hipótese da feature ser ou não relacionada àquela vairavel target, precisamos realizar os calculos com classes binárias, portanto faremos 5 variações de classificação uma para cada classe diferente.


Uma vez encontradas as melhores features com chi2, podemos utiliza-las para gerar nosso gráficos e obter nossa informação.
Entretanto como pode ser visualizado nos resultados obtidos, as palavras mais relevantes não são tão relacionadas a nossa classe desejada, isto porque palavras que tem baixa ocorrencia mas aparecem apenas quando a classe é desejada, se sobressaem em relação a palavras com alta ocorrencia mas que tem uma variação maior entre as classes analisadas.
Portanto, para contornar este problema podemos analisar a frequencia das palavras de forma a penalizar palavras com baixa ocorrencia em nossos documentos, desta forma obtemos um ajuste melhor dos resultados encontrados com o chi2. Podemos utilzar outros tipos de analise de frequencia como TFIDF, porém a fim de simplificar, utilizaremos apenas a frenquencia da feature em relação a todos os documentos.

## 7 - Final Results
