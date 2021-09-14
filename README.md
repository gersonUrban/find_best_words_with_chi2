# Find most relevant words for each class using Chi² and TFIDF

### Suposing we want analyse which words is most importante accoding to each class. Which analysis we could do and how visualize this words?

Supondo que quero analisar quais as palavras mais importantes de acordo com cada classe dos meus dados, quais analises devo fazer e como posso visualizar essas palavras?

### We can do a word cloud from our text, but when we do that, the bigger words in our cloud(the most important words) not necessarily are the most important words to us and our classes. In general are the most repeated words.
Quando fazemos uma nuvem de palavras clássica as palavras com maior relevancia não são necessariamente as mais representativas para aquela categoria específica. Em geral são as palavras que mais aparecem que acabam tendo um tamanho maior em nossa nuvem.

### Therefore, if we want analyze witch words have a bigger relevance to each categoty, in order to get these keywords to check if there is words that must be removed or to understand how our model is working, we should understanding our data and find the most important features to each class. 
Portanto, se quisermos fazer uma análise de quais palavras tem maior importancia para cada categoria, a fim de obter essas palavras chaves, ou entender como um possivel modelo de similaridade de textos ou classificação pode funcionar, devemos entender a fundo quais os termos mais relevantes para nossa modelagem, possibilitando também melhorar o préprocessamento de texto realizado inicialmente.

### So we will discuss some points related to this problematic and later a posible solution to this case, in this way we will talk about:
 1 - DataBase used in this case
 2 - PreProccess DataBase
 3 - Basic Word Cloud
 4 - Chi²
 5 - TFIDF
 6 - Final Results
 
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




