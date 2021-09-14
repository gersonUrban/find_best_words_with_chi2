from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Show one word cloud to each Sentiment
def plot_word_cloud(df, text_col, class_col):
    '''
    plot a basic word cloud to each class of data frame
    df: Pandas DataFrame with text column and text class column
    text_col: string with text column name
    class_col: string with class column name
    '''
    d = {}
    plt.figure(1,figsize=(20, 20))
    n_classes = range(0, len(df[class_col].value_counts().index))
    for i in n_classes:
        print("Doing {} graph".format(i))
        d[i] = df[df[class_col]==i]
        words = ' '.join(d[i][text_col])
        split_word = " ".join([word for word in words.split()])
        wordcloud = WordCloud(background_color='black',width=3000,height=2500).generate(split_word)
        plt.subplot(320+(i+1))
        plt.imshow(wordcloud)
        plt.title('Sentiment {}'.format(i))
        plt.axis('off')
    plt.show()
    

def plot_finish_word_cloud(s):
    '''
    Function to plot simple word_cloud
    s: string with words to be ploted
    '''
    wordcloud = WordCloud(background_color='black',width=3000,height=2500,collocations=False).generate(s)
    plt.imshow(wordcloud)
    plt.show()
    
