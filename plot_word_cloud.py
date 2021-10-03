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
    

def plot_finish_word_cloud(s):
    '''
    Function to plot simple word_cloud
    s: string with words to be ploted
    '''
    wordcloud = WordCloud(background_color='black',width=3000,height=2500,collocations=False).generate(s)
    plt.imshow(wordcloud)
    plt.show()
    
