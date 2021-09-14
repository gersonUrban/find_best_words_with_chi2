# Getting top chi2 to all values

import numpy as np
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

def get_top_chi2(X,y,feature_names, n):
    '''
    Function to receive X with columns and y as target and return the n best features according with chi2
    X: pandas Dataframe with
    y: pandas series with 1 in index that be analysed and 0 for another ones
    return: list of n tuples with features name and chi2 value, ordered by chi2 value
    '''
    # Getting feature names
    #feature_names = X.columns
    # Getting chi2 score
    chi2score = chi2(X, y)[0]
    # Join feature names and chi2 results
    wscores = zip(feature_names, chi2score, range(0,len(feature_names)))
    # Sorting results
    wchi2 = sorted(wscores, key=lambda x:x[1])
    # Select only the n best values
    topchi2 = wchi2[::-1][:n]
    
    return topchi2

def get_doc_freq(X, cols):
    '''
    This function get all features frequency from a setted cols
    and return as a list with freq values according to the features
    X: dataFrame to get frequency of each feature(word)
    cols: list with cols names to be analysed
    '''
    # Getting n sample size
    n = X.shape[0]
    # Getting Frequencies for each feature (document frequency)
    doc_freq = X[:,[cols]].sum(axis=0)/n

    return doc_freq

def get_chi2_x_doc_freq(topchi2, doc_freq,n=40):
    '''
    This Function multiply top chi2 values with document frequencies from this features, 
    to aim get the most relevant features from a type o text
    topchi2: list with tuples containing name features and its chi2 values 
    doc_freq: document frequency from a feature(the features must be ordered as topchi2 features)
    n: max features to be returned
    return: list of tuples containing feature names and its importance values
    '''
    
    # Getting chi values
    chi2_values = np.array(topchi2)[:,1]
    # Transforming to float(because was as string)
    chi2_values = chi2_values.astype(float)
    # Multiplying frequencies and chi2 values
    result = np.array(doc_freq) *np.array(list(chi2_values))
    # Join result with top chi feature names
    result_chi2_freq = zip(np.array(topchi2)[:,0], result)
    # Sorting results by values
    result_chi2_freq = sorted(result_chi2_freq, key=lambda x:x[1])
    # Getting the n most important features
    result_chi2_freq = result_chi2_freq[::-1][:n]
    
    return result_chi2_freq

def plot_main_features(result_chi2_freq):
    '''
    Function to create a most important features barplot
    result_chi2_freq: tuple returned by get_main_words_pipeline
    '''
    y_pos = range(0,len(result_chi2_freq))
    plt.figure(figsize=(8,7))
    chi_values_final = [round(float(i),2) for i in list(np.array(result_chi2_freq)[:,1])]

    plt.bar(y_pos, chi_values_final, align='center', alpha=0.5)
    plt.xticks(y_pos, np.array(result_chi2_freq)[:,0],rotation='vertical')
    plt.plot(y_pos, chi_values_final, '-o', markersize=5,alpha=0.8)
    plt.ylabel('Chi2 * Freq')
    plt.xlabel('Top Tokens')
    plt.title('Top Tokens')
    plt.show()

    
def get_new_string(result_chi2_freq):
    '''
    Function to create a string with main words used to plot a word cloud graph
    result_chi2_freq: list of tuple containing feature name (word) and importance
    return: String containing words repeatedly according to word importance
    '''
    # Getting a multiplier coeficient to apply in words weights
    min_value = 1/result_chi2_freq[len(result_chi2_freq)-1][1]
    # Getting values
    res_chi = np.array(result_chi2_freq)[:,1]
    # Multiplying with coefficient
    res_chi = np.ceil(res_chi.astype(float) * min_value)
    # Change to int
    res_chi = res_chi.astype(int)
    # Creating a string reapeting each word n times, according to it importance
    s = ""
    for i, word in enumerate(np.array(result_chi2_freq)[:,0]):
        for j in range(0,res_chi[i]):
            s+= word + ' '

    return s
