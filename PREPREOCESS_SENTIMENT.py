import sys, csv
from textblob import TextBlob 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report   

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from threading import Thread
from functools import wraps



from numpy import log, dot, e, where
from numpy.random import rand




class LogisticRegression():
    
    def sigmoid(self, z): return 1 / (1 + e**(-z))
    
    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)
    
   
    
    def fit(self, X, y, epochs=25, lr=0.05):        
        loss = []
        weights = rand(X.shape[1])
        N = len(X)
                 
        for _ in range(epochs):        
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            weights -= lr * dot(X.T,  y_hat - y) / N            
            # Saving Progress
            loss.append(self.cost_function(X, y, weights)) 
            
        self.weights = weights
        self.loss = loss
    
    
    def predict(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        x = []
        print (z)
        for i in z:
            if i > 0.0:
                x.append(1)
                print("{:.2f}".format(i),': 1')
            else:
                x.append(0)
                print("{:.2f}".format(i),': 0')
                
        return x

class logistic_regression():
    
    
    
    
    def Preprocess(self):
        try:
            pos = 1
            
            neg = 1
            #tokened_text = str('')
            #stemmed_text = str('')
            polarity = 0
            col_list = ["ID", "TWEETS"]   
        
            #-----READS THE DATASET OF THE TWEETS-----------
            df = pd.read_csv("train.csv", usecols=col_list)
        
            reader = df["TWEETS"]
            with open('test.csv', 'w',encoding='utf8', newline='') as filed: 
                writer = csv.writer(filed)
                writer.writerow(["ID", "TWEETS","TOKENIZED","STOP WORDS", "STEMMED", "POLARITY", "SUBJECTIVITY", "SENTIMENT"])
                counter = 0
                tweetted = '' 
                for row in reader:
                    tweetted = row
                    #-----TOKENIZING TWEETS FROM THE TWEETS ROW-----------
                    tokenized_text=sent_tokenize(row)
                    #print(tokenized_text)
                    for k in tokenized_text:
                        tokenized_sentence=sent_tokenize(k)
                        


                    filtered_sent=[]
                    for tokword in tokenized_sentence:
                        stop_words=set(stopwords.words('english'))
                        #print(tokword)
                        new_sentence = ' '.join([word for word in tokword.split() if word not in stop_words])
                        filtered_sent.append(new_sentence)
                
                        

                    
                #--------STEMMED TWEETS-----------------------
                    stemmed_tweet=[]
                    ps = PorterStemmer()
                    
                    for getsteam in filtered_sent:
                        getstoped = getsteam
                        stemmed_tweet.append(ps.stem(getsteam))
                        #steam_sentence=ps.stem(get_stop)
                    

                    

                    
                    
                    
                #--------------------GET SENTIMENT VALUE----------- 
                    for gather_steam in stemmed_tweet:
                        #steam_tweets = gather_steam
                        analysis = TextBlob(gather_steam)
                        counter+=1
                        sentied = 0
                        if analysis.sentiment.polarity > 0:
                            sentied = 1
                            #pos+=1
                        #elif analysis.sentiment.polarity == 0:
                            #sentied = 0
                            #pos+=1
                        else: 
                            neg+=1
                            sentied = -1
                        
                        
                        #--------WRITE TO CSV FILE FOR SPLIT DATASETS-----------------------
                        writer.writerow([counter, tweetted, tokenized_sentence,getstoped, gather_steam, analysis.sentiment.polarity, analysis.sentiment.subjectivity, sentied])        
                    polarity += analysis.sentiment.polarity # adding up polarities to find the average later
            polarity = polarity / counter 
        except:
            pass
                

    
                
object = logistic_regression()
object.Preprocess()


