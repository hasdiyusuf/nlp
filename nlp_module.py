
import numpy as np
import json
import re

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional

#%% EDA

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def data_cleaning(self, data):
        '''
        remove the string punctuation in the texts

        Parameters
        ----------
        data : series
            DESCRIPTION.

        Returns
        -------
        data : series
            return list of text that has been lowercase and split

        '''
        for index, text in enumerate(data): 
            data[index] = re.sub('[^a-zA-z]', ' ', text).lower().split()

        return data
    
    def text_token(self, token_save_path,data, num_words = 10000, oov_token = '<oov>'):
        '''
        convert texts into numeric values

        Parameters
        ----------
        token_save_path : json file
            saved the tokenizer data into the path
        data : series
            DESCRIPTION.
        num_words : int, optional
            DESCRIPTION. The default is 10000.
        oov_token : str, optional
            DESCRIPTION. The default is '<oov>'.

        Returns
        -------
        data : list
            return list of interger

        '''
        token = Tokenizer(num_words= num_words, oov_token= oov_token)
        token.fit_on_texts(data)
        data = token.texts_to_sequences(data)
        
        # to save the tokenizer for deployment purpose
        token_json = token.to_json()
        
        
        with open(token_save_path,'w') as json_file:
            json.dump(token_json, json_file)
        
        
        
        return data
    
    
    def text_pad_sequences(self,data):
        '''
        adding zero to equal the number of length

        Parameters
        ----------
        data : 
            DESCRIPTION.

        Returns
        -------
        array
            DESCRIPTION

        '''
        return pad_sequences(data, maxlen = 200,
                             padding = 'post',
                             truncating = 'post')
    
    def one_hot_encoder(self, label):
        '''
        encode target label into numerical values

        Parameters
        ----------
        label : str
            DESCRIPTION.

        Returns
        -------
        data : float64
            DESCRIPTION.

        '''
        ohe = OneHotEncoder(sparse = False)
        data = ohe.fit_transform(np.expand_dims(label,axis=-1))
        return data 
    
class ModelCreation():
    def lstm_layer(self, model_path, num_words,nb_categories, embedding_output = 64, nodes = 32, dropout = (0.2)):
        '''
        Sequential model creation

        Parameters
        ----------
        model_path : TYPE
            path where the model will saved
        num_words : int
            number of words or input
        nb_categories : int
            number of outputs
        embedding_output : TYPE, optional
            DESCRIPTION. The default is 64.
        nodes : TYPE, optional
            DESCRIPTION. The default is 32.
        dropout : TYPE, optional
            DESCRIPTION. The default is (0.2).

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        
        model.save(model_path)
        
        return model

class ModelEvaluate():
    def report(self, y_true, y_pred):
        '''
        report predicted score of actual and predicted values
        
        Parameters
        ----------
        y_true : array
            DESCRIPTION.
        y_pred : array
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))




