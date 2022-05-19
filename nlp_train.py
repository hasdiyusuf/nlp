
import os
import pandas as pd
import numpy as np 
import datetime

from nlp_module import ExploratoryDataAnalysis, ModelCreation, ModelEvaluate

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

#%% Load the data
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
PATH_LOG = os.path.join(os.getcwd(),'log')
JSON_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
MODEL_PATH = os.path.join(os.getcwd(), 'model_saved','model.h5')

df = pd.read_csv(URL)

#%% data inspection
df.shape
df.info()

# split features and target label 
category = df['category']
category_dummy = category.copy()

text = df['text']
text_dummy = text.copy()

# checking unique values on category
category.unique()

# review the text data
text[3]
text[16]

text_dummy.isna().sum()
category_dummy.isna().sum()


#%%
eda = ExploratoryDataAnalysis()

text_dummy = eda.data_cleaning(text_dummy)
# remove punctuation from string
text_dummy = eda.text_token(JSON_PATH, text_dummy)
# first arguement saving the tokenizer data into JSON_PATH
# second arguement convert str into numeric
text_dummy = eda.text_pad_sequences(text_dummy)

category_dummy = eda.one_hot_encoder(category_dummy)
# one hot encoder is used for multilabel target 

#%% create model 

mc = ModelCreation() # calling model class and assign it as mc
nb_categories = 5 # no. of label
num_words = 10000 # no. of vocabs

#%% test model


model = mc.lstm_layer(MODEL_PATH,num_words,nb_categories)
# model creation 
# MODEL_PATH : saved model into the path 
# num_words : no. of inputs
# nb_categorires : no. of outputs

plot_model(model) # visualize the model architecture

#%% train test split the data
x_train, x_test, y_train, y_test = train_test_split(text_dummy,
                                                    category_dummy,
                                                    test_size=0.3, 
                                                    random_state = 123)

x_try = np.expand_dims(x_train, axis = -1)

#%% compile/fit model
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = 'acc')

# assign the log path directory
log_dir = os.path.join(PATH_LOG,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir = log_dir)

model.fit(x_train, y_train, epochs = 5, validation_data = (x_test, y_test),
           callbacks= tensorboard_callback)



#%% model prediction
# pre-allocation approac
predicted = np.empty([len(x_test), 5]) # creating empty cell
for index, test in enumerate(x_test):
    predicted[index,:] = model.predict(np.expand_dims(test, axis = 0))

y_pred = np.argmax(predicted, axis = 1)
y_true = np.argmax(y_test, axis = 1)

#%% Model evaluate
me = ModelEvaluate()
me.report(y_true, y_pred)

# from the report, we can see the accuracy score 
# we can see that the model has predicted the test very accurate 
# with around 85%

# using Embedding to the model help to boost the accuracy of the performance

# issues
# LabelEncoder is used before this for the target label
# the performance does not improve much compare to OneHotEncoder
# it is prefer to used onehotencoder for multilabel target.

