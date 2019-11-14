#!/usr/bin/env python
# coding: utf-8

# In[47]:



from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
from keras.layers import Dense
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import ExcelFile
from pandas import ExcelWriter

from scipy import ndimage
from scipy.stats import randint as sp_randint
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn import metrics
from sklearn import pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.python.framework import ops

import keras
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import scipy
import tensorflow as tf
import xlsxwriter


import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input


from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import tensorflow as tf
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


import keras.backend as K

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import tensorflow as tf
import keras
from keras import backend as K

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

import keras
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils import shuffle

from multiprocessing import freeze_support
from sklearn import preprocessing
from rdkit import Chem

from mordred import Calculator, descriptors

from padelpy import from_smiles
from padelpy import padeldescriptor


# In[2]:


from padelpy import from_smiles



# In[48]:




train_data = pd.read_excel(r'train.xlsx')


# In[49]:


test_data = pd.read_excel(r'test.xlsx')


# In[50]:


X_train_smiles=np.array(train_data ['smiles'])


# In[51]:


X_test_smiles=np.array(test_data ['smiles'])


# In[52]:


print(X_train_smiles.shape)
print(X_test_smiles.shape)


# In[53]:


print(X_train_smiles.shape[0])


# In[54]:


Y_train=np.array(train_data ['ACTIVITY'])


# In[55]:


Y_test=np.array(test_data ['ACTIVITY'])


# In[56]:


full_data = pd.read_excel(r'data.xlsx')


# In[57]:




charset = set("".join(list(full_data.smiles))+"!E")
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in full_data.smiles]) + 5
print (str(charset))
print(len(charset), embed)



# In[58]:


char_to_int


# In[59]:


def vectorize(smiles):
        one_hot =  np.zeros((smiles.shape[0], embed , len(charset)),dtype=np.int8)
        for i,smile in enumerate(smiles):
            #encode the startchar
            one_hot[i,0,char_to_int["!"]] = 1
            #encode the rest of the chars
            for j,c in enumerate(smile):
                one_hot[i,j+1,char_to_int[c]] = 1
            #Encode endchar
            one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
        #Return two, one for input and the other for output
        return one_hot[:,0:-1,:], one_hot[:,1:,:]


# In[60]:


X_train, _ = vectorize(X_train_smiles)
X_test, _ = vectorize(X_test_smiles)


# In[61]:


X_train[8].shape


# In[62]:




vocab_size=len(charset)



# In[ ]:





# In[ ]:





# In[63]:




from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding



# In[64]:


print(np.shape(np.argmax(X_train, axis=2)))


# In[65]:


print(np.shape(X_train))


# In[66]:


print(np.shape(X_test))


# In[67]:


dataX_train=np.argmax(X_train, axis=2)
dataX_test=np.argmax(X_test, axis=2)


dataY_train=Y_train
dataY_test=Y_test


# In[68]:


print('dataX_test Shape: '+str(np.shape(dataX_test)))
print('dataY_test Shape: '+str(np.shape(dataY_test)))


# In[69]:


print('dataX_train Shape: '+str(np.shape(dataX_train)))
print('dataY_train Shape: '+str(np.shape(dataY_train)))


# In[70]:


data_y_train = (np.array(dataY_train, dtype=np.float32)).reshape(dataY_train.shape[0],1)


# In[71]:


print('data_y_train Shape: '+str(np.shape(data_y_train)))


# In[72]:


data_y_test = (np.array(dataY_test, dtype=np.float32)).reshape(dataY_test.shape[0],1)


# In[73]:


print('data_y_test Shape: '+str(np.shape(data_y_test)))


# In[74]:


data_x_test=dataX_test


# In[75]:


data_x_train=dataX_train


# In[76]:


print(np.shape(data_x_train))


# In[77]:


Max_len=data_x_train.shape[1]


# In[78]:


X = tf.placeholder(tf.float32, [None, Max_len])
Y = tf.placeholder(tf.float64, [None, 1])


# In[79]:


py_x =keras.layers.Embedding(1025, 400, input_length=Max_len)(X)


# In[80]:


py_x=keras.layers.Conv1D(192,10,activation='relu')(py_x)
py_x=keras.layers.BatchNormalization()(py_x)
py_x=keras.layers.Conv1D(192,5,activation='relu')(py_x)
py_x=keras.layers.Conv1D(192,3,activation='relu')(py_x)
py_x=keras.layers.Flatten()(py_x)


# In[81]:


py_x1_keras = keras.layers.Dense(100, activation='relu')(py_x)
py_x1_keras = keras.layers.Dropout(0.7)(py_x1_keras)


# In[82]:


py_x1 = keras.layers.Dense(1, activation='linear')(py_x1_keras)


# In[83]:


cost1 = tf.losses.mean_squared_error(labels=Y, predictions=py_x1)


# In[84]:


train_op1 = tf.train.AdamOptimizer(learning_rate = 5e-6).minimize(cost1)


# In[85]:


prediction_error1 = tf.sqrt(cost1)


# In[86]:


import tensorflow as tf


# In[ ]:





# In[87]:


batch_size = 32


# In[88]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

SAVER_DIR = "model_ld50"
saver = tf.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "model_ld50")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    
    
    best_rmse = 10
    best_idx = 0

    LD50_R2_train = []
    #LD50_R2_valid = []
    LD50_R2_test = []

    
    LD50_RMSE_train = []
    #LD50_RMSE_valid = []
    LD50_RMSE_test = []
    
    

    LD50_MAE_train = []
    #LD50_MAE_valid = []
    LD50_MAE_test = []

    
    
    
   

    
    steps=[]
    for i in range(5000):
        steps.append(i)
        training_batch = zip(range(0, len(data_x_train), batch_size),
                             range(batch_size, len(data_x_train)+1, batch_size))
   #for start, end in tqdm.tqdm(training_batch):
        for start, end in training_batch:
            sess.run(train_op1, feed_dict={X: data_x_train[start:end], Y: data_y_train[start:end]})
            

 
        
      
        merr_train_1 = sess.run(prediction_error1, feed_dict={X: data_x_train, Y: data_y_train})     
        print('Epoch Number: '+str(i))
        print('RMSE_Train: '+str(merr_train_1))
        LD50_RMSE_train.append(merr_train_1)
        
        train_preds1 = sess.run(py_x1, feed_dict={X: data_x_train})
        train_r1 = r2_score(data_y_train, train_preds1)   
        train_mae = mean_absolute_error(data_y_train, train_preds1)
        print('R^2_Train: '+str(train_r1))
        LD50_R2_train.append(train_r1)
        print('MAE_Train: '+str(train_mae))
        LD50_MAE_train.append(train_mae)
        print("   ")
       



    

    
    
    
      
        merr_test_1 = sess.run(prediction_error1, feed_dict={X: data_x_test, Y: data_y_test})     
        print('Epoch Number: '+str(i))
        print('RMSE_test: '+str(merr_test_1))
        LD50_RMSE_test.append(merr_test_1)
        
        test_preds1 = sess.run(py_x1, feed_dict={X: data_x_test})
        test_r1 = r2_score(data_y_test, test_preds1)   
        test_mae = mean_absolute_error(data_y_test, test_preds1)
        print('R^2_test: '+str(test_r1))
        LD50_R2_test.append(test_r1)
        print('MAE_test: '+str(test_mae))
        LD50_MAE_test.append(test_mae)
        print("   ")
        
        
        
        if best_rmse > merr_test_1:
            best_idx = i
            best_rmse = merr_test_1
            save_path = saver.save(sess, ckpt_path)
            print('model saved!')
    
       

        print("###########################################################################")


# In[89]:


####################################################################
#=========================== test part ============================#
####################################################################
saver = tf.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, ckpt.model_checkpoint_path)
    print("model loaded successfully!")
    test_rmse = sess.run(prediction_error1, feed_dict={X: data_x_test, Y: data_y_test})
    print('RMSE of the test after loading the best model: '+str(test_rmse))
    
    test_preds = sess.run(py_x1, feed_dict={X: data_x_test})
    test_r = r2_score(data_y_test, test_preds1)   
    test_mae = mean_absolute_error(data_y_test, test_preds1)
    print('R^2_test after loading the best model: '+str(test_r))
  
    print('MAE_test after loading the best model: '+str(test_mae))
    print(test_preds)
    

    test_preds = pd.DataFrame(test_preds)
    print(test_preds)
    
    test_preds.columns = ['C1DS_out']

    writer = pd.ExcelWriter('test_preds_C1DS.xlsx',engine='xlsxwriter')
    test_preds.to_excel(writer,sheet_name='test_preds')
    
    writer.save()
