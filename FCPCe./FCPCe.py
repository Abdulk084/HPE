#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:


trfile = open('train.csv', 'r')
line = trfile.readline()

mols_train=[]
dataY_train=[]

for i, line in enumerate(trfile):
    line = line.rstrip().split(',')
    smiles = str(line[1])
    Activity = str(line[0])
    mol = Chem.MolFromSmiles(smiles)
    mols_train.append(mol)
    dataY_train.append(Activity)

trfile.close()

dataY_train = np.array(dataY_train)

print("SMIES are extracted in list in mols_train and activity in an array dataY_train")

print('dataY_train Shape: '+str(np.shape(dataY_train)))


# In[5]:


if __name__ == "__main__":
    freeze_support()



    # Create Calculator
    calc = Calculator(descriptors)

    # map method calculate multiple molecules (return generator)
    #print(list(calc.map(mols)))

    # pandas method calculate multiple molecules (return pandas DataFrame)
    dataX_train=calc.pandas(mols_train)


# In[6]:


trfile = open('test.csv', 'r')
line = trfile.readline()

mols_test=[]
dataY_test=[]

for i, line in enumerate(trfile):
    line = line.rstrip().split(',')
    smiles = str(line[1])
    Activity = str(line[0])
    mol = Chem.MolFromSmiles(smiles)
    mols_test.append(mol)
    dataY_test.append(Activity)

trfile.close()

dataY_test = np.array(dataY_test)

print("SMIES are extracted in list in mols_test and activity in an array dataY_test")

print('dataY_test Shape: '+str(np.shape(dataY_test)))


# In[7]:


if __name__ == "__main__":
    freeze_support()



    # Create Calculator
    calc = Calculator(descriptors)

    # map method calculate multiple molecules (return generator)
    #print(list(calc.map(mols)))

    # pandas method calculate multiple molecules (return pandas DataFrame)
    dataX_test=calc.pandas(mols_test)


# In[ ]:





# In[ ]:





# In[8]:


desc_number=len(calc.descriptors)


# In[9]:


print(type(dataX_test))


# In[ ]:





# In[10]:


fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
dataX_train = pd.DataFrame(fill_NaN.fit_transform(dataX_train))


# In[ ]:





# In[ ]:





# In[11]:


dataX_test = pd.DataFrame(fill_NaN.fit_transform(dataX_test))


# In[12]:


normalizer = preprocessing.Normalizer().fit(dataX_train)


# In[13]:


dataX_train=normalizer.transform(dataX_train) 


# In[14]:


dataX_test=normalizer.transform(dataX_test) 


# In[15]:


print(type(dataX_train))


# In[16]:


print(np.shape(dataX_train))


# In[17]:


print(np.shape(dataX_test))


# In[ ]:





# In[18]:


X = tf.placeholder(tf.float32, [None, desc_number])
Y = tf.placeholder(tf.float64, [None, 1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


py_x =keras.layers.Dense(1000, activation='relu')(X)
py_x = keras.layers.Dropout(0.7)(py_x)
py_x =keras.layers.Dense(1000, activation='sigmoid')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)

py_x =keras.layers.Dense(1000, activation='relu')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)

py_x =keras.layers.Dense(1000, activation='sigmoid')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)

py_x =keras.layers.Dense(1000, activation='relu')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)

py_x =keras.layers.Dense(1000, activation='sigmoid')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)

py_x =keras.layers.Dense(1000, activation='relu')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)

py_x =keras.layers.Dense(1000, activation='sigmoid')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)
py_x =keras.layers.Dense(1000, activation='sigmoid')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)
py_x =keras.layers.Dense(1000, activation='sigmoid')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)

py_x =keras.layers.Dense(1000, activation='relu')(py_x )
py_x = keras.layers.Dropout(0.5)(py_x)


py_x =keras.layers.Dense(10, activation='relu')(py_x )
py_x = keras.layers.Dropout(0.2)(py_x)
py_x1 = keras.layers.Dense(1, activation='linear')(py_x)


# In[ ]:





# In[20]:


cost1 = tf.losses.mean_squared_error(labels=Y, predictions=py_x1)


# In[21]:


train_op1 = tf.train.AdamOptimizer(learning_rate = 5e-6).minimize(cost1)


# In[22]:


prediction_error1 = tf.sqrt(cost1)


# In[23]:


import tensorflow as tf


# In[24]:


data_x_train=dataX_train
data_y_train=dataY_train

data_x_test=dataX_test
data_y_test=dataY_test


# In[25]:


print(np.shape(data_y_test))


# In[26]:


data_y_test.shape[0]


# In[27]:


data_y_test = (np.array(dataY_test, dtype=np.float32)).reshape(dataY_test.shape[0],1)


# In[28]:


data_y_train = (np.array(dataY_train, dtype=np.float32)).reshape(dataY_train.shape[0],1)


# In[29]:


batch_size = 16


# In[30]:


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
    for i in range(8000):
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


# In[31]:


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
    test_r = r2_score(data_y_test, test_preds)   
    test_mae = mean_absolute_error(data_y_test, test_preds)
    print('R^2_test after loading the best model: '+str(test_r))
  
    print('MAE_test after loading the best model: '+str(test_mae))
    print(test_preds)
    

    test_preds = pd.DataFrame(test_preds)
    print(test_preds)
    
    test_preds.columns = ['FCPCe_out']

    writer = pd.ExcelWriter('test_preds_FCPCe.xlsx',engine='xlsxwriter')
    test_preds.to_excel(writer,sheet_name='test_preds')
    
    writer.save()


