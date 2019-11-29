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


# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


trfile = open('train.csv', 'r')
line = trfile.readline()

mols_train=[]
dataY_train=[]
smiles_train=[]

for i, line in enumerate(trfile):
    line = line.rstrip().split(',')
    smiles = str(line[1])
    
    smiles_train.append(smiles)
    
    
    Activity = str(line[0])
    mol = Chem.MolFromSmiles(smiles)
    mols_train.append(mol)
    dataY_train.append(Activity)

trfile.close()

dataY_train = np.array(dataY_train)

print("SMIES are extracted in list in mols_train and activity in an array dataY_train")

print('dataY_train Shape: '+str(np.shape(dataY_train)))


# In[5]:







with open('train.smi', 'w') as filehandle:
    for listitem in smiles_train:
        filehandle.write('%s\n' % listitem)







trfile = open('test.csv', 'r')
line = trfile.readline()

mols_test=[]
dataY_test=[]
smiles_test=[]
for i, line in enumerate(trfile):
    line = line.rstrip().split(',')
    smiles = str(line[1])
    smiles_test.append(smiles)
    Activity = str(line[0])
    mol = Chem.MolFromSmiles(smiles)
    mols_test.append(mol)
    dataY_test.append(Activity)

trfile.close()

dataY_test = np.array(dataY_test)

print("SMIES are extracted in list in mols_test and activity in an array dataY_test")

print('dataY_test Shape: '+str(np.shape(dataY_test)))


# In[6]:



with open('test.smi', 'w') as filehandle:
    for listitem in smiles_test:
        filehandle.write('%s\n' % listitem)






padeldescriptor(mol_dir='test.smi',d_2d=True, d_3d=False,fingerprints=False, removesalt=True, retainorder=True,
               
               d_file='test_2D.csv',
              
               maxruntime=100000, threads=1)


# In[3]:


padeldescriptor(mol_dir='train.smi',d_2d=True, d_3d=False,fingerprints=False, removesalt=True, retainorder=True,
               
               d_file='train_2D.csv',
               
               maxruntime=100000, threads=1)







dataX_train=pd.read_csv('train_2D.csv')


# In[7]:


print(np.shape(dataX_train))


# In[8]:


dataX_test=pd.read_csv('test_2D.csv')


# In[9]:


print(np.shape(dataX_test))


# In[10]:


dataX=pd.concat([dataX_train,dataX_test])


# In[11]:


print(np.shape(dataX))


# In[12]:


dataX


# In[ ]:





# In[13]:


#This function gets the raw data and clean it 
def data_clean(data):

    print("Data shape before cleaning:"  + str(np.shape(data)))
    
             
    #Change the data type of any column if necessary.
    print("Now it will print only those columns with non-numeric values")
    print(data.select_dtypes(exclude=[np.number]))
    

    #Now dropping those columns with zero values entirely or which sums to zero
    data= data.loc[:, (data != 0).any(axis=0)]

    #Now dropping those columns with NAN values entirely 
    data=data.dropna(axis=1, how='all')
    data=data.dropna(axis=0, how='all')

    #Keep track of the columns which are exculded after NAN and column zero sum operation above
    print("Data shape after cleaning:"  + str(np.shape(data)))

    return data


# In[14]:


dataX= data_clean(dataX)


# In[15]:


#This function impute the missing values with features (column mean)
def data_impute(data):
      
    #Seprating out the NAMES of the molecules column and ACTIVITY column because they are not the features to be normalized.
    data_input=data.drop(['Name'], axis=1)
  
    data_names = data.Name
    

    #Imputing the missing values with features mean values
    fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    Imputed_Data_input = pd.DataFrame(fill_NaN.fit_transform(data_input))
    Imputed_Data_input.index = data_input.index
    
    
    print(np.shape(Imputed_Data_input))
    print("Data shape after imputation:"  + str(np.shape(Imputed_Data_input)))
    nanmask = np.isnan(fill_NaN.statistics_)
    print(nanmask)
    
    
    
    return Imputed_Data_input, data_names


# In[16]:


Imputed_Data_input, data_names=data_impute(dataX)


# In[17]:


print(np.shape(Imputed_Data_input))


# In[18]:


Imputed_Data_input


# In[19]:


#This function is to normalize features  
def data_norm(Imputed_Data_input):   
    #Calculatig the mean and STD of the imputed input data set
    Imputed_Data_input_mean=Imputed_Data_input.mean()
    Imputed_Data_input_std=Imputed_Data_input.std()

    #z-score normalizing the whole input data:
    Imputed_Data_input_norm = (Imputed_Data_input - Imputed_Data_input_mean)/Imputed_Data_input_std

    #Adding names and labels to the data again
    #frames = [data_names,data_labels, Imputed_Data_input_norm]
    #full_data_norm = pd.concat(frames,axis=1)
    
    return Imputed_Data_input_norm


# In[20]:


full_data_norm=data_norm(Imputed_Data_input)


# In[21]:


full_data_norm


# In[22]:


print(np.shape(dataX_train))


# In[23]:


dataX_train=full_data_norm[0:dataX_train.shape[0]]


# In[24]:


dataX_test=full_data_norm[dataX_train.shape[0]:]


# In[25]:


dataX_test


# In[ ]:





# In[26]:


print(np.shape(dataX_train))


# In[27]:


print(np.shape(dataX_test))


# In[28]:


desc_number=dataX_train.shape[1]


# In[29]:


X = tf.placeholder(tf.float32, [None, desc_number])
Y = tf.placeholder(tf.float64, [None, 1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


py_x =keras.layers.Dense(1000, kernel_initializer ='glorot_normal', activation='sigmoid')(X)
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





# In[31]:


cost1 = tf.losses.mean_squared_error(labels=Y, predictions=py_x1)


# In[45]:


train_op1 = tf.train.AdamOptimizer(learning_rate = 5e-6).minimize(cost1)


# In[46]:


prediction_error1 = tf.sqrt(cost1)


# In[47]:


import tensorflow as tf


# In[48]:


data_x_train=dataX_train
data_y_train=dataY_train

data_x_test=dataX_test
data_y_test=dataY_test


# In[49]:


print(np.shape(data_y_test))


# In[50]:


data_y_test.shape[0]


# In[51]:


data_y_test = (np.array(dataY_test, dtype=np.float32)).reshape(dataY_test.shape[0],1)


# In[52]:


data_y_train = (np.array(dataY_train, dtype=np.float32)).reshape(dataY_train.shape[0],1)


# In[53]:


batch_size = 32


# In[54]:


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
    for i in range(1000):
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


# In[55]:


####################################################################
#=========================== test part ============================#
####################################################################
saver = tf.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "model_FCPC")
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
    
    test_preds.columns = ['FCPC_out']

    writer = pd.ExcelWriter('test_preds_FCPC.xlsx',engine='xlsxwriter')
    test_preds.to_excel(writer,sheet_name='test_preds')
    
    writer.save()
