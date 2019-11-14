#!/usr/bin/env python
# coding: utf-8

# In[76]:


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

tf_sess = tf.Session()
K.set_session(tf_sess)
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

print("RDKit: %s"%rdkit.__version__)

import keras
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils import shuffle
print("Keras: %s"%keras.__version__)


print("All the libraries are loaded")


# In[78]:


#############Loading data from the csv file#########################


print("Prepairing data for train set")


bit_size=1024



trfile = open('train.csv', 'r')
line = trfile.readline()
dataX_train = []
dataY_train=[]
for i, line in enumerate(trfile):
    line = line.rstrip().split(',')
    smiles = str(line[1])
    Activity = str(line[0])
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bit_size)
    fp = np.array(fp)
    dataX_train.append(fp)
    dataY_train.append(Activity)
trfile.close()

dataX_train = np.array(dataX_train)
dataY_train = np.array(dataY_train)

print("1024 Binary Finger prints are created in dataX_train and dataY_train")


print('dataX_train Shape: '+str(np.shape(dataX_train)))
print('dataY_train Shape: '+str(np.shape(dataY_train)))


Max_len_train=np.max((np.count_nonzero(dataX_train, axis=1)))
print('Maximum number of 1s in dataX_train: '+str(Max_len_train))


# In[79]:





print("Prepairing data for test set")


bit_size=1024



trfile = open('test.csv', 'r')
line = trfile.readline()
dataX_test = []
dataY_test=[]
for i, line in enumerate(trfile):
    line = line.rstrip().split(',')
    smiles = str(line[1])
    Activity = str(line[0])
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bit_size)
    fp = np.array(fp)
    dataX_test.append(fp)
    dataY_test.append(Activity)
trfile.close()

dataX_test = np.array(dataX_test)
dataY_test = np.array(dataY_test)

print("1024 Binary Finger prints are created in dataX_test and dataY_test")


print('dataX_test Shape: '+str(np.shape(dataX_test)))
print('dataY_test Shape: '+str(np.shape(dataY_test)))


Max_len_test=np.max((np.count_nonzero(dataX_test, axis=1)))
print('Maximum number of 1s in dataX_test: '+str(Max_len_test))


# In[80]:


Max_len=max(Max_len_train,Max_len_train)
print('Max_len: '+str(Max_len))


# In[81]:






print("Taking the 1 index in all train samples")

data_x_train = []
    
for i in range(len(dataX_train)):
    fp = [0] * Max_len
    n_ones = 0
    for j in range(bit_size):
        if dataX_train[i][j] == 1:
            fp[n_ones] = j+1
            n_ones += 1
    data_x_train.append(fp)
        
data_x_train = np.array(data_x_train, dtype=np.int32)
data_y_train = (np.array(dataY_train, dtype=np.float32)).reshape(dataY_train.shape[0],1)




print('data_x_train Shape: '+str(np.shape(data_x_train)))
print('data_y_train Shape: '+str(np.shape(data_y_train)))


# In[82]:





print("Taking the 1 index in all test samples")

data_x_test = []
    
for i in range(len(dataX_test)):
    fp = [0] * Max_len
    n_ones = 0
    for j in range(bit_size):
        if dataX_test[i][j] == 1:
            fp[n_ones] = j+1
            n_ones += 1
    data_x_test.append(fp)
        
data_x_test = np.array(data_x_test, dtype=np.int32)
data_y_test = (np.array(dataY_test, dtype=np.float32)).reshape(dataY_test.shape[0],1)




print('data_x_test Shape: '+str(np.shape(data_x_test)))
print('data_y_test Shape: '+str(np.shape(data_y_test)))


print("Defining hyper parameters")
batch_size = 32

embedding_size = 400
n_hid = 2048 # number of feature maps
win_size = 4 # window size of kernel
lr = 5e-6 # learning rate of optimzier


bit_size = 1024 # circular fingerprint
emb = tf.Variable(tf.random_uniform([bit_size, embedding_size], -1, 1), dtype=tf.float32)
pads = tf.constant([[1,0], [0,0]])
embeddings = tf.pad(emb, pads)



def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

p_keep_conv = tf.placeholder(dtype=tf.float32)

class model():
    def __init__(self, embedding_size, n_hid, win_size, p_keep_conv, Max_len):
         self.Max_len = Max_len
         self.nhid  = n_hid
         self.kernel_size = win_size
         self.w2 = init_weights([self.kernel_size, embedding_size, 1, self.nhid]) # 64
         self.w_o = init_weights([self.nhid, 1])
         
         self.b2 = bias_variable([1, self.nhid])
         self.b_o = bias_variable([1])
         self.p_keep_conv = p_keep_conv
             
    def conv_model(self, X): 
        l2 = tf.nn.relu(tf.nn.conv2d(X, self.w2, strides=[1, 1, 1, 1], padding='VALID') + self.b2)
        l2 = tf.squeeze(l2, [2])
        l2 = tf.nn.pool(l2, window_shape=[self.Max_len-self.kernel_size+1], pooling_type='MAX', padding='VALID')
        l2 = tf.nn.dropout(l2, self.p_keep_conv)         
        lout = tf.reshape(l2, [-1, self.w_o.get_shape().as_list()[0]])
        return lout



X = tf.placeholder(tf.int32, [None, Max_len])
Y = tf.placeholder(tf.float32, [None, 1])






X_em = tf.nn.embedding_lookup(embeddings, X)
X_em = tf.reshape(X_em, [-1, Max_len, embedding_size, 1])

model = model(embedding_size, n_hid, win_size, p_keep_conv, Max_len)

py_x = model.conv_model(X_em)

temp_hid = n_hid
w1 = init_weights([temp_hid, 1])
b1 = bias_variable([1])



py_x1 = (tf.matmul(py_x, w1) + b1 )
cost1 = tf.losses.mean_squared_error(labels=Y, predictions=py_x1)
train_op1 = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost1)

prediction_error1 = tf.sqrt(cost1)

import tensorflow as tf



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
    for i in range(3):
        steps.append(i)
        training_batch = zip(range(0, len(data_x_train), batch_size),
                             range(batch_size, len(data_x_train)+1, batch_size))
   #for start, end in tqdm.tqdm(training_batch):
        for start, end in training_batch:
            sess.run(train_op1, feed_dict={X: data_x_train[start:end], Y: data_y_train[start:end] ,p_keep_conv: 0.5})
            

 
        
      
        merr_train_1 = sess.run(prediction_error1, feed_dict={X: data_x_train, Y: data_y_train, p_keep_conv: 1.0})     
        print('Epoch Number: '+str(i))
        print('RMSE_Train: '+str(merr_train_1))
        LD50_RMSE_train.append(merr_train_1)
        
        train_preds1 = sess.run(py_x1, feed_dict={X: data_x_train, p_keep_conv: 1.0})
        train_r1 = r2_score(data_y_train, train_preds1)   
        train_mae = mean_absolute_error(data_y_train, train_preds1)
        print('R^2_Train: '+str(train_r1))
        LD50_R2_train.append(train_r1)
        print('MAE_Train: '+str(train_mae))
        LD50_MAE_train.append(train_mae)
        print("   ")
       



    
    

    
    
    
      
        merr_test_1 = sess.run(prediction_error1, feed_dict={X: data_x_test, Y: data_y_test, p_keep_conv: 1.0})     
        print('Epoch Number: '+str(i))
        print('RMSE_test: '+str(merr_test_1))
        LD50_RMSE_test.append(merr_test_1)
        
        test_preds1 = sess.run(py_x1, feed_dict={X: data_x_test, p_keep_conv: 1.0})
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
    test_rmse = sess.run(prediction_error1, feed_dict={X: data_x_test, Y: data_y_test, p_keep_conv: 1})
    print('RMSE of the test after loading the best model: '+str(test_rmse))
    
    test_preds = sess.run(py_x1, feed_dict={X: data_x_test, p_keep_conv: 1.0})
    test_r = r2_score(data_y_test, test_preds1)   
    test_mae = mean_absolute_error(data_y_test, test_preds1)
    print('R^2_test after loading the best model: '+str(test_r))
  
    print('MAE_test after loading the best model: '+str(test_mae))
    print(test_preds)
    
  
    test_preds = pd.DataFrame(test_preds)
    print(test_preds)
    
    test_preds.columns = ['C2DF_out']

    writer = pd.ExcelWriter('test_preds_C2DF.xlsx',engine='xlsxwriter')
    test_preds.to_excel(writer,sheet_name='test_preds')
    
    writer.save()



