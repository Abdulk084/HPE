#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel


# In[2]:


import tensorflow as tf
import deepchem as dc
import numpy as np
graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
loader_train = dc.data.data_loader.CSVLoader( tasks=['ACTIVITY'], smiles_field="smiles",featurizer=graph_featurizer)
dataset_train = loader_train.featurize( './train.csv')


# In[3]:


loader_test = dc.data.data_loader.CSVLoader( tasks=['ACTIVITY'], smiles_field="smiles",featurizer=graph_featurizer)
dataset_test = loader_test.featurize( './test.csv')


# In[9]:


model = GraphConvModel( n_tasks = 1, mode ='regression', dropout = 0.2) 

model.fit( dataset_train, nb_epoch = 1000)



# In[10]:


metric = dc.metrics.Metric( dc.metrics.pearson_r2_score) 

print( model.evaluate( dataset_train, [metric])) 

print( model.evaluate( dataset_test, [metric]))



# In[11]:



test_preds=model.predict(dataset_test)


# In[12]:


import pandas as pd


# In[13]:



print(test_preds)


test_preds = pd.DataFrame(test_preds)
print(test_preds)

test_preds.columns = ['MGC_out']

writer = pd.ExcelWriter('test_preds_MGC.xlsx',engine='xlsxwriter')
test_preds.to_excel(writer,sheet_name='test_preds')

writer.save()

# In[ ]:




