#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import WeaveModel


# In[2]:


featurizer = dc.feat.WeaveFeaturizer()


# In[ ]:





# In[ ]:





# In[3]:


import tensorflow as tf
import deepchem as dc
import numpy as np
graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
loader_train = dc.data.data_loader.CSVLoader(tasks=['ACTIVITY'], smiles_field="smiles",featurizer=featurizer)
dataset_train = loader_train.featurize( './train.csv')


# In[4]:


loader_test = dc.data.data_loader.CSVLoader( tasks=['ACTIVITY'], smiles_field="smiles",featurizer=featurizer)
dataset_test = loader_test.featurize('./test.csv')



# In[ ]:





# In[5]:


model = dc.models.WeaveModel( n_tasks = 1, mode ='regression') 

model.fit( dataset_train, nb_epoch = 2000)



# In[6]:


metric = dc.metrics.Metric( dc.metrics.pearson_r2_score) 

print( model.evaluate( dataset_train, [metric])) 

print( model.evaluate( dataset_test, [metric]))



# In[7]:



test_preds=model.predict(dataset_test)


# In[8]:


import pandas as pd


# In[9]:

print(test_preds)


test_preds = pd.DataFrame(test_preds)
print(test_preds)

test_preds.columns = ['MWC_out']

writer = pd.ExcelWriter('test_preds_MWC.xlsx',engine='xlsxwriter')
test_preds.to_excel(writer,sheet_name='test_preds')

writer.save()



