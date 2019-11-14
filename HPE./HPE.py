#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import scipy

import xlsxwriter







import os
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

print("All the libraries are loaded")


# In[27]:


FCPC = pd.read_excel(r"test_preds_FCPC.xlsx")
FCPCe = pd.read_excel(r'test_preds_FCPCe.xlsx')


C1DS = pd.read_excel(r'test_preds_C1DS.xlsx')
C2DF = pd.read_excel(r'test_preds_C2DF.xlsx')

MGC = pd.read_excel(r'test_preds_MGC.xlsx')
MWC = pd.read_excel(r'test_preds_MWC.xlsx')

ACTIVITY = pd.read_excel(r'test.xlsx')

ACTIVITY=ACTIVITY['ACTIVITY']


# In[28]:


Average_output=(FCPC['FCPC_out']+FCPCe['FCPCe_out']+C1DS['C1DS_out']+C2DF['C2DF_out']+MGC['MGC_out']+MWC['MWC_out'])/6


# In[29]:


print(type(Average_output))


# In[30]:


Average_output=pd.DataFrame(Average_output)
Average_output.columns = ['ave']


# In[34]:


ACTIVITY=pd.DataFrame(ACTIVITY)
ACTIVITY.columns = ['ACTIVITY']

print(type(ACTIVITY))


# In[37]:


print("R2= "+str(r2_score(ACTIVITY['ACTIVITY'], Average_output['ave'])))
print("RMSE= "+str(sqrt(mean_squared_error(ACTIVITY['ACTIVITY'], Average_output['ave']))))
print("MAE= "+str(mean_absolute_error(ACTIVITY['ACTIVITY'], Average_output['ave'])))


# In[ ]:




