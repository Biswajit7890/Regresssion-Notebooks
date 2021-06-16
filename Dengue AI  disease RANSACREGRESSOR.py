#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import io


# In[3]:


pd.set_option('display.max_rows', 200000)
pd.set_option('display.max_columns', 200)
np.set_printoptions(suppress=True)


# In[4]:


dftrain=pd.read_csv('dengue_features_train.csv')
dftest=pd.read_csv('dengue_features_test.csv')
Targetlabels=pd.read_csv('dengue_labels_train.csv')


# In[5]:


dftrain.head(10)


# In[6]:


dftest.head()


# In[7]:


Targetlabels.shape


# In[8]:


dftrain.shape


# In[9]:


dftrain['total_cases']=Targetlabels['total_cases']


# In[10]:


pd.plotting.hist_frame(dftrain, figsize=(15,5))


# In[10]:


pd.plotting.hist_frame(dftest, figsize=(15,5))


# In[11]:


dftrain=dftrain.drop(labels=['week_start_date','year','city'], axis=1)
#dftest=dftest.drop(labels=['week_start_date','year','city'], axis=1)


# In[12]:


dftrain.isnull().sum()


# In[13]:


dftest.isnull().sum()


# In[14]:


dftrain.nunique()


# In[15]:


Missingcols=['ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm',
'reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k' ,'station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm' ]


# In[16]:


dftrain[['station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm']].median()


# In[17]:


dftest[['station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm']].median()


# In[18]:


for feautre in Missingcols:
    meadian_sum=dftrain[feautre].median()
    dftrain[feautre]=dftrain[feautre].fillna(meadian_sum)
    dftest[feautre]=dftest[feautre].fillna(meadian_sum)


# In[19]:


dftrain.corrwith(dftrain['total_cases'], axis=0)


# In[20]:


dftrain.columns


# In[93]:


selected_Feat=['weekofyear', 'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw','precipitation_amt_mm', 
'reanalysis_air_temp_k','reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c','station_min_temp_c', 'station_precip_mm']

Target=['total_cases']


# In[166]:


X=dftrain[selected_Feat].values
y=dftrain[Target].values


# In[167]:


X=np.log1p(X)
y=np.log1p(y)


# In[168]:


from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RANSACRegressor
from astropy.stats import median_absolute_deviation


# In[169]:


from xgboost import XGBRegressor


# In[170]:


mad=median_absolute_deviation(dftrain['total_cases'])
print(mad)


# In[171]:


paramGRID={
            'max_trials':[50,200,400,1000,1500,1900],
             'stop_probability':[0.35,0.5,0.65]
          }


# ## Best Submission

# In[174]:


RAN= RANSACRegressor(max_skips=5,min_samples=0.5,stop_n_inliers=(0.45*1456))
GV=GridSearchCV(RAN,param_grid=paramGRID,scoring='neg_mean_absolute_error',cv=3)
model=GV.fit(X,y)
prediction=model.predict(X)
print("The R2 score",metrics.r2_score(y,prediction))
print("The Accuracy",100-(metrics.mean_absolute_error(y,prediction)))


# In[175]:


GV.best_params_


# In[176]:


print("The Accuracy on original",100-(metrics.mean_absolute_error(y,prediction)))


# In[ ]:





# In[ ]:





# In[177]:


X_test=dftest[selected_Feat].values


# In[178]:


X_test=np.log1p(X_test)


# In[179]:


pred_test=model.predict(X_test)


# In[180]:


pred_test=np.exp(pred_test)


# In[181]:


sample_df=pd.DataFrame(dftest, columns=['city','year','weekofyear'])
PredictionTest=np.round(pred_test,0)
sample_df['total_cases']=(PredictionTest).astype(int)


# In[182]:


sample_df.dtypes


# In[183]:


sample_df.head(20)


# In[184]:


sample_df['total_cases'].hist(figsize=(15,5))


# In[186]:


sample_df.to_csv('sample_df.csv', index=False)


# In[ ]:




