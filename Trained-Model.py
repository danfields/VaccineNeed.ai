#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Getting rid of warnings, which are useless variants of errors
import warnings
warnings.filterwarnings('ignore')

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from pprint import pprint
from IPython.display import clear_output
from sklearn.metrics import roc_curve

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import LeakyReLU
from keras.utils.np_utils import to_categorical

from keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt

state_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}
abbrev_state = dict(map(reversed, state_abbrev.items()))


# In[2]:


path = '/Users/22danielf/Desktop/county_vax_trend.csv'
df_raw = pd.read_csv(path, index_col='Date',parse_dates=True)

df_cut = df_raw[['Recip_County','Recip_State','Series_Complete_Yes']]
df_cut


# In[3]:


#for ind in df_cut.index:
   # print(df_cut['Recip_County'][ind])
df_cut_sorted = df_cut.sort_values(by=['Recip_County', 'Recip_State'])
df_cut_sorted


# In[4]:


indices = []
county = 'Abbeville County'
state = 'SC'
i=0
for index, row in df_cut_sorted.iterrows():
    curr_county = row['Recip_County']
    curr_state = row['Recip_State']
    if((curr_county != county) or (curr_state != state)):
        county = curr_county
        state = curr_state
        indices.append(i)
    i+=1 


# In[5]:


list_dfs = []
list_dfs.append(df_cut_sorted.iloc[:287,:])
index = 0
while index < len(indices) - 1:
    lower_bound = indices[index]
    upper_bound = indices[index + 1]
    list_dfs.append(df_cut_sorted.iloc[lower_bound:upper_bound,:])
    index+=1
list_dfs.append(df_cut_sorted.iloc[indices[len(indices)-2]:indices[len(indices)-1],:])


# In[6]:


list_dfs


# In[7]:


# # Making all the county data frames have the same number of rows (dates)
# min = 1000
# list_dfs_cut = []
# for df in list_dfs:
#     if len(df.index) < min:
#         min = len(df.index)
# print('The minimum # of rows is ' + str(min))
# for df in list_dfs:
#     if len(df.index) > min:
#         df_cut = df.iloc[0:min,:]
#         list_dfs_cut.append(df_cut)
# for df in list_dfs_cut:
#     print(len(df.index))
# # Use percents for training and testing sizes


# In[8]:


# for row in list_dfs[1].iterrows():
#     print(row[0])


# In[23]:


uploads = []

# In[10]:

del_frames = []

for dataframe in list_dfs:
    if(dataframe['Recip_State'][0]=='DE'):
        del_frames.append(dataframe)

for frame in del_frames:
    nameofcounty = frame['Recip_County'][0]

    state_Name = abbrev_state[frame['Recip_State'][0]]
    print(frame)
    df = frame

    # for df in list_dfs:

    df_rev_0 = df.iloc[::-1]
    # Assuming vaccine was created on January 4th (Fact-Check)
    N = 22
    df_rev = df_rev_0.iloc[N: , :]

    results = seasonal_decompose(df_rev["Series_Complete_Yes"])

    df = df_rev[['Series_Complete_Yes']]

    # numTraining = int(0.89*len(df.index))

    train = df.iloc[0:234]
    test = df.iloc[234:]

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_train = scaler.fit_transform(df.iloc[:234])
    # scaled_test = scaler.fit_transform(df.iloc[234:])
    # scaled_train = pd.DataFrame(scaled_train)
    # scaled_test = pd.DataFrame(scaled_test)
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df)
    scaled_train = scaled_df.iloc[:234]
    scaled_test = scaled_df.iloc[234:]

    scaled_train = scaled_train.to_numpy()
    scaled_test = scaled_test.to_numpy()

    # n_input = int(0.08*len(df.index))
    n_input = 25
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    #     X,y = generator[0]

    # model definition
    # Using one layer is kinda sus
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    #     model.summary()

    #change this back to higher number later
    model.fit(generator,epochs=2)

    loss_per_epoch = model.history.history['loss']
    #     plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

    # last_train_batch = np.array(scaled_train[-int(0.08*len(df.index)):])\
    last_train_batch = np.array(scaled_train[-25:])

    last_train_batch = last_train_batch.reshape((1, n_input, n_features))

    model.predict(last_train_batch)

    test_predictions = []

    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))

    for i in range(len(test)):

        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


    # In[11]:


    test_predictions


    # In[12]:


    # test.head()

    true_predictions = scaler.inverse_transform(test_predictions)
    true_predictions


    # In[13]:


    test['Series_Complete_Yes']


    # In[14]:


    test['Predictions'] = true_predictions

    test.plot(figsize=(12,6))

    rmse=sqrt(mean_squared_error(test['Series_Complete_Yes'],test['Predictions']))
    print('RMSE = ' + str(rmse))

    true_pred_percent = true_predictions / 24557
    true_pred_final = 1-true_pred_percent

    actual_vals = df_rev[["Series_Complete_Yes"]]
    actual_vals /= 24557
    actual_vals_final = 1 - actual_vals

    test['Predictions'] = true_pred_final
    test['Series_Complete_Yes'] = actual_vals_final

    test.plot(figsize=(12,6))

    size_test = len(test)
    name_list = []
    for i in range(size_test):
        name_list.append(nameofcounty + " (" + state_Name + ")")
    test['County_Name'] = name_list
    uploads.append(test)


df_upload = pd.concat(uploads)   

# df_upload['County_Name'] = delframe['Recip_County'].to_list()

df_upload.to_csv('/Users/22danielf/Desktop/VA_Files/uploadedDF.csv')


print(df_upload)
