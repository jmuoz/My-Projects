# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:13:43 2018

@author: xxzac
"""

import pandas as pd
import numpy as np
import sqlite3
import datetime
import scipy.stats 
#%%
con = sqlite3.connect('ff.sqlite')
cur = con.cursor()
result = cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(result.fetchall())
#%%
fires = pd.read_sql("select * from Fires order by random() limit 100000", con)
#%%
#fires['DURATION_DAYS'] = fires.CONT_DOY - fires.DISCOVERY_DOY
#fires.DISCOVERY_TIME = pd.to_numeric(fires.DISCOVERY_TIME)
#fires.CONT_TIME = pd.to_numeric(fires.CONT_TIME)
#fires['DURATION_TIME'] = fires.CONT_TIME - fires.DISCOVERY_TIME
#%%
#fires['DURATION_TIME'] = np.abs(fires.DURATION_TIME)
#print(fires.columns)
#fires_2 = fires[['STAT_CAUSE_DESCR', 'FIRE_SIZE', 'FIRE_YEAR', 'STATE', 'OWNER_DESCR']]
#%%
fires = fires.dropna(subset=['DISCOVERY_TIME', 'DISCOVERY_DOY', 'CONT_TIME', 'CONT_DOY'])
fires = fires.reset_index()
del fires['index']
print(len(fires))
#%%
parsed_dates = np.empty(len(fires), dtype='O')
for i in range(len(fires)):
    happy_new_year = datetime.datetime(year=fires.iloc[i]['FIRE_YEAR'],
        month=1,day=1)
    the_time = fires.iloc[i]['DISCOVERY_TIME']
    hours = int(the_time[0:2])
    mins = int(the_time[2:4])
    time_into_the_year = datetime.timedelta(
        days=int(fires.iloc[i]['DISCOVERY_DOY']),hours=hours,minutes=mins)
    parsed_dates[i] = happy_new_year + time_into_the_year

fires['THE_FREAKING_DATETIME'] = parsed_dates
#%%
parsed_dates2 = np.empty(len(fires), dtype='O')
for i in range(len(fires)):
    happy_new_year = datetime.datetime(year=fires.iloc[i]['FIRE_YEAR'],
        month=1,day=1)
    the_time = fires.iloc[i]['CONT_TIME']
    hours = int(the_time[0:2])
    mins = int(the_time[2:4])
    time_into_the_year = datetime.timedelta(
        days=int(fires.iloc[i]['CONT_DOY']),hours=hours,minutes=mins)
    parsed_dates2[i] = happy_new_year + time_into_the_year

fires['THE_FUARKING_DATETIME'] = parsed_dates2
#%%
fires['DURATION'] = fires.THE_FUARKING_DATETIME - fires.THE_FREAKING_DATETIME
#print(fires.DURATION.head())

integer_time = []
for i in range(len(fires)):
    integer_time.append(fires['DURATION'][i].total_seconds())
fires['INT_DURATION'] = integer_time #duration in seconds
fires['INT_DURATION'] = fires['INT_DURATION']/60 #duration in minutes
fires['INT_DURATION'] = fires['INT_DURATION']/60 #duration in hours

#%%
fires_2 = fires[['FIRE_SIZE', 'FIRE_YEAR', 'STATE', 'OWNER_DESCR', 'INT_DURATION', 'STAT_CAUSE_DESCR']]
print(fires_2.head())
#%%
t = fires_2.sample(frac=.7)
test_data = fires_2.drop(t.index)
test_data_r = test_data  
#%%
print(type(fires_2['FIRE_YEAR'][0]))
#%%
densities = {}
probs = {}
numeric_features = [c for c in t.columns[:-1] if np.issubdtype(t[c].dtype, np.number)]
categorical_features = [c for c in t.columns[:-1] if not np.issubdtype(t[c].dtype, np.number)]
target = t.columns[-1]

for label in t[target].drop_duplicates():
    rows_with_this_label = t[t[target] == label]
    probs[label] = len(rows_with_this_label)/len(t)
    for feature in numeric_features:
        density = scipy.stats.norm(rows_with_this_label[feature].mean(),
                                   rows_with_this_label[feature].std())
        densities[feature+"|"+label] = density
    for feature in categorical_features:
        for value in t[feature].drop_duplicates():
            probs[value+"|"+label] = (len(t[(t[target] == label) & (t[feature] == value)]) /len(t[t[target] == label]))    

#%%
#print(probs)
#%%
#HOW TO USE predict():
    # Values is a list, and must be enclosed in square brackets [].
    # Exactly 5 values must go into the brackets as follows:
    # The SIZE of the fire. Input it as FIRE_SIZE=YOUR_NUMBER. Example: FIRE_SIZE=500. SIZE is in acres.
    # The YEAR the fire happened in. Must be a whole number, from 1992-2015. Input it as FIRE_YEAR=YOUR_YEAR. Example: FIRE_YEAR=2007
    # The STATE the fire was in. Use the two-character state code. Input it as YOUR_STATE_CODE. Example: VA
    # The OWNER OF THE LAND at the time of the fire. Input as YOUR_OWNER. Example: USFS
#OWNER can only be one of the following (case sensitive): PRIVATE, MISSING/NOT SPECIFIED, USFS,
#BIA, BLM, STATE OR PRIVATE, STATE, NPS, FWS, TRIBAL, OTHER FEDERAL, MUNICIPAL/LOCAL, UNDEFINED FEDERAL,
#COUNTY, BOR, or FOREIGN.
    #The INT_DURATION of the fire. Formatted as follows INT_DURATION=NUMBER. Example: INT_DURATION=100    
#Keep in mind that the INT_DURATION is the duration measured in hours, so don't choose extremely large numbers.
#INT_duration can be a decimal.    
    #Each value must be separated by a comma (,), and enclosed in quotations (''). 
    #Example: predict(['FIRE_SIZE=2011', 'FIRE_YEAR=1999', 'OR', 'USFS', 'INT_DURATION=200'])
    #The function will then try to predict the cause of the fire based on the information you input.
def predict(values): 
    scores = []
    top_prob = 0
    top_answer = ''
    for label in test_data_r[target].drop_duplicates():
        prob = probs[label] 
        for value in values:
            if "=" in value:
                numeric_feature, numeric_value = value.split("=")
                numeric_value = float(numeric_value)
                prob *= densities[numeric_feature+"|"+label].pdf(numeric_value)
            else:
                prob *= probs[value+"|"+label]
        scores.append(prob)
        if prob > top_prob:
            top_prob = prob
            top_answer = label
    return [ top_answer, top_prob / sum(scores) ]
#%%
#print(fires_2['INT_DURATION'].head(20))
#%%
print(pd.value_counts(fires_2['STATE']))
print(pd.value_counts(fires_2['OWNER_DESCR'])) 
print(pd.value_counts(fires_2['FIRE_YEAR']))
#%%
predict(['FIRE_SIZE=300', 'FIRE_YEAR=2002', 'OR', 'MISSING/NOT SPECIFIED', 'INT_DURATION=33'])
#%%
correct = 0
for i in range(len(test_data_r)):
    row = test_data_r.iloc[i]
    pred = predict(['FIRE_SIZE=' + str(row['FIRE_SIZE']), 'FIRE_YEAR=' + str(row['FIRE_YEAR']), row['STATE'], row['OWNER_DESCR'], 'INT_DURATION=' + str(row['INT_DURATION'])])
    if pred[0] == row['STAT_CAUSE_DESCR']:
        correct+=1

print('This algorithm got '+str(correct/len(test_data_r)*100)+ '% correct on the test data.')