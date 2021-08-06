# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:13:10 2019

@author: Matthew D'Ambrosio
"""
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#We'll be coorelating LW with State, Subject, and University
categories = ['LW Ratio', 'Stipend', 'State', 'University', 'Subject']
stip = pd.read_csv('stipend_and_locations.csv', encoding = 'latin-1')[categories]
stip = stip.replace(r'^\s*$', np.nan, regex=True)

stip[stip['State']=='Alaska']
stip = stip.dropna()
print(len(stip))

#Must be an int
stip['LW Ratio'] = stip['LW Ratio']*100
stip['LW Ratio'] = stip['LW Ratio'].astype(int)
stip['LW Ratio'] = stip['LW Ratio']/100
#%%
"""
find unique values in subjects

for i in unique subjects find nearish string and put in dictrionary for 
each unique category with the unique value as the key and the near neighbors
as values 

then merge the keys based on how close they are to eachother 

finally rename all of the subjects as their simplified counterpart
"""
#subject containing dictionary
sub_dict = {}

#lsit of all subjects
subs = stip['Subject'].str.lower()
#list of unique subjects
uniq_subs = subs.unique()

#location of all subjects
for idx,unique_subject in enumerate(uniq_subs):
    sub_dict[unique_subject] = subs[uniq_subs[idx] == subs].index

#%%
from fuzzywuzzy import process

#find dictionary keys that are similar and merge them
sub_dict = {}

sort = pd.Series(sorted(subs, key=len))
#find matches
for i, subject in enumerate(sort):
    try:
        #sort out smaller matches first
        if i < 198:
            cutoff = 100
        else:
            cutoff = 100
            
            #top 5 matches
        matches = process.extractBests(sort[i], sort, score_cutoff=cutoff, limit = len(sort))
        #index of matches
        indxs = [item[2] for item in matches]
        #match subject and its matched subjects in a dictionary
        sub_dict[subject] = [item[0] for item in matches]
        #remove matches from list
        sort = sort.drop(indxs)
    except:
        pass


simp_subs = []
for subject in subs:
    simp_subs.append([sub for sub, matches in sub_dict.items() 
               if subject in matches][0])

'''
with open('sub_dict.csv', 'w') as f:
    for key in sub_dict.keys():
        f.write("%s,%s\n"%(key,sub_dict[key]))
f.close()
'''

#cursory look
pd.DataFrame([simp_subs, subs]).T

simp_stip = stip
simp_stip['Subject'] = simp_subs
simp_stip = simp_stip.reset_index().iloc[:,1:]

#%% Preprocess Data
#remove outliers
#f = simp_stip[abs(simp_stip['Stipend'])<=46000]
#f = f[f['Stipend']>5000]

f = simp_stip[simp_stip['Stipend'].between(simp_stip['Stipend'].quantile(.08), simp_stip['Stipend'].quantile(.94))]

#remove sci-notation
pd.options.display.float_format = '{:.2f}'.format

#onehot encode categorical data
from sklearn.model_selection import train_test_split
X = f.iloc[:,2:] 
#X = simp_stip['Subject'].values.reshape(-1,1) #State, Uni, Sub
y = f['Stipend'].values.reshape(-1,1) #LW ratio
#y = simp_stip['Stipend'].values.reshape(-1,1) #LW ratio

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression


#encoder = OrdinalEncoder()
encoder = OneHotEncoder(sparse = True, categories='auto')
X_encoded = encoder.fit_transform(X)

#plit data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#%% LINEAR REGRESSION
model = LinearRegression()
model.fit(X=X_train, y=y_train)

#Test model
y_pred = model.predict(X_test)
df = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten(), 'Diff':y_test.flatten()-y_pred.flatten()})

print(df.head())

from sklearn import metrics
print('\nSummary')
print('R^2 Score:', metrics.r2_score(y_test, y_pred)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

x = df['Actual']
y = df['Predicted']

plt.scatter(x=x, y=y, color='gray')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

model.fit(x.values.reshape(-1,1), y.values.reshape(-1,1))
print('Slope:', model.coef_)  
print('Intercept:', model.intercept_)


#%% NEURAL NETWORK
#scale data
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,1000),max_iter=500,random_state=42)
mlp.fit(X_train, y_train.ravel())

mlp_y_pred = mlp.predict(X_test)

print('\nSummary')
mlp_df = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':mlp_y_pred.flatten(), 'Diff':y_test.flatten()-mlp_y_pred.flatten()})
print(mlp_df.head())

print('\nSummary')
print('R^2 Score:', metrics.r2_score(y_test, mlp_y_pred)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, mlp_y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, mlp_y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, mlp_y_pred)))

plt.scatter(y_test, mlp_y_pred, color='gray')
#plt.ylim(1,1.2)
#plt.xlim(0,50000)
plt.show()

model.fit(mlp_df['Actual'].values.reshape(-1,1), mlp_df['Predicted'].values.reshape(-1,1))
print('Slope:', model.coef_)  
print('Intercept:', model.intercept_)

#%% RANDOM FOREST
f = simp_stip['LW Ratio'].round(0)

#remove sci-notation
pd.options.display.float_format = '{:.2f}'.format

#onehot encode categorical data
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=50, random_state=42)
rfr.fit(X_train, y_train.ravel())
rfr_y_pred = rfr.predict(X_test)

print('\nSummary')
rfr_df = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':rfr_y_pred.flatten(), 'Diff':y_test.flatten()-rfr_y_pred.flatten()})
print(rfr_df.head())

print('\nSummary')
print('R^2 Score:', metrics.r2_score(y_test, rfr_y_pred)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rfr_y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rfr_y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rfr_y_pred)))

plt.scatter(y_test, rfr_y_pred, color='gray')
#plt.ylim(1,1.2)
#plt.xlim(0,50000)
plt.show()

model.fit(rfr_df['Actual'].values.reshape(-1,1), rfr_df['Predicted'].values.reshape(-1,1))
print('Slope:', model.coef_)  
print('Intercept:', model.intercept_)
