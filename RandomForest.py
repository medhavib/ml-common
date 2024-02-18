# more information about this code and output can be found on:
# http://turngeek.blogspot.com/2017/01/random-forest-analysis-of-us-national.html

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier

#Load the dataset

#response column
respcols= ['S12Q1']

#explanatory variabls
expcols = ['BUILDTYP', 'CYEAR', 'NUMPERS', 'NUMPER18',  'NUMREL18', 'S1Q1E',
'DGSTATUS','S9Q1A', 'SMOKER', 'S3AQ3B1', 'S3AQ3B2','S6Q1',
'S1Q9B','S2AQ5G','S2AQ9','S2AQ10','S2AQ12B','S2AQ12F',
'S10Q1A3','S10Q1A16','S10Q1A20','S10Q1A22','S10Q1A25','S10Q1A32', 
'S10Q1A43','S10Q1A45','S10Q1A46','S10Q1A47','S10Q1A52','S10Q1A58',
'S11AQ1A1','S11AQ1A2', 'S11AQ1A14','S11AQ1A15', 'S11AQ1A22',
'S11AQ1A25','S11BQ1', 'CHLD0_17','DOBY',  'S1Q1D5', 'S1Q1G', 'MARITAL', 
'S1Q10A']

#read and convert to numeric
nesarc_data = pd.read_csv("nesarc_pds.csv", na_values=' ', usecols = ['IDNUM'] + respcols + expcols)
nesarc_data = nesarc_data.apply(lambda x: pd.to_numeric(x, errors='ignore'))

#evaluate the age
nesarc_data['AGE'] = nesarc_data['CYEAR'] - nesarc_data['DOBY']

#populate some default data to avoid record reduction due to n/a deletion later on
nesarc_data['BUILDTYP'].fillna(value=99, inplace=True)
nesarc_data['S1Q1G'].fillna(value=nesarc_data['AGE'], inplace=True)
nesarc_data['S2AQ5G'].fillna(value=11, inplace=True)
nesarc_data['S2AQ9'].fillna(value=11, inplace=True)
nesarc_data['S2AQ10'].fillna(value=11, inplace=True)
nesarc_data['S2AQ12B'].fillna(value=11, inplace=True)
nesarc_data['S2AQ12F'].fillna(value=11, inplace=True)
nesarc_data.fillna(value=2, inplace=True)

#drop the NaN records for now
nesarc_data = nesarc_data.dropna()

#we need to create the boolean response variable which indicates gambling tendency 

nesarc_data['GAMBLER'] = (nesarc_data[respcols] == 1).any(axis=1)

#Split into training and testing sets

predictors = nesarc_data.drop(['DOBY', 'CYEAR', 'IDNUM'] + respcols + ['GAMBLER'], 1)
targets = nesarc_data.GAMBLER

pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=0.4)

#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

print ('Confusion Matrix:', sklearn.metrics.confusion_matrix(tar_test,predictions))
print ('Accuracy Score is: ', sklearn.metrics.accuracy_score(tar_test, predictions))

# fit an Extra Trees model to the data
model = ExtraTreesClassifier(n_estimators=25)
model.fit(pred_train,tar_train)

print ('Variable Importance:\n')

# constuct dataframe to sort values so we can print out the variables
df = pd.DataFrame(data=model.feature_importances_, index=predictors.columns, columns=['score'])

df.sort_values(axis=0, ascending=False, by='score', inplace=True)

print (df)

"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)

