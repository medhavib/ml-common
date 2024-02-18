#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 01:08:41 2017

@author: medhavi
For an explanation of this analysis, please visit: http://turngeek.blogspot.com/2017/01/analyzing-gapminder-data.html

"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

"""
Data Engineering and Analysis
"""
#Load the dataset

country_data = pd.read_csv("gapminder.csv")

country_data = country_data.apply(lambda x: x.str.strip()).replace('', np.nan)

data_clean = country_data.dropna()

data_clean = data_clean.apply(lambda x: pd.to_numeric(x, errors='ignore'))

# calculate the average alcohol consumption and then use that 
# as the dependent variable
data_clean['suicideper100thlevel'] = data_clean['suicideper100th'] >= data_clean['suicideper100th'].mean()

predictors = data_clean.drop(['suicideper100th', 'suicideper100thlevel', 'country', 'lifeexpectancy'], 1)
targets = data_clean.suicideper100thlevel

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.25)

classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())