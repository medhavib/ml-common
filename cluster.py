# for more information on this, please take a look at http://turngeek.blogspot.com/2017/01/k-means-cluster-analysis-on-outlook-on.html

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

"""
Data Management
"""

evaluator = ['W1_B4']  
clustervars = ['W1_H1', 'W1_F6', 'PPINCIMP', 'W1_F4_D', 'W1_O5', 'W1_M5'] 

clustervars2 = ['W1_H1', 'W1_F6', 'W1_P2', 'W2_QL3', 'PPINCIMP', 
 'W1_K1_C', 'W1_F4_D', 'W1_L4_D', 
  'W1_K4', 'W1_L5_D', 'PPWORK',
  'W1_O5', 'W1_I2', 'W1_O1', 'W1_O3', 'W1_O4', 'W1_QA2', 
'W1_QA4A', 'W1_QB1', 'W1_QB2', 'W1_K1_B', 'W1_H8', 'W1_H5',
'W1_H6', 'W1_H7', 'W1_H4', 'W2_QK1', 'W1_M3',
'W1_M1', 'W1_M5', 'W1_M10', 'W1_M11', 'W1_M9', 'W1_M8', 'W1_M7', 'W1_M6', 'W1_M2',
'c', 'W1_E61_C']            

#'W1_L1_A', 'W1_L5_A', 'W1_P15', 'W1_L5_B','W1_F4_B', 'W1_P6', 'W1_M6', 'W1_J3A_C',
# 'PPGENDER','W1_F4_A', 'W1_M5', 'W1_I1', 'W1_K5', 'W1_P2'

data = pd.read_csv("ool_pds.csv", na_values=' ', usecols = clustervars + evaluator)

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)
data = data.apply(lambda x: pd.to_numeric(x, errors='ignore'))

# Data Management
data = data.dropna()
target = data.W1_B4

data = data.drop('W1_B4', 1)

# standardize clustering variables to have mean=0 and sd=1
clustervar = data.copy()
clustervar = preprocessing.scale(clustervar.astype('float64'))
clustervar = pd.DataFrame(data = clustervar, columns = data.columns)

# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

"""
merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
merged_train = clus_train.assign(cluster = Series(model3.labels_))

# cluster frequencies
print (merged_train.cluster.value_counts())

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)

merged_train_all = merged_train.merge(target.to_frame(), how= 'left', left_index=True, right_index=True)

sub1 = merged_train_all[['W1_B4', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

gpamod = smf.ols(formula='W1_B4 ~ C(cluster)', data=sub1).fit()
print (gpamod.summary())

print ('means for W1_B4 by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for W1_B4 by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['W1_B4'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())
