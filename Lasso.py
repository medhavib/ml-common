#details on this lasso regression can be found at: 
#http://turngeek.blogspot.com/2017/01/lasso-regression-for-us-national.html

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
 
respcols= ['S12Q1']
expcols = ['CYEAR', 'NUMPERS', 'NUMPER18',  'NUMREL18', 'S1Q1E',
'DGSTATUS','S9Q1A', 'SMOKER', 'S3AQ3B1', 'S3AQ3B2','S6Q1',
'S1Q9B','S2AQ5G','S2AQ9','S2AQ10','S2AQ12B','S2AQ12F',
'S10Q1A3','S10Q1A16','S10Q1A20','S10Q1A22','S10Q1A25','S10Q1A32', 
'S10Q1A43','S10Q1A45','S10Q1A46','S10Q1A47','S10Q1A52','S10Q1A58',
'S11AQ1A1','S11AQ1A2', 'S11AQ1A14','S11AQ1A15', 'S11AQ1A22',
'S11AQ1A25','S11BQ1', 'CHLD0_17','DOBY',  'S1Q1D5', 'S1Q1G', 
'S1Q10A', 'MARITAL']

nesarc_data = pd.read_csv("nesarc_pds.csv", na_values=' ', usecols = ['IDNUM'] + respcols + expcols)
nesarc_data = nesarc_data.apply(lambda x: pd.to_numeric(x, errors='ignore'))

#evaluate the age
nesarc_data['AGE'] = nesarc_data['CYEAR'] - nesarc_data['DOBY']

nesarc_data['S1Q1G'].fillna(value=nesarc_data['AGE'], inplace=True)
nesarc_data['S2AQ5G'].fillna(value=11, inplace=True)
nesarc_data['S2AQ9'].fillna(value=11, inplace=True)
nesarc_data['S2AQ10'].fillna(value=11, inplace=True)
nesarc_data['S2AQ12B'].fillna(value=11, inplace=True)
nesarc_data['S2AQ12F'].fillna(value=11, inplace=True)

nesarc_data.fillna(value=2, inplace=True)

#drop the NaN records for now
nesarc_data = nesarc_data.dropna()

#Split into training and testing sets

predvar = nesarc_data.drop(['DOBY', 'CYEAR', 'IDNUM'] + respcols, 1)
target = nesarc_data.S12Q1
 
# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()
from sklearn import preprocessing
predictors['NUMREL18']=preprocessing.scale(predictors['NUMREL18'].astype('float64'))
predictors['NUMPERS']=preprocessing.scale(predictors['NUMPERS'].astype('float64'))
predictors['NUMPER18']=preprocessing.scale(predictors['NUMPER18'].astype('float64'))
predictors['S1Q1E']=preprocessing.scale(predictors['S1Q1E'].astype('float64'))
predictors['DGSTATUS']=preprocessing.scale(predictors['DGSTATUS'].astype('float64'))
predictors['S9Q1A']=preprocessing.scale(predictors['S9Q1A'].astype('float64'))
predictors['SMOKER']=preprocessing.scale(predictors['SMOKER'].astype('float64'))
predictors['S3AQ3B1']=preprocessing.scale(predictors['S3AQ3B1'].astype('float64'))
predictors['S3AQ3B2']=preprocessing.scale(predictors['S3AQ3B2'].astype('float64'))
predictors['S6Q1']=preprocessing.scale(predictors['S6Q1'].astype('float64'))
predictors['S1Q9B']=preprocessing.scale(predictors['S1Q9B'].astype('float64'))
predictors['S2AQ5G']=preprocessing.scale(predictors['S2AQ5G'].astype('float64'))
predictors['S2AQ9']=preprocessing.scale(predictors['S2AQ9'].astype('float64'))
predictors['S2AQ10']=preprocessing.scale(predictors['S2AQ10'].astype('float64'))
predictors['S2AQ12B']=preprocessing.scale(predictors['S2AQ12B'].astype('float64'))
predictors['S2AQ12F']=preprocessing.scale(predictors['S2AQ12F'].astype('float64'))
predictors['S10Q1A3']=preprocessing.scale(predictors['S10Q1A3'].astype('float64'))
predictors['S10Q1A16']=preprocessing.scale(predictors['S10Q1A16'].astype('float64'))
predictors['S10Q1A20']=preprocessing.scale(predictors['S10Q1A20'].astype('float64'))
predictors['S10Q1A22']=preprocessing.scale(predictors['S10Q1A22'].astype('float64'))
predictors['S11AQ1A25']=preprocessing.scale(predictors['S11AQ1A25'].astype('float64'))
predictors['S11BQ1']=preprocessing.scale(predictors['S11BQ1'].astype('float64'))
predictors['CHLD0_17']=preprocessing.scale(predictors['CHLD0_17'].astype('float64'))
predictors['AGE']=preprocessing.scale(predictors['AGE'].astype('float64'))
predictors['S1Q1D5']=preprocessing.scale(predictors['S1Q1D5'].astype('float64'))
predictors['S1Q1G']=preprocessing.scale(predictors['S1Q1G'].astype('float64'))
predictors['S1Q10A']=preprocessing.scale(predictors['S1Q10A'].astype('float64'))
predictors['MARITAL']=preprocessing.scale(predictors['MARITAL'].astype('float64'))

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=45682)

# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# constuct dataframe to sort values so we can print out the variables
df = pd.DataFrame(data=model.coef_, index=predictors.columns, columns=['score'])

df['ascore'] = df.score.abs()

df.sort_values(axis=0, ascending=False, by='ascore', inplace=True)

print (df)

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')         

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
