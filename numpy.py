# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:18:32 2017

@author: jaturman
"""
import matplotlib as plt
import numpy as np
import pandas as pd

m = np.arange(5)
plt.plot(m) #requires pyplot

df = pd.read_csv("C:/Users/jaturman/Desktop/Practicum/python/train.csv")

df.head(10)
df.describe
df['Property_Area'].value_counts()

df['ApplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome')
df.boxplot(column='ApplicantIncome',by='Education')
df.boxplot(column='ApplicantIncome',by='Gender')
df['LoanAmount'].hist(bins=50)

temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],
                       aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print 'Frequency Table for Credit History'
print temp1
print '\nProbability of getting loan for each credit history class:'
print temp2

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('count of applicants')
ax1.set_title('applicants by Credit_History')
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind='bar')
ax2.set_xlabel('credit_history')
ax2.set_ylabel('probability of getting loan')
ax2.set_title('probability of getting loan by credit history')

temp3 = pd.crosstab(df['Credit_History'],df['Loan_Status'])
temp3.plot(kind='bar',stacked=True,color=['red','green'],grid=False)

# check missing values
df.apply(lambda x: sum(x.isnull()),axis=0)
# replace missing values with mean of each
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)

df.boxplot(column='LoanAmount',by=['Education','Self_Employed'])

df['Self_Employed'].value_counts()
# impute missing values as no since 86% are no anyway
df['Self_Employed'].fillna('No',inplace=True)

# create pivot table with median values for all groups of unique values for
# self employed and education features
table = df.pivot_table(values='LoanAmount',index='Self_Employed',
                       columns=['Education','Gender'],aggfunc=np.median)

print table

def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]

# replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage,axis=1),inplace=True)

df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

# building predictive model
from sklearn.preprocessing import LabelEncoder

# encode all categorical variables to numeric
var_mod = ['Gender','Married','Dependents','Education',
'Self_Employed','Property_Area','Loan_Status']

le = LabelEncoder()

for i in var_mod:
    df[i] = le.fit_transform(df[i])
    
df.dtypes

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

# function for classification model and performance
def classification_model(model,data,predictors,outcome):
    #fit model
    model.fit(data[predictors],data[outcome])
    # make predictions on training set
    predictions = model.predict(data[predictors])
    #print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print "Accuracy : %s" % "{0:.3%}".format(accuracy)
    #k-fold cross validation with 5 folds
    kf = KFold(data.shape[0],n_folds=5)
    error = []
    
    for train, test in kf:
        #filter training data
        train_predictors = (data[predictors].iloc[train,:])
        #target to train algorithm
        model.fit(train_predictors,train_target)
        #record error from cross validation
        error.append(model.score(data[predictors].iloc[test,:],data[outcome].iloc[test]))
    
    print "Cross-Validation score :%s" % "{0:.3%}".format(np.mean(error))
    # fit model again to refer outside of function
    model.fit(data[predictors],data[outcome])
    

outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model,df,predictor_var,outcome_var)    


    
    
