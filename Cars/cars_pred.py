# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:35:19 2017

use cars data to 

@author: jaturman
"""

import pandas as pd
import matplotlib.pyplot as pyp
import matplotlib as plt
import numpy as np
import scipy.stats as stats
import pylab
import seaborn as sns

cars = pd.read_csv('FuelEfficiency.csv')
cars.head()
cars.describe()

# qq plot for normality
stats.probplot(cars['MPG'],dist="norm",plot=pylab)

matshow(cars)

# correlation matrix to determine correlation to mpg
corr = cars
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# scatter matrix
pd.scatter_matrix(cars,alpha=.5,figsize=(14,8),diagonal='kde')
# to see skewness
sns.pairplot(cars)
# sns heatmap
f, ax = subplots(figsize=(10,8))
corr = cars
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),
            cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)

# first model
import statsmodels.formula.api as smf
# first degree model
lm1 = smf.ols(formula='MPG ~ 1 + DIS',data=cars).fit()
pyp.plot(cars.DIS,lm1.predict(cars),'b-',alpha=0.9)
# 2nd degree polynomial for fun
lm2 = smf.ols(formula='MPG ~ 1 + DIS + I(DIS ** 2.0)',data=cars).fit()
pyp.plot(cars.DIS,lm2.predict(cars),'g-',alpha=0.9)
# 3rd degree for more fun
lm3 = smf.ols(formula='MPG ~ 1 + DIS + I(DIS ** 2.0)+I(DIS ** 3.0)',data=cars).fit()
pyp.plot(cars.DIS,lm3.predict(cars),'r-',alpha=0.9)
pyp.legend

# 2nd degreee polynomial adj R2 = .687 !!!! winner !!!
lm1.summary()
lm2.summary()
lm3.summary()

# do a better version with train and test splits












