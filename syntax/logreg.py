# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# read data from web
df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

df.head()

#   admit  gre   gpa  rank
#0      0  380  3.61     3
#1      1  660  3.67     3
#2      1  800  4.00     1
#3      1  640  3.19     4
#4      0  520  2.93     4

df.columns = ["admit","gre","gpa","prestige"]
print df.columns

print df.describe()
#            admit         gre         gpa   prestige
#count  400.000000  400.000000  400.000000  400.00000
#mean     0.317500  587.700000    3.389900    2.48500
#std      0.466087  115.516536    0.380567    0.94446
#min      0.000000  220.000000    2.260000    1.00000
#25%      0.000000  520.000000    3.130000    2.00000
#50%      0.000000  580.000000    3.395000    2.00000
#75%      1.000000  660.000000    3.670000    3.00000
#max      1.000000  800.000000    4.000000    4.00000

print df.std()
#admit         0.466087
#gre         115.516536
#gpa           0.380567
#prestige      0.944460

# frequency table with prestige and admittance
print pd.crosstab(df['admit'],df['prestige'],rownames=['admit'])

#prestige   1   2   3   4
#admit                   
#0         28  97  93  55
#1         33  54  28  12

df.hist()
pl.show()

# dummify ranks
dummy_ranks = pd.get_dummies(df['prestige'],prefix='prestige')
print dummy_ranks.head()

# clean data frame for regression
keep = ['admit','gre','gpa']
data = df[keep].join(dummy_ranks.ix[:,'prestige_2':])
print data.head()
# add intercept manually
data['intercept'] = 1.0

# performing regression
train_cols = data.columns[1:]

logit = sm.Logit(data['admit'],data[train_cols])

# fit model
result = logit.fit()
# print results
print result.summary()

# odds ratio
# results of 1 unit change for each variable
print np.exp(result.params)

# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%','97.5%','OR']
print np.exp(conf)

# use evenly spaced range of 10 values from min to max
gres = np.linspace(data['gre'].min(),data['gre'].max(),10)
print gres

gpas = np.linspace(data['gpa'].min(),data['gpa'].max(),10)
print gpas

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
# enumerate all possibilities
combos = pd.DataFrame(cartesian([gres,gpas,[1,2,3,4],[1.]]))
# recreate dummy variables 
combos.columns = ['gre','gpa','prestige','intercept']

dummy_ranks = pd.get_dummies(combos['prestige'],prefix='prestige')
dummy_ranks.columns = ['prestige_1','prestige_2','prestige_3','prestige_4']
#keep only what we need for predictions
cols_to_keep = ['gre','gpa','prestige','intercept']
combos = combos[cols_to_keep].join(dummy_ranks.ix[:,'prestige_2':])
# make predictions on enumerated set
combos['admit_pred'] = result.predict(combos[train_cols])
print combos.head()

def isolate_and_plot(variable):
    grouped = pd.pivot_table(combos,values=['admit_pred'],index=[variable,'prestige'],
                             aggfunc=np.mean)
    colors = 'rbgyrbgy'
    for col in combos.prestige.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0),plt_data['admit_pred'],
                color=colors[int(col)])
        
    pl.xlabel(variable)
    pl.ylabel("P(admit=1)")
    pl.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
    pl.title("Prob(admit=1) isolating " + variable + " and presitge")
    pl.show()
    
isolate_and_plot('gre')
isolate_and_plot('gpa')