# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:26:21 2017

@author: jaturman
"""

import pandas as pd
datafile = "PlantGrowth.csv"
data = pd.read_csv(datafile)

# boxplot
data.boxplot('weight',by='group',figsize=(12,8))

ctrl = data['weight'][data.group=='ctrl']
grps = pd.unique(data.group.values)
d_data = {grp:data['weight'][data.group==grp]for grp in grps}

k = len(pd.unique(data.group)) # number of conditions
N = len(data.values) # conditions times participants
n = data.groupby('group').size()[0] # participants in each condition

from scipy import stats

F,p = stats.f_oneway(d_data['ctrl'],d_data['trt1'],d_data['trt2'])

# use stats models
import statsmodels.api as sm
from statsmodels.formula.api import ols

mod = ols('weight~group',data=data).fit()
aov_table = sm.stats.anova_lm(mod,type=2)
print aov_table

esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
esq_sm







