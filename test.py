import datetime

import numpy as np
import scipy.stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

dicct = {'a': [1]}

second = {'b': [4.0], 'c': ['k']}
dicct.update(second)
print(dicct)

d1 = {'a': [4], 'b': [4.5], 'c':['k']}
d2 = [4, 4.5, 'k']

from pandas import *

# dataset = DataFrame.from_dict(d1, orient='index')
# dataset.loc[len(dataset.index)] = d2
# print(dataset)


# df = DataFrame(d1)
# print(df)
# df.loc[1] = [4, 4.5, 'k']
# print(df)
#
# for row in df.values:
#     print(row)
#
# print(list(d1.values()))
#
# print(datetime.datetime.strptime('2018-01-01', "%Y-%m-%d").date().toordinal())
#
# print(datetime.date.min.toordinal(), datetime.date.max.toordinal())

print(np.arange(7))

data = read_csv("./resources/employments.csv", delimiter=';')

# Przeprowadzenie ANCOVA z kontrolą zmiennej kowariant
model = ols('rate_per_hour ~ C(sex)', data=data).fit()
# * C(country)* C(languages)* C(speciality)'
#             '* C(core_programming_language)* C(academic_title)* C(company_country)'
#             '* C(company_type)* C(work_form)* C(team_type)* C(form_of_employment)
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

